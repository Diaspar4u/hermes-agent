"""
Transport-agnostic WhatsApp behavior shared by the Baileys bridge adapter and the
Cloud API adapter: allow-list / DM / group gating, mention detection, quoted-reply-
to-bot detection, broadcast filtering, WhatsApp markdown conversion, chunk budgeting.

Mixin contract — the host adapter sets these on ``self`` before calling any mixin
method: ``config`` (PlatformConfig), ``name``, ``_dm_policy`` / ``_group_policy``
("open" | "allowlist" | "disabled"), ``_allow_from`` / ``_group_allow_from`` (set[str]),
``_mention_patterns`` (list[re.Pattern]), ``_reply_prefix`` (Optional[str]).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

try:  # pragma: no cover - platform-specific locking
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None
try:  # pragma: no cover - Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None

from gateway.platforms._shared import get_scoped_secret as _get_wsecret


logger = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes", "on"}
_OPTIN_TRUTHY = {"true", "1", "yes"}


def _stash(pattern: str, text: str, tag: str) -> tuple[str, list[str]]:
    """Replace every ``pattern`` match with a ``\\x00<tag><n>\\x00`` placeholder."""
    saved: list[str] = []

    def keep(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"\x00{tag}{len(saved) - 1}\x00"

    return re.sub(pattern, keep, text), saved


def _header_to_bold(m: re.Match) -> str:
    """``# Header`` → ``*Header*``, stripping already-bolded ``*...*`` so ``# **Title**``
    doesn't render with literal asterisks."""
    inner = m.group(1).strip()
    while len(inner) > 1 and inner.startswith("*") and inner.endswith("*"):
        inner = inner[1:-1].strip()
    return f"*{inner}*"


class WhatsAppBehaviorMixin:
    """Shared behavior for all WhatsApp adapters (Baileys + Cloud API); owns no state
    of its own — see the module docstring for the host adapter's attribute contract."""

    # Practical UX limit, not the ~65K protocol max (long messages are unreadable on mobile).
    MAX_MESSAGE_LENGTH: int = 4096
    supports_code_blocks = True  # WhatsApp renders fenced code blocks (monospace)

    DEFAULT_REPLY_PREFIX: str = "⚕ *Hermes Agent*\n────────────\n"

    _OUTBOUND_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u2060\u2063\ufeff]")
    _OUTBOUND_ODD_SPACE_RE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]")

    @classmethod
    def _sanitize_outbound_text(cls, content: str) -> str:
        """Strip zero-width format chars (WORD JOINER etc.) and normalize odd unicode
        spaces — WhatsApp renders them as mojibake prefixes. Emoji joiners are kept."""
        if not content:
            return content
        return cls._OUTBOUND_ODD_SPACE_RE.sub(" ", cls._OUTBOUND_INVISIBLE_CHARS_RE.sub("", content))

    @property
    def enforces_own_access_policy(self) -> bool:
        """WhatsApp gates DM/group access at intake via dm_policy/group_policy."""
        return True

    def _effective_reply_prefix(self) -> str:
        """Prefix for outgoing replies in self-chat mode (Cloud API overrides to ``""``)."""
        if (_get_wsecret("WHATSAPP_MODE", default="self-chat") or "self-chat") != "self-chat":
            return ""
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        env_prefix = _get_wsecret("WHATSAPP_REPLY_PREFIX")
        if env_prefix is not None:
            return env_prefix.replace("\\n", "\n")
        return self.DEFAULT_REPLY_PREFIX

    def _outgoing_chunk_limit(self) -> int:
        """Reserve room for the reply prefix; floor keeps space for pagination/fence repair."""
        return max(1024, self.MAX_MESSAGE_LENGTH - len(self._effective_reply_prefix()))

    def _whatsapp_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is None:
            configured = _get_wsecret("WHATSAPP_REQUIRE_MENTION", default="false") or "false"
        if isinstance(configured, str):
            return configured.lower() in _TRUTHY
        return bool(configured)

    def _whatsapp_free_response_chats(self) -> set[str]:
        raw = self.config.extra.get("free_response_chats")
        if raw is None:
            raw = _get_wsecret("WHATSAPP_FREE_RESPONSE_CHATS", default="") or ""
        return self._coerce_allow_list(raw)

    @staticmethod
    def _coerce_allow_list(raw) -> set[str]:
        """Parse allow_from / group_allow_from from config (list) or env var (CSV)."""
        if raw is None:
            return set()
        parts = raw if isinstance(raw, list) else str(raw).split(",")
        return {str(part).strip() for part in parts if str(part).strip()}

    def _select_dm_allowlist(self, extra: Dict[str, Any], env_keys, read_env) -> Any:
        """Pick the raw DM allowlist by key *presence*: ``allow_from``/``allowFrom`` in config (an
        explicit empty list stays authoritative), then the first truthy env carrier. Records the
        winning source in ``_dm_allowlist_source`` so live DM checks keep the same precedence."""
        for key in ("allow_from", "allowFrom"):
            if key in extra:
                self._dm_allowlist_source = "config"
                return extra.get(key)
        for env in env_keys:
            if read_env(env):
                self._dm_allowlist_source = env
                return read_env(env)
        self._dm_allowlist_source = None
        return None

    def _live_dm_allow_from(self) -> set[str]:
        """Allowlist currently enforced for DM intake / strict DM auth. Env-seeded adapters re-read
        the same key so pairing approve/revoke takes effect without restart; a removed key (sole-entry
        revoke) means empty, not the construction snapshot. Config-seeded adapters keep the in-memory
        set (pairing revoke purges it in place) — a stale env value must not broaden access."""
        source = getattr(self, "_dm_allowlist_source", None)
        if isinstance(source, str) and source != "config":
            return self._coerce_allow_list(os.environ[source]) if source in os.environ else set()
        return set(self._allow_from or ())

    # ------------------------------------------------------------------ JID helpers
    @staticmethod
    def _normalize_whatsapp_id(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = str(value).strip()
        if ":" in normalized and "@" in normalized:
            normalized = normalized.replace(":", "@", 1)
        return normalized

    @staticmethod
    def _is_broadcast_chat(chat_id: str) -> bool:
        """Status updates (Stories) and Channel/Newsletter broadcasts — never reply
        (answering a Story spams the status feed; Channel posts aren't addressable)."""
        cid = (chat_id or "").strip().lower()
        return cid == "status@broadcast" or cid.endswith(("@broadcast", "@newsletter"))

    # ------------------------------------------------------------------ gating
    def _open_dm_opted_in(self) -> bool:
        if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in _OPTIN_TRUTHY:
            return True
        return (_get_wsecret("WHATSAPP_ALLOW_ALL_USERS", default="") or "").lower() in _OPTIN_TRUTHY

    @staticmethod
    def _matches_whatsapp_allowlist(candidate: str, allow_from) -> bool:
        """Match a WhatsApp identifier against an allowlist across phone/LID forms. Inbound senders
        arrive as ``<id>@lid`` while allowlists hold phone numbers (or vice versa), so resolve both
        sides through the bridge's lid-mapping files via ``gateway.whatsapp_identity``."""
        if not allow_from:
            return False
        if candidate in allow_from:
            return True
        from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier
        candidate_aliases = expand_whatsapp_aliases(candidate)
        if not candidate_aliases:
            return False
        return any(
            entry == "*"
            or normalize_whatsapp_identifier(entry) in candidate_aliases
            or expand_whatsapp_aliases(entry) & candidate_aliases
            for entry in allow_from
        )

    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Strict DM authorization — pairing does not imply access."""
        if self._dm_policy == "allowlist":
            return self._matches_whatsapp_allowlist(sender_id, self._live_dm_allow_from())
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        """Whether a DM may reach the gateway intake (pairing handshake path)."""
        principal = str(sender_id or "").strip()
        if not principal:
            return False
        if self._dm_policy == "allowlist":
            return self._matches_whatsapp_allowlist(principal, self._live_dm_allow_from())
        if self._dm_policy == "pairing":
            return True
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_group_allowed(self, chat_id: str) -> bool:
        """Check whether a group chat should be processed."""
        if self._group_policy == "allowlist":
            return self._matches_whatsapp_allowlist(chat_id, self._group_allow_from)
        return self._group_policy == "open"

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = (_get_wsecret("WHATSAPP_MENTION_PATTERNS", default="") or "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    # Plain text: one pattern per line, else comma-separated.
                    patterns = [p.strip() for p in raw.splitlines() if p.strip()]
                    patterns = patterns or [p.strip() for p in raw.split(",") if p.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning("[%s] whatsapp mention_patterns must be a list or string; got %s", self.name, type(patterns).__name__)
            return []
        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid WhatsApp mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d WhatsApp mention pattern(s)", self.name, len(compiled))
        return compiled

    def _bot_ids_from_message(self, data: Dict[str, Any]) -> set[str]:
        return {nid for c in (data.get("botIds") or []) if (nid := self._normalize_whatsapp_id(c))}

    def _message_is_reply_to_bot(self, data: Dict[str, Any]) -> bool:
        quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
        return bool(quoted_participant) and quoted_participant in self._bot_ids_from_message(data)

    def _message_mentions_bot(self, data: Dict[str, Any]) -> bool:
        bot_ids = self._bot_ids_from_message(data)
        if not bot_ids:
            return False
        mentioned = {nid for c in (data.get("mentionedIds") or []) if (nid := self._normalize_whatsapp_id(c))}
        if mentioned & bot_ids:
            return True
        lower_body = str(data.get("body") or "").lower()
        return any(
            bare and (f"@{bare}" in lower_body or bare in lower_body)
            for bare in (bot_id.split("@", 1)[0].lower() for bot_id in bot_ids)
        )

    def _message_matches_mention_patterns(self, data: Dict[str, Any]) -> bool:
        body = str(data.get("body") or "")
        return any(pattern.search(body) for pattern in self._mention_patterns or ())

    def _clean_bot_mention_text(self, text: str, data: Dict[str, Any]) -> str:
        if not text:
            return text
        cleaned = text
        for bot_id in self._bot_ids_from_message(data):
            bare_id = bot_id.split("@", 1)[0]
            if bare_id:
                cleaned = re.sub(rf"@{re.escape(bare_id)}\b[,:\-]*\s*", "", cleaned)
        return cleaned.strip() or text

    def _should_process_message(self, data: Dict[str, Any]) -> bool:
        chat_id = str(data.get("chatId") or "")
        # Broadcast pseudo-chats are filtered even in self-chat mode (fromMe events).
        if self._is_broadcast_chat(chat_id):
            return False
        if not data.get("isGroup", False):
            # DMs that pass the policy gate are always processed
            return self._is_dm_intake_allowed(str(data.get("senderId") or data.get("from") or ""))
        if not self._is_group_allowed(chat_id):
            return False
        # Group messages: check mention / free-response settings
        if chat_id in self._whatsapp_free_response_chats() or not self._whatsapp_require_mention():
            return True
        return (
            str(data.get("body") or "").strip().startswith("/")
            or self._message_is_reply_to_bot(data)
            or self._message_mentions_bot(data)
            or self._message_matches_mention_patterns(data)
        )

    # ------------------------------------------------------------------ formatting
    def format_message(self, content: str) -> str:
        """Convert markdown to WhatsApp syntax (*bold*, _italic_, ~strike~); fenced and
        inline code are protected via placeholder substitution."""
        if not content:
            return content
        result, fences = _stash(r"```[\s\S]*?```", self._sanitize_outbound_text(content), "FENCE")
        result, codes = _stash(r"`[^`\n]+`", result, "CODE")
        # Italic *text* → _text_ BEFORE bold so **bold** doesn't become italic;
        # lookarounds skip list bullets and bold delimiters.
        result = re.sub(r"(?<!\*)\*(?!\s|\*)([^*\n]*?\S[^*\n]*?)\*(?!\*)", r"_\1_", result)
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
        result = re.sub(r"__(.+?)__", r"*\1*", result)
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)
        result = re.sub(r"^#{1,6}\s+(.+)$", _header_to_bold, result, flags=re.MULTILINE)
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)  # [text](url) → text (url)
        for tag, saved in (("FENCE", fences), ("CODE", codes)):
            for i, original in enumerate(saved):
                result = result.replace(f"\x00{tag}{i}\x00", original)
        return result


_WHATSAPP_DEPENDENCY_STAMP = ".hermes-pkg-hash"


def whatsapp_bridge_dependency_fingerprint(bridge_dir: Path) -> str:
    """Return a stable fingerprint for the checked-in bridge manifests."""
    digest = hashlib.sha256()
    for name in ("package.json", "package-lock.json"):
        path = bridge_dir / name
        try:
            content = path.read_bytes()
        except OSError:
            return ""
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def whatsapp_bridge_dependencies_fresh(bridge_dir: Path) -> bool:
    """Return whether installed bridge dependencies match both manifests."""
    node_modules = bridge_dir / "node_modules"
    fingerprint = whatsapp_bridge_dependency_fingerprint(bridge_dir)
    if not node_modules.is_dir() or not fingerprint:
        return False
    try:
        recorded = (node_modules / _WHATSAPP_DEPENDENCY_STAMP).read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return False
    return recorded == fingerprint


def record_whatsapp_bridge_dependency_fingerprint(bridge_dir: Path) -> bool:
    """Stamp a successful explicit install with its manifest fingerprint."""
    fingerprint = whatsapp_bridge_dependency_fingerprint(bridge_dir)
    if not fingerprint:
        return False
    try:
        (bridge_dir / "node_modules" / _WHATSAPP_DEPENDENCY_STAMP).write_text(
            fingerprint, encoding="utf-8"
        )
    except OSError:
        return False
    return True


_WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS: Optional[float] = None
_WHATSAPP_BRIDGE_LOCK_POLL_SECONDS = 0.05


class WhatsAppBridgeDependencyError(RuntimeError):
    """A deterministic WhatsApp dependency transaction could not complete."""


class WhatsAppBridgeBusyError(WhatsAppBridgeDependencyError):
    """Another process retained the bounded bridge transaction lock."""


class WhatsAppBridgeUnavailableError(WhatsAppBridgeDependencyError):
    """A required local executable is unavailable without a PATH fallback."""


def _is_windows_reparse_point(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _canonical_bridge_identity(bridge_dir: Path) -> str:
    """Canonical location for bridge I/O, not a physical lock identity."""
    return os.path.normcase(os.path.realpath(os.fspath(bridge_dir)))


def _whatsapp_bridge_transaction_lock_path(bridge_dir: Path) -> Path:
    """One retained native-user lock, independent of profiles and target changes."""
    from hermes_constants import _get_platform_default_hermes_home

    return (
        _get_platform_default_hermes_home()
        / ".whatsapp-bridge-locks" / "transaction.lock"
    )


def _validate_private_lock_root(lock_root: Path) -> None:
    """Create a 0700 real directory and reject aliases/reparse points."""
    try:
        os.makedirs(lock_root, mode=0o700, exist_ok=True)
        metadata = lock_root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or _is_windows_reparse_point(metadata)
        ):
            raise OSError(errno.ELOOP, "lock root is a symlink or reparse point")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise OSError(errno.EACCES, "lock root is not owned by the current user")
        if os.name != "nt":
            os.chmod(lock_root, 0o700)
            metadata = lock_root.lstat()
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise OSError(errno.EACCES, "lock root is accessible by other users")
    except OSError as exc:
        detail = _bounded_redacted_dependency_output(str(exc))
        raise WhatsAppBridgeDependencyError(
            f"Could not secure the WhatsApp bridge lock directory: {detail}"
        ) from exc


def _validate_lock_file_metadata(
    path_metadata: os.stat_result,
    descriptor_metadata: os.stat_result,
) -> None:
    if (
        not stat.S_ISREG(path_metadata.st_mode)
        or not stat.S_ISREG(descriptor_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or _is_windows_reparse_point(path_metadata)
        or _is_windows_reparse_point(descriptor_metadata)
        or not _same_file_identity(path_metadata, descriptor_metadata)
    ):
        raise OSError(errno.ELOOP, "lock path is not the opened regular file")
    if hasattr(os, "getuid") and descriptor_metadata.st_uid != os.getuid():
        raise OSError(errno.EACCES, "lock file is not owned by the current user")
    if os.name != "nt" and stat.S_IMODE(descriptor_metadata.st_mode) & 0o077:
        raise OSError(errno.EACCES, "lock file is accessible by other users")


def _secure_open_whatsapp_bridge_lock(lock_path: Path) -> BinaryIO:
    """Open the stable lock inode without following attacker-planted links."""
    _validate_private_lock_root(lock_path.parent)
    before: Optional[os.stat_result]
    try:
        before = lock_path.lstat()
    except FileNotFoundError:
        before = None
    if before is not None and (
        stat.S_ISLNK(before.st_mode) or _is_windows_reparse_point(before)
    ):
        raise WhatsAppBridgeDependencyError(
            "Could not open the WhatsApp bridge transaction lock: the lock "
            "path is a symlink or reparse point."
        )

    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        path_metadata = lock_path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        _validate_lock_file_metadata(path_metadata, descriptor_metadata)
        if before is not None and not _same_file_identity(before, path_metadata):
            raise OSError(errno.EAGAIN, "lock path changed while it was opened")
        return os.fdopen(descriptor, "r+b", buffering=0)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        detail = _bounded_redacted_dependency_output(str(exc))
        raise WhatsAppBridgeDependencyError(
            f"Could not open the WhatsApp bridge transaction lock: {detail}"
        ) from exc


def _uses_windows_file_locking() -> bool:
    return os.name == "nt"


def _try_acquire_whatsapp_bridge_file_lock(lock_file) -> None:
    """Attempt one non-blocking, portable advisory lock acquisition."""
    lock_file.seek(0)
    if _uses_windows_file_locking():
        if _msvcrt is None:
            raise OSError(errno.ENOSYS, "Windows file locking is unavailable")
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_NBLCK, 1)
    else:
        if _fcntl is None:
            raise OSError(errno.ENOSYS, "POSIX file locking is unavailable")
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)


def _release_whatsapp_bridge_file_lock(lock_file) -> None:
    lock_file.seek(0)
    if _uses_windows_file_locking():
        if _msvcrt is None:
            raise OSError(errno.ENOSYS, "Windows file locking is unavailable")
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
    else:
        if _fcntl is None:
            raise OSError(errno.ENOSYS, "POSIX file locking is unavailable")
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)


def _whatsapp_bridge_lock_is_busy(exc: OSError) -> bool:
    return (
        isinstance(exc, BlockingIOError)
        or exc.errno in {errno.EACCES, errno.EAGAIN}
        or getattr(exc, "winerror", None) in {33, 36}
    )


@contextmanager
def _exclusive_whatsapp_bridge_transaction(
    bridge_dir: Path,
    *,
    timeout: Optional[float] = None,
):
    """Fail-closed per-user maintenance lock; yield the validated target path.

    All explicit bridge maintenance for this native OS home serializes, including
    different profiles and targets. Retain the lock outside replaceable roots:
    unlinking it lets new callers lock a different inode than an existing owner.
    Cooperating callers must share the native home; this is not cross-user locking.
    """
    from utils import env_int

    lock_path = _whatsapp_bridge_transaction_lock_path(bridge_dir)
    if timeout is None:
        configured_timeout = _WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS
        timeout = (
            configured_timeout
            if configured_timeout is not None
            else max(30, env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300) + 30)
        )
    try:
        timeout = max(0.0, float(timeout))
    except (TypeError, ValueError, OverflowError) as exc:
        raise WhatsAppBridgeDependencyError(
            "WhatsApp bridge lock timeout must be a finite number."
        ) from exc
    if timeout != timeout or timeout == float("inf"):
        raise WhatsAppBridgeDependencyError(
            "WhatsApp bridge lock timeout must be bounded."
        ) from None

    lock_file = _secure_open_whatsapp_bridge_lock(lock_path)
    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                # A peer can lock its first byte before our initialization write.
                # Windows mandatory write/flush conflicts use the same bounded retry.
                if os.fstat(lock_file.fileno()).st_size == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                _try_acquire_whatsapp_bridge_file_lock(lock_file)
                acquired = True
                break
            except OSError as exc:
                if not _whatsapp_bridge_lock_is_busy(exc):
                    detail = _bounded_redacted_dependency_output(str(exc))
                    raise WhatsAppBridgeDependencyError(
                        f"Could not lock the WhatsApp bridge dependencies: {detail}"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WhatsAppBridgeBusyError(
                        "Timed out waiting for another Hermes process to finish "
                        "the WhatsApp bridge transaction."
                    ) from exc
                time.sleep(min(_WHATSAPP_BRIDGE_LOCK_POLL_SECONDS, remaining))
        # Resolve and validate only after acquisition: the previous owner may
        # replace the bridge root, or create a previously absent mirror.
        canonical = Path(_canonical_bridge_identity(bridge_dir))
        try:
            metadata = canonical.lstat()
        except FileNotFoundError:
            pass  # The shared owner also covers creation of an absent target.
        else:
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or _is_windows_reparse_point(metadata)
                or metadata.st_dev < 0
                or metadata.st_ino <= 0
            ):
                raise WhatsAppBridgeDependencyError(
                    "WhatsApp bridge target has no usable physical directory identity."
                )
        yield canonical
    finally:
        cleanup_errors: list[str] = []
        if acquired:
            try:
                _release_whatsapp_bridge_file_lock(lock_file)
            except OSError as exc:
                cleanup_errors.append(str(exc))
        try:
            lock_file.close()
        except OSError as exc:
            cleanup_errors.append(str(exc))
        if cleanup_errors:
            detail = _bounded_redacted_dependency_output("; ".join(cleanup_errors))
            logger.warning(
                "[whatsapp] WhatsApp bridge lock release needed descriptor-close "
                "fallback: %s",
                detail,
            )


def _bounded_redacted_dependency_output(output: str) -> str:
    """Return a short diagnostic without credentials or registry URLs."""
    text = str(output or "").strip()
    if not text:
        return "no output"
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    text = re.sub(r"(?i)\bhttps?://\S+", "<redacted-url>", text)
    bounded = "\n".join(text.splitlines()[-10:])
    if len(bounded) > 1200:
        bounded = bounded[-1200:]
    return bounded or "no output"


def _path_exists_without_following(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_path_without_following(path: Path) -> None:
    if not _path_exists_without_following(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _minimal_whatsapp_npm_environment(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Keep npm launch/build and transport settings without unrelated secrets."""
    from hermes_constants import get_hermes_home, with_hermes_node_path

    allowed = {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PYTHON",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
    }
    transport_allowed = {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "NODE_EXTRA_CA_CERTS",
        "NPM_CONFIG_CA",
        "NPM_CONFIG_CACHE",
        "NPM_CONFIG_CAFILE",
        "NPM_CONFIG_HTTPS_PROXY",
        "NPM_CONFIG_NOPROXY",
        "NPM_CONFIG_PROXY",
        "NPM_CONFIG_REGISTRY",
        "NPM_CONFIG_STRICT_SSL",
        "NPM_CONFIG_USERCONFIG",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
    # Preserve proxy/npm spelling and coexistence so their own precedence applies.
    base = {
        (key if key.upper() in transport_allowed else key.upper()): value
        for key, value in (os.environ if env is None else env).items()
        if key.upper() in allowed or key.upper() in transport_allowed
    }
    # Match the native npm lifecycle's profile npmrc fallback, but honor every
    # explicit userconfig spelling (including an intentionally empty value).
    if not any(key.upper() == "NPM_CONFIG_USERCONFIG" for key in base):
        npmrc = get_hermes_home() / "npmrc"
        if npmrc.is_file():
            base["NPM_CONFIG_USERCONFIG"] = os.fspath(npmrc)
    return with_hermes_node_path(base)


def _ensure_whatsapp_bridge_dependencies(
    bridge_dir: Path, *, npm: Optional[str], env: Optional[dict[str, str]]
) -> bool:
    """Implement a staged node_modules replacement under the caller's lock."""
    from hermes_constants import find_node_executable
    from hermes_cli._subprocess_compat import windows_hide_flags
    from utils import env_int

    bridge_dir = Path(bridge_dir)
    node_modules = bridge_dir / "node_modules"
    fingerprint = whatsapp_bridge_dependency_fingerprint(bridge_dir)
    if not fingerprint:
        raise WhatsAppBridgeDependencyError(
            "WhatsApp dependency manifests are missing or unreadable."
        )
    if whatsapp_bridge_dependencies_fresh(bridge_dir):
        return False

    staging: Optional[Path] = None
    backup: Optional[Path] = None
    committed = False
    preserve_staging = False
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=".node_modules.staging-", dir=bridge_dir)
        )
        for name in ("package.json", "package-lock.json"):
            shutil.copy2(bridge_dir / name, staging / name)

        npm_bin = npm if npm is not None else find_node_executable("npm")
        if not npm_bin:
            raise WhatsAppBridgeUnavailableError(
                "npm is unavailable; install or repair Node.js before preparing "
                "WhatsApp dependencies."
            )
        npm_install_timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
        try:
            install_result = subprocess.run(
                [npm_bin, "ci", "--silent"],
                cwd=str(staging),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=windows_hide_flags(),
                timeout=npm_install_timeout,
                env=_minimal_whatsapp_npm_environment(env),
            )
        except subprocess.TimeoutExpired as exc:
            raise WhatsAppBridgeDependencyError(
                "WhatsApp dependency installation timed out."
            ) from exc
        except OSError as exc:
            detail = _bounded_redacted_dependency_output(str(exc))
            raise WhatsAppBridgeDependencyError(
                f"Could not run npm ci for WhatsApp: {detail}"
            ) from exc

        if install_result.returncode != 0:
            detail = _bounded_redacted_dependency_output(
                install_result.stderr or install_result.stdout or ""
            )
            raise WhatsAppBridgeDependencyError(
                f"npm ci failed for WhatsApp dependencies: {detail}"
            )

        staged_modules = staging / "node_modules"
        if not staged_modules.is_dir():
            raise WhatsAppBridgeDependencyError(
                "npm ci succeeded without creating node_modules."
            )
        if whatsapp_bridge_dependency_fingerprint(staging) != fingerprint:
            raise WhatsAppBridgeDependencyError(
                "Dependency manifests changed while npm ci was running."
            )
        if whatsapp_bridge_dependency_fingerprint(bridge_dir) != fingerprint:
            raise WhatsAppBridgeDependencyError(
                "Live dependency manifests changed while npm ci was running."
            )
        (staged_modules / _WHATSAPP_DEPENDENCY_STAMP).write_text(
            fingerprint, encoding="utf-8"
        )

        rejected_modules = staging / ".rejected-node_modules"

        def rollback_activation() -> None:
            nonlocal backup, committed
            committed = False
            rollback_errors: list[str] = []
            # A rename may succeed and then raise a control exception before a
            # Python flag is assigned. Reserved paths tell us which moves ran.
            if (
                not _path_exists_without_following(staged_modules)
                and _path_exists_without_following(node_modules)
            ):
                try:
                    os.replace(node_modules, rejected_modules)
                except BaseException as rollback_error:
                    rollback_errors.append(
                        "could not quarantine the promoted tree: "
                        + _bounded_redacted_dependency_output(str(rollback_error))
                    )
            if (
                backup is not None
                and _path_exists_without_following(backup)
                and not _path_exists_without_following(node_modules)
            ):
                try:
                    os.replace(backup, node_modules)
                    backup = None
                except BaseException as rollback_error:
                    rollback_errors.append(
                        "could not restore the prior tree: "
                        + _bounded_redacted_dependency_output(str(rollback_error))
                    )
            if rollback_errors:
                raise WhatsAppBridgeDependencyError("; ".join(rollback_errors))

        try:
            if _path_exists_without_following(node_modules):
                backup = bridge_dir / f".node_modules.backup-{uuid.uuid4().hex}"
                os.replace(node_modules, backup)
            os.replace(staged_modules, node_modules)
            if (
                whatsapp_bridge_dependency_fingerprint(staging) != fingerprint
                or whatsapp_bridge_dependency_fingerprint(bridge_dir) != fingerprint
                or not whatsapp_bridge_dependencies_fresh(bridge_dir)
            ):
                raise WhatsAppBridgeDependencyError(
                    "Dependency manifests or their fingerprint changed after promotion"
                )
            committed = True
        except BaseException as activation_error:
            reason = (
                "Could not activate WhatsApp dependencies "
                f"({_bounded_redacted_dependency_output(str(activation_error))});"
            )
            try:
                rollback_activation()
            except BaseException as rollback_failure:
                preserve_staging = True
                detail = _bounded_redacted_dependency_output(str(rollback_failure))
                recovery_detail = _bounded_redacted_dependency_output(
                    ", ".join(str(path) for path in (backup, staging) if path is not None)
                )
                diagnostic = (
                    f"{reason} Rollback also failed ({detail}). "
                    f"Recovery data was preserved at: {recovery_detail}."
                )
                if isinstance(activation_error, Exception):
                    raise WhatsAppBridgeDependencyError(diagnostic) from activation_error
                activation_error.add_note(diagnostic)
            else:
                if isinstance(activation_error, Exception):
                    raise WhatsAppBridgeDependencyError(
                        f"{reason} the prior dependency state was restored."
                    ) from activation_error
            # Preserve KeyboardInterrupt/SystemExit identity and exit semantics.
            raise

        if backup is not None and _path_exists_without_following(backup):
            try:
                _remove_path_without_following(backup)
            except Exception as cleanup_error:
                detail = _bounded_redacted_dependency_output(str(cleanup_error))
                logger.warning(
                    "[whatsapp] Installed dependencies but could not remove "
                    "the old node_modules backup at %s: %s",
                    backup,
                    detail,
                )
            else:
                backup = None
        return True
    except WhatsAppBridgeDependencyError:
        raise
    except Exception as exc:
        detail = _bounded_redacted_dependency_output(str(exc))
        raise WhatsAppBridgeDependencyError(
            f"WhatsApp dependency transaction failed: {detail}"
        ) from exc
    finally:
        if (
            staging is not None
            and _path_exists_without_following(staging)
            and not preserve_staging
        ):
            try:
                _remove_path_without_following(staging)
            except Exception as cleanup_error:
                # Cleanup is subordinate to the transaction outcome. Before
                # activation an already-raised DependencyError remains primary;
                # after activation cleanup cannot turn committed success into a
                # new, untyped failure.
                detail = _bounded_redacted_dependency_output(str(cleanup_error))
                phase = "after activation" if committed else "after failure"
                logger.warning(
                    "[whatsapp] Could not remove dependency staging %s: %s",
                    phase,
                    detail,
                )


def ensure_whatsapp_bridge_dependencies(
    bridge_dir: Path, *, npm: Optional[str] = None, env: Optional[dict[str, str]] = None
) -> bool:
    """Explicit maintenance: True when replaced, False when already fresh.

    A stable native-user lock covers freshness through verified promotion. Caught
    failures restore the prior dependency tree; failed rollback retains recovery
    data. This is not a power-loss journal or atomic visibility to runtime readers.
    The updater may supply its resolved npm and build environment; only launch,
    home, temporary/cache, npm transport settings and node-gyp's PYTHON reach npm.
    """
    try:
        with _exclusive_whatsapp_bridge_transaction(bridge_dir) as canonical:
            return _ensure_whatsapp_bridge_dependencies(canonical, npm=npm, env=env)
    except WhatsAppBridgeDependencyError:
        raise
    except Exception as exc:
        detail = _bounded_redacted_dependency_output(str(exc))
        raise WhatsAppBridgeDependencyError(
            f"WhatsApp dependency transaction failed: {detail}"
        ) from exc


def resolve_whatsapp_bridge_dir() -> Path:
    """Bridge directory for CLI and adapter. A read-only install tree (e.g. Docker
    /opt/hermes) is mirrored to HERMES_HOME so npm install works."""
    import shutil
    from hermes_constants import get_hermes_home
    install_bridge = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"
    hermes_home_bridge = get_hermes_home() / "scripts" / "whatsapp-bridge"
    try:
        (install_bridge / ".write_test").touch()
        (install_bridge / ".write_test").unlink()
        return install_bridge
    except OSError:
        pass
    if hermes_home_bridge.exists():
        return hermes_home_bridge
    try:
        hermes_home_bridge.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(install_bridge, hermes_home_bridge, dirs_exist_ok=False)
        return hermes_home_bridge
    except Exception:
        return install_bridge
