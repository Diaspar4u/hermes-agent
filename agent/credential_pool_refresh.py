"""Owner-serialized refresh for independently persisted Codex pool grants."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import CredentialPool, PooledCredential


def sync_codex_pool_entry(pool: CredentialPool, entry: PooledCredential) -> PooledCredential:
    """Read the exact owning row without the best-effort global-store read cache."""
    from agent import credential_pool as cp

    owner = None
    if not cp._profile_owns_pool_provider(pool.provider):
        owner = cp._borrowed_single_use_pool_root()
        if owner is None:
            raise RuntimeError("Cannot resolve the owner of the borrowed Codex pool")
    rows = cp._load_auth_store(owner).get("credential_pool", {}).get(pool.provider, [])
    row = next((row for row in rows if isinstance(row, dict) and row.get("id") == entry.id), None)
    if row is None:
        if entry.id in pool._persisted_token_pairs:
            raise RuntimeError("Codex pool row was removed; reload the credential pool")
        return entry  # Unsaved entries have no persisted generation yet.
    pair = cp.auth_mod._credential_token_pair(row)
    if not any(pair):
        raise RuntimeError("Codex pool row has no token generation; reload the credential pool")
    if entry.last_error_reason == cp.CREDENTIAL_PERSIST_FAILED_REASON and pair == pool._persisted_token_pairs.get(entry.id):
        raise RuntimeError("Codex refresh rotation was not durably persisted")
    pool._persisted_token_pairs[entry.id] = pair
    if pair != (entry.access_token, entry.refresh_token):
        stored = cp.PooledCredential.from_dict(pool.provider, row)
        pool._replace_entry(entry, stored)
        return stored
    return entry


def refresh_codex_pool_entry(
    pool: CredentialPool, entry: PooledCredential, *, force: bool, failed_key: str | None,
    quota_probe: bool = False,
) -> PooledCredential | None:
    # Late import keeps the facade's persistence/lock seams authoritative.
    from agent import credential_pool as cp

    current = pool._find(lambda candidate: candidate.id == entry.id)
    if current is not None and current.last_error_reason == cp.CREDENTIAL_PERSIST_FAILED_REASON:
        raise RuntimeError("Codex refresh rotation was not durably persisted")
    timeout = pool._single_use_refresh_lock_timeout()
    with cp._auth_store_lock(timeout_seconds=timeout), ExitStack() as locks:
        owns = cp._profile_owns_pool_provider(pool.provider)
        if owns and entry.id in pool._borrowed_root_ids:
            raise RuntimeError("Codex pool ownership changed; reload the credential pool")
        if not owns:
            owner = cp._borrowed_single_use_pool_root()
            if owner is None:
                raise RuntimeError("Cannot resolve the owner of the borrowed Codex pool")
            # Active -> root matches persistence's lock order. Holding only the active
            # profile lock lets root and sibling profiles spend the same grant concurrently.
            locks.enter_context(cp._auth_store_lock(timeout_seconds=timeout, target_path=owner))
            cp.auth_mod._global_auth_store_cache = None
        synced = sync_codex_pool_entry(pool, entry)
        synced = pool._sync_entry_from_auth_store(synced)
        try:
            if quota_probe:
                fresh = cp.auth_mod._refresh_expired_codex_probe_token(synced.access_token, synced.refresh_token)
                if fresh:
                    synced = pool._adopt(synced, **fresh)
                    pool._sync_device_code_entry_to_auth_store(synced)
                return synced
            changed = (synced.access_token, synced.refresh_token) != (entry.access_token, entry.refresh_token)
            if changed and not pool._entry_needs_refresh(synced) and (
                not force or synced.runtime_api_key != (failed_key or entry.runtime_api_key)
            ):
                return synced
            return pool._refresh_entry_impl(synced, force=force, failed_key=failed_key)
        except OSError:
            # Do not expose a rotated-but-uncommitted pair as a successful refresh.
            current = pool._find(lambda candidate: candidate.id == entry.id)
            if current is not None:
                pool._replace_entry(current, replace(
                    current, last_status=cp.STATUS_DEAD, last_error_reason=cp.CREDENTIAL_PERSIST_FAILED_REASON))
            raise
