"""Exact completion retries reuse the queued admission without folding into themselves."""

import pytest

from hermes_state_runtime import list_session_admissions
from tests.gateway.test_automation_notification_category import _native_owner


@pytest.mark.asyncio
async def test_retried_completion_preserves_its_queued_payload(tmp_path, monkeypatch):
    runner, authority, _adapter, _source, metadata = await _native_owner(
        tmp_path, monkeypatch, 'human-1')
    try:
        owner = authority.logical_owner(metadata['gateway_session_id'])
        event = {
            'type': 'completion', 'session_id': 'process-1', 'started_at': 1.0,
            'session_key': metadata['gateway_session_key'],
            'parent_session_id': metadata['gateway_session_id'],
            'platform': 'discord', 'chat_id': '42', 'chat_type': 'dm',
            'exit_code': 0, 'output': 'finished',
        }
        assert await runner._inject_watch_notification('finished', event, raise_not_accepted=True) is True
        before = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        assert before[-1]['status'] == 'queued'
        assert (before[-1]['payload'].get('native_text_v1') or {}).get('automation')

        assert await runner._inject_watch_notification('finished', event, raise_not_accepted=True) is True
        after = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        assert after == before
    finally:
        authority.db.close()
