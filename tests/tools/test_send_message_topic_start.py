"""Behavior tests for send_message create_topic_and_start."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
import tools.send_message_tool as send_message


def test_create_topic_and_start_creates_telegram_topic_and_wakes_it():
    create_thread = AsyncMock(return_value="444")
    adapter = SimpleNamespace(create_handoff_thread=create_thread)
    wake = AsyncMock()

    with patch("gateway.wake.deliver_wake", wake):
        result = asyncio.run(
            send_message._create_topic_and_start(
                adapter,
                chat_id="-100123",
                topic_name="Research",
                prompt="Research the launch.",
                user_id="42",
                profile="dev",
            )
        )

    assert result == {
        "success": True,
        "platform": "telegram",
        "chat_id": "-100123",
        "thread_id": "444",
        "topic_name": "Research",
        "started": True,
    }
    create_thread.assert_awaited_once_with("-100123", "Research")
    wake.assert_awaited_once()
    call = wake.await_args
    assert call.args == (adapter,)
    assert call.kwargs["text"] == "Research the launch."
    source = call.kwargs["source"]
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100123"
    assert source.chat_type == "forum"
    assert source.thread_id == "444"
    assert source.user_id == "42"
    assert source.profile == "dev"


def test_send_message_exposes_and_dispatches_create_topic_and_start(monkeypatch):
    actions = send_message.SEND_MESSAGE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "create_topic_and_start" in actions

    args = {
        "action": "create_topic_and_start",
        "target": "telegram:-100123",
        "topic_name": "Research",
        "message": "Research the launch.",
    }
    handler = MagicMock(return_value='{"success": true}')
    monkeypatch.setattr(send_message, "_handle_create_topic_and_start", handler, raising=False)

    assert send_message.send_message_tool(args) == '{"success": true}'
    handler.assert_called_once_with(args)


def test_create_topic_and_start_action_uses_live_telegram_gateway(monkeypatch):
    create_thread = AsyncMock(return_value="444")
    adapter = SimpleNamespace(create_handoff_thread=create_thread)
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    runner = SimpleNamespace(
        adapters={Platform.TELEGRAM: adapter},
        _gateway_loop=loop,
    )
    wake = AsyncMock()

    import gateway.run as gateway_run
    import gateway.session_context as session_context

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(send_message, "prepare_send_message_platforms", lambda: None)
    values = {
        "HERMES_SESSION_USER_ID": "42",
        "HERMES_SESSION_PROFILE": "dev",
    }
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": values.get(key, default),
    )

    try:
        with patch("gateway.wake.deliver_wake", wake):
            result = send_message.send_message_tool(
                {
                    "action": "create_topic_and_start",
                    "target": "telegram:-100123",
                    "topic_name": "Research",
                    "message": "Research the launch.",
                }
            )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert json.loads(result) == {
        "success": True,
        "platform": "telegram",
        "chat_id": "-100123",
        "thread_id": "444",
        "topic_name": "Research",
        "started": True,
    }
    wake.assert_awaited_once()
