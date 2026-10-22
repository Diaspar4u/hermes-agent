"""Native admission preserves streamed replies for the A2A completion callback."""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
from plugins.platforms.a2a.adapter import A2AAdapter
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_native_streamed_a2a_returns_complete_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform("a2a"): adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="a2a-stream")
    runner._wire_adapter_handlers(adapter)
    try:
        send = adapter.send
        for preview in (False, True):
            task_id, context_id, body = f"task-{preview}", f"ctx-{preview}", "STREAMED_REPLY"
            adapter.tasks.create(task_id, context_id, "peer", "", "")
            future = adapter._add_pending(task_id, context_id)
            original = MessageEvent(text="Reply STREAMED_REPLY", message_id=task_id,
                                    source=adapter.build_source(chat_id=context_id, user_id="peer", chat_type="dm"))
            trace, executed = [], []

            async def traced_send(chat_id, content, reply_to=None, metadata=None):
                trace.append((content, dict(metadata or {})))
                return await send(chat_id, content, reply_to=reply_to, metadata=metadata)

            async def turn(event, *_args, **_kwargs):
                executed.append(event)
                consumer = GatewayStreamConsumer(adapter, context_id,
                                                 StreamConsumerConfig(cursor=" ▉", transport="auto"),
                                                 initial_reply_to_id=task_id)
                consumer._accumulated = body
                if preview:
                    await consumer._push_update(_Tick())
                final = _Tick(got_done=True)
                await consumer._push_update(final)
                await consumer._finalize_turn(final)
                assert consumer.already_sent and consumer.final_content_delivered
                execution_result.get()["result"] = {"final_response": body, "messages": []}
                return await runner._hmwa_deliver_turn_response(
                    event, event.source, None, "a2a-stream", 0, {"already_sent": consumer.already_sent}, [], body, "", False)

            monkeypatch.setattr(adapter, "send", traced_send)
            monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
            try:
                await adapter.handle_message(original)
                await async_wait_until(lambda: future.done() and not adapter._background_tasks,
                                       timeout=20, what="native A2A completion")
                assert executed and executed[0] is not original
                assert future.result(timeout=0)[1] == body, {
                    "reply": future.result(timeout=0), "sends": trace,
                    "execution_stash": getattr(executed[0], "_streamed_final_response", None),
                    "original_stash": getattr(original, "_streamed_final_response", None),
                }
            finally:
                adapter._pop_pending(task_id)

    finally:
        authority.db.close()
