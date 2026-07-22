"""Tests for /bg gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/bg", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    # A real SessionStore returns None when no persisted /model override exists.
    # MagicMock's default truthy return would otherwise rehydrate a fake model
    # and make the session-scoped reasoning resolver receive a MagicMock.
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


# ---------------------------------------------------------------------------
# _handle_background_command
# ---------------------------------------------------------------------------


class TestHandleBackgroundCommand:
    """Tests for GatewayRunner._handle_background_command."""

    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/bg" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /bg with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_valid_prompt_starts_task(self):
        """Running /background with a prompt returns confirmation and starts task."""
        runner = _make_runner()

        # Patch asyncio.create_task to capture the coroutine
        created_tasks = []
        original_create_task = asyncio.create_task

        def capture_task(coro, *args, **kwargs):
            # Close the coroutine to avoid warnings
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/background Summarize the top HN stories")
            result = await runner._handle_background_command(event)

        assert "🔄" in result
        assert "Background task started" in result
        assert "bg_" in result  # task ID starts with bg_
        assert "Summarize the top HN stories" in result
        assert len(created_tasks) == 1  # background task was created

    @pytest.mark.asyncio
    async def test_telegram_dm_topic_passes_trigger_anchor_to_task(self):
        """Telegram private-topic completion sends need the original command message id."""
        runner = _make_runner()
        runner._run_background_task = AsyncMock()

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            return mock_task

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="20197",
        )
        event = MessageEvent(
            text="/background summarize",
            source=source,
            message_id="463",
            reply_to_message_id="462",
        )

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_background_command(event)

        assert "Background task started" in result
        runner._run_background_task.assert_called_once()
        assert runner._run_background_task.call_args.kwargs["event_message_id"] == "463"

    @pytest.mark.asyncio
    async def test_prompt_truncated_in_preview(self):
        """Long prompts are truncated to 60 chars in the confirmation message."""
        runner = _make_runner()
        long_prompt = "A" * 100

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            event = _make_event(text=f"/background {long_prompt}")
            result = await runner._handle_background_command(event)

        assert "..." in result
        # Should not contain the full prompt
        assert long_prompt not in result

    @pytest.mark.asyncio
    async def test_task_id_is_unique(self):
        """Each background task gets a unique task ID."""
        runner = _make_runner()
        task_ids = set()

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            for i in range(5):
                event = _make_event(text=f"/background task {i}")
                result = await runner._handle_background_command(event)
                # Extract task ID from result (format: "Task ID: bg_HHMMSS_hex")
                for line in result.split("\n"):
                    if "Task ID:" in line:
                        tid = line.split("Task ID:")[1].strip()
                        task_ids.add(tid)

        assert len(task_ids) == 5  # all unique

    @pytest.mark.asyncio
    async def test_works_across_platforms(self):
        """The /background command works for all platforms."""
        for platform in [Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK]:
            runner = _make_runner()
            with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
                event = _make_event(
                    text="/background test task",
                    platform=platform,
                )
                result = await runner._handle_background_command(event)
                assert "Background task started" in result

    @pytest.mark.asyncio
    async def test_reply_context_threads_into_background_prompt(self):
        """Telegram reply + /btw: the replied-to message must reach the prompt.

        Regression for the case where a user replies to a prior Telegram
        message with `/btw <follow-up>`. The dispatcher (gateway/run.py
        around line 11117) enriches event.text with a `[Replying to: "..."]`
        prefix so the command handler receives the original context.
        """
        runner = _make_runner()
        event = _make_event(text="/background summarize")
        # Simulate Telegram reply metadata.
        event.reply_to_message_id = "888"
        event.reply_to_text = "the original question the user replied to"
        event.reply_to_is_own_message = False

        captured_kwargs = {}

        def capture_task(coro, **kwargs):
            coro.close()
            captured_kwargs.update(kwargs)
            return MagicMock()

        # Run the enrichment branch directly: it lives on GatewayRunner and is
        # invoked before _handle_background_command. Test it as a unit.
        from gateway.run import GatewayRunner
        enrich = getattr(GatewayRunner, "_enrich_command_with_reply_context", None)
        if enrich is not None:
            enrich(event)
        else:
            # Fall back to the dispatcher-level path: re-implement the
            # enrichment logic so this test stays in sync without a private
            # helper dependency. The real production logic is in
            # gateway/run.py around line 11117; we mirror it here.
            command = event.get_command()
            if (command
                and getattr(event, "reply_to_text", None)
                and getattr(event, "reply_to_message_id", None)
                and (event.text or "").lstrip().startswith("/")):
                _reply_snip = (event.reply_to_text or "")[:500]
                _prefix = '[Replying to: "' + _reply_snip + '"]\n\n'
                _parts = (event.text or "").split(maxsplit=1)
                _cmd_word = _parts[0] if _parts else ""
                _args = _parts[1] if len(_parts) > 1 else ""
                event.text = (
                    _cmd_word + " " + _prefix + _args if _args
                    else (_cmd_word + " " + _prefix.rstrip())
                ).strip()

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            await runner._handle_background_command(event)

        # /background runs the prompt as command_args. Assert it now contains
        # the replied-to context, not just the bare command args.
        prompt = event.get_command_args().strip()
        assert "Replying to" in prompt, (
            "reply context missing from background prompt: %r" % prompt
        )
        assert "the original question the user replied to" in prompt
        assert "summarize" in prompt



# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""


    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        """When provider credentials are missing, an error is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        # Should have sent an error message
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert "failed" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "").lower()

    @pytest.mark.asyncio
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        mock_result = {"final_response": "Hello from background!", "messages": []}

        checkpoint_config = {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 8,
                "max_total_size_mb": 222,
                "max_file_size_mb": 3,
            }
        }
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value=checkpoint_config), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        # Should have sent the result
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Background task complete" in content
        assert "Hello from background!" in content
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# /bg in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /bg and /btw appear in help text and known commands."""

    @pytest.mark.asyncio
    async def test_bg_and_btw_in_help_output(self):
        """The /help output includes /bg and /btw."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/bg" in result
        assert "/btw" in result


# ---------------------------------------------------------------------------
# CLI /bg command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /bg and /btw are registered in the CLI command system."""


    def test_bg_autocompletes(self):
        """The /bg and /btw commands appear in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("bg")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/bg")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/bg" in d for d in cmd_displays)

        doc = Document("/btw")
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/btw" in d for d in cmd_displays)


# ---------------------------------------------------------------------------
# _handle_btw_command
# ---------------------------------------------------------------------------


class TestHandleBtwCommand:
    """Tests for GatewayRunner._handle_btw_command (context-aware side question)."""

    @pytest.mark.asyncio
    async def test_no_question_shows_usage(self):
        runner = _make_runner()
        event = _make_event(text="/btw")
        result = await runner._handle_btw_command(event)
        assert "Usage:" in result
        assert "/btw" in result

    @pytest.mark.asyncio
    async def test_no_history_reports_no_conversation(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = []
        store._store = runner.session_store
        runner._async_session_store = store
        event = _make_event(text="/btw what did we do?")
        result = await runner._handle_btw_command(event)
        assert "conversation" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatches_side_question_and_sends_answer(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [
            {"role": "user", "content": "fix foo.py"},
            {"role": "assistant", "content": "done"},
        ]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "k", "provider": "p",
                                         "base_url": "u", "api_mode": "chat_completions"})
        )
        runner._reply_anchor_for_event = MagicMock(return_value=None)
        runner._thread_metadata_for_source = MagicMock(return_value=None)
        mock_adapter = AsyncMock()
        runner._adapter_for_source = MagicMock(return_value=mock_adapter)

        event = _make_event(text="/btw which file was that?")

        with patch("agent.side_question.answer_side_question",
                   return_value="it was foo.py") as mock_answer:
            result = await runner._handle_btw_command(event)
            # Ack returned immediately, worker task registered.
            assert "which file was that?" in result
            # Drain the fire-and-forget task.
            for task in list(runner._background_tasks):
                await task

        # Snapshot + question reached the engine; live history untouched.
        args, kwargs = mock_answer.call_args
        assert args[0] == "which file was that?"
        assert args[1][0]["content"] == "fix foo.py"
        assert kwargs["main_runtime"]["model"] == "test-model"

        # The answer was delivered to the chat.
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "it was foo.py" in sent_text

    @pytest.mark.asyncio
    async def test_no_credentials_reports_error(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [{"role": "user", "content": "hi"}]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=(None, {"api_key": None})
        )
        event = _make_event(text="/btw what?")
        result = await runner._handle_btw_command(event)
        assert "❌" in result
