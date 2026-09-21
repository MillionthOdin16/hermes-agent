"""Unit tests for Telegram topic routing in send operations.

Run with: pytest tests/gateway/test_telegram_topic_send.py -v
"""

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestMetadataThreadIdExtraction:
    """Test that _metadata_thread_id extracts thread from metadata."""

    def test_extracts_thread_id_from_metadata(self):
        """thread_id in metadata should be extracted."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._dm_topics = {}

        result = adapter._metadata_thread_id({"thread_id": "42"})
        assert result == "42"

    def test_extracts_message_thread_id_as_fallback(self):
        """message_thread_id used when thread_id not present."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._dm_topics = {}

        result = adapter._metadata_thread_id({"message_thread_id": "99"})
        assert result == "99"

    def test_returns_none_when_no_metadata(self):
        """Returns None when metadata is None."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._dm_topics = {}

        result = adapter._metadata_thread_id(None)
        assert result is None

    def test_returns_none_when_empty(self):
        """Returns None when metadata is empty dict."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._dm_topics = {}

        result = adapter._metadata_thread_id({})
        assert result is None


class TestThreadKwargsForSend:
    """Test that _thread_kwargs_for_send computes correct kwargs."""

    def test_with_thread_id_includes_message_thread_id(self):
        """When thread_id provided, kwargs should include it."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.config = MagicMock(extra={})
        adapter._GENERAL_TOPIC_THREAD_ID = "1"

        kwargs = adapter._thread_kwargs_for_send(
            chat_id="208214988",
            thread_id="5",
            metadata={"thread_id": "5"},
        )
        assert kwargs.get("message_thread_id") == 5

    def test_none_thread_id_gives_empty_kwargs(self):
        """When no thread_id, kwargs should be empty (routes to General/root)."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.config = MagicMock(extra={})
        adapter._GENERAL_TOPIC_THREAD_ID = "1"

        kwargs = adapter._thread_kwargs_for_send(
            chat_id="208214988",
            thread_id=None,
            metadata=None,
        )
        # Should NOT have message_thread_id set (or be None)
        assert kwargs.get("message_thread_id") is None


class TestEditOverflowSplit:
    """Test that _edit_overflow_split preserves thread routing."""

    @pytest.mark.asyncio
    async def test_overflow_preserves_thread_id(self):
        """All chunks from overflow should go to same topic."""
        from gateway.platforms.telegram import TelegramAdapter
        from gateway.config import Platform
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        
        # Setup minimal config - MUST set self.platform (not _platform)
        adapter.platform = Platform.TELEGRAM
        adapter.config = MagicMock(extra={})
        adapter._dm_topics = {"Dev": 5, "Ops": 7}
        adapter._GENERAL_TOPIC_THREAD_ID = "1"

        # Setup mock bot to track calls
        mock_bot = MagicMock()
        call_count = [0]
        
        async def mock_send(**kwargs):
            call_count[0] += 1
            msg = MagicMock()
            msg.message_id = call_count[0]
            return msg
        
        mock_bot.send_message = AsyncMock(side_effect=mock_send)
        adapter._bot = mock_bot
        
        # Use class constant (can't override with small value for test)
        # This tests the actual splitting behavior
        content = "x" * 6000  # Long enough to split

        result = await adapter._edit_overflow_split(
            chat_id="123",
            message_id="1",
            content=content,
            finalize=True,
            metadata={"thread_id": "7"},
        )

        # All chunks must have the same thread_id
        calls = adapter._bot.send_message.call_args_list
        for i, call in enumerate(calls):
            kwargs = call.kwargs
            thread_id = kwargs.get("message_thread_id")
            assert thread_id == 7, f"Chunk {i}: expected 7, got {thread_id}"


class TestEditMessageWithMetadata:
    """Test that edit_message accepts and propagates metadata to overflow split."""

    @pytest.mark.asyncio
    async def test_edit_message_accepts_metadata_param(self):
        """edit_message should accept metadata parameter."""
        import inspect
        from gateway.platforms.telegram import TelegramAdapter
        
        sig = inspect.signature(TelegramAdapter.edit_message)
        params = list(sig.parameters.keys())
        assert "metadata" in params, "edit_message should have metadata param"

    @pytest.mark.asyncio
    async def test_edit_message_with_metadata_uses_correct_thread(self):
        """When metadata passed, continuation chunks go to correct topic."""
        from gateway.platforms.telegram import TelegramAdapter
        from gateway.config import Platform
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        
        adapter.platform = Platform.TELEGRAM  # Must set, not None
        adapter.config = MagicMock(extra={})
        adapter._dm_topics = {}
        adapter._GENERAL_TOPIC_THREAD_ID = "1"

        # Mock bot - track all send_message calls
        mock_bot = MagicMock()
        call_count = [0]
        
        async def mock_send(**kwargs):
            call_count[0] += 1
            msg = MagicMock()
            msg.message_id = call_count[0]
            return msg
        
        async def mock_edit(**kwargs):
            raise Exception("Message too long")  # Force overflow path
        
        mock_bot.send_message = AsyncMock(side_effect=mock_send)
        mock_bot.edit_message_text = AsyncMock(side_effect=mock_edit)
        adapter._bot = mock_bot

        # Long content triggers overflow
        content = "x" * 6000

        result = await adapter.edit_message(
            chat_id="123",
            message_id="1",
            content=content,
            finalize=True,
            metadata={"thread_id": "7"},
        )

        # All continuation chunks must use thread_id=7
        calls = adapter._bot.send_message.call_args_list
        for i, call in enumerate(calls):
            kwargs = call.kwargs
            thread_id = kwargs.get("message_thread_id")
            assert thread_id == 7, f"Chunk {i}: expected 7, got {thread_id}"