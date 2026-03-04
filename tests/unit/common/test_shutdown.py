"""
Comprehensive tests for ``common.shutdown``.

Covers the shutdown flag lifecycle, signal handler registration, and
simulated signal delivery.
"""

from __future__ import annotations

import signal

import pytest

from common.shutdown import (
    is_shutdown_requested,
    register_signal_handlers,
    request_shutdown,
    reset_shutdown,
)


# ---------------------------------------------------------------------------
# Ensure a clean slate for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_shutdown():
    """Reset the shutdown flag before and after each test."""
    reset_shutdown()
    yield
    reset_shutdown()


# ===================================================================
# Initial state: not shutdown
# ===================================================================

class TestInitialState:

    def test_not_shutdown_initially(self):
        assert is_shutdown_requested() is False


# ===================================================================
# request_shutdown sets the flag
# ===================================================================

class TestRequestShutdown:

    def test_sets_flag(self):
        request_shutdown()
        assert is_shutdown_requested() is True

    def test_idempotent(self):
        request_shutdown()
        request_shutdown()
        assert is_shutdown_requested() is True


# ===================================================================
# is_shutdown_requested returns True after request
# ===================================================================

class TestIsShutdownRequested:

    def test_returns_false_before_request(self):
        assert is_shutdown_requested() is False

    def test_returns_true_after_request(self):
        request_shutdown()
        assert is_shutdown_requested() is True


# ===================================================================
# reset_shutdown clears the flag
# ===================================================================

class TestResetShutdown:

    def test_clears_flag(self):
        request_shutdown()
        assert is_shutdown_requested() is True
        reset_shutdown()
        assert is_shutdown_requested() is False

    def test_reset_when_already_clear(self):
        """Resetting an already-clear flag is a no-op."""
        reset_shutdown()
        assert is_shutdown_requested() is False


# ===================================================================
# register_signal_handlers installs SIGTERM and SIGINT handlers
# ===================================================================

class TestRegisterSignalHandlers:

    def test_installs_sigterm_handler(self):
        register_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        assert handler is not signal.SIG_DFL
        assert handler is not signal.SIG_IGN

    def test_installs_sigint_handler(self):
        register_signal_handlers()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        assert handler is not signal.SIG_DFL
        assert handler is not signal.SIG_IGN


# ===================================================================
# Signal handler calls request_shutdown
# ===================================================================

class TestSignalHandlerBehavior:

    def test_sigterm_handler_triggers_shutdown(self):
        register_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        # Simulate calling the handler directly
        handler(signal.SIGTERM, None)
        assert is_shutdown_requested() is True

    def test_sigint_handler_triggers_shutdown(self):
        register_signal_handlers()
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert is_shutdown_requested() is True

    def test_handler_is_idempotent(self):
        register_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        handler(signal.SIGTERM, None)
        assert is_shutdown_requested() is True
