from litellm.exceptions import MidStreamFallbackError


class _FakeProviderError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_mid_stream_fallback_error_handles_non_numeric_status_code():
    original_exception = _FakeProviderError(status_code="tool_use_failed")

    error = MidStreamFallbackError(
        message="stream failed",
        model="groq/openai/gpt-oss-120b",
        llm_provider="groq",
        original_exception=original_exception,
    )

    assert error.status_code == 503


def test_mid_stream_fallback_error_preserves_numeric_status_code():
    original_exception = _FakeProviderError(status_code=429)

    error = MidStreamFallbackError(
        message="stream failed",
        model="groq/openai/gpt-oss-120b",
        llm_provider="groq",
        original_exception=original_exception,
    )

    assert error.status_code == 429
