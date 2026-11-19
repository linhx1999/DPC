"""Response failures that should bypass retries with an unchanged token budget."""
from __future__ import annotations


class LLMResponseError(RuntimeError):
    """The provider returned a response that cannot be used by an agent."""

    outcome = "invalid_response"


class OutputTokenLimitError(LLMResponseError):
    """The response ended at the output limit, including partial final answers."""

    outcome = "output_limit"

    def __init__(self, max_tokens: int):
        super().__init__(
            f"LLM output truncated (finish_reason=length, max_tokens={max_tokens}); "
            "not retrying with the same budget."
        )


class EmptyResponseError(LLMResponseError):
    """A response has no final text for the agent to parse."""

    outcome = "empty_response"

    def __init__(self):
        super().__init__("LLM returned no final text; not retrying the empty response.")
