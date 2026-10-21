"""Project-agnostic evaluation harness for AI agents."""

from agent_eval.models import AgentRequest, AgentResult, ToolCallRecord

__version__ = "0.1.0"

__all__ = [
    "AgentRequest",
    "AgentResult",
    "ToolCallRecord",
    "__version__",
]
