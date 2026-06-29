import logging
from typing import Any

logger = logging.getLogger(__name__)

def report_error(error: Exception, context: dict[str, Any] | None = None) -> None:
    """Centralized error reporting function.

    All code paths handling unexpected errors must funnel through this function.
    If a remote error tracking system (like Sentry) is added later, this will
    serve as the single point of integration.

    Args:
        error: The caught exception.
        context: Optional dictionary with additional context about the error.
    """
    error_msg = f"Unexpected error: {error}"
    if context:
        error_msg += f" | Context: {context}"

    logger.error(error_msg, exc_info=error)
