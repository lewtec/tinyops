import logging
import sys
import traceback

logger = logging.getLogger(__name__)


def report_error(e: Exception, context: str = "") -> None:
    """
    Centralized error reporting function.
    All code paths that catch unexpected errors must route them through here.
    If Sentry or another observability tool is configured in the environment,
    this function will forward the error to it. Otherwise, it logs to stderr.
    """
    error_msg = f"Reported Error: {type(e).__name__}: {e}"
    if context:
        error_msg = f"{context} - {error_msg}"

    # Attempt to capture via Sentry if available in the environment
    try:
        import sentry_sdk

        if sentry_sdk.Hub.current.client:
            sentry_sdk.capture_exception(e)
            return
    except ImportError:
        pass  # Sentry not installed, fallback to standard logging

    logger.error(error_msg, exc_info=True)
    # Also ensure it prints to stderr for environments that might swallow python logs
    print(error_msg, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
