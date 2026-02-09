import logging
import sys
import traceback


def report_error(e: Exception, context: dict | None = None) -> None:
    """
    Centralized error reporting function.
    Logs the error and traceback to stderr using the logging module.
    """
    logging.error(f"Error: {e}")
    if context:
        logging.error(f"Context: {context}")
    traceback.print_exc(file=sys.stderr)
