"""
net_timeout.py - Enforce a hard wall-clock timeout around any blocking call.

Why this exists: the google-genai SDK's own http_options timeout does not
reliably abort a stalled socket (this is a known open issue upstream —
requests can hang indefinitely instead of raising). Since a single stuck
Gemini call can hang past gunicorn's worker timeout and take the whole
worker down with it (SIGKILL, not just a 500), we run these calls in a
worker thread and give up on them ourselves after `timeout` seconds,
regardless of what's happening inside the SDK.

The stuck thread is abandoned (Python can't force-kill a thread), but the
request itself returns promptly and the caller's existing except-block /
fallback path takes over, exactly like any other Gemini failure.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger("finsight.net_timeout")

# One small shared pool is enough — these calls are short-lived and infrequent
# relative to a single Flask worker's request rate.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="net-timeout")


class NetworkTimeoutError(Exception):
    """Raised when a wrapped call exceeds its allotted wall-clock time."""
    pass


def call_with_timeout(func, *args, timeout=8, **kwargs):
    """
    Run func(*args, **kwargs) with a hard timeout in seconds.
    Raises NetworkTimeoutError if it doesn't complete in time.
    Any exception raised by func itself propagates unchanged.
    """
    future = _executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.warning("Call to %s timed out after %ss", getattr(func, "__name__", func), timeout)
        raise NetworkTimeoutError(f"{getattr(func, '__name__', 'call')} timed out after {timeout}s")
