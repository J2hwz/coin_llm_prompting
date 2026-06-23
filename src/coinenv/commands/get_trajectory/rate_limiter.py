"""Rate limiter using token bucket algorithm for API request throttling."""

import time
import logging
from threading import Lock
from typing import Optional

logger = logging.getLogger(__file__)


class RateLimiter:
    """Thread-safe token bucket rate limiter.

    Tokens replenish at a constant rate over time. When the bucket is full,
    requests are blocked until tokens become available.

    Example:
        # Allow 1000 requests per 5 minutes (300 seconds)
        limiter = RateLimiter(rate_limit=1000, period=300)

        with limiter:
            # Make API request
            response = api_call()

    Args:
        rate_limit: Maximum number of requests allowed per period.
        period: Time period in seconds over which rate_limit is applied.
        initial_tokens: Initial number of tokens (defaults to rate_limit).
    """

    def __init__(
        self, rate_limit: int, period: float, initial_tokens: Optional[int] = None
    ):
        if rate_limit <= 0:
            raise ValueError(f"rate_limit must be positive, got {rate_limit}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")

        self.rate_limit = rate_limit
        self.period = period
        self.tokens_per_second = rate_limit / period
        self._tokens = initial_tokens if initial_tokens is not None else rate_limit
        self._last_refill = time.time()
        self._lock = Lock()

    @property
    def tokens(self) -> float:
        """Current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self._last_refill

        # Calculate tokens to add (capped at rate_limit)
        tokens_to_add = elapsed * self.tokens_per_second
        self._tokens = min(self.rate_limit, self._tokens + tokens_to_add)
        self._last_refill = now

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token from the bucket.

        Args:
            block: If True, block until a token is available. If False, return
                   False immediately if no token is available.
            timeout: Maximum time to wait for a token (only if block=True).
                     If None, wait indefinitely.

        Returns:
            True if a token was acquired, False if timeout expired or block=False
            and no token was available.
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            if not block:
                return False

            # Calculate wait time needed for one token
            wait_time = (1 - self._tokens) / self.tokens_per_second

            if timeout is not None and wait_time > timeout:
                return False

        # Release lock while waiting
        start_wait = time.time()
        while True:
            time.sleep(0.01)  # Small sleep to avoid busy waiting

            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

                if timeout is not None and (time.time() - start_wait) >= timeout:
                    return False

    def __enter__(self):
        """Context manager entry - acquire a token."""
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - no cleanup needed."""
        return False

    def wait_time(self, n_tokens: int = 1) -> float:
        """Calculate time to wait for n tokens to be available.

        Args:
            n_tokens: Number of tokens to wait for.

        Returns:
            Estimated wait time in seconds. Returns 0 if tokens are available.
        """
        with self._lock:
            self._refill()
            if self._tokens >= n_tokens:
                return 0.0
            return (n_tokens - self._tokens) / self.tokens_per_second


class RateLimitInfo:
    """Container for rate limit configuration and status.

    Attributes:
        rate_limit: Maximum requests per period.
        period: Time period in seconds.
        requests_per_second: Rate of requests.
        description: Human-readable description.
    """

    def __init__(self, rate_limit: int, period: float, description: str = ""):
        self.rate_limit = rate_limit
        self.period = period
        self.requests_per_second = rate_limit / period
        self.description = description

    @classmethod
    def from_requests_per_minute(
        cls, requests_per_minute: int, description: str = ""
    ) -> "RateLimitInfo":
        """Create rate limit info from requests per minute.

        Args:
            requests_per_minute: Maximum requests per minute.
            description: Human-readable description.

        Returns:
            RateLimitInfo instance.
        """
        return cls(rate_limit=requests_per_minute, period=60.0, description=description)

    @classmethod
    def from_custom_period(
        cls, rate_limit: int, period_minutes: float, description: str = ""
    ) -> "RateLimitInfo":
        """Create rate limit info from custom period.

        Args:
            rate_limit: Maximum requests per period.
            period_minutes: Time period in minutes.
            description: Human-readable description.

        Returns:
            RateLimitInfo instance.
        """
        return cls(
            rate_limit=rate_limit, period=period_minutes * 60, description=description
        )

    def __repr__(self) -> str:
        if self.description:
            return (
                f"RateLimitInfo({self.rate_limit}/{self.period}s ({self.description}))"
            )
        return f"RateLimitInfo({self.rate_limit}/{self.period}s)"


# Common rate limit presets
TOGETHER_AI_FREE_TIER = RateLimitInfo(
    rate_limit=1000,
    period=300,
    description="Together AI free tier: 1000 requests per 5 minutes",
)

TOGETHER_AI_PAID_TIER_1 = RateLimitInfo(
    rate_limit=6000,
    period=60,
    description="Together AI paid tier 1: 6000 requests per minute",
)

OPENAI_TIER_1 = RateLimitInfo(
    rate_limit=3000, period=60, description="OpenAI tier 1: 3000 requests per minute"
)

ANTHROPIC_TIER_1 = RateLimitInfo(
    rate_limit=1000, period=60, description="Anthropic tier 1: 1000 requests per minute"
)
