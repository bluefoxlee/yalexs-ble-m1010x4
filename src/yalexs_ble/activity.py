from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Protocol

from bleak.exc import BleakError

from .const import (
    LOCK_ACTIVITY_POLL_INTERVAL,
    LOCK_ACTIVITY_POLL_RETRIES,
    LOCK_ACTIVITY_POLL_RETRY_EXPONENTIAL_BACKOFF_SECONDS,
    ConnectionInfo,
    DoorActivity,
    LockActivity,
    LockActivityValue,
    LockInfo,
    RawActivity,
)
from .lock import Lock
from .session import DisconnectedError, ResponseError

_LOGGER = logging.getLogger(__name__)
_DIAG_LOGGER = logging.getLogger("yalexs_ble.diagnostics")

# Some lock variants use an activity type that is not decoded yet. Allow a
# larger bounded drain so a batch of unknown records is not truncated while
# retaining a guard against a firmware response that repeats forever.
MAX_CONSECUTIVE_UNKNOWN_ACTIVITY_RECORDS = 10


def _activity_summary(activity: LockActivityValue | None) -> str:
    """Return a safe, compact description for temporary diagnostics."""
    if activity is None:
        return "none"
    if isinstance(activity, RawActivity):
        return (
            f"RawActivity type=0x{activity.activity_type:02X} "
            f"pin_id={activity.pin_id!r} timestamp={activity.timestamp.isoformat()}"
        )
    if isinstance(activity, DoorActivity):
        return (
            f"DoorActivity status={activity.status.name} "
            f"timestamp={activity.timestamp.isoformat()}"
        )
    if isinstance(activity, LockActivity):
        return (
            f"LockActivity status={activity.status.name} "
            f"source={activity.source.name} slot={activity.slot!r} "
            f"timestamp={activity.timestamp.isoformat()}"
        )
    return type(activity).__name__


class LockBridge(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def lock_info(self) -> LockInfo | None: ...

    @property
    def connection_info(self) -> ConnectionInfo | None: ...

    @property
    def loop(self) -> Any: ...

    @property
    def operation_lock(self) -> asyncio.Lock: ...

    async def ensure_connected(self) -> Lock: ...

    async def handle_disconnected(self, exc: Exception) -> None: ...


class ActivityManager:
    """A manager for getting lock activity (history)."""

    def __init__(
        self,
        lock: LockBridge,
    ) -> None:
        self._lock = lock
        self._activity_callbacks: list[
            Callable[
                [DoorActivity | LockActivity | RawActivity, LockInfo, ConnectionInfo],
                None,
            ]
        ] = []
        self._activity_poll_task: asyncio.Task[None] | None = None
        self._cancel_deferred_activity_poll: asyncio.TimerHandle | None = None
        self._poll_sequence = 0

    def register_activity_callback(
        self,
        callback: Callable[
            [DoorActivity | LockActivity | RawActivity, LockInfo, ConnectionInfo], None
        ],
        *,
        request_update: bool = False,
    ) -> Callable[[], None]:
        """Register an activity callback to be called when the lock state changes."""
        self._activity_callbacks.append(callback)

        if request_update:
            self.schedule_activity_poll(
                0, max_retries=0, reason="initial_request"
            )

        return partial(self._activity_callbacks.remove, callback)

    def handle_activities(self, activities: Iterable[LockActivityValue]) -> None:
        """Handle activity update."""
        _LOGGER.debug("%s: Activity updates: %s", self._lock.name, activities)

        for activity in activities:
            _DIAG_LOGGER.debug(
                "%s activity callback dispatch: %s callbacks=%d",
                self._lock.name,
                _activity_summary(activity),
                len(self._activity_callbacks),
            )
            self._callback_activity(activity)

    async def execute_forced_disconnect(self) -> None:
        """Cancel an activity poll before its connection is torn down."""
        if (
            activity_poll_task := self._activity_poll_task
        ) and not activity_poll_task.done():
            self._activity_poll_task = None
            activity_poll_task.cancel()
            _DIAG_LOGGER.debug(
                "%s activity poll cancelled for forced disconnect", self._lock.name
            )
            await activity_poll_task

    def _callback_activity(self, activity: LockActivityValue) -> None:
        """Call the activity callbacks."""
        _LOGGER.debug(
            "%s: New activity: %s %s %s",
            self._lock.name,
            activity,
            self._lock.lock_info,
            self._lock.connection_info,
        )
        if not self._activity_callbacks:
            return
        assert self._lock.lock_info is not None  # nosec
        connection_info = self._lock.connection_info
        assert connection_info is not None  # nosec
        for callback in self._activity_callbacks:
            try:
                callback(activity, self._lock.lock_info, connection_info)
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception(
                    "%s: Error calling activity callback", self._lock.name
                )

    def _cancel_activity_poll(self) -> None:
        """Cancel polling for activity."""
        if self._cancel_deferred_activity_poll:
            self._cancel_deferred_activity_poll.cancel()
            self._cancel_deferred_activity_poll = None

    def schedule_activity_poll(
        self,
        seconds: float,
        retries: int = 0,
        max_retries: int = LOCK_ACTIVITY_POLL_RETRIES,
        backoff: float = LOCK_ACTIVITY_POLL_RETRY_EXPONENTIAL_BACKOFF_SECONDS,
        *,
        replace: bool = True,
        reason: str = "unspecified",
    ) -> None:
        """Schedule an activity poll in future seconds.

        This does nothing if no activity callbacks are registered (leaving the
        activity for the Yale/August app to consume).
        """
        if not self._activity_callbacks:
            return

        _DIAG_LOGGER.debug(
            "%s activity poll scheduled: delay=%.1fs retry=%d/%d replace=%s "
            "reason=%s pending=%s running=%s",
            self._lock.name,
            seconds,
            retries,
            max_retries,
            replace,
            reason,
            self._cancel_deferred_activity_poll is not None,
            self._activity_poll_task is not None
            and not self._activity_poll_task.done(),
        )

        if not replace and (
            self._cancel_deferred_activity_poll
            or (self._activity_poll_task and not self._activity_poll_task.done())
        ):
            _LOGGER.debug(
                "%s: Keeping already scheduled activity poll",
                self._lock.name,
            )
            return

        _LOGGER.debug(
            "%s: Scheduling activity poll to happen in %s seconds%s",
            self._lock.name,
            seconds,
            f" (retry {retries})" if retries and max_retries else "",
        )
        self._cancel_activity_poll()
        self._cancel_deferred_activity_poll = self._lock.loop.call_later(
            seconds,
            partial(
                self._deferred_activity_poll,
                retries=retries,
                max_retries=max_retries,
                backoff=backoff,
            ),
        )

    def _deferred_activity_poll(
        self,
        retries: int,
        max_retries: int,
        backoff: float,
    ) -> None:
        """Start an activity poll after its timer fires."""
        self._cancel_activity_poll()
        if self._activity_poll_task and not self._activity_poll_task.done():
            _LOGGER.debug(
                "%s: Skipping activity poll since one already in progress",
                self._lock.name,
            )
            _DIAG_LOGGER.debug(
                "%s activity poll skipped: another poll is already running",
                self._lock.name,
            )
            return
        self._activity_poll_task = asyncio.create_task(
            self._execute_activity_poll(
                retries=retries,
                max_retries=max_retries,
                backoff=backoff,
            )
        )

    async def _execute_activity_poll(
        self, retries: int, max_retries: int, backoff: float
    ) -> None:
        if not self._activity_callbacks:
            return

        self._poll_sequence += 1
        poll_id = self._poll_sequence
        started = time.monotonic()
        _LOGGER.debug("%s: Starting deferred activity update", self._lock.name)
        _DIAG_LOGGER.debug(
            "%s activity poll[%d] start: retry=%d/%d",
            self._lock.name,
            poll_id,
            retries,
            max_retries,
        )

        try:
            # Activity polling shares the PushLock connection with the normal
            # YBA state/update path. Serialize the history command with lock
            # operations and state reads so the ESPHome proxy never receives
            # two GATT writes on the same Yale session at once.
            lock_wait_started = time.monotonic()
            _DIAG_LOGGER.debug(
                "%s activity poll[%d] waiting for operation lock",
                self._lock.name,
                poll_id,
            )
            async with self._lock.operation_lock:
                _DIAG_LOGGER.debug(
                    "%s activity poll[%d] operation lock acquired after %.3fs",
                    self._lock.name,
                    poll_id,
                    time.monotonic() - lock_wait_started,
                )
                _DIAG_LOGGER.debug(
                    "%s activity poll[%d] ensure_connected start",
                    self._lock.name,
                    poll_id,
                )
                lock = await self._lock.ensure_connected()
                _DIAG_LOGGER.debug(
                    "%s activity poll[%d] ensure_connected done: connected=%s",
                    self._lock.name,
                    poll_id,
                    lock.is_connected,
                )
                request_number = 1
                first_result = await lock.lock_activity()
                first_was_unknown = lock.last_activity_was_unknown
                _DIAG_LOGGER.debug(
                    "%s activity poll[%d] request[%d] result=%s unknown=%s",
                    self._lock.name,
                    poll_id,
                    request_number,
                    _activity_summary(first_result),
                    first_was_unknown,
                )

                if not first_result and not first_was_unknown:
                    if retries < max_retries:
                        _LOGGER.debug(
                            "%s: No activity found while polling on attempt %s; "
                            "retrying up to %s more times",
                            self._lock.name,
                            retries,
                            max_retries - retries,
                        )
                        self.schedule_activity_poll(
                            backoff * (2**retries),
                            retries=retries + 1,
                            max_retries=max_retries,
                            backoff=backoff,
                            reason="empty_response_retry",
                        )
                    else:
                        _LOGGER.debug(
                            "%s: No activity found while polling after maximum of "
                            "%s retries",
                            self._lock.name,
                            max_retries,
                        )
                        self.schedule_activity_poll(
                            LOCK_ACTIVITY_POLL_INTERVAL,
                            reason="empty_response_interval",
                        )
                    return

                # Unknown records are still records. Do not mistake one for
                # the end-of-history marker, or a later valid unlock/door
                # record would remain unread. Keep a cap in case a firmware
                # repeats the same unknown response forever.
                unknown_records = int(first_was_unknown)
                while unknown_records < MAX_CONSECUTIVE_UNKNOWN_ACTIVITY_RECORDS:
                    request_number += 1
                    result = await lock.lock_activity()
                    _DIAG_LOGGER.debug(
                        "%s activity poll[%d] request[%d] result=%s unknown=%s "
                        "unknown_streak=%d",
                        self._lock.name,
                        poll_id,
                        request_number,
                        _activity_summary(result),
                        lock.last_activity_was_unknown,
                        unknown_records,
                    )
                    if result is None:
                        if not lock.last_activity_was_unknown:
                            _DIAG_LOGGER.debug(
                                "%s activity poll[%d] history end marker after "
                                "%d request(s)",
                                self._lock.name,
                                poll_id,
                                request_number,
                            )
                            break
                        unknown_records += 1
                        continue
                    unknown_records = 0

                if unknown_records >= MAX_CONSECUTIVE_UNKNOWN_ACTIVITY_RECORDS:
                    _LOGGER.warning(
                        "%s: Stopped activity drain after %s consecutive "
                        "unknown records",
                        self._lock.name,
                        MAX_CONSECUTIVE_UNKNOWN_ACTIVITY_RECORDS,
                    )
                self.schedule_activity_poll(
                    LOCK_ACTIVITY_POLL_INTERVAL,
                    reason="poll_complete_interval",
                )
                _DIAG_LOGGER.debug(
                    "%s activity poll[%d] complete duration=%.3fs requests=%d",
                    self._lock.name,
                    poll_id,
                    time.monotonic() - started,
                    request_number,
                )
        except asyncio.CancelledError:
            _DIAG_LOGGER.debug(
                "%s activity poll[%d] cancelled after %.3fs",
                self._lock.name,
                poll_id,
                time.monotonic() - started,
            )
            raise
        except (BleakError, DisconnectedError, ResponseError) as err:
            # Activity polling used to be the one path that bypassed the
            # PushLock retry wrapper. A transient GATT 133 therefore escaped
            # from this background task and stopped activity collection until
            # another update happened to schedule a new poll.
            _LOGGER.warning(
                "%s: Activity poll lost the Bluetooth connection: %s",
                self._lock.name,
                err,
            )
            _DIAG_LOGGER.warning(
                "%s activity poll[%d] failed after %.3fs: %s: %s",
                self._lock.name,
                poll_id,
                time.monotonic() - started,
                type(err).__name__,
                err,
            )
            try:
                await self._lock.handle_disconnected(err)
            except Exception:  # pylint: disable=broad-except
                _LOGGER.debug(
                    "%s: Failed to clean up after activity poll disconnect",
                    self._lock.name,
                    exc_info=True,
                )

            if retries < max_retries:
                self.schedule_activity_poll(
                    backoff * (2**retries),
                    retries=retries + 1,
                    max_retries=max_retries,
                    backoff=backoff,
                    reason="connection_error_retry",
                )
            else:
                _LOGGER.warning(
                    "%s: Activity poll stopped after the maximum of %s retries",
                    self._lock.name,
                    max_retries,
                )
                self.schedule_activity_poll(
                    LOCK_ACTIVITY_POLL_INTERVAL,
                    reason="connection_error_interval",
                )
        except Exception:  # pylint: disable=broad-except
            # Do not leave an unobserved exception in the background task. Keep
            # the traceback in the log for actual programming/protocol errors.
            _LOGGER.exception("%s: Unexpected activity poll failure", self._lock.name)
            _DIAG_LOGGER.warning(
                "%s activity poll[%d] unexpected failure after %.3fs",
                self._lock.name,
                poll_id,
                time.monotonic() - started,
                exc_info=True,
            )
            self.schedule_activity_poll(
                LOCK_ACTIVITY_POLL_INTERVAL,
                reason="unexpected_error_interval",
            )
