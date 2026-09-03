from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Protocol

from bleak.exc import BleakError

from .const import (
    LOCK_ACTIVITY_POLL_RETRIES,
    LOCK_ACTIVITY_POLL_RETRY_EXPONENTIAL_BACKOFF_SECONDS,
    ConnectionInfo,
    DoorActivity,
    LockActivity,
    LockActivityValue,
    LockInfo,
)
from .lock import Lock
from .session import DisconnectedError, ResponseError

_LOGGER = logging.getLogger(__name__)


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
            Callable[[DoorActivity | LockActivity, LockInfo, ConnectionInfo], None]
        ] = []
        self._activity_poll_task: asyncio.Task[None] | None = None
        self._cancel_deferred_activity_poll: asyncio.TimerHandle | None = None

    def register_activity_callback(
        self,
        callback: Callable[
            [DoorActivity | LockActivity, LockInfo, ConnectionInfo], None
        ],
        *,
        request_update: bool = False,
    ) -> Callable[[], None]:
        """Register an activity callback to be called when the lock state changes."""
        self._activity_callbacks.append(callback)

        if request_update:
            self.schedule_activity_poll(0, max_retries=0)

        return partial(self._activity_callbacks.remove, callback)

    def handle_activities(self, activities: Iterable[LockActivityValue]) -> None:
        """Handle activity update."""
        _LOGGER.debug("%s: Activity updates: %s", self._lock.name, activities)

        for activity in activities:
            self._callback_activity(activity)

    async def execute_forced_disconnect(self) -> None:
        """Cancel an activity poll before its connection is torn down."""
        if (
            activity_poll_task := self._activity_poll_task
        ) and not activity_poll_task.done():
            self._activity_poll_task = None
            activity_poll_task.cancel()
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
    ) -> None:
        """Schedule an activity poll in future seconds.

        This does nothing if no activity callbacks are registered (leaving the
        activity for the Yale/August app to consume).
        """
        if not self._activity_callbacks:
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

        _LOGGER.debug("%s: Starting deferred activity update", self._lock.name)

        try:
            # Activity polling shares the PushLock connection with the normal
            # YBA state/update path. Serialize the history command with lock
            # operations and state reads so the ESPHome proxy never receives
            # two GATT writes on the same Yale session at once.
            async with self._lock.operation_lock:
                lock = await self._lock.ensure_connected()
                first_result = await lock.lock_activity()

                if not first_result:
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
                        )
                    else:
                        _LOGGER.debug(
                            "%s: No activity found while polling after maximum of "
                            "%s retries",
                            self._lock.name,
                            max_retries,
                        )
                    return

                # Continue to fetch activity while some is available.
                while (await lock.lock_activity()) is not None:
                    pass
        except asyncio.CancelledError:
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
                )
            else:
                _LOGGER.warning(
                    "%s: Activity poll stopped after the maximum of %s retries",
                    self._lock.name,
                    max_retries,
                )
        except Exception:  # pylint: disable=broad-except
            # Do not leave an unobserved exception in the background task. Keep
            # the traceback in the log for actual programming/protocol errors.
            _LOGGER.exception("%s: Unexpected activity poll failure", self._lock.name)
