import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from yalexs_ble.activity import ActivityManager
from yalexs_ble.const import (
    LOCK_ACTIVITY_POLL_INTERVAL,
    ConnectionInfo,
    DoorActivity,
    LockActivity,
    LockInfo,
    LockOperationSource,
    LockStatus,
)
from yalexs_ble.session import DisconnectedError


def _activity() -> LockActivity:
    return LockActivity(
        timestamp=datetime(2026, 1, 1),
        status=LockStatus.LOCKED,
        source=LockOperationSource.MANUAL,
    )


def test_activity_callbacks_can_be_registered_and_removed() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    manager = ActivityManager(bridge)
    received: list[DoorActivity | LockActivity] = []

    unregister = manager.register_activity_callback(
        lambda activity, _info, _connection: received.append(activity)
    )
    manager.handle_activities([_activity()])
    unregister()
    manager.handle_activities([_activity()])

    assert received == [_activity()]


@pytest.mark.asyncio
async def test_activity_poll_drains_until_lock_reports_no_activity() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    bridge.operation_lock = asyncio.Lock()
    lock = MagicMock()
    lock.last_activity_was_unknown = False
    lock.lock_activity = AsyncMock(side_effect=[_activity(), _activity(), None])
    bridge.ensure_connected = AsyncMock(return_value=lock)
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)

    await manager._execute_activity_poll(retries=0, max_retries=0, backoff=1)

    assert lock.lock_activity.await_count == 3


@pytest.mark.asyncio
async def test_activity_poll_schedules_follow_up_when_history_is_empty() -> None:
    """Activity polling continues without a live YBA state callback."""
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    bridge.operation_lock = asyncio.Lock()
    lock = MagicMock()
    lock.last_activity_was_unknown = False
    lock.lock_activity = AsyncMock(return_value=None)
    bridge.ensure_connected = AsyncMock(return_value=lock)
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)
    manager.schedule_activity_poll = MagicMock()

    await manager._execute_activity_poll(retries=0, max_retries=0, backoff=1)

    manager.schedule_activity_poll.assert_called_once_with(
        LOCK_ACTIVITY_POLL_INTERVAL,
        reason="empty_response_interval",
    )


def test_activity_poll_does_not_replace_existing_poll_when_not_requested() -> None:
    """Live YBA updates must not starve a pending activity poll."""
    bridge = MagicMock(name="front door")
    manager = ActivityManager(bridge)
    manager._activity_callbacks.append(lambda *_args: None)
    timer = MagicMock()
    manager._cancel_deferred_activity_poll = timer

    manager.schedule_activity_poll(30, replace=False)

    assert manager._cancel_deferred_activity_poll is timer
    timer.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_activity_poll_recovers_from_disconnected_error() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    bridge.operation_lock = asyncio.Lock()
    lock = MagicMock()
    lock.last_activity_was_unknown = False
    lock.lock_activity = AsyncMock(side_effect=DisconnectedError("GATT 133"))
    bridge.ensure_connected = AsyncMock(return_value=lock)
    bridge.handle_disconnected = AsyncMock()
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)
    manager.schedule_activity_poll = MagicMock()

    await manager._execute_activity_poll(retries=0, max_retries=1, backoff=1)

    bridge.handle_disconnected.assert_awaited_once()
    manager.schedule_activity_poll.assert_called_once_with(
        1,
        retries=1,
        max_retries=1,
        backoff=1,
        reason="connection_error_retry",
    )


@pytest.mark.asyncio
async def test_activity_poll_skips_unknown_record_and_drains_following_records() -> (
    None
):
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    bridge.operation_lock = asyncio.Lock()
    lock = MagicMock()
    lock.last_activity_was_unknown = False
    results = [None, _activity(), None]
    unknown_flags = [True, False, False]

    async def lock_activity() -> LockActivity | None:
        result = results.pop(0)
        lock.last_activity_was_unknown = unknown_flags.pop(0)
        return result

    lock.lock_activity = lock_activity
    bridge.ensure_connected = AsyncMock(return_value=lock)
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)

    await manager._execute_activity_poll(retries=0, max_retries=0, backoff=1)

    assert not results


@pytest.mark.asyncio
async def test_forced_disconnect_cancels_pending_activity_poll() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    bridge.operation_lock = asyncio.Lock()
    lock = MagicMock()
    blocker = asyncio.Event()

    async def wait_forever() -> None:
        await blocker.wait()

    lock.lock_activity = wait_forever
    bridge.ensure_connected = AsyncMock(return_value=lock)
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)
    manager._deferred_activity_poll(retries=0, max_retries=0, backoff=1)
    await asyncio.sleep(0)

    with contextlib.suppress(asyncio.CancelledError):
        await manager.execute_forced_disconnect()

    assert manager._activity_poll_task is None
