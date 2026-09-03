import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from yalexs_ble.activity import ActivityManager
from yalexs_ble.const import (
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
    lock = MagicMock()
    lock.lock_activity = AsyncMock(side_effect=[_activity(), _activity(), None])
    bridge.ensure_connected = AsyncMock(return_value=lock)
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)

    await manager._execute_activity_poll(retries=0, max_retries=0, backoff=1)

    assert lock.lock_activity.await_count == 3


@pytest.mark.asyncio
async def test_activity_poll_recovers_from_disconnected_error() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
    lock = MagicMock()
    lock.lock_activity = AsyncMock(side_effect=DisconnectedError("GATT 133"))
    bridge.ensure_connected = AsyncMock(return_value=lock)
    bridge.handle_disconnected = AsyncMock()
    manager = ActivityManager(bridge)
    manager.register_activity_callback(lambda *_args: None)
    manager.schedule_activity_poll = MagicMock()

    await manager._execute_activity_poll(retries=0, max_retries=1, backoff=1)

    bridge.handle_disconnected.assert_awaited_once()
    manager.schedule_activity_poll.assert_called_once_with(
        1, retries=1, max_retries=1, backoff=1
    )


@pytest.mark.asyncio
async def test_forced_disconnect_cancels_pending_activity_poll() -> None:
    bridge = MagicMock(
        name="front door",
        lock_info=LockInfo("Yale", "ASL-03", "123", "1.0"),
        connection_info=ConnectionInfo(-42),
    )
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
