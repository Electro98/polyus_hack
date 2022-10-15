import asyncio
from contextlib import suppress


class Periodic:
    """Class for periodic async task."""

    def __init__(self, func, time):
        """Create simple asynchronous task."""
        self.func = func
        self.time = time
        self.is_started = False
        self._task = None

    async def start(self):
        """Start task execution."""
        if not self.is_started:
            self.is_started = True
            # Start task to call func periodically:
            self._task = asyncio.ensure_future(self._run())

    async def stop(self):
        """Stop task execution."""
        if self.is_started:
            self.is_started = False
            # Stop task and await it stopped:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self):
        while True:
            timer = asyncio.sleep(self.time)
            self.func()
            await timer
