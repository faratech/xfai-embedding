"""Exact, bounded pair score cache with cancellation-safe async miss coalescing."""
import asyncio
import math
import hashlib
from collections import OrderedDict


class PairScoreCache:
    def __init__(self, maximum=50_000):
        self.maximum = maximum
        self.values = OrderedDict()
        self.pending = {}
        self.tasks = set()
        self.hits = self.misses = self.coalesced = 0

    async def scores(self, identity, pairs, predict, executor):
        loop = asyncio.get_running_loop()
        waiting, owned, missing = [], [], []
        # No await before registering all misses: concurrent requests on this
        # event loop observe pending work before either dispatches inference.
        for pair in pairs:
            a,b=(v.encode('utf-8') for v in pair)
            digest=hashlib.sha256(len(a).to_bytes(8,'big')+a+len(b).to_bytes(8,'big')+b).digest()
            key = (identity,digest)
            if key in self.values:
                self.hits += 1
                self.values.move_to_end(key)
                future = loop.create_future()
                future.set_result(self.values[key])
            elif key in self.pending:
                self.coalesced += 1
                future = self.pending[key]
            else:
                self.misses += 1
                future = loop.create_future()
                # An abandoned request must not leave an unobserved exception.
                future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
                self.pending[key] = future
                owned.append((key, future))
                missing.append(pair)
            waiting.append(future)
        if owned:
            task = asyncio.create_task(self._fill(owned, missing, predict, executor))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return await asyncio.gather(*(asyncio.shield(f) for f in waiting))

    async def _fill(self, owned, pairs, predict, executor):
        try:
            raw = await asyncio.get_running_loop().run_in_executor(executor, predict, pairs)
            values = [float(v) for v in raw]
            if len(values) != len(owned) or not all(math.isfinite(v) for v in values):
                raise ValueError('Incomplete or nonfinite cross-encoder output')
            for (key, future), value in zip(owned, values):
                self.values[key] = value
                self.values.move_to_end(key)
                while len(self.values) > self.maximum:
                    self.values.popitem(last=False)
                future.set_result(value)
        except BaseException as exc:
            for _, future in owned:
                if not future.done():
                    future.set_exception(exc)
        finally:
            for key, future in owned:
                if self.pending.get(key) is future:
                    del self.pending[key]

    async def finish(self):
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks))
