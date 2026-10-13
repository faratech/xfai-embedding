import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pair_score_cache import PairScoreCache


def test_exact_order_duplicates_lru_and_model_identity():
    async def run():
        cache = PairScoreCache(2)
        calls = []
        def predict(pairs):
            calls.append(pairs)
            return [float(len(a)-len(b)) for a,b in pairs]
        with ThreadPoolExecutor(1) as pool:
            pairs=[(' A','b'),(' A','b'),('a','B')]
            assert await cache.scores('m',pairs,predict,pool)==[1.,1.,0.]
            assert len(calls[0])==2
            await cache.scores('m',[(' A','b')],predict,pool)
            assert len(calls)==1
            await cache.scores('other',[(' A','b')],predict,pool)
            assert len(calls)==2 and len(cache.values)==2
            await cache.scores('m',[('b',' A')],predict,pool)
            assert len(calls)==3
    asyncio.run(run())


def test_overlapping_requests_cancelled_owner_keeps_shared_inference():
    async def run():
        cache=PairScoreCache();entered=threading.Event();release=threading.Event();calls=[]
        def predict(pairs):
            calls.append(pairs);entered.set();assert release.wait(2)
            return [3.]*len(pairs)
        with ThreadPoolExecutor(2) as pool:
            first=asyncio.create_task(cache.scores('m',[('x','y')],predict,pool))
            await asyncio.to_thread(entered.wait,2)
            second=asyncio.create_task(cache.scores('m',[('x','y')],predict,pool))
            await asyncio.sleep(0);first.cancel()
            with pytest.raises(asyncio.CancelledError):await first
            release.set();assert await second==[3.]
            assert len(calls)==1 and not cache.pending
            assert await cache.scores('m',[('x','y')],predict,pool)==[3.]
    asyncio.run(run())


@pytest.mark.parametrize('output', [[],[1.,2.],[float('nan')],[float('inf')]])
def test_invalid_outputs_never_cached_and_retry(output):
    async def run():
        cache=PairScoreCache()
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(ValueError):
                await cache.scores('m',[('x','y')],lambda _:output,pool)
            assert not cache.values and not cache.pending
            assert await cache.scores('m',[('x','y')],lambda _:[4.],pool)==[4.]
    asyncio.run(run())


def test_backend_failure_releases_every_waiter():
    async def run():
        cache=PairScoreCache()
        def fail(_):raise RuntimeError('offline failure')
        with ThreadPoolExecutor(1) as pool:
            result=await asyncio.gather(cache.scores('m',[('x','y')],fail,pool),cache.scores('m',[('x','y')],fail,pool),return_exceptions=True)
            assert all(isinstance(r,RuntimeError) for r in result)
            assert not cache.pending and not cache.values
    asyncio.run(run())
