import asyncio
import pytest
import server as S
from pair_score_cache import PairScoreCache


def test_pairs_cache_ids_order_reverse_direction_topk_and_documents(monkeypatch):
    class Model:
        def __init__(self):self.calls=[]
        def predict(self,pairs,**kw):
            self.calls.append(pairs)
            return [float(len(a)-len(b)) for a,b in pairs]
    model=Model();monkeypatch.setattr(S,'RERANK_MODEL',model)
    monkeypatch.setattr(S,'PAIR_CACHE_ENABLED',True);monkeypatch.setattr(S,'_PAIR_SCORE_CACHE',PairScoreCache())
    async def run():
        request=S.RerankRequest(pairs=[S.RerankPair(id='left',query='abc',document='d'),S.RerankPair(id='reverse',query='d',document='abc'),S.RerankPair(id='tie',query='abc',document='d')])
        first=await S.rerank(request)
        assert [(v.id,v.index,v.score) for v in first.results]==[('left',0,2.),('tie',2,2.),('reverse',1,-2.)]
        request.pairs[0].id='new';request.top_k=1
        second=await S.rerank(request);assert second.results[0].id=='new' and len(second.results)==1
        assert len(model.calls)==1 and len(model.calls[0])==2
        docs=await S.rerank(S.RerankRequest(query='abc',documents=[S.RerankDocument(id='doc',text='d')]))
        assert docs.results[0].id=='doc' and docs.results[0].score==2.
    asyncio.run(run())


def test_pairs_no_default_truncation_and_validation(monkeypatch):
    class Model:
        def predict(self,pairs,**kw):return [1.]*len(pairs)
    monkeypatch.setattr(S,'RERANK_MODEL',Model());monkeypatch.setattr(S,'PAIR_CACHE_ENABLED',True)
    monkeypatch.setattr(S,'_PAIR_SCORE_CACHE',PairScoreCache())
    async def run():
        result=await S.rerank(S.RerankRequest(pairs=[S.RerankPair(query='a',document='b') for _ in range(22)]))
        assert len(result.results)==22 and [r.id for r in result.results]==[str(i) for i in range(22)]
        with pytest.raises(S.HTTPException) as exc:
            await S.rerank(S.RerankRequest(query='a',pairs=[S.RerankPair(query='a',document='b')]))
        assert exc.value.status_code==422
    asyncio.run(run())
