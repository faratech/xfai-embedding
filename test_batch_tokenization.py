import asyncio
import httpx
import server as S


def fake_provider(monkeypatch):
    payloads = []
    class Client:
        async def post(self, url, **kwargs):
            payloads.append(kwargs['json']['input'])
            return httpx.Response(200, json={'data': [
                {'index': i, 'embedding': [float(i)]} for i in range(len(payloads[-1]))
            ]})
    async def client(): return Client()
    monkeypatch.setattr(S, 'get_http_client', client)
    monkeypatch.setattr(S, 'OPENAI_API_KEY', 'offline-fake')
    monkeypatch.setattr(S, 'schedule_meter', lambda coroutine: coroutine.close())
    monkeypatch.setattr(S, 'BATCH_REQUEST_DELAY_SECONDS', 0)
    return payloads


def test_untruncated_batch_encodes_each_text_once(monkeypatch):
    payloads = fake_provider(monkeypatch)
    class Encoder:
        def __init__(self): self.calls = []
        def encode(self, text): self.calls.append(text); return [1] * len(text)
        def decode(self, tokens): raise AssertionError('untruncated text decoded')
    encoder = Encoder(); monkeypatch.setattr(S, 'TOKENIZER', encoder)
    asyncio.run(S.get_openai_embeddings_batch(['one', 'two']))
    assert encoder.calls == ['one', 'two'] and payloads == [['one', 'two']]


def test_unicode_truncation_retains_exact_decoded_payload_and_batch_counts(monkeypatch):
    payloads = fake_provider(monkeypatch)
    monkeypatch.setattr(S, 'MAX_TOKENS', 3)
    monkeypatch.setattr(S, 'MAX_BATCH_TOKENS', 4)
    texts = ['🐈हिन्दीé' * 20, '中文🐈' * 20, 'ok']
    prepared = [S.truncate_to_tokens(x, 3) for x in texts]
    expected = []; current = []; count = 0
    for text in prepared:
        n = len(S.TOKENIZER.encode(text))
        if current and count + n > 4: expected.append(current); current = []; count = 0
        current.append(text); count += n
    if current: expected.append(current)
    asyncio.run(S.get_openai_embeddings_batch(texts))
    assert payloads == expected
