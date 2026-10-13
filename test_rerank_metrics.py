from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import rerank_metrics as M


def test_overlapping_inference_cpu_is_counted_once(monkeypatch):
    cpu = [0.0]
    monkeypatch.setattr(M.time, 'process_time', lambda: cpu[0])
    first = threading.Event(); second = threading.Event()
    release_first = threading.Event(); release_second = threading.Event()
    class Model:
        def predict(self, pairs, **kwargs):
            event, release = ((first, release_first) if pairs[0][0] == 'first' else (second, release_second))
            event.set(); assert release.wait(3); return [1.0]
    metrics = M.InferenceMetrics()
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(metrics.predict, Model(), [('first', 'private input')], 'pairs')
        assert first.wait(3)
        b = pool.submit(metrics.predict, Model(), [('second', 'private input')], 'documents')
        assert second.wait(3)
        cpu[0] = 5.0; release_first.set(); assert a.result() == [1.0]
        assert metrics.snapshot()['inference_busy_process_cpu_seconds'] == 5.0
        cpu[0] = 7.0; release_second.set(); assert b.result() == [1.0]
    result = metrics.snapshot()
    assert result['inference_busy_process_cpu_seconds'] == 7.0
    assert result['inference_active'] == 0 and result['inference_maximum_active'] == 2
    assert result['inferred_pairs'] == 2 and result['inference_batch_sizes'] == {1: 2}
    assert 'private input' not in str(result)


def test_failed_prediction_releases_active_window():
    class Model:
        def predict(self, pairs, **kwargs): raise RuntimeError('model failure')
    metrics = M.InferenceMetrics()
    with pytest.raises(RuntimeError): metrics.predict(Model(), [('a', 'b')], 'pairs')
    assert metrics.snapshot()['inference_active'] == 0
    assert metrics.snapshot()['inference_errors'] == 1
