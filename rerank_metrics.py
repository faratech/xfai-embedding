"""Private inference counters; process CPU windows are counted once when calls overlap."""
from collections import Counter
import threading
import time


class InferenceMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.maximum_active = 0
        self.started_cpu = 0.0
        self.busy_cpu = 0.0
        self.calls = self.pairs = self.errors = 0
        self.wall = self.thread_cpu = self.queue_wall = 0.0
        self.batch_sizes = Counter()
        self.kinds = Counter()

    def predict(self, model, pairs, kind, queued_at=None):
        wall = time.monotonic()
        thread_cpu = time.thread_time()
        with self.lock:
            if self.active == 0:
                self.started_cpu = time.process_time()
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls += 1
            self.pairs += len(pairs)
            self.kinds[kind] += len(pairs)
            self.batch_sizes[len(pairs)] += 1
            if queued_at is not None:
                self.queue_wall += max(0.0, wall - queued_at)
        failed = False
        try:
            return model.predict(pairs, show_progress_bar=False)
        except BaseException:
            failed = True
            raise
        finally:
            with self.lock:
                self.active -= 1
                if self.active == 0:
                    self.busy_cpu += time.process_time() - self.started_cpu
                self.errors += int(failed)
                self.wall += time.monotonic() - wall
                self.thread_cpu += time.thread_time() - thread_cpu

    def snapshot(self):
        with self.lock:
            busy = self.busy_cpu + (time.process_time() - self.started_cpu if self.active else 0.0)
            return {'inference_calls': self.calls, 'inferred_pairs': self.pairs,
                    'inference_errors': self.errors, 'inference_active': self.active,
                    'inference_maximum_active': self.maximum_active,
                    'inference_busy_process_cpu_seconds': busy,
                    'inference_worker_thread_cpu_seconds': self.thread_cpu,
                    'inference_call_wall_seconds': self.wall,
                    'inference_queue_wall_seconds': self.queue_wall,
                    'inference_pairs_by_kind': dict(self.kinds),
                    'inference_batch_sizes': dict(self.batch_sizes)}
