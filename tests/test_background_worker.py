import threading
import time

from src.background_worker import BackgroundWorker


def wait_for_result(worker, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result, is_new = worker.fetch()
        if is_new:
            return result
        time.sleep(0.01)
    raise TimeoutError


def test_result_is_fetched_only_once():
    worker = BackgroundWorker(lambda x: x * 2)
    assert worker.submit(3)
    assert wait_for_result(worker) == 6
    assert worker.fetch() == (6, False)
    worker.stop()


def test_frames_submitted_while_busy_are_ignored():
    release = threading.Event()
    worker = BackgroundWorker(lambda x: release.wait() and x)
    assert worker.submit(1)
    time.sleep(0.05)
    assert not worker.submit(2)
    release.set()
    assert wait_for_result(worker) == 1
    worker.stop()


def test_function_error_does_not_stop_the_thread():
    worker = BackgroundWorker(lambda x: 1 / x)
    worker.submit(0)
    assert wait_for_result(worker) is None
    worker.submit(4)
    assert wait_for_result(worker) == 0.25
    worker.stop()


def test_stop_ends_the_thread():
    worker = BackgroundWorker(lambda x: x)
    worker.stop()
    assert not worker._thread.is_alive()
