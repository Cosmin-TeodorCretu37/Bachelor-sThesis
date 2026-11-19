import logging
import threading

log = logging.getLogger(__name__)


class BackgroundWorker:
    def __init__(self, function):
        self._function = function
        self._condition = threading.Condition()
        self._input = None
        self._result = None
        self._has_new_result = False
        self._busy = False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, data):
        with self._condition:
            if self._busy or self._input is not None:
                return False
            self._input = data
            self._condition.notify()
            return True

    def fetch(self):
        with self._condition:
            is_new, self._has_new_result = self._has_new_result, False
            return self._result, is_new

    def stop(self, timeout=2.0):
        with self._condition:
            self._running = False
            self._condition.notify()
        self._thread.join(timeout)

    def _run(self):
        while True:
            with self._condition:
                while self._running and self._input is None:
                    self._condition.wait()
                if not self._running:
                    return
                data, self._input = self._input, None
                self._busy = True

            try:
                result = self._function(data)
            except Exception:
                log.exception("Error in background processing")
                result = None

            with self._condition:
                self._result = result
                self._has_new_result = True
                self._busy = False
