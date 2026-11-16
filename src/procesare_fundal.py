import logging
import threading

log = logging.getLogger(__name__)


class ProcesareFundal:
    def __init__(self, functie):
        self._functie = functie
        self._conditie = threading.Condition()
        self._intrare = None
        self._rezultat = None
        self._rezultat_nou = False
        self._ocupat = False
        self._activ = True
        self._fir = threading.Thread(target=self._ruleaza, daemon=True)
        self._fir.start()

    def trimite(self, date):
        with self._conditie:
            if self._ocupat or self._intrare is not None:
                return False
            self._intrare = date
            self._conditie.notify()
            return True

    def preia(self):
        with self._conditie:
            nou, self._rezultat_nou = self._rezultat_nou, False
            return self._rezultat, nou

    def opreste(self, timeout=2.0):
        with self._conditie:
            self._activ = False
            self._conditie.notify()
        self._fir.join(timeout)

    def _ruleaza(self):
        while True:
            with self._conditie:
                while self._activ and self._intrare is None:
                    self._conditie.wait()
                if not self._activ:
                    return
                date, self._intrare = self._intrare, None
                self._ocupat = True

            try:
                rezultat = self._functie(date)
            except Exception:
                log.exception("Eroare in procesarea de fundal")
                rezultat = None

            with self._conditie:
                self._rezultat = rezultat
                self._rezultat_nou = True
                self._ocupat = False
