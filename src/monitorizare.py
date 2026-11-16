class ConfirmareGest:
    def __init__(self, grace):
        self.grace = grace
        self.reseteaza()

    def reseteaza(self):
        self.gest = None
        self.start = None
        self.ultima_detectie = 0.0
        self.confirmat = None

    def actualizeaza(self, gest, now, timp_confirmare):
        if gest is not None:
            self.ultima_detectie = now
            if gest != self.gest:
                self.gest = gest
                self.start = now
        elif self.gest is not None and now - self.ultima_detectie > self.grace:
            self.reseteaza()

        if self.gest is None:
            return 0.0, False

        progres = min((now - self.start) / timp_confirmare, 1.0)
        confirmat_acum = progres >= 1.0 and self.confirmat != self.gest
        if confirmat_acum:
            self.confirmat = self.gest
        return progres, confirmat_acum


class StareSustinuta:
    def __init__(self, interval_repetare, grace=0.0):
        self.interval_repetare = interval_repetare
        self.grace = grace
        self.reseteaza()

    def reseteaza(self):
        self.start = None
        self.ultima_detectie = 0.0
        self.activa = False
        self.ultima_anuntare = None

    def actualizeaza(self, conditie, now, prag):
        if conditie:
            if self.start is None:
                self.start = now
            self.ultima_detectie = now
        elif self.start is not None and not self.activa and now - self.ultima_detectie > self.grace:
            self.reseteaza()

        if self.start is None:
            return 0.0, False

        progres = min((now - self.start) / prag, 1.0)
        if progres < 1.0:
            return progres, False

        self.activa = True
        if self.ultima_anuntare is None or now - self.ultima_anuntare > self.interval_repetare:
            self.ultima_anuntare = now
            return progres, True
        return progres, False
