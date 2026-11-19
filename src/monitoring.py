class GestureConfirmation:
    def __init__(self, grace):
        self.grace = grace
        self.reset()

    def reset(self):
        self.gesture = None
        self.start = None
        self.last_seen = 0.0
        self.confirmed = None

    def update(self, gesture, now, confirmation_time):
        if gesture is not None:
            self.last_seen = now
            if gesture != self.gesture:
                self.gesture = gesture
                self.start = now
        elif self.gesture is not None and now - self.last_seen > self.grace:
            self.reset()

        if self.gesture is None:
            return 0.0, False

        progress = min((now - self.start) / confirmation_time, 1.0)
        just_confirmed = progress >= 1.0 and self.confirmed != self.gesture
        if just_confirmed:
            self.confirmed = self.gesture
        return progress, just_confirmed


class SustainedState:
    def __init__(self, repeat_interval, grace=0.0):
        self.repeat_interval = repeat_interval
        self.grace = grace
        self.reset()

    def reset(self):
        self.start = None
        self.last_seen = 0.0
        self.active = False
        self.last_announcement = None

    def update(self, condition, now, threshold):
        if condition:
            if self.start is None:
                self.start = now
            self.last_seen = now
        elif self.start is not None and not self.active and now - self.last_seen > self.grace:
            self.reset()

        if self.start is None:
            return 0.0, False

        progress = min((now - self.start) / threshold, 1.0)
        if progress < 1.0:
            return progress, False

        self.active = True
        if self.last_announcement is None or now - self.last_announcement > self.repeat_interval:
            self.last_announcement = now
            return progress, True
        return progress, False
