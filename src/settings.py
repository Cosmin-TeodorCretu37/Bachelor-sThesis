import json
import logging
from dataclasses import asdict, dataclass, fields

from src.config import ALERT_THRESHOLD, GESTURE_CONFIDENCE_THRESHOLD, GESTURE_CONFIRMATION_TIME, SETTINGS_FILE

log = logging.getLogger(__name__)


def _is_valid(value, default):
    if isinstance(default, bool):
        return isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class PatientSettings:
    gesture_threshold: float = GESTURE_CONFIDENCE_THRESHOLD
    confirmation_time: float = GESTURE_CONFIRMATION_TIME
    alert_threshold: float = ALERT_THRESHOLD
    gestures_enabled: bool = True
    emotions_enabled: bool = True
    alerts_enabled: bool = True
    history_enabled: bool = True

    @classmethod
    def load(cls, path=SETTINGS_FILE):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            defaults = cls()
            values = {}
            for field in fields(cls):
                default = getattr(defaults, field.name)
                value = data.get(field.name, default)
                values[field.name] = value if _is_valid(value, default) else default
            return cls(**values)
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, AttributeError):
            log.warning("The settings file could not be read, using default values.")
            return cls()

    def save(self, path=SETTINGS_FILE):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except OSError:
            log.exception("The settings could not be saved")
