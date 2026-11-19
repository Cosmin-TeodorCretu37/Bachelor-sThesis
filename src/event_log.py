import csv
import logging
from datetime import datetime

from src.config import EVENT_LOG_DIR

log = logging.getLogger(__name__)

START = "START"
STOP = "STOP"
COMMAND = "COMMAND"
CANCEL = "CANCEL"
EMERGENCY_ALERT = "EMERGENCY_ALERT"
DISCOMFORT_ALERT = "DISCOMFORT_ALERT"


class EventLog:
    def __init__(self, directory=EVENT_LOG_DIR):
        self.directory = directory

    def record(self, event, details=""):
        now = datetime.now()
        path = self.directory / f"events_{now:%Y-%m-%d}.csv"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            is_new_file = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_new_file:
                    writer.writerow(["time", "event", "details"])
                writer.writerow([now.strftime("%H:%M:%S"), event, details.replace("\n", " ")])
        except OSError:
            log.exception("The event could not be written to the log")
