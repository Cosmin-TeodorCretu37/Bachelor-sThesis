import csv

from src import event_log
from src.event_log import EventLog
from src.settings import PatientSettings


def test_saved_settings_are_reloaded(tmp_path):
    path = tmp_path / "settings.json"
    PatientSettings(gesture_threshold=0.9, alert_threshold=12.0, emotions_enabled=False).save(path)
    assert PatientSettings.load(path) == PatientSettings(gesture_threshold=0.9, alert_threshold=12.0,
                                                         emotions_enabled=False)


def test_missing_file_gives_defaults(tmp_path):
    assert PatientSettings.load(tmp_path / "missing.json") == PatientSettings()


def test_corrupted_file_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert PatientSettings.load(path) == PatientSettings()


def test_invalid_values_are_replaced_with_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"gesture_threshold": "high", "gestures_enabled": 1, "alert_threshold": 10, "unknown": 5}',
                    encoding="utf-8")
    settings = PatientSettings.load(path)
    assert settings.gesture_threshold == PatientSettings().gesture_threshold
    assert settings.gestures_enabled is True
    assert settings.alert_threshold == 10


def test_event_log_writes_header_and_events(tmp_path):
    log = EventLog(tmp_path / "logs")
    log.record(event_log.START)
    log.record(event_log.COMMAND, "I am in pain. \nNeed painkillers.")

    files = list((tmp_path / "logs").glob("events_*.csv"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time", "event", "details"]
    assert [r[1:] for r in rows[1:]] == [[event_log.START, ""], [event_log.COMMAND, "I am in pain.  Need painkillers."]]
