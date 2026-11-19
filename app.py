import logging
import time
import tkinter as tk
from collections import Counter, deque

import cv2
from PIL import Image, ImageTk

from src import event_log
from src import ui_components as ui
from src.background_worker import BackgroundWorker
from src.config import (
    ALERT_THRESHOLD,
    COLOR_ACCENT,
    COLOR_ALERT_1,
    COLOR_ALERT_2,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_SEPARATOR,
    COLOR_SUCCESS,
    COLOR_TEXT_LABEL,
    COLOR_TEXT_SECONDARY,
    COMMAND_HISTORY_LENGTH,
    DISCOMFORT_MESSAGE,
    DISCOMFORT_REALERT_INTERVAL,
    EMERGENCY_MESSAGE,
    EMERGENCY_REALERT_INTERVAL,
    EMOTION_FRAME_INTERVAL,
    EMOTION_HISTORY_LENGTH,
    FACE_GRACE,
    GESTURE_CONFIDENCE_THRESHOLD,
    GESTURE_CONFIRMATION_TIME,
    GESTURE_GRACE,
    LOOP_INTERVAL_MS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.detectors import draw_hand
from src.emotion_classifier import EmotionClassifier
from src.event_log import EventLog
from src.gesture_classifier import GestureClassifier
from src.monitoring import GestureConfirmation, SustainedState
from src.settings import PatientSettings
from src.speech import TextToSpeech

log = logging.getLogger(__name__)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Assistive Medical System")
        self.root.state("zoomed")
        self.root.configure(bg=COLOR_BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        log.info("Loading models...")
        self.emotion_classifier = EmotionClassifier()
        self.gesture_classifier = GestureClassifier()
        self.emotion_worker = BackgroundWorker(self.emotion_classifier.process_frame)
        self.tts = TextToSpeech()
        self.event_log = EventLog()
        log.info("Models loaded.")

        #emotion/discomfort
        self.emotion_history = deque(maxlen=EMOTION_HISTORY_LENGTH)
        self.last_emotion_result = None
        self.patient_state = None
        self.frame_counter = 0
        self.last_face_seen = 0.0
        self.discomfort = SustainedState(DISCOMFORT_REALERT_INTERVAL)

        #gesture/command
        self.gesture_confirmation = GestureConfirmation(GESTURE_GRACE)
        self.current_gesture = None
        self.command_history = deque(maxlen=COMMAND_HISTORY_LENGTH)

        #emergency state
        self.emergency = SustainedState(EMERGENCY_REALERT_INTERVAL, grace=GESTURE_GRACE)

        self.fps_time = time.time()
        self.fps = 0.0

        settings = PatientSettings.load()

        #toggle on/off
        self.gestures_enabled = tk.BooleanVar(value=settings.gestures_enabled)
        self.emotions_enabled = tk.BooleanVar(value=settings.emotions_enabled)
        self.alerts_enabled = tk.BooleanVar(value=settings.alerts_enabled)
        self.history_enabled = tk.BooleanVar(value=settings.history_enabled)

        # live adjustable settings
        self.gesture_threshold = tk.DoubleVar(value=settings.gesture_threshold)
        self.confirmation_time = tk.DoubleVar(value=settings.confirmation_time)
        self.alert_threshold = tk.DoubleVar(value=settings.alert_threshold)

        self._build_ui()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.camera_available = self.cap.isOpened()

        self.running = True
        self.loop_id = None
        self.event_log.record(event_log.START)
        self._update()

    #UI
    def _build_ui(self):
        header = tk.Frame(self.root, bg=COLOR_BACKGROUND)
        header.pack(fill="x", padx=20, pady=(15, 0))

        tk.Label(header, text="ASSISTIVE MEDICAL SYSTEM", font=(ui.FONT, 16, "bold"),
                 bg=COLOR_BACKGROUND, fg=COLOR_ACCENT).pack(side="left")

        self.fps_label = tk.Label(header, text="FPS: 0", font=(ui.FONT, 11),
                                  bg=COLOR_BACKGROUND, fg=COLOR_TEXT_LABEL)
        self.fps_label.pack(side="right", padx=10)

        self.banner = tk.Label(self.root, text="", font=(ui.FONT, 13, "bold"),
                               bg=COLOR_BACKGROUND, fg="white", pady=8)
        self.banner.pack(fill="x", padx=20, pady=(10, 0))

        body = tk.Frame(self.root, bg=COLOR_BACKGROUND)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        video_frame = tk.Frame(body, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        video_frame.pack(side="left", padx=(0, 15))
        self.video_label = tk.Label(video_frame, bg=COLOR_CARD)
        self.video_label.pack(padx=2, pady=2)

        panel = ui.scrollable_panel(body)

        frame = ui.card(panel, "ASSISTIVE COMMAND", self.gestures_enabled, COLOR_ACCENT)
        self.gesture_label = ui.large_text(frame, COLOR_ACCENT)
        self.confirmation_label = ui.value_row(frame, "Command confirmation:", COLOR_TEXT_LABEL,
                                               COLOR_ACCENT, pady=(4, 4))
        self.gesture_bar = ui.ProgressBar(frame, COLOR_ACCENT, (0, 12))
        ui.separator(panel)

        frame = ui.card(panel, "PATIENT STATE", self.emotions_enabled, COLOR_SUCCESS)
        self.emotion_label = ui.large_text(frame, COLOR_SUCCESS)
        self.emotion_confidence_label = tk.Label(frame, text="", font=(ui.FONT, 9),
                                                 bg=COLOR_CARD, fg=COLOR_TEXT_SECONDARY)
        self.emotion_confidence_label.pack(anchor="w", padx=15, pady=(0, 8))
        self.emotion_bar = ui.ProgressBar(frame, COLOR_SUCCESS, (0, 12))
        ui.separator(panel)

        frame = ui.card(panel, "ALERT MONITORING", self.alerts_enabled, COLOR_ALERT_1, title_pady=(10, 8))
        self.discomfort_indicator = ui.AlertIndicator(frame, "Sustained discomfort / pain", (2, 10))
        self.emergency_indicator = ui.AlertIndicator(frame, "Sustained emergency signal", (2, 12))
        ui.separator(panel)

        frame = ui.card(panel, "COMMAND HISTORY", self.history_enabled, COLOR_ACCENT, title_pady=(10, 6))
        self.history_label = tk.Label(frame, text="—", font=(ui.FONT, 10), bg=COLOR_CARD,
                                      fg=COLOR_TEXT_SECONDARY, wraplength=340, justify="left")
        self.history_label.pack(anchor="w", padx=15, pady=(0, 12))
        ui.separator(panel)

        frame = ui.card(panel, "PATIENT SETTINGS", title_pady=(10, 6))
        ui.slider(frame, "Min. gesture confidence", self.gesture_threshold, 0.50, 0.95, 0.01, lambda v: f"{v:.0%}")
        ui.slider(frame, "Confirmation time (s)", self.confirmation_time, 1.0, 5.0, 0.1, lambda v: f"{v:.1f}s")
        ui.slider(frame, "Alert time (s)", self.alert_threshold, 3.0, 15.0, 0.5, lambda v: f"{v:.1f}s")
        tk.Frame(frame, bg=COLOR_BORDER, height=1).pack(fill="x", padx=15)
        tk.Button(frame, text="Reset to defaults", font=(ui.FONT, 8), bg=COLOR_CARD,
                  fg=COLOR_TEXT_SECONDARY, activebackground=COLOR_SEPARATOR, activeforeground=COLOR_ACCENT,
                  bd=0, padx=10, pady=6, cursor="hand2",
                  command=self._reset_settings).pack(anchor="e", padx=15, pady=8)

        tk.Button(self.root, text="⏹  Stop application", font=(ui.FONT, 11, "bold"),
                  bg=COLOR_SEPARATOR, fg=COLOR_ALERT_1, activebackground="#2d1b1b",
                  activeforeground=COLOR_ALERT_1, bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._close).pack(pady=(5, 15))

    def _reset_settings(self):
        self.gesture_threshold.set(GESTURE_CONFIDENCE_THRESHOLD)
        self.confirmation_time.set(GESTURE_CONFIRMATION_TIME)
        self.alert_threshold.set(ALERT_THRESHOLD)

    def _current_settings(self):
        return PatientSettings(
            gesture_threshold=self.gesture_threshold.get(),
            confirmation_time=self.confirmation_time.get(),
            alert_threshold=self.alert_threshold.get(),
            gestures_enabled=self.gestures_enabled.get(),
            emotions_enabled=self.emotions_enabled.get(),
            alerts_enabled=self.alerts_enabled.get(),
            history_enabled=self.history_enabled.get(),
        )

    def _cancel_last_command(self):
        self.emergency.reset()
        self.discomfort.reset()

        if self.command_history:
            self.command_history.pop()
            log.info("The last command was removed and the alerts were stopped.")
        else:
            log.info("Empty history. There is no command to cancel.")

    #MAIN LOOP
    def _update(self):
        if not self.running:
            return

        success, frame = self.cap.read() if self.camera_available else (False, None)
        if not success:
            self.banner.config(text="Camera unavailable - check the webcam connection",
                               bg=COLOR_ALERT_2, fg="white")
            self.loop_id = self.root.after(500, self._update)
            return

        frame = cv2.flip(frame, 1)
        if frame.shape[1] != VIDEO_WIDTH or frame.shape[0] != VIDEO_HEIGHT:
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        now = time.time()

        dt = now - self.fps_time
        self.fps_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt

        display = {"command": "—", "gesture_progress": 0.0,
                   "emotion": "—", "emotion_color": COLOR_SUCCESS, "emotion_confidence": 0.0}

        self._submit_emotion_frame(frame)
        self._process_gesture(frame, now, display)
        self._process_emotion(frame, now, display)
        emergency_progress, discomfort_progress = self._process_alerts(now)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self._update_ui(display, discomfort_progress, emergency_progress)

        self.loop_id = self.root.after(LOOP_INTERVAL_MS, self._update)

    def _submit_emotion_frame(self, frame):
        if not self.emotions_enabled.get():
            return
        self.frame_counter += 1
        if self.frame_counter % EMOTION_FRAME_INTERVAL == 0:
            self.emotion_worker.submit(frame.copy())

    def _process_gesture(self, frame, now, display):
        self.current_gesture = None
        if self.gestures_enabled.get():
            result = self.gesture_classifier.process_frame(frame)
            if result and result.confidence > self.gesture_threshold.get():
                draw_hand(frame, result.landmarks)
                self.current_gesture = result.gesture
        else:
            self.gesture_confirmation.reset()

        progress, confirmed = self.gesture_confirmation.update(
            self.current_gesture, now, self.confirmation_time.get())
        gesture = self.gesture_confirmation.gesture
        if gesture is None:
            return

        command = GestureClassifier.MEDICAL_COMMANDS[gesture]
        if confirmed:
            self.tts.say(command)
            if gesture == GestureClassifier.CANCEL_GESTURE:
                self.event_log.record(event_log.CANCEL)
                self._cancel_last_command()
            else:
                self.event_log.record(event_log.COMMAND, command)
                if self.history_enabled.get():
                    self.command_history.append(command)

        display["command"] = command
        display["gesture_progress"] = progress

    def _process_emotion(self, frame, now, display):
        result, is_new = self.emotion_worker.fetch()
        if not self.emotions_enabled.get():
            self.frame_counter = 0
            self.last_emotion_result = None
        elif is_new:
            self.last_emotion_result = result
            if result:
                self.emotion_history.append(result.state)
                self.last_face_seen = now

        if now - self.last_face_seen > FACE_GRACE:
            self.emotion_history.clear()

        self.patient_state = Counter(self.emotion_history).most_common(1)[0][0] if self.emotion_history else None
        res = self.last_emotion_result
        if self.patient_state is None or res is None:
            return

        color_bgr, color_hex = EmotionClassifier.STATE_COLORS[self.patient_state]
        x, y, w, h = res.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color_bgr, -1)
        cv2.putText(frame, f"{self.patient_state} ({res.confidence * 100:.0f}%)", (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 0, 0) if color_bgr != (0, 0, 255) else (255, 255, 255), 2, cv2.LINE_AA)

        display["emotion"] = f"{self.patient_state} ({res.emotion_name})"
        display["emotion_color"] = color_hex
        display["emotion_confidence"] = res.confidence

    def _process_alerts(self, now):
        if not self.alerts_enabled.get():
            self.emergency.reset()
            self.discomfort.reset()
            return 0.0, 0.0

        threshold = self.alert_threshold.get()
        emergency_was_active, discomfort_was_active = self.emergency.active, self.discomfort.active
        emergency_progress, announce_emergency = self.emergency.update(
            self.current_gesture == GestureClassifier.EMERGENCY_GESTURE, now, threshold)
        discomfort_progress, announce_discomfort = self.discomfort.update(
            self.patient_state in EmotionClassifier.DISCOMFORT_STATES, now, threshold)

        if self.emergency.active and not emergency_was_active:
            self.event_log.record(event_log.EMERGENCY_ALERT)
        if self.discomfort.active and not discomfort_was_active:
            self.event_log.record(event_log.DISCOMFORT_ALERT)

        if announce_emergency:
            self.tts.say_urgent(EMERGENCY_MESSAGE)
        elif announce_discomfort:
            self.tts.say_urgent(DISCOMFORT_MESSAGE)
        return emergency_progress, discomfort_progress

    #SIDE PANELS
    def _update_ui(self, display, discomfort_progress, emergency_progress):
        self.gesture_label.config(text=display["command"])
        self.confirmation_label.config(text=f"{display['gesture_progress'] * 100:.0f}%")
        self.gesture_bar.set(display['gesture_progress'],
                             COLOR_SUCCESS if display['gesture_progress'] >= 1.0 else COLOR_ACCENT)

        self.emotion_label.config(text=display["emotion"], fg=display["emotion_color"])
        self.emotion_confidence_label.config(text=f"Model confidence: {display['emotion_confidence'] * 100:.0f}%")
        self.emotion_bar.set(display['emotion_confidence'], display["emotion_color"])

        self.discomfort_indicator.set(discomfort_progress, self.discomfort.active)
        self.emergency_indicator.set(emergency_progress, self.emergency.active)

        flash = int(time.time() * 2) % 2 == 0
        banner_color = COLOR_ALERT_1 if flash else COLOR_ALERT_2
        if self.emergency.active:
            self.banner.config(text="CRITICAL: sustained emergency signal", bg=banner_color, fg="white")
        elif self.discomfort.active:
            self.banner.config(
                text=f"ALERT: discomfort sustained for over {self.alert_threshold.get():.0f} seconds",
                bg=banner_color, fg="white")
        else:
            self.banner.config(text="", bg=COLOR_BACKGROUND)

        #command history
        if self.command_history:
            self.history_label.config(text=" → ".join(c.replace("\n", " ") for c in self.command_history))
        else:
            self.history_label.config(text="—")

        self.fps_label.config(text=f"FPS: {self.fps:.0f}")

    def _close(self):
        self.running = False
        self._current_settings().save()
        self.event_log.record(event_log.STOP)
        if self.loop_id is not None:
            self.root.after_cancel(self.loop_id)
        self.cap.release()
        self.emotion_worker.stop()
        self.emotion_classifier.close()
        self.gesture_classifier.close()
        self.tts.stop()
        self.root.destroy()


def main():
    from src.splash_screen import SplashScreen

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = tk.Tk()
    root.withdraw()

    def start_app():
        root.deiconify()
        App(root)

    SplashScreen(root, start_app)
    root.mainloop()


if __name__ == "__main__":
    main()
