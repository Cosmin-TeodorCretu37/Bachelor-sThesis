import tkinter as tk

from src.config import ALERT_THRESHOLD, COLOR_ACCENT, COLOR_BACKGROUND, COLOR_BORDER, COLOR_CARD
from src.config import COLOR_ALERT_1 as COLOR_EMERGENCY
from src.config import COLOR_TEXT_SECONDARY as COLOR_GRAY
from src.gesture_classifier import GestureClassifier

GESTURE_GUIDE = [
    ("Closed fist", 'fist', "#f85149"),
    ("OK", 'ok', "#58a6ff"),
    ("One", 'one', "#f0883e"),
    ("Two", 'two', "#bc8cff"),
    ("Three", 'three', "#3fb950"),
    ("Open palm", 'open_palm', "#8b949e"),
]


class SplashScreen(tk.Toplevel):

    def __init__(self, root, on_start, alert_threshold=ALERT_THRESHOLD):
        super().__init__(root)
        self.root_ref = root
        self.on_start = on_start
        self.alert_threshold = alert_threshold

        self.title("Assistive Medical System - User Guide")
        self.geometry("820x640")
        self.configure(bg=COLOR_BACKGROUND)
        self.resizable(False, False)

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 820) // 2
        y = (self.winfo_screenheight() - 640) // 2
        self.geometry(f"820x640+{x}+{y}")

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close_all)

        self._build_ui()

    def _build_ui(self):
        tk.Frame(self, bg=COLOR_ACCENT, height=4).pack(fill="x")

        header = tk.Frame(self, bg=COLOR_BACKGROUND)
        header.pack(fill="x", padx=30, pady=(20, 0))

        tk.Label(
            header,
            text="ASSISTIVE MEDICAL SYSTEM",
            font=("Courier", 18, "bold"),
            bg=COLOR_BACKGROUND, fg=COLOR_ACCENT,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Non-verbal communication for patients with reduced mobility",
            font=("Courier", 10),
            bg=COLOR_BACKGROUND, fg=COLOR_GRAY,
        ).pack(anchor="w", pady=(4, 0))

        tk.Frame(self, bg=COLOR_BORDER, height=1).pack(fill="x", padx=30, pady=16)

        tk.Label(
            self,
            text="GESTURE GUIDE - place your hand in front of the camera and hold the gesture for 2-3 seconds",
            font=("Courier", 9, "bold"),
            bg=COLOR_BACKGROUND, fg=COLOR_GRAY,
        ).pack(anchor="w", padx=30, pady=(0, 10))

        grid = tk.Frame(self, bg=COLOR_BACKGROUND)
        grid.pack(padx=30, fill="x")

        for col in range(3):
            grid.columnconfigure(col, weight=1, uniform="col")

        for i, (name, gesture, color) in enumerate(GESTURE_GUIDE):
            row = i // 3
            col = i % 3
            command = GestureClassifier.MEDICAL_COMMANDS[gesture]
            self._gesture_card(grid, name, command, color).grid(
                row=row, column=col, padx=6, pady=6, sticky="nsew"
            )

        tk.Frame(self, bg=COLOR_BORDER, height=1).pack(fill="x", padx=30, pady=(18, 12))

        note = tk.Frame(self, bg=COLOR_CARD,
                        highlightbackground=COLOR_EMERGENCY, highlightthickness=1)
        note.pack(fill="x", padx=30)

        tk.Label(
            note,
            text=f"EMERGENCY: keep your fist closed for {self.alert_threshold:.0f} seconds "
                 "to trigger the emergency alarm with automatic voice alert.",
            font=("Courier", 9),
            bg=COLOR_CARD, fg=COLOR_EMERGENCY,
            wraplength=740, justify="left", pady=10, padx=12,
        ).pack(anchor="w")

        tk.Button(
            self,
            text="Start system",
            font=("Courier", 12, "bold"),
            bg=COLOR_ACCENT, fg=COLOR_BACKGROUND,
            activebackground="#79c0ff", activeforeground=COLOR_BACKGROUND,
            bd=0, padx=24, pady=10, cursor="hand2",
            command=self._start,
        ).pack(pady=20)

    def _gesture_card(self, parent, name, command, color):
        frame = tk.Frame(
            parent, bg=COLOR_CARD,
            highlightbackground=color, highlightthickness=1,
        )

        tk.Label(
            frame, text=name,
            font=("Courier", 13, "bold"),
            bg=COLOR_CARD, fg=color,
        ).pack(pady=(18, 4))

        tk.Frame(frame, bg=color, height=1).pack(fill="x", padx=20, pady=6)

        tk.Label(
            frame, text=command,
            font=("Courier", 9),
            bg=COLOR_CARD, fg=COLOR_GRAY,
            wraplength=200, justify="center",
        ).pack(pady=(0, 18), padx=10)

        return frame

    def _start(self):
        self.destroy()
        self.on_start()

    def _close_all(self):
        self.root_ref.destroy()
