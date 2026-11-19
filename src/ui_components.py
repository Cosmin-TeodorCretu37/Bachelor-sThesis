import tkinter as tk

from src.config import (
    COLOR_ACCENT,
    COLOR_ALERT_1,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_SEPARATOR,
    COLOR_TEXT_LABEL,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
)

FONT = "Courier"


def separator(parent):
    tk.Frame(parent, bg=COLOR_SEPARATOR, height=1).pack(fill="x", pady=10)


def card(parent, title, variable=None, on_color=COLOR_ACCENT, title_pady=(10, 2)):
    frame = tk.Frame(parent, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
    frame.pack(fill="x")

    header = tk.Frame(frame, bg=COLOR_CARD)
    header.pack(fill="x", padx=15, pady=title_pady)
    tk.Label(header, text=title, font=(FONT, 9, "bold"),
             bg=COLOR_CARD, fg=COLOR_TEXT_LABEL).pack(side="left")
    if variable is not None:
        toggle_button(header, variable, on_color)
    return frame


def toggle_button(parent, variable, on_color):
    button = tk.Button(parent, font=(FONT, 9, "bold"), bd=0, cursor="hand2", width=5)

    def refresh():
        if variable.get():
            button.config(text="ON", bg=on_color, fg="white")
        else:
            button.config(text="OFF", bg=COLOR_BORDER, fg=COLOR_TEXT_SECONDARY)

    def on_click():
        variable.set(not variable.get())
        refresh()

    button.config(command=on_click)
    refresh()
    button.pack(side="right")


def large_text(frame, color):
    label = tk.Label(frame, text="—", font=(FONT, 24, "bold"), bg=COLOR_CARD, fg=color, justify="left")
    label.pack(anchor="w", padx=15)
    return label


def value_row(frame, text, text_color=COLOR_TEXT_SECONDARY, value_color=COLOR_TEXT_SECONDARY, pady=(0, 0)):
    row = tk.Frame(frame, bg=COLOR_CARD)
    row.pack(fill="x", padx=15, pady=pady)
    tk.Label(row, text=text, font=(FONT, 9), bg=COLOR_CARD, fg=text_color).pack(side="left")
    value = tk.Label(row, text="0%", font=(FONT, 9, "bold"), bg=COLOR_CARD, fg=value_color)
    value.pack(side="right")
    return value


class ProgressBar:
    def __init__(self, frame, color, pady):
        self.background = tk.Frame(frame, bg=COLOR_SEPARATOR, height=6)
        self.background.pack(fill="x", padx=15, pady=pady)
        self.bar = tk.Frame(self.background, bg=color, height=6)
        self.bar.place(x=0, y=0, relheight=1, width=0)

    def set(self, progress, color):
        self.bar.config(bg=color)
        self.bar.place(width=int(progress * self.background.winfo_width()))


class AlertIndicator:
    def __init__(self, frame, text, bar_pady):
        self.percent_label = value_row(frame, text)
        self.bar = ProgressBar(frame, COLOR_TEXT_LABEL, bar_pady)

    def set(self, progress, active):
        color = COLOR_ALERT_1 if active else (COLOR_WARNING if progress > 0 else COLOR_TEXT_LABEL)
        self.bar.set(progress, color)
        self.percent_label.config(text=f"{progress * 100:.0f}%", fg=color)


def slider(frame, text, variable, min_value, max_value, step, format_value):
    row = tk.Frame(frame, bg=COLOR_CARD)
    row.pack(fill="x", padx=15, pady=(0, 8))

    tk.Label(row, text=text, font=(FONT, 9), bg=COLOR_CARD,
             fg=COLOR_TEXT_SECONDARY, width=22, anchor="w").pack(side="left")

    value_label = tk.Label(row, text=format_value(variable.get()), font=(FONT, 9, "bold"),
                           bg=COLOR_CARD, fg=COLOR_ACCENT, width=6, anchor="e")
    value_label.pack(side="right")
    variable.trace_add("write", lambda *_: value_label.config(text=format_value(variable.get())))

    tk.Scale(row, variable=variable, from_=min_value, to=max_value, resolution=step,
             orient="horizontal", showvalue=False, bg=COLOR_CARD, fg=COLOR_ACCENT,
             highlightthickness=0, troughcolor=COLOR_SEPARATOR, activebackground=COLOR_ACCENT
             ).pack(side="left", fill="x", expand=True, padx=(8, 8))


def scrollable_panel(parent):
    container = tk.Frame(parent, bg=parent["bg"])
    container.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(container, bg=parent["bg"], highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    panel = tk.Frame(canvas, bg=parent["bg"])
    window_id = canvas.create_window((0, 0), window=panel, anchor="nw")

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(window_id, width=canvas.winfo_width())

    panel.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", update_scroll_region)
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    return panel
