import tkinter as tk

from src.config import (
    CULOARE_ACCENT,
    CULOARE_ALERTA_1,
    CULOARE_AVERTIZARE,
    CULOARE_BORDURA,
    CULOARE_CARD,
    CULOARE_SEPARATOR,
    CULOARE_TEXT_LABEL,
    CULOARE_TEXT_SECUNDAR,
)

FONT = "Courier"


def separator(parinte):
    tk.Frame(parinte, bg=CULOARE_SEPARATOR, height=1).pack(fill="x", pady=10)


def card(parinte, titlu, variabila=None, culoare_on=CULOARE_ACCENT, pady_titlu=(10, 2)):
    cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA, highlightthickness=1)
    cadru.pack(fill="x")

    antet = tk.Frame(cadru, bg=CULOARE_CARD)
    antet.pack(fill="x", padx=15, pady=pady_titlu)
    tk.Label(antet, text=titlu, font=(FONT, 9, "bold"),
             bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
    if variabila is not None:
        buton_toggle(antet, variabila, culoare_on)
    return cadru


def buton_toggle(parinte, variabila, culoare_on):
    buton = tk.Button(parinte, font=(FONT, 9, "bold"), bd=0, cursor="hand2", width=5)

    def update_vizual():
        if variabila.get():
            buton.config(text="ON", bg=culoare_on, fg="white")
        else:
            buton.config(text="OFF", bg=CULOARE_BORDURA, fg=CULOARE_TEXT_SECUNDAR)

    def la_click():
        variabila.set(not variabila.get())
        update_vizual()

    buton.config(command=la_click)
    update_vizual()
    buton.pack(side="right")


def text_mare(cadru, culoare):
    eticheta = tk.Label(cadru, text="—", font=(FONT, 24, "bold"), bg=CULOARE_CARD, fg=culoare, justify="left")
    eticheta.pack(anchor="w", padx=15)
    return eticheta


def rand_valoare(cadru, eticheta, culoare_text=CULOARE_TEXT_SECUNDAR, culoare_valoare=CULOARE_TEXT_SECUNDAR,
                 pady=(0, 0)):
    rand = tk.Frame(cadru, bg=CULOARE_CARD)
    rand.pack(fill="x", padx=15, pady=pady)
    tk.Label(rand, text=eticheta, font=(FONT, 9), bg=CULOARE_CARD, fg=culoare_text).pack(side="left")
    valoare = tk.Label(rand, text="0%", font=(FONT, 9, "bold"), bg=CULOARE_CARD, fg=culoare_valoare)
    valoare.pack(side="right")
    return valoare


class BaraProgres:
    def __init__(self, cadru, culoare, pady):
        self.fundal = tk.Frame(cadru, bg=CULOARE_SEPARATOR, height=6)
        self.fundal.pack(fill="x", padx=15, pady=pady)
        self.bara = tk.Frame(self.fundal, bg=culoare, height=6)
        self.bara.place(x=0, y=0, relheight=1, width=0)

    def seteaza(self, progres, culoare):
        self.bara.config(bg=culoare)
        self.bara.place(width=int(progres * self.fundal.winfo_width()))


class IndicatorAlerta:
    def __init__(self, cadru, eticheta, pady_bara):
        self.lbl_procent = rand_valoare(cadru, eticheta)
        self.bara = BaraProgres(cadru, CULOARE_TEXT_LABEL, pady_bara)

    def seteaza(self, progres, activa):
        culoare = CULOARE_ALERTA_1 if activa else (CULOARE_AVERTIZARE if progres > 0 else CULOARE_TEXT_LABEL)
        self.bara.seteaza(progres, culoare)
        self.lbl_procent.config(text=f"{progres * 100:.0f}%", fg=culoare)


def slider(cadru, eticheta, variabila, val_min, val_max, pas, format_val):
    rand = tk.Frame(cadru, bg=CULOARE_CARD)
    rand.pack(fill="x", padx=15, pady=(0, 8))

    tk.Label(rand, text=eticheta, font=(FONT, 9), bg=CULOARE_CARD,
             fg=CULOARE_TEXT_SECUNDAR, width=22, anchor="w").pack(side="left")

    lbl_val = tk.Label(rand, text=format_val(variabila.get()), font=(FONT, 9, "bold"),
                       bg=CULOARE_CARD, fg=CULOARE_ACCENT, width=6, anchor="e")
    lbl_val.pack(side="right")
    variabila.trace_add("write", lambda *_: lbl_val.config(text=format_val(variabila.get())))

    tk.Scale(rand, variable=variabila, from_=val_min, to=val_max, resolution=pas,
             orient="horizontal", showvalue=False, bg=CULOARE_CARD, fg=CULOARE_ACCENT,
             highlightthickness=0, troughcolor=CULOARE_SEPARATOR, activebackground=CULOARE_ACCENT
             ).pack(side="left", fill="x", expand=True, padx=(8, 8))


def panou_derulant(parinte):
    container = tk.Frame(parinte, bg=parinte["bg"])
    container.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(container, bg=parinte["bg"], highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    panou = tk.Frame(canvas, bg=parinte["bg"])
    id_fereastra = canvas.create_window((0, 0), window=panou, anchor="nw")

    def actualizeaza_scroll(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(id_fereastra, width=canvas.winfo_width())

    panou.bind("<Configure>", actualizeaza_scroll)
    canvas.bind("<Configure>", actualizeaza_scroll)
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    return panou
