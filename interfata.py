import time
import tkinter as tk
from collections import deque, Counter

import cv2
from PIL import Image, ImageTk

from src import repere_mana
from src.clasificator_emotii import ClasificatorEmotii
from src.clasificator_gesturi import ClasificatorGesturi
from src.monitorizare import ConfirmareGest, StareSustinuta
from src.procesare_fundal import ProcesareFundal
from src.sinteza_vocala import SintezaVocala

#config
PRAG_INCREDERE_GEST = 0.75          #prag ignorare gest
TIMP_CONFIRMARE_GEST = 2.5          #sec. gest -> comanda
LUNGIME_ISTORIC_EMOTII = 10          #cadre vot majoritar
PRAG_ALERTA = 7.0

GRACE_FATA = 1.5                    #toleranta fata
GRACE_GEST = 1.0                    #toleranta gest

INTERVAL_RE_ALERTA_DISCONFORT = 15  #re-anunt vocal
INTERVAL_RE_ALERTA_URGENTA = 5

INTERVAL_EMOTII = 3
INTERVAL_BUCLA_MS = 1
LATIME_VIDEO, INALTIME_VIDEO = 640, 480

STARI_DISCONFORT = ('Discomfort/Pain', 'Discomfort')
MESAJ_URGENTA = "Critical alert. Emergency signal sustained. Please respond immediately."
MESAJ_DISCONFORT = "Attention. Patient has been showing signs of discomfort for an extended period."

CULOARE_FUNDAL = "#0d1117"
CULOARE_CARD = "#161b22"
CULOARE_BORDURA = "#30363d"
CULOARE_TEXT_SECUNDAR = "#8b949e"
CULOARE_TEXT_LABEL = "#484f58"
CULOARE_ACCENT = "#58a6ff"
CULOARE_ALERTA_1 = "#f85149"
CULOARE_ALERTA_2 = "#7a1f1a"
CULOARE_SEPARATOR = "#21262d"
CULOARE_SUCCES = "#3fb950"
CULOARE_AVERTIZARE = "#d29922"


class Aplicatie:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Medical Asistiv")
        self.root.state("zoomed")
        self.root.configure(bg=CULOARE_FUNDAL)
        self.root.protocol("WM_DELETE_WINDOW", self._inchide)

        print("Se incarca modelele...")
        self.detector_emotii = ClasificatorEmotii()
        self.detector_gesturi = ClasificatorGesturi()
        self.analiza_emotii = ProcesareFundal(self.detector_emotii.proceseaza_cadru)
        self.tts = SintezaVocala()
        print("Modele incarcate.")

        #emotie/disconfort
        self.istoric_emotii = deque(maxlen=LUNGIME_ISTORIC_EMOTII)
        self.ultimul_rezultat_emotie = None
        self.stare_pacient = None
        self.contor_cadre = 0
        self.ultima_detectie_fata = 0.0
        self.disconfort = StareSustinuta(INTERVAL_RE_ALERTA_DISCONFORT)

        #gest/comanda
        self.confirmare_gest = ConfirmareGest(GRACE_GEST)
        self.gest_curent = None
        self.istoric_comenzi = deque(maxlen=7)

        #stare urgenta
        self.urgenta = StareSustinuta(INTERVAL_RE_ALERTA_URGENTA, grace=GRACE_GEST)

        self.fps_time = time.time()
        self.fps = 0.0

        #toggle on/off
        self.activ_gest = tk.BooleanVar(value=True)
        self.activ_emotie = tk.BooleanVar(value=True)
        self.activ_alerte = tk.BooleanVar(value=True)
        self.activ_istoric = tk.BooleanVar(value=True)

        # setari reglabile live
        self.var_prag_gest = tk.DoubleVar(value=PRAG_INCREDERE_GEST)
        self.var_timp_confirmare = tk.DoubleVar(value=TIMP_CONFIRMARE_GEST)
        self.var_prag_alerta = tk.DoubleVar(value=PRAG_ALERTA)

        self._construieste_ui()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LATIME_VIDEO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INALTIME_VIDEO)
        self.camera_disponibila = self.cap.isOpened()

        self.activ = True
        self.id_bucla = None
        self._update()

    #UI
    def _construieste_ui(self):
        header = tk.Frame(self.root, bg=CULOARE_FUNDAL, height=60)
        header.pack(fill="x", padx=20, pady=(15, 0))

        tk.Label(header, text="SISTEM MEDICAL ASISTIV", font=("Courier", 16, "bold"),
                 bg=CULOARE_FUNDAL, fg=CULOARE_ACCENT).pack(side="left")

        self.lbl_fps = tk.Label(header, text="FPS: 0", font=("Courier", 11),
                                bg=CULOARE_FUNDAL, fg=CULOARE_TEXT_LABEL)
        self.lbl_fps.pack(side="right", padx=10)

        self.lbl_banner = tk.Label(self.root, text="", font=("Courier", 13, "bold"),
                                   bg=CULOARE_FUNDAL, fg="white", pady=8)
        self.lbl_banner.pack(fill="x", padx=20, pady=(10, 0))

        corp = tk.Frame(self.root, bg=CULOARE_FUNDAL)
        corp.pack(fill="both", expand=True, padx=20, pady=10)

        cadru_video = tk.Frame(corp, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                               highlightthickness=1)
        cadru_video.pack(side="left", padx=(0, 15))
        self.lbl_video = tk.Label(cadru_video, bg=CULOARE_CARD)
        self.lbl_video.pack(padx=2, pady=2)

        panou_container = tk.Frame(corp, bg=CULOARE_FUNDAL)
        panou_container.pack(side="left", fill="both", expand=True)

        canvas_panou = tk.Canvas(panou_container, bg=CULOARE_FUNDAL, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(panou_container, orient="vertical", command=canvas_panou.yview)
        canvas_panou.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas_panou.pack(side="left", fill="both", expand=True)

        panou = tk.Frame(canvas_panou, bg=CULOARE_FUNDAL)
        id_fereastra_panou = canvas_panou.create_window((0, 0), window=panou, anchor="nw")

        def _actualizeaza_scroll(event=None):
            canvas_panou.configure(scrollregion=canvas_panou.bbox("all"))
            canvas_panou.itemconfig(id_fereastra_panou, width=canvas_panou.winfo_width())

        panou.bind("<Configure>", _actualizeaza_scroll)
        canvas_panou.bind("<Configure>", _actualizeaza_scroll)

        def _scroll_mouse(event):
            canvas_panou.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_panou.bind_all("<MouseWheel>", _scroll_mouse)

        self._card_gest(panou)
        self._separator(panou)
        self._card_emotie(panou)
        self._separator(panou)
        self._card_alerte(panou)
        self._separator(panou)
        self._card_istoric(panou)
        self._separator(panou)
        self._panou_setari(panou)

        tk.Button(self.root, text="⏹  Oprește aplicația", font=("Courier", 11, "bold"),
                  bg=CULOARE_SEPARATOR, fg=CULOARE_ALERTA_1, activebackground="#2d1b1b",
                  activeforeground=CULOARE_ALERTA_1, bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._inchide).pack(pady=(5, 15))

    @staticmethod
    def _separator(parinte):
        tk.Frame(parinte, bg=CULOARE_SEPARATOR, height=1).pack(fill="x", pady=10)

    @staticmethod
    def _card(parinte):
        cadru = tk.Frame(parinte, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA,
                         highlightthickness=1)
        cadru.pack(fill="x")
        return cadru

    def _antet_card(self, cadru, titlu, variabila, culoare_on, pady):
        antet = tk.Frame(cadru, bg=CULOARE_CARD)
        antet.pack(fill="x", padx=15, pady=pady)
        tk.Label(antet, text=titlu, font=("Courier", 9, "bold"),
                 bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        self._adauga_buton_toggle(antet, variabila, culoare_on)

    @staticmethod
    def _bara_progres(cadru, culoare, pady):
        bara_bg = tk.Frame(cadru, bg=CULOARE_SEPARATOR, height=6)
        bara_bg.pack(fill="x", padx=15, pady=pady)
        bara = tk.Frame(bara_bg, bg=culoare, height=6)
        bara.place(x=0, y=0, relheight=1, width=0)
        return bara, bara_bg

    @staticmethod
    def _adauga_buton_toggle(parinte, variabila, culoare_on):
        buton = tk.Button(parinte, font=("Courier", 9, "bold"), bd=0, cursor="hand2", width=5)

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

    def _card_gest(self, parinte):
        cadru = self._card(parinte)
        self._antet_card(cadru, "COMANDA ASISTIVA", self.activ_gest, CULOARE_ACCENT, (10, 2))

        self.lbl_gest = tk.Label(cadru, text="—", font=("Courier", 24, "bold"),
                                 bg=CULOARE_CARD, fg=CULOARE_ACCENT, justify="left")
        self.lbl_gest.pack(anchor="w", padx=15)

        info_frame = tk.Frame(cadru, bg=CULOARE_CARD)
        info_frame.pack(fill="x", padx=15, pady=(4, 4))
        tk.Label(info_frame, text="Confirmare comanda:", font=("Courier", 9),
                 bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(side="left")
        self.lbl_conf_gest = tk.Label(info_frame, text="0%", font=("Courier", 9, "bold"),
                                      bg=CULOARE_CARD, fg=CULOARE_ACCENT)
        self.lbl_conf_gest.pack(side="right")

        self.bara_gest, self.bara_gest_bg = self._bara_progres(cadru, CULOARE_ACCENT, (0, 12))

    def _card_emotie(self, parinte):
        cadru = self._card(parinte)
        self._antet_card(cadru, "STARE PACIENT", self.activ_emotie, CULOARE_SUCCES, (10, 2))

        self.lbl_emotie = tk.Label(cadru, text="—", font=("Courier", 24, "bold"),
                                   bg=CULOARE_CARD, fg=CULOARE_SUCCES)
        self.lbl_emotie.pack(anchor="w", padx=15)

        self.lbl_emotie_raw = tk.Label(cadru, text="", font=("Courier", 9),
                                       bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        self.lbl_emotie_raw.pack(anchor="w", padx=15, pady=(0, 8))

        self.bara_emotie, self.bara_emotie_bg = self._bara_progres(cadru, CULOARE_SUCCES, (0, 12))

    def _card_alerte(self, parinte):
        cadru = self._card(parinte)
        self._antet_card(cadru, "MONITORIZARE ALERTE", self.activ_alerte, CULOARE_ALERTA_1, (10, 8))

        self.lbl_pct_disconfort = self._rand_alerta(cadru, "Disconfort / Durere sustinut(a)")
        self.bara_disconfort, self.bara_disconfort_bg = self._bara_progres(
            cadru, CULOARE_TEXT_LABEL, (2, 10))

        self.lbl_pct_urgenta = self._rand_alerta(cadru, "Semnal de urgenta sustinut")
        self.bara_urgenta, self.bara_urgenta_bg = self._bara_progres(
            cadru, CULOARE_TEXT_LABEL, (2, 12))

    @staticmethod
    def _rand_alerta(cadru, eticheta):
        rand = tk.Frame(cadru, bg=CULOARE_CARD)
        rand.pack(fill="x", padx=15)
        tk.Label(rand, text=eticheta, font=("Courier", 9),
                 bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR).pack(side="left")
        lbl_pct = tk.Label(rand, text="0%", font=("Courier", 9, "bold"),
                           bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        lbl_pct.pack(side="right")
        return lbl_pct

    def _card_istoric(self, parinte):
        cadru = self._card(parinte)
        self._antet_card(cadru, "ISTORIC COMENZI", self.activ_istoric, CULOARE_ACCENT, (10, 6))

        self.lbl_istoric = tk.Label(cadru, text="—", font=("Courier", 10),
                                    bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR,
                                    wraplength=340, justify="left")
        self.lbl_istoric.pack(anchor="w", padx=15, pady=(0, 12))

    def _panou_setari(self, parinte):
        cadru = self._card(parinte)

        tk.Label(cadru, text="SETARI PACIENT", font=("Courier", 9, "bold"),
                 bg=CULOARE_CARD, fg=CULOARE_TEXT_LABEL).pack(anchor="w", padx=15, pady=(10, 6))

        def _slider(eticheta, variabila, val_min, val_max, format_val):
            rand = tk.Frame(cadru, bg=CULOARE_CARD)
            rand.pack(fill="x", padx=15, pady=(0, 8))

            tk.Label(rand, text=eticheta, font=("Courier", 9), bg=CULOARE_CARD,
                     fg=CULOARE_TEXT_SECUNDAR, width=22, anchor="w").pack(side="left")

            lbl_val = tk.Label(rand, text=format_val(variabila.get()),
                               font=("Courier", 9, "bold"),
                               bg=CULOARE_CARD, fg=CULOARE_ACCENT, width=6, anchor="e")
            lbl_val.pack(side="right")

            variabila.trace_add("write", lambda *_: lbl_val.config(text=format_val(variabila.get())))

            tk.Scale(rand, variable=variabila,
                     from_=val_min, to=val_max, resolution=(val_max - val_min) / 100,
                     orient="horizontal", showvalue=False,
                     bg=CULOARE_CARD, fg=CULOARE_ACCENT,
                     highlightthickness=0, troughcolor=CULOARE_SEPARATOR,
                     activebackground=CULOARE_ACCENT
                     ).pack(side="left", fill="x", expand=True, padx=(8, 8))

        _slider("Incredere minima gest", self.var_prag_gest, 0.50, 0.95, lambda v: f"{v:.0%}")
        _slider("Timp confirmare (sec)", self.var_timp_confirmare, 1.0, 5.0, lambda v: f"{v:.1f}s")
        _slider("Timp alerta (sec)", self.var_prag_alerta, 3.0, 15.0, lambda v: f"{v:.0f}s")

        tk.Frame(cadru, bg=CULOARE_BORDURA, height=1).pack(fill="x", padx=15)

        tk.Button(cadru, text="Reseteaza valorile implicite",
                  font=("Courier", 8), bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR,
                  activebackground=CULOARE_SEPARATOR, activeforeground=CULOARE_ACCENT,
                  bd=0, padx=10, pady=6, cursor="hand2",
                  command=self._reseteaza_setari).pack(anchor="e", padx=15, pady=8)

    def _reseteaza_setari(self):
        self.var_prag_gest.set(PRAG_INCREDERE_GEST)
        self.var_timp_confirmare.set(TIMP_CONFIRMARE_GEST)
        self.var_prag_alerta.set(PRAG_ALERTA)

    def _anuleaza_ultima_comanda(self):
        self.urgenta.reseteaza()
        self.disconfort.reseteaza()

        if self.istoric_comenzi:
            self.istoric_comenzi.pop()
            print("[!] Ultima comanda a fost stearsa, iar alertele au fost oprite.")
        else:
            print("[!] Istoric gol. Nu exista comenzi de anulat.")

    #LOOP PRINCIPAL
    def _update(self):
        if not self.activ:
            return

        succes, frame = self.cap.read() if self.camera_disponibila else (False, None)
        if not succes:
            self.lbl_banner.config(text="Camera indisponibila - verificati conexiunea camerei web",
                                   bg=CULOARE_ALERTA_2, fg="white")
            self.id_bucla = self.root.after(500, self._update)
            return

        frame = cv2.flip(frame, 1)
        if frame.shape[1] != LATIME_VIDEO or frame.shape[0] != INALTIME_VIDEO:
            frame = cv2.resize(frame, (LATIME_VIDEO, INALTIME_VIDEO))
        now = time.time()

        dt = now - self.fps_time
        self.fps_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt

        stare_afisata = {"comanda": "—", "progres_gest": 0.0,
                         "emotie_ro": "—", "emotie_hex": CULOARE_SUCCES, "incredere_emotie": 0.0}

        self._trimite_cadru_emotii(frame)
        self._proceseaza_gest(frame, now, stare_afisata)
        self._proceseaza_emotie(frame, now, stare_afisata)
        progres_urgenta, progres_disconfort = self._proceseaza_alerte(now)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.lbl_video.imgtk = img
        self.lbl_video.configure(image=img)

        self._update_ui(stare_afisata, progres_disconfort, progres_urgenta)

        self.id_bucla = self.root.after(INTERVAL_BUCLA_MS, self._update)

    def _trimite_cadru_emotii(self, frame):
        if not self.activ_emotie.get():
            return
        self.contor_cadre += 1
        if self.contor_cadre % INTERVAL_EMOTII == 0:
            self.analiza_emotii.trimite(frame.copy())

    def _proceseaza_gest(self, frame, now, stare_afisata):
        self.gest_curent = None
        if self.activ_gest.get():
            landmarks, gest, incredere = self.detector_gesturi.proceseaza_cadru(frame)
            if landmarks is not None and incredere > self.var_prag_gest.get():
                repere_mana.deseneaza(frame, landmarks)
                self.gest_curent = gest
        else:
            self.confirmare_gest.reseteaza()

        progres, confirmat = self.confirmare_gest.actualizeaza(
            self.gest_curent, now, self.var_timp_confirmare.get())
        gest = self.confirmare_gest.gest
        if gest is None:
            return

        comanda = ClasificatorGesturi.COMENZI_MEDICALE[gest]
        if confirmat:
            self.tts.roteste(comanda)
            if gest == ClasificatorGesturi.GEST_ANULARE:
                self._anuleaza_ultima_comanda()
            elif self.activ_istoric.get():
                self.istoric_comenzi.append(comanda)

        stare_afisata["comanda"] = comanda
        stare_afisata["progres_gest"] = progres

    def _proceseaza_emotie(self, frame, now, stare_afisata):
        rezultat, nou = self.analiza_emotii.preia()
        if not self.activ_emotie.get():
            self.contor_cadre = 0
            self.ultimul_rezultat_emotie = None
        elif nou:
            self.ultimul_rezultat_emotie = rezultat
            if rezultat:
                self.istoric_emotii.append(rezultat['stare'])
                self.ultima_detectie_fata = now

        if now - self.ultima_detectie_fata > GRACE_FATA:
            self.istoric_emotii.clear()

        self.stare_pacient = Counter(self.istoric_emotii).most_common(1)[0][0] if self.istoric_emotii else None
        res = self.ultimul_rezultat_emotie
        if self.stare_pacient is None or res is None:
            return

        culoare_bgr, culoare_hex = ClasificatorEmotii.CULORI_STARI[self.stare_pacient]
        x, y, w, h = res['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), culoare_bgr, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), culoare_bgr, -1)
        cv2.putText(frame, f"{self.stare_pacient} ({res['incredere'] * 100:.0f}%)", (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 0, 0) if culoare_bgr != (0, 0, 255) else (255, 255, 255), 2, cv2.LINE_AA)

        stare_afisata["emotie_ro"] = f"{self.stare_pacient} ({res['emotie_ro']})"
        stare_afisata["emotie_hex"] = culoare_hex
        stare_afisata["incredere_emotie"] = res['incredere']

    def _proceseaza_alerte(self, now):
        if not self.activ_alerte.get():
            self.urgenta.reseteaza()
            self.disconfort.reseteaza()
            return 0.0, 0.0

        prag = self.var_prag_alerta.get()
        progres_urgenta, anunta_urgenta = self.urgenta.actualizeaza(
            self.gest_curent == ClasificatorGesturi.GEST_URGENTA, now, prag)
        progres_disconfort, anunta_disconfort = self.disconfort.actualizeaza(
            self.stare_pacient in STARI_DISCONFORT, now, prag)

        if anunta_urgenta:
            self.tts.roteste_prioritar(MESAJ_URGENTA)
        elif anunta_disconfort:
            self.tts.roteste_prioritar(MESAJ_DISCONFORT)
        return progres_urgenta, progres_disconfort

    #PANOURI LATERALE
    def _update_ui(self, stare, progres_disconfort, progres_urgenta):
        self.lbl_gest.config(text=stare["comanda"])
        self.lbl_conf_gest.config(text=f"{stare['progres_gest'] * 100:.0f}%")
        culoare_bara_gest = CULOARE_SUCCES if stare['progres_gest'] >= 1.0 else CULOARE_ACCENT
        self.bara_gest.config(bg=culoare_bara_gest)
        self.bara_gest.place(width=int(stare['progres_gest'] * self.bara_gest_bg.winfo_width()))

        self.lbl_emotie.config(text=stare["emotie_ro"], fg=stare["emotie_hex"])
        self.lbl_emotie_raw.config(text=f"Incredere model: {stare['incredere_emotie'] * 100:.0f}%")
        self.bara_emotie.config(bg=stare["emotie_hex"])
        self.bara_emotie.place(width=int(stare['incredere_emotie'] * self.bara_emotie_bg.winfo_width()))

        self._seteaza_bara_alerta(self.bara_disconfort, self.bara_disconfort_bg,
                                  self.lbl_pct_disconfort, progres_disconfort, self.disconfort.activa)
        self._seteaza_bara_alerta(self.bara_urgenta, self.bara_urgenta_bg,
                                  self.lbl_pct_urgenta, progres_urgenta, self.urgenta.activa)

        flash = int(time.time() * 2) % 2 == 0
        culoare_banner = CULOARE_ALERTA_1 if flash else CULOARE_ALERTA_2
        if self.urgenta.activa:
            self.lbl_banner.config(text="CRITICAL: semnal de urgenta sustinut",
                                   bg=culoare_banner, fg="white")
        elif self.disconfort.activa:
            self.lbl_banner.config(
                text=f"ALERTA: disconfort sustinut de peste {self.var_prag_alerta.get():.0f} secunde",
                bg=culoare_banner, fg="white")
        else:
            self.lbl_banner.config(text="", bg=CULOARE_FUNDAL)

        #istoric comenzi
        if self.istoric_comenzi:
            self.lbl_istoric.config(text=" → ".join(c.replace("\n", " ") for c in self.istoric_comenzi))
        else:
            self.lbl_istoric.config(text="—")

        self.lbl_fps.config(text=f"FPS: {self.fps:.0f}")

    @staticmethod
    def _seteaza_bara_alerta(bara, bara_bg, label_pct, progres, activa):
        culoare = CULOARE_ALERTA_1 if activa else (CULOARE_AVERTIZARE if progres > 0 else CULOARE_TEXT_LABEL)
        bara.config(bg=culoare)
        bara.place(width=int(progres * bara_bg.winfo_width()))
        label_pct.config(text=f"{progres * 100:.0f}%", fg=culoare)

    def _inchide(self):
        self.activ = False
        if self.id_bucla is not None:
            self.root.after_cancel(self.id_bucla)
        self.cap.release()
        self.detector_gesturi.inchide()
        self.analiza_emotii.opreste()
        self.tts.opreste()
        self.root.destroy()


if __name__ == "__main__":
    from src.ecran_pornire import EcranPornire

    root = tk.Tk()
    root.withdraw()

    def porneste_aplicatia():
        root.deiconify()
        Aplicatie(root)

    EcranPornire(root, porneste_aplicatia, prag_alerta=PRAG_ALERTA)
    root.mainloop()
