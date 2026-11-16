import logging
import time
import tkinter as tk
from collections import Counter, deque

import cv2
from PIL import Image, ImageTk

from src import componente_ui as ui
from src import jurnal
from src.clasificator_emotii import ClasificatorEmotii
from src.clasificator_gesturi import ClasificatorGesturi
from src.config import (
    CULOARE_ACCENT,
    CULOARE_ALERTA_1,
    CULOARE_ALERTA_2,
    CULOARE_BORDURA,
    CULOARE_CARD,
    CULOARE_FUNDAL,
    CULOARE_SEPARATOR,
    CULOARE_SUCCES,
    CULOARE_TEXT_LABEL,
    CULOARE_TEXT_SECUNDAR,
    GRACE_FATA,
    GRACE_GEST,
    INALTIME_VIDEO,
    INTERVAL_BUCLA_MS,
    INTERVAL_EMOTII,
    INTERVAL_RE_ALERTA_DISCONFORT,
    INTERVAL_RE_ALERTA_URGENTA,
    LATIME_VIDEO,
    LUNGIME_ISTORIC_COMENZI,
    LUNGIME_ISTORIC_EMOTII,
    MESAJ_DISCONFORT,
    MESAJ_URGENTA,
    PRAG_ALERTA,
    PRAG_INCREDERE_GEST,
    TIMP_CONFIRMARE_GEST,
)
from src.detectoare import deseneaza_mana
from src.jurnal import JurnalEvenimente
from src.monitorizare import ConfirmareGest, StareSustinuta
from src.procesare_fundal import ProcesareFundal
from src.setari import SetariPacient
from src.sinteza_vocala import SintezaVocala

log = logging.getLogger(__name__)


class Aplicatie:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Medical Asistiv")
        self.root.state("zoomed")
        self.root.configure(bg=CULOARE_FUNDAL)
        self.root.protocol("WM_DELETE_WINDOW", self._inchide)

        log.info("Se incarca modelele...")
        self.detector_emotii = ClasificatorEmotii()
        self.detector_gesturi = ClasificatorGesturi()
        self.analiza_emotii = ProcesareFundal(self.detector_emotii.proceseaza_cadru)
        self.tts = SintezaVocala()
        self.jurnal = JurnalEvenimente()
        log.info("Modele incarcate.")

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
        self.istoric_comenzi = deque(maxlen=LUNGIME_ISTORIC_COMENZI)

        #stare urgenta
        self.urgenta = StareSustinuta(INTERVAL_RE_ALERTA_URGENTA, grace=GRACE_GEST)

        self.fps_time = time.time()
        self.fps = 0.0

        setari = SetariPacient.incarca()

        #toggle on/off
        self.activ_gest = tk.BooleanVar(value=setari.activ_gest)
        self.activ_emotie = tk.BooleanVar(value=setari.activ_emotie)
        self.activ_alerte = tk.BooleanVar(value=setari.activ_alerte)
        self.activ_istoric = tk.BooleanVar(value=setari.activ_istoric)

        # setari reglabile live
        self.var_prag_gest = tk.DoubleVar(value=setari.prag_gest)
        self.var_timp_confirmare = tk.DoubleVar(value=setari.timp_confirmare)
        self.var_prag_alerta = tk.DoubleVar(value=setari.prag_alerta)

        self._construieste_ui()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LATIME_VIDEO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INALTIME_VIDEO)
        self.camera_disponibila = self.cap.isOpened()

        self.activ = True
        self.id_bucla = None
        self.jurnal.inregistreaza(jurnal.PORNIRE)
        self._update()

    #UI
    def _construieste_ui(self):
        header = tk.Frame(self.root, bg=CULOARE_FUNDAL)
        header.pack(fill="x", padx=20, pady=(15, 0))

        tk.Label(header, text="SISTEM MEDICAL ASISTIV", font=(ui.FONT, 16, "bold"),
                 bg=CULOARE_FUNDAL, fg=CULOARE_ACCENT).pack(side="left")

        self.lbl_fps = tk.Label(header, text="FPS: 0", font=(ui.FONT, 11),
                                bg=CULOARE_FUNDAL, fg=CULOARE_TEXT_LABEL)
        self.lbl_fps.pack(side="right", padx=10)

        self.lbl_banner = tk.Label(self.root, text="", font=(ui.FONT, 13, "bold"),
                                   bg=CULOARE_FUNDAL, fg="white", pady=8)
        self.lbl_banner.pack(fill="x", padx=20, pady=(10, 0))

        corp = tk.Frame(self.root, bg=CULOARE_FUNDAL)
        corp.pack(fill="both", expand=True, padx=20, pady=10)

        cadru_video = tk.Frame(corp, bg=CULOARE_CARD, highlightbackground=CULOARE_BORDURA, highlightthickness=1)
        cadru_video.pack(side="left", padx=(0, 15))
        self.lbl_video = tk.Label(cadru_video, bg=CULOARE_CARD)
        self.lbl_video.pack(padx=2, pady=2)

        panou = ui.panou_derulant(corp)

        cadru = ui.card(panou, "COMANDA ASISTIVA", self.activ_gest, CULOARE_ACCENT)
        self.lbl_gest = ui.text_mare(cadru, CULOARE_ACCENT)
        self.lbl_conf_gest = ui.rand_valoare(cadru, "Confirmare comanda:", CULOARE_TEXT_LABEL,
                                             CULOARE_ACCENT, pady=(4, 4))
        self.bara_gest = ui.BaraProgres(cadru, CULOARE_ACCENT, (0, 12))
        ui.separator(panou)

        cadru = ui.card(panou, "STARE PACIENT", self.activ_emotie, CULOARE_SUCCES)
        self.lbl_emotie = ui.text_mare(cadru, CULOARE_SUCCES)
        self.lbl_emotie_raw = tk.Label(cadru, text="", font=(ui.FONT, 9),
                                       bg=CULOARE_CARD, fg=CULOARE_TEXT_SECUNDAR)
        self.lbl_emotie_raw.pack(anchor="w", padx=15, pady=(0, 8))
        self.bara_emotie = ui.BaraProgres(cadru, CULOARE_SUCCES, (0, 12))
        ui.separator(panou)

        cadru = ui.card(panou, "MONITORIZARE ALERTE", self.activ_alerte, CULOARE_ALERTA_1, pady_titlu=(10, 8))
        self.indicator_disconfort = ui.IndicatorAlerta(cadru, "Disconfort / Durere sustinut(a)", (2, 10))
        self.indicator_urgenta = ui.IndicatorAlerta(cadru, "Semnal de urgenta sustinut", (2, 12))
        ui.separator(panou)

        cadru = ui.card(panou, "ISTORIC COMENZI", self.activ_istoric, CULOARE_ACCENT, pady_titlu=(10, 6))
        self.lbl_istoric = tk.Label(cadru, text="—", font=(ui.FONT, 10), bg=CULOARE_CARD,
                                    fg=CULOARE_TEXT_SECUNDAR, wraplength=340, justify="left")
        self.lbl_istoric.pack(anchor="w", padx=15, pady=(0, 12))
        ui.separator(panou)

        cadru = ui.card(panou, "SETARI PACIENT", pady_titlu=(10, 6))
        ui.slider(cadru, "Incredere minima gest", self.var_prag_gest, 0.50, 0.95, 0.01, lambda v: f"{v:.0%}")
        ui.slider(cadru, "Timp confirmare (sec)", self.var_timp_confirmare, 1.0, 5.0, 0.1, lambda v: f"{v:.1f}s")
        ui.slider(cadru, "Timp alerta (sec)", self.var_prag_alerta, 3.0, 15.0, 0.5, lambda v: f"{v:.1f}s")
        tk.Frame(cadru, bg=CULOARE_BORDURA, height=1).pack(fill="x", padx=15)
        tk.Button(cadru, text="Reseteaza valorile implicite", font=(ui.FONT, 8), bg=CULOARE_CARD,
                  fg=CULOARE_TEXT_SECUNDAR, activebackground=CULOARE_SEPARATOR, activeforeground=CULOARE_ACCENT,
                  bd=0, padx=10, pady=6, cursor="hand2",
                  command=self._reseteaza_setari).pack(anchor="e", padx=15, pady=8)

        tk.Button(self.root, text="⏹  Oprește aplicația", font=(ui.FONT, 11, "bold"),
                  bg=CULOARE_SEPARATOR, fg=CULOARE_ALERTA_1, activebackground="#2d1b1b",
                  activeforeground=CULOARE_ALERTA_1, bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._inchide).pack(pady=(5, 15))

    def _reseteaza_setari(self):
        self.var_prag_gest.set(PRAG_INCREDERE_GEST)
        self.var_timp_confirmare.set(TIMP_CONFIRMARE_GEST)
        self.var_prag_alerta.set(PRAG_ALERTA)

    def _setari_curente(self):
        return SetariPacient(
            prag_gest=self.var_prag_gest.get(),
            timp_confirmare=self.var_timp_confirmare.get(),
            prag_alerta=self.var_prag_alerta.get(),
            activ_gest=self.activ_gest.get(),
            activ_emotie=self.activ_emotie.get(),
            activ_alerte=self.activ_alerte.get(),
            activ_istoric=self.activ_istoric.get(),
        )

    def _anuleaza_ultima_comanda(self):
        self.urgenta.reseteaza()
        self.disconfort.reseteaza()

        if self.istoric_comenzi:
            self.istoric_comenzi.pop()
            log.info("Ultima comanda a fost stearsa, iar alertele au fost oprite.")
        else:
            log.info("Istoric gol. Nu exista comenzi de anulat.")

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
            rezultat = self.detector_gesturi.proceseaza_cadru(frame)
            if rezultat and rezultat.incredere > self.var_prag_gest.get():
                deseneaza_mana(frame, rezultat.landmarks)
                self.gest_curent = rezultat.gest
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
                self.jurnal.inregistreaza(jurnal.ANULARE)
                self._anuleaza_ultima_comanda()
            else:
                self.jurnal.inregistreaza(jurnal.COMANDA, comanda)
                if self.activ_istoric.get():
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
                self.istoric_emotii.append(rezultat.stare)
                self.ultima_detectie_fata = now

        if now - self.ultima_detectie_fata > GRACE_FATA:
            self.istoric_emotii.clear()

        self.stare_pacient = Counter(self.istoric_emotii).most_common(1)[0][0] if self.istoric_emotii else None
        res = self.ultimul_rezultat_emotie
        if self.stare_pacient is None or res is None:
            return

        culoare_bgr, culoare_hex = ClasificatorEmotii.CULORI_STARI[self.stare_pacient]
        x, y, w, h = res.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), culoare_bgr, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), culoare_bgr, -1)
        cv2.putText(frame, f"{self.stare_pacient} ({res.incredere * 100:.0f}%)", (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 0, 0) if culoare_bgr != (0, 0, 255) else (255, 255, 255), 2, cv2.LINE_AA)

        stare_afisata["emotie_ro"] = f"{self.stare_pacient} ({res.emotie_ro})"
        stare_afisata["emotie_hex"] = culoare_hex
        stare_afisata["incredere_emotie"] = res.incredere

    def _proceseaza_alerte(self, now):
        if not self.activ_alerte.get():
            self.urgenta.reseteaza()
            self.disconfort.reseteaza()
            return 0.0, 0.0

        prag = self.var_prag_alerta.get()
        urgenta_era_activa, disconfort_era_activ = self.urgenta.activa, self.disconfort.activa
        progres_urgenta, anunta_urgenta = self.urgenta.actualizeaza(
            self.gest_curent == ClasificatorGesturi.GEST_URGENTA, now, prag)
        progres_disconfort, anunta_disconfort = self.disconfort.actualizeaza(
            self.stare_pacient in ClasificatorEmotii.STARI_DISCONFORT, now, prag)

        if self.urgenta.activa and not urgenta_era_activa:
            self.jurnal.inregistreaza(jurnal.ALERTA_URGENTA)
        if self.disconfort.activa and not disconfort_era_activ:
            self.jurnal.inregistreaza(jurnal.ALERTA_DISCONFORT)

        if anunta_urgenta:
            self.tts.roteste_prioritar(MESAJ_URGENTA)
        elif anunta_disconfort:
            self.tts.roteste_prioritar(MESAJ_DISCONFORT)
        return progres_urgenta, progres_disconfort

    #PANOURI LATERALE
    def _update_ui(self, stare, progres_disconfort, progres_urgenta):
        self.lbl_gest.config(text=stare["comanda"])
        self.lbl_conf_gest.config(text=f"{stare['progres_gest'] * 100:.0f}%")
        self.bara_gest.seteaza(stare['progres_gest'],
                               CULOARE_SUCCES if stare['progres_gest'] >= 1.0 else CULOARE_ACCENT)

        self.lbl_emotie.config(text=stare["emotie_ro"], fg=stare["emotie_hex"])
        self.lbl_emotie_raw.config(text=f"Incredere model: {stare['incredere_emotie'] * 100:.0f}%")
        self.bara_emotie.seteaza(stare['incredere_emotie'], stare["emotie_hex"])

        self.indicator_disconfort.seteaza(progres_disconfort, self.disconfort.activa)
        self.indicator_urgenta.seteaza(progres_urgenta, self.urgenta.activa)

        flash = int(time.time() * 2) % 2 == 0
        culoare_banner = CULOARE_ALERTA_1 if flash else CULOARE_ALERTA_2
        if self.urgenta.activa:
            self.lbl_banner.config(text="CRITICAL: semnal de urgenta sustinut", bg=culoare_banner, fg="white")
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

    def _inchide(self):
        self.activ = False
        self._setari_curente().salveaza()
        self.jurnal.inregistreaza(jurnal.OPRIRE)
        if self.id_bucla is not None:
            self.root.after_cancel(self.id_bucla)
        self.cap.release()
        self.analiza_emotii.opreste()
        self.detector_emotii.inchide()
        self.detector_gesturi.inchide()
        self.tts.opreste()
        self.root.destroy()


def main():
    from src.ecran_pornire import EcranPornire

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = tk.Tk()
    root.withdraw()

    def porneste_aplicatia():
        root.deiconify()
        Aplicatie(root)

    EcranPornire(root, porneste_aplicatia)
    root.mainloop()


if __name__ == "__main__":
    main()
