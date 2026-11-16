from pathlib import Path

DIR_PROIECT = Path(__file__).resolve().parent.parent
DIR_MODELE = DIR_PROIECT / "modele"
DIR_ANTRENARE = DIR_PROIECT / "antrenare"

MODEL_GESTURI = DIR_MODELE / "model_gesturi.keras"
MAPARE_GESTURI = DIR_MODELE / "mapare_gesturi.json"
MODEL_EMOTII = DIR_MODELE / "model_emotii.keras"
MAPARE_EMOTII = DIR_MODELE / "mapare_emotii.json"
TASK_MANA = DIR_MODELE / "hand_landmarker.task"
TASK_FATA = DIR_MODELE / "blaze_face_short_range.tflite"

FISIER_SETARI = DIR_PROIECT / "setari_pacient.json"
DIR_JURNAL = DIR_PROIECT / "jurnal"

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
LUNGIME_ISTORIC_COMENZI = 7

MESAJ_URGENTA = "Critical alert. Emergency signal sustained. Please respond immediately."
MESAJ_DISCONFORT = "Attention. Patient has been showing signs of discomfort for an extended period."

CULOARE_FUNDAL = "#0d1117"
CULOARE_CARD = "#161b22"
CULOARE_BORDURA = "#30363d"
CULOARE_SEPARATOR = "#21262d"
CULOARE_TEXT_SECUNDAR = "#8b949e"
CULOARE_TEXT_LABEL = "#484f58"
CULOARE_ACCENT = "#58a6ff"
CULOARE_SUCCES = "#3fb950"
CULOARE_AVERTIZARE = "#d29922"
CULOARE_ALERTA_1 = "#f85149"
CULOARE_ALERTA_2 = "#7a1f1a"
