from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "portfolio" / "build"

PyInstaller.__main__.run([
    str(ROOT / "app.py"),
    "--name", "AssistiveMedicalSystem",
    "--noconfirm",
    "--windowed",
    "--distpath", str(OUT / "dist"),
    "--workpath", str(OUT / "work"),
    "--specpath", str(OUT),
    "--paths", str(ROOT),
    "--add-data", f"{ROOT / 'models'};models",
    "--collect-data", "mediapipe",
    "--collect-binaries", "mediapipe",
    "--hidden-import", "pyttsx3.drivers",
    "--hidden-import", "pyttsx3.drivers.sapi5",
])
