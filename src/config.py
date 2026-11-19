from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
TRAINING_DIR = PROJECT_DIR / "training"

GESTURE_MODEL = MODELS_DIR / "gesture_model.keras"
GESTURE_LABELS = MODELS_DIR / "gesture_labels.json"
EMOTION_MODEL = MODELS_DIR / "emotion_model.keras"
EMOTION_LABELS = MODELS_DIR / "emotion_labels.json"
HAND_TASK = MODELS_DIR / "hand_landmarker.task"
FACE_TASK = MODELS_DIR / "blaze_face_short_range.tflite"

SETTINGS_FILE = PROJECT_DIR / "patient_settings.json"
EVENT_LOG_DIR = PROJECT_DIR / "logs"

#config
GESTURE_CONFIDENCE_THRESHOLD = 0.75  #gesture ignore threshold
GESTURE_CONFIRMATION_TIME = 2.5      #sec. gesture -> command
EMOTION_HISTORY_LENGTH = 10          #majority vote frames
ALERT_THRESHOLD = 7.0

FACE_GRACE = 1.5                     #face tolerance
GESTURE_GRACE = 1.0                  #gesture tolerance

DISCOMFORT_REALERT_INTERVAL = 15     #voice re-announcement
EMERGENCY_REALERT_INTERVAL = 5

EMOTION_FRAME_INTERVAL = 3
LOOP_INTERVAL_MS = 1
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
COMMAND_HISTORY_LENGTH = 7

EMERGENCY_MESSAGE = "Critical alert. Emergency signal sustained. Please respond immediately."
DISCOMFORT_MESSAGE = "Attention. Patient has been showing signs of discomfort for an extended period."

COLOR_BACKGROUND = "#0d1117"
COLOR_CARD = "#161b22"
COLOR_BORDER = "#30363d"
COLOR_SEPARATOR = "#21262d"
COLOR_TEXT_SECONDARY = "#8b949e"
COLOR_TEXT_LABEL = "#484f58"
COLOR_ACCENT = "#58a6ff"
COLOR_SUCCESS = "#3fb950"
COLOR_WARNING = "#d29922"
COLOR_ALERT_1 = "#f85149"
COLOR_ALERT_2 = "#7a1f1a"
