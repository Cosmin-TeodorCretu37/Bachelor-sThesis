import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAINING_DIR
from src.detectors import HandDetector, draw_hand, normalize_hand

#config
GESTURES = ["fist", "open_palm", "one", "two", "three", "ok"]
SAMPLES_PER_GESTURE = 500
DATA_DIR = TRAINING_DIR / "gesture_data"
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]

#gesture instructions
INSTRUCTIONS = {
    "fist": "Close ALL fingers into a fist",
    "open_palm": "Open ALL 5 fingers wide",
    "one": "Raise ONLY the index finger",
    "two": "Raise ONLY index + middle",
    "three": "Raise ONLY index + middle + ring",
    "ok": "Make a circle with thumb + index"
}


def raised_fingers(hand_landmarks):
    #returns which fingers are raised
    #visual feedback
    tips = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    return [hand_landmarks[tip].y < hand_landmarks[base].y for tip, base in zip(tips, bases, strict=True)]


def is_valid_gesture(gesture, fingers):
    expected = {
        "fist": [False, False, False, False],
        "open_palm": [True, True, True, True],
        "one": [True, False, False, False],
        "two": [True, True, False, False],
        "three": [True, True, True, False],
    }
    return gesture not in expected or fingers == expected[gesture]


def show_fingers(frame, fingers):
    for i, (raised, name) in enumerate(zip(fingers, FINGER_NAMES, strict=True)):
        color = (0, 255, 0) if raised else (0, 0, 255)
        cv2.putText(frame, f"{name}: {'UP' if raised else 'DOWN'}", (10, 160 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def read_frame(cap):
    success, frame = cap.read()
    return cv2.flip(frame, 1) if success else None


def wait_for_start(cap, detector, gesture_index, gesture):
    while True:
        frame = read_frame(cap)
        if frame is None:
            return False
        landmarks = detector.detect(frame)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"GESTURE {gesture_index + 1}/{len(GESTURES)}: {gesture.upper()}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, INSTRUCTIONS[gesture],
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE = start, Q = quit",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        if landmarks is not None:
            draw_hand(frame, landmarks)
            show_fingers(frame, raised_fingers(landmarks))

        cv2.imshow("Data collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


def collect(cap, detector, gesture):
    collected = 0
    with open(DATA_DIR / f"{gesture}.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        while collected < SAMPLES_PER_GESTURE:
            frame = read_frame(cap)
            if frame is None:
                break
            landmarks = detector.detect(frame)

            if landmarks is not None:
                #gesture correctness check
                fingers = raised_fingers(landmarks)
                valid = is_valid_gesture(gesture, fingers)
                if valid:
                    writer.writerow(normalize_hand(landmarks))
                    collected += 1

                draw_hand(frame, landmarks)

                #color feedback: green=valid, red=invalid
                status_color = (0, 255, 0) if valid else (0, 0, 255)
                status_text = "VALID - saving!" if valid else "INVALID - fix the gesture!"
                cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                show_fingers(frame, fingers)

            #progress bar
            progress = int((collected / SAMPLES_PER_GESTURE) * 400)
            cv2.rectangle(frame, (10, 450), (410, 472), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 450), (10 + progress, 472), (0, 255, 100), -1)
            cv2.putText(frame, f"{gesture.upper()}: {collected}/{SAMPLES_PER_GESTURE}", (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Data collection", frame)
            cv2.waitKey(1)

    print(f"{gesture}: {collected} samples saved")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(confidence=0.7)
    cap = cv2.VideoCapture(0)

    try:
        for gesture_index, gesture in enumerate(GESTURES):
            print(f"\n---{gesture.upper()}---\nInstruction: {INSTRUCTIONS[gesture]}")
            if not wait_for_start(cap, detector, gesture_index, gesture):
                print("Data collection stopped.")
                return
            collect(cap, detector, gesture)
        print(f"\nData collection finished! The CSV files are in '{DATA_DIR}'")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
