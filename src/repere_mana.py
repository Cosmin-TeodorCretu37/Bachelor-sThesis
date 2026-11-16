import cv2

CONEXIUNI_PALMA = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]


def normalizeaza(hand_landmarks):
    baza = hand_landmarks[0]
    coordonate_relative = []
    for lm in hand_landmarks:
        coordonate_relative.extend([lm.x - baza.x, lm.y - baza.y, lm.z - baza.z])

    val_max = max(map(abs, coordonate_relative)) or 1.0
    return [v / val_max for v in coordonate_relative]


def deseneaza(frame, hand_landmarks):
    h, w = frame.shape[:2]
    puncte = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in CONEXIUNI_PALMA:
        cv2.line(frame, puncte[start], puncte[end], (0, 215, 255), 2, cv2.LINE_AA)
    for (x, y) in puncte:
        cv2.circle(frame, (x, y), 5, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
