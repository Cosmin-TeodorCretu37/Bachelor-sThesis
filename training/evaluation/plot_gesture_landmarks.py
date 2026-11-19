import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import TRAINING_DIR
from src.detectors import HAND_CONNECTIONS

GESTURES = ["fist", "open_palm", "one", "two", "three", "ok"]


fig, axs = plt.subplots(2, 3, figsize=(14, 9))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('MediaPipe Landmarks — Samples per Gesture',
             fontsize=15, fontweight='bold', color='white', y=1.01)

finger_colors = {
    range(0, 1): '#ffffff',   #wrist
    range(1, 5): '#58a6ff',   #thumb
    range(5, 9): '#3fb950',   #index
    range(9, 13): '#ff8c00',  #middle
    range(13, 17): '#aa44ff', #ring
    range(17, 21): '#ff4444', #pinky
}

def point_color(idx):
    for r, c in finger_colors.items():
        if idx in r:
            return c
    return 'white'

for ax, gesture in zip(axs.flat, GESTURES, strict=True):
    ax.set_facecolor('#161b22')
    ax.set_title(gesture.replace("_", " ").upper(),
                 fontsize=10, fontweight='bold', color='#58a6ff', pad=6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    df = pd.read_csv(TRAINING_DIR / "gesture_data" / f"{gesture}.csv", header=None)

    #draw the first 5 samples overlapped (transparency)
    for _, row in df.head(5).iterrows():
        points = row.values.reshape(21, 3)
        xs = points[:, 0]
        ys = points[:, 1]

        for start, end in HAND_CONNECTIONS:
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]],
                    color='#30363d', linewidth=1.5, alpha=0.6, zorder=1)

        for i in range(21):
            ax.scatter(xs[i], ys[i], color=point_color(i), s=30, zorder=2, alpha=0.8)

    #draw the mean of the samples (main line)
    mean = df.mean().values.reshape(21, 3)
    xs_m, ys_m = mean[:, 0], mean[:, 1]

    for start, end in HAND_CONNECTIONS:
        ax.plot([xs_m[start], xs_m[end]], [ys_m[start], ys_m[end]],
                color='white', linewidth=2, zorder=3)
    for i in range(21):
        ax.scatter(xs_m[i], ys_m[i], color=point_color(i), s=60, zorder=4, edgecolors='white', linewidth=0.5)

#finger legend
legend_items = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#58a6ff', markersize=8, label='Thumb'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3fb950', markersize=8, label='Index'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff8c00', markersize=8, label='Middle'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aa44ff', markersize=8, label='Ring'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', markersize=8, label='Pinky'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=5,
           facecolor='#161b22', labelcolor='white', fontsize=9,
           framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(TRAINING_DIR / 'gesture_landmarks.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.show()
print(f"Saved: {TRAINING_DIR / 'gesture_landmarks.png'}")
