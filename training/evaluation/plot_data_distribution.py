import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import TRAINING_DIR

GESTURES = ["fist", "open_palm", "one", "two", "three", "ok"]

#count samples per gesture
sample_counts = []
for gesture in GESTURES:
    df = pd.read_csv(TRAINING_DIR / "gesture_data" / f"{gesture}.csv", header=None)
    sample_counts.append(len(df))

#count FER+ images per emotion
FER_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
train_counts = []
test_counts = []

for emotion in FER_EMOTIONS:
    train_folder = TRAINING_DIR / "ferplus" / "train" / emotion
    test_folder = TRAINING_DIR / "ferplus" / "test" / emotion
    train_counts.append(len(os.listdir(train_folder)) if os.path.exists(train_folder) else 0)
    test_counts.append(len(os.listdir(test_folder)) if os.path.exists(test_folder) else 0)

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

#plot 1: gesture distribution
ax1 = fig.add_subplot(gs[0, 0])
gesture_colors = ['#58a6ff', '#3fb950', '#ff8c00', '#ff4444', '#aa44ff', '#ffff44']
bars = ax1.bar([g.replace("_", "\n") for g in GESTURES], sample_counts,
               color=gesture_colors, edgecolor='#30363d', linewidth=0.8)
ax1.set_title('Data Distribution - Gestures', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Gesture', fontsize=10)
ax1.set_ylabel('Number of samples', fontsize=10)
ax1.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')
ax1.tick_params(colors='white')
ax1.title.set_color('white')
ax1.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#30363d')
for bar, val in zip(bars, sample_counts, strict=True):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(val), ha='center', va='bottom', fontsize=9, color='white')

#plot 2: training emotion distribution
ax2 = fig.add_subplot(gs[0, 1])
emotion_colors = ['#ff4444', '#ff8c00', '#aa44ff', '#ffff44', '#aaaaaa', '#4488ff', '#44ffaa']
bars2 = ax2.bar(emotion_names, train_counts, color=emotion_colors,
                edgecolor='#30363d', linewidth=0.8)
ax2.set_title('Data Distribution - Emotions (Train)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Emotion', fontsize=10)
ax2.set_ylabel('Number of images', fontsize=10)
ax2.set_facecolor('#0d1117')
ax2.tick_params(colors='white', axis='both')
ax2.title.set_color('white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
for spine in ax2.spines.values():
    spine.set_edgecolor('#30363d')
for bar, val in zip(bars2, train_counts, strict=True):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', va='bottom', fontsize=8, color='white')

#plot 3: gesture pie chart
ax3 = fig.add_subplot(gs[1, 0])
ax3.pie(sample_counts, labels=[g.replace("_", " ") for g in GESTURES],
        colors=gesture_colors, autopct='%1.1f%%',
        startangle=90, textprops={'color': 'white', 'fontsize': 9})
ax3.set_title('Class Proportions - Gestures', fontsize=13, fontweight='bold', pad=10)
ax3.title.set_color('white')
ax3.set_facecolor('#0d1117')

#plot 4: train vs test emotions
ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(len(emotion_names))
width = 0.35
ax4.bar(x - width/2, train_counts, width, label='Train', color='#58a6ff', alpha=0.85)
ax4.bar(x + width/2, test_counts, width, label='Test', color='#3fb950', alpha=0.85)
ax4.set_title('Train vs Test - Emotions', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Emotion', fontsize=10)
ax4.set_ylabel('Number of images', fontsize=10)
ax4.set_xticks(x)
ax4.set_xticklabels(emotion_names, rotation=30, ha='right')
ax4.legend(facecolor='#161b22', labelcolor='white')
ax4.set_facecolor('#0d1117')
ax4.tick_params(colors='white')
ax4.title.set_color('white')
ax4.xaxis.label.set_color('white')
ax4.yaxis.label.set_color('white')
for spine in ax4.spines.values():
    spine.set_edgecolor('#30363d')

plt.savefig(TRAINING_DIR / 'data_distribution.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.show()
print(f"Saved: {TRAINING_DIR / 'data_distribution.png'}")
