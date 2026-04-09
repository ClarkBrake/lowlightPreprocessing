import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

labels = [
    "Dark",
    "CLAHE",
    "CLAHE+Gamma",
    "Normalized",
    "Norm+Gamma",
    "AB-1",
    "AB-2",
    "AB-3",
    "Grayscale",
    "Gray+CLAHE",
]

map50 = [
    0.3662,
    0.3553,
    0.2723,
    0.2513,
    0.2444,
    0.3270,
    0.3193,
    0.2737,
    0.3909,
    0.3700,
]

map5095 = [
    0.1814,
    0.1788,
    0.1366,
    0.1201,
    0.1101,
    0.1587,
    0.1514,
    0.1335,
    0.1918,
    0.1794,
]

x = np.arange(len(labels))
width = 0.38

Path("figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))  # slightly wider for readability
plt.bar(x - width / 2, map50, width, label="mAP50")
plt.bar(x + width / 2, map5095, width, label="mAP50-95")

plt.xticks(x, labels, rotation=30)
plt.ylabel("Score")
plt.title("Detection Performance Across Preprocessing Methods")

plt.legend()
plt.tight_layout()
plt.savefig("figures/all_methods_results.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: figures/all_methods_results.png")