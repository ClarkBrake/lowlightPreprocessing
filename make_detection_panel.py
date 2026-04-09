import cv2
import matplotlib.pyplot as plt
from pathlib import Path

dark_det = "runs/detect/predict3/frame_000608.jpg"
ab_det = "runs/detect/predict6/frame_000608.jpg"

out_path = "figures/detection_panel.png"


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    imgs = [
        ("Dark Detection", read_rgb(dark_det)),
        ("Greyscale Detection", read_rgb(ab_det)),
    ]

    plt.figure(figsize=(10, 5))

    for i, (title, img) in enumerate(imgs, start=1):
        plt.subplot(1, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()