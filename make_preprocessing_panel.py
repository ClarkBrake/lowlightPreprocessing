import cv2
import matplotlib.pyplot as plt
from pathlib import Path

dark_img = "data_preprocessed/dark/images/val/frame_000576.png"
clahe_img = "data_preprocessed/dark_clahe/images/val/frame_000576.png"
norm_img = "data_preprocessed/dark_normalized/images/val/frame_000576.png"
ab_img = "data_preprocessed/dark_ab_1/images/val/frame_000576.png"
gray_img = "data_preprocessed/dark_gray/images/val/frame_000576.png"
gray_clahe_img = "data_preprocessed/dark_gray_clahe/images/val/frame_000576.png"

out_path = "figures/preprocessing_panel.png"


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    imgs = [
        ("Dark", read_rgb(dark_img)),
        ("CLAHE", read_rgb(clahe_img)),
        ("Normalized", read_rgb(norm_img)),
        ("Alpha-Beta", read_rgb(ab_img)),
        ("Grayscale", read_rgb(gray_img)),
        ("Gray + CLAHE", read_rgb(gray_clahe_img)),
    ]

    plt.figure(figsize=(12, 10))

    for i, (title, img) in enumerate(imgs, start=1):
        plt.subplot(3, 2, i)  # 3 rows, 2 columns
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