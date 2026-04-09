import os
import cv2
from pathlib import Path

input_dir = "data_preprocessed/dark/images"
output_dir = "data_preprocessed/dark_gray_clahe/images"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clahe_on_grayscale(image, clip_limit=2.5, tile_grid_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    gray_clahe = clahe.apply(gray)

    # Convert back to 3 channels for YOLO
    return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)


for root, _, files in os.walk(input_dir):
    for file in files:
        if Path(file).suffix.lower() not in VALID_EXTS:
            continue

        in_path = os.path.join(root, file)
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(in_path)
        if img is None:
            print(f"Skipping unreadable image: {in_path}")
            continue

        processed = clahe_on_grayscale(img)
        cv2.imwrite(out_path, processed)

print("Done grayscale + CLAHE preprocessing")