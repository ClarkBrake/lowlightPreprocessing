import os
import cv2
from pathlib import Path

input_dir = "data_preprocessed/dark/images"
output_dir = "data_preprocessed/dark_gray/images"

VALID_EXTS = {".jpg", ".jpeg", ".png"}

for root, _, files in os.walk(input_dir):
    for file in files:
        if Path(file).suffix.lower() not in VALID_EXTS:
            continue

        in_path = os.path.join(root, file)
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(in_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert back to 3-channel (VERY IMPORTANT for YOLO)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(out_path, gray_3ch)

print("Done grayscale preprocessing")