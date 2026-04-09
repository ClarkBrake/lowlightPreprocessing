import os
import cv2
import argparse
import numpy as np
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gamma_correction(image, gamma=1.5):
    if gamma <= 0:
        raise ValueError("Gamma must be > 0")

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def clahe_bgr(image, clip_limit=2.0, tile_grid_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    l_enhanced = clahe.apply(l)

    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def hist_equalization_bgr(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def process_image(image, method="clahe_gamma", gamma=1.6, clip_limit=2.5, tile_grid_size=8):
    if method == "gamma":
        return gamma_correction(image, gamma=gamma)

    elif method == "hist_eq":
        return hist_equalization_bgr(image)

    elif method == "clahe":
        return clahe_bgr(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    elif method == "clahe_gamma":
        out = clahe_bgr(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        out = gamma_correction(out, gamma=gamma)
        return out

    else:
        raise ValueError(f"Unknown method: {method}")


def collect_images(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in VALID_EXTS:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_processed_image(input_path, input_base, output_base, processed):
    rel_path = os.path.relpath(input_path, input_base)
    out_path = os.path.join(output_base, rel_path)
    ensure_parent(out_path)
    cv2.imwrite(out_path, processed)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess dark images for improved pedestrian detection.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output image folder")
    parser.add_argument(
        "--method",
        type=str,
        default="clahe_gamma",
        choices=["gamma", "hist_eq", "clahe", "clahe_gamma"],
        help="Enhancement method"
    )
    parser.add_argument("--gamma", type=float, default=1.6, help="Gamma value")
    parser.add_argument("--clip_limit", type=float, default=2.5, help="CLAHE clip limit")
    parser.add_argument("--tile_grid_size", type=int, default=8, help="CLAHE tile size")
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Using method: {args.method}")

    for i, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        processed = process_image(
            image,
            method=args.method,
            gamma=args.gamma,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size
        )

        out_path = save_processed_image(img_path, args.input_dir, args.output_dir, processed)
        print(f"[{i}/{len(image_paths)}] Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()