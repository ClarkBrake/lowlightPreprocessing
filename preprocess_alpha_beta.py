import os
import cv2
import argparse
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def process_image(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def main():
    parser = argparse.ArgumentParser(description="Generate alpha/beta brightness-contrast variants")
    parser.add_argument("--input_dir", required=True, help="Input image folder")
    parser.add_argument("--output_dir", required=True, help="Output image folder")
    parser.add_argument("--alpha", type=float, required=True, help="Contrast scale")
    parser.add_argument("--beta", type=int, required=True, help="Brightness shift")
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)

    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        return

    print(f"Processing {len(image_paths)} images with alpha={args.alpha}, beta={args.beta}")

    for i, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        processed = process_image(image, args.alpha, args.beta)
        out_path = save_processed_image(img_path, args.input_dir, args.output_dir, processed)
        print(f"[{i}/{len(image_paths)}] Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()