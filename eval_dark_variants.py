import os
import json
from ultralytics import YOLO


def evaluate_dataset(model_path, data_yaml, imgsz=640, conf=0.25, iou=0.6, split="val"):
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        split=split,
        plots=False,
        save_json=False,
        verbose=False
    )

    metrics = {
        "dataset": os.path.basename(data_yaml),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }

    return metrics


def print_comparison_table(results_list):
    print("\n" + "=" * 72)
    print(f"{'Dataset':25s} {'Precision':>10s} {'Recall':>10s} {'mAP50':>10s} {'mAP50-95':>12s}")
    print("=" * 72)

    for r in results_list:
        print(
            f"{r['dataset']:25s} "
            f"{r['precision']:10.4f} "
            f"{r['recall']:10.4f} "
            f"{r['map50']:10.4f} "
            f"{r['map50_95']:12.4f}"
        )

    print("=" * 72)


def save_results_json(results_list, output_path):
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=4)
    print(f"\nSaved results to: {output_path}")


def main():
    model_path = "yolov8n.pt"

    datasets = [
        "dark.yaml",
        "dark_ab_1.yaml",
        "dark_ab_2.yaml",
        "dark_ab_3.yaml",
        "dark_clahe.yaml",
        "dark_clahe_gamma.yaml",
        "dark_normalized.yaml",
        "dark_normalized_gamma.yaml",
        
        "dark_gray.yaml",
        "dark_gray_clahe.yaml",
        
    ]

    all_results = []

    for data_yaml in datasets:
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Could not find dataset YAML: {data_yaml}")

        print(f"Evaluating: {data_yaml}")
        metrics = evaluate_dataset(
            model_path=model_path,
            data_yaml=data_yaml,
            imgsz=640,
            conf=0.25,
            iou=0.6,
            split="val"
        )
        all_results.append(metrics)

    print_comparison_table(all_results)
    save_results_json(all_results, "dark_variant_results.json")


if __name__ == "__main__":
    main()