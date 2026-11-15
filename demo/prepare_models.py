"""Chuẩn bị ba checkpoint chính thức và đối chiếu dự đoán trên CPU."""

import csv
from copy import deepcopy
import json
from pathlib import Path
import hashlib
import shutil

import numpy as np
from PIL import Image

from inference import MODEL_DIR, MODEL_SPECS, load_classifier, prepare_image, tf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "casting_full_3models_01"


def main():
    base_metadata = json.loads((SOURCE / "demo/metadata.json").read_text(encoding="utf-8"))
    prepared = {}
    verification_report = {}
    for key, spec in MODEL_SPECS.items():
        fit = json.loads((SOURCE / f"fit_{key}.json").read_text(encoding="utf-8"))
        source_model = SOURCE / f"{key}.keras"
        metadata = deepcopy(base_metadata)
        metadata.update(
            model=key, model_file=f"{key}.keras", model_sha256=fit["checkpoint_sha256"],
            image_size=list(spec["shape"][:2]), channels=spec["shape"][2],
            selection={"model": key, "criterion": "minimum_validation_loss_within_model",
                       "best_epoch": fit["best_epoch"], "validation_loss": fit["best_validation_loss"]},
        )
        if key == "mobilenetv2":
            metadata = deepcopy(base_metadata)
        if key == "cnn":
            metadata["preprocessing"].update(transfer_resize=None, repeat_grayscale_channels=False)
        classifier = load_classifier(source_model, metadata["model_sha256"], key)
        with (SOURCE / f"test_predictions_{key}.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        references = []
        for label in ("0", "1"):
            references.extend([row for row in rows if row["true"] == label][:5])
        if len(references) != 10:
            raise ValueError(f"{key}: cần đủ 5 ảnh lỗi và 5 ảnh đạt để đối chiếu.")
        actual, expected, records = [], [], []
        for row in references:
            image_path = SOURCE / "demo_test" / Path(row["path"]).relative_to("test")
            with Image.open(image_path) as image:
                _, batch = prepare_image(image, metadata["preprocessing"])
            with tf.device("/CPU:0"):
                probability = float(classifier(batch, training=False).numpy()[0, 0])
            actual.append(probability)
            expected.append(float(row["p_ok"]))
            records.append({"image": row["path"], "reference_p_ok": float(row["p_ok"]),
                            "cpu_p_ok": probability})
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        labels_agree = bool(np.array_equal(actual >= 0.5, expected >= 0.5))
        if not (np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
                and labels_agree and np.allclose(actual, expected, rtol=1e-3, atol=1e-4)):
            raise ValueError(f"{key}: dự đoán CPU khác kết quả tham chiếu.")
        verification = {
            "reference": f"test_predictions_{key}.csv", "dataset": "test", "count": 10,
            "labels_agree": labels_agree, "max_abs_error": float(np.max(np.abs(actual-expected))),
            "rtol": 1e-3, "atol": 1e-4,
        }
        if key != "mobilenetv2":
            metadata["verification"] = verification
        prepared[key] = (source_model, metadata)
        verification_report[key] = {**verification, "records": records}
        print(f"{spec['title']}: epoch {fit['best_epoch']}, nhãn khớp 10/10, "
              f"sai lệch lớn nhất {verification['max_abs_error']:.8g}", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    for key, (source_model, metadata) in prepared.items():
        destination = MODEL_DIR / metadata["model_file"]
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != metadata["model_sha256"]:
            temporary = destination.with_suffix(".pending")
            shutil.copy2(source_model, temporary)
            temporary.replace(destination)
        metadata_path = MODEL_DIR / MODEL_SPECS[key]["metadata"]
        temporary = metadata_path.with_suffix(".pending")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
    (MODEL_DIR / "comparison_verification.json").write_text(
        json.dumps(verification_report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Đã chuẩn bị ba mô hình trong {MODEL_DIR}")


if __name__ == "__main__":
    main()
