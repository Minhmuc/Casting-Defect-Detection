import csv
import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import (
    CLASS_TITLES, MODEL_DIR, MODEL_SPECS, load_classifier, prepare_image, read_metadata, tf,
)


load_cached_classifier = st.cache_resource(load_classifier)

st.set_page_config(page_title="Kiểm tra sản phẩm đúc", layout="wide")
styles = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
st.html(f"<style>{styles}</style>")
st.title("Kiểm tra lỗi sản phẩm đúc")
st.write("Chọn mô hình và tải ảnh để xem, so sánh kết quả phân loại.")

selected_models = st.multiselect(
    "Mô hình sử dụng", list(MODEL_SPECS), default=list(MODEL_SPECS),
    format_func=lambda key: MODEL_SPECS[key]["title"],
    help="Giữ một mô hình để chạy riêng hoặc chọn cả ba để so sánh trên cùng ảnh.",
)
model_metadata = {}
for model_key in selected_models:
    try:
        model_metadata[model_key] = read_metadata(model_key)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"{MODEL_SPECS[model_key]['title']}. {error}")
        st.caption("Xem bước chuẩn bị ba mô hình trong demo/README.md.")

if model_metadata:
    checkpoints = [f"{MODEL_SPECS[key]['title']} · epoch {value['selection']['best_epoch']}"
                   for key, value in model_metadata.items()]
    st.caption(" | ".join(checkpoints))
st.caption("Chạy trên CPU. Nhãn đạt được chọn khi điểm đạt ≥ 0,5.")

with st.container(border=True):
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh sản phẩm", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    st.caption(f"Đã chọn {len(uploaded_files)} ảnh và {len(selected_models)} mô hình")
    ready = bool(uploaded_files and selected_models and len(model_metadata) == len(selected_models))
    classify = st.button("Phân loại tất cả", type="primary", disabled=not ready)

selection = (
    tuple(selected_models),
    tuple((key, value["model_sha256"], value["threshold"]) for key, value in model_metadata.items()),
    tuple((upload.name, hashlib.sha256(upload.getvalue()).hexdigest()) for upload in uploaded_files),
)
if st.session_state.get("image_selection") != selection:
    st.session_state["image_selection"] = selection
    st.session_state.pop("classification_results", None)
    st.session_state.pop("image_previews", None)

if classify:
    classifiers = {}
    loading_errors = {}
    with st.spinner("Đang nạp các checkpoint đã chọn..."):
        for key, metadata in model_metadata.items():
            try:
                classifiers[key] = load_cached_classifier(
                    str(MODEL_DIR / metadata["model_file"]), metadata["model_sha256"], key,
                )
            except (ImportError, OSError, ValueError, TypeError, tf.errors.OpError) as error:
                loading_errors[key] = str(error)

    results = []
    previews = {}
    progress = st.progress(0, text="Đang phân loại ảnh...")
    total_predictions = len(uploaded_files) * len(selected_models)
    for index, upload in enumerate(uploaded_files, start=1):
        for key, metadata in model_metadata.items():
            result = {
                "index": index, "image": upload.name, "model": key,
                "best_epoch": metadata["selection"]["best_epoch"],
                "model_sha256": metadata["model_sha256"], "status": "error",
                "predicted_label": None, "label": None, "p_ok": None, "p_defect": None, "error": "",
            }
            try:
                if key in loading_errors:
                    raise ValueError(loading_errors[key])
                upload.seek(0)
                with Image.open(upload) as source:
                    preview, batch = prepare_image(source, metadata["preprocessing"])
                preview.thumbnail((600, 400))
                previews[index] = preview
                with tf.device("/CPU:0"):
                    p_ok = float(classifiers[key](batch, training=False).numpy()[0, 0])
                if not np.isfinite(p_ok) or not 0 <= p_ok <= 1:
                    raise ValueError("Điểm dự đoán nằm ngoài khoảng hợp lệ 0–1.")
                predicted_label = int(p_ok >= metadata["threshold"])
                result.update(
                    status="success", predicted_label=predicted_label,
                    label=CLASS_TITLES[predicted_label], p_ok=p_ok, p_defect=1 - p_ok,
                )
            except (OSError, ValueError, TypeError, UnidentifiedImageError,
                    Image.DecompressionBombError, tf.errors.OpError) as error:
                result["error"] = str(error)
            results.append(result)
            progress.progress(len(results) / total_predictions,
                              text=f"Đã xử lý {len(results)}/{total_predictions} lượt phân loại")
    progress.empty()
    st.session_state["classification_results"] = results
    st.session_state["image_previews"] = previews

results = st.session_state.get("classification_results", [])
if results:
    st.subheader("Kết quả phân loại")
    images, models, predictions = st.columns(3)
    images.metric("Số ảnh", len(uploaded_files))
    models.metric("Số mô hình", len(selected_models))
    predictions.metric("Lượt thành công", sum(row["status"] == "success" for row in results))
    failed = sum(row["status"] == "error" for row in results)
    if failed:
        st.warning(f"{failed} lượt gặp lỗi. Xem thông báo ở từng ảnh bên dưới.")

    summary = []
    for key in selected_models:
        rows = [row for row in results if row["model"] == key]
        summary.append({
            "Mô hình": MODEL_SPECS[key]["title"],
            "Dự đoán lỗi": sum(row["predicted_label"] == 0 for row in rows),
            "Dự đoán đạt": sum(row["predicted_label"] == 1 for row in rows),
            "Lỗi xử lý": sum(row["status"] == "error" for row in rows),
        })
    st.dataframe(summary, hide_index=True, width="stretch")

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(results[0]))
    writer.writeheader()
    for result in results:
        csv_row = {}
        for key, value in result.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            csv_row[key] = value
        writer.writerow(csv_row)
    csv_button, json_button = st.columns(2)
    csv_button.download_button(
        "Tải kết quả CSV", csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="ket_qua_phan_loai.csv", mime="text/csv", on_click="ignore",
    )
    json_button.download_button(
        "Tải kết quả JSON", json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        file_name="ket_qua_phan_loai.json", mime="application/json", on_click="ignore",
    )
    st.caption("Điểm nằm trong khoảng 0–1. Điểm càng lớn, mô hình càng nghiêng về lớp tương ứng.")

    for index, upload in enumerate(uploaded_files, start=1):
        with st.container(border=True):
            st.text(f"{index}. {upload.name}")
            preview = st.session_state["image_previews"].get(index)
            if preview is not None:
                st.image(preview, width=180)
            image_results = [row for row in results if row["index"] == index]
            display_rows = [{
                "Mô hình": MODEL_SPECS[row["model"]]["title"],
                "Kết quả": row["label"] or "Lỗi xử lý",
                "Điểm lỗi": row["p_defect"], "Điểm đạt": row["p_ok"],
            } for row in image_results]
            st.dataframe(
                display_rows, hide_index=True, width="stretch",
                column_config={
                    "Điểm lỗi": st.column_config.NumberColumn(format="%.6f"),
                    "Điểm đạt": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            for row in image_results:
                if row["status"] == "error":
                    st.error(f"{MODEL_SPECS[row['model']]['title']}. {row['error']}")

st.caption("Nhóm 10 · Học phần Học sâu · ICTU")
