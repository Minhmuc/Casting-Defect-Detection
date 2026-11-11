import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import hashlib
import json
from pathlib import Path

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_DIR = Path(__file__).resolve().parent / "models"
CLASS_TITLES = ["Sản phẩm lỗi", "Sản phẩm đạt yêu cầu"]


@st.cache_resource
def load_classifier(model_file, expected_hash):
    model_path = Path(model_file)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Trọng số khác mô hình đã kiểm chứng. Kiểm tra lại gói demo.")
    with tf.device("/CPU:0"):
        model = keras.models.load_model(model_path, compile=False)
    model.jit_compile = False
    if tuple(model.input_shape[1:]) != (224, 224, 3) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError("Kích thước đầu vào/đầu ra khác cấu hình MobileNetV2 đã chốt.")
    return model


def prepare_image(source, preprocessing):
    gray = ImageOps.exif_transpose(source).convert("L")
    height, width = preprocessing["base_resize"]
    resized = gray.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)[..., None]
    with tf.device("/CPU:0"):
        array = tf.image.resize(array, preprocessing["transfer_resize"],
                                method="bilinear", antialias=True)
        array = tf.image.grayscale_to_rgb(array).numpy()
    return gray, array[None, ...]


st.set_page_config(page_title="Kiểm tra sản phẩm đúc", layout="centered")
st.title("Kiểm tra lỗi sản phẩm đúc")
st.write("Tải một ảnh sản phẩm để phân loại lỗi hoặc đạt yêu cầu bằng MobileNetV2.")

try:
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    preprocessing = metadata["preprocessing"]
    if metadata["model"] != "mobilenetv2" or metadata["classes"] != ["def_front", "ok_front"]:
        raise ValueError("Gói mô hình cần là MobileNetV2 của lần thực nghiệm chính thức.")
    verification = metadata.get("verification")
    if not isinstance(verification, dict) or not verification.get("labels_agree"):
        raise ValueError("Gói mô hình chưa hoàn tất kiểm chứng trên CPU.")
    if (metadata["image_size"] != [224, 224] or metadata["channels"] != 3
            or metadata["output_meaning"] != "p_ok" or metadata["input_range"] != [0, 255]
            or preprocessing["base_resize"] != [256, 256]
            or preprocessing["transfer_resize"] != [224, 224]
            or preprocessing["normalization"] != "included in model"):
        raise ValueError("Tiền xử lý trong metadata khác cấu hình đã triển khai.")
    model_path = (MODEL_DIR / metadata["model_file"]).resolve()
    if model_path.parent != MODEL_DIR.resolve():
        raise ValueError("Đường dẫn mô hình trong metadata không hợp lệ.")
    classifier = load_classifier(str(model_path), metadata["model_sha256"])
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Chưa nạp được mô hình. {error}")
    st.stop()

uploaded = st.file_uploader("Chọn ảnh sản phẩm", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    try:
        with Image.open(uploaded) as source:
            preview, batch = prepare_image(source, preprocessing)
        st.image(preview, caption=uploaded.name)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        st.error(f"Không đọc được ảnh. {error}")
        st.stop()

    if st.button("Phân loại ảnh", type="primary"):
        with st.spinner("Đang phân loại..."):
            with tf.device("/CPU:0"):
                p_ok = float(classifier(batch, training=False).numpy()[0, 0])
        if not np.isfinite(p_ok) or not 0 <= p_ok <= 1:
            st.error("Đầu ra mô hình không hợp lệ. Kiểm tra lại gói trọng số và ảnh đầu vào.")
            st.stop()
        predicted_label = int(p_ok >= metadata["threshold"])
        st.subheader(CLASS_TITLES[predicted_label])
        left, right = st.columns(2)
        left.metric("Xác suất lỗi", f"{1 - p_ok:.2%}")
        right.metric("Xác suất đạt", f"{p_ok:.2%}")
        st.caption("Xác suất do mô hình ước lượng. Ngưỡng phân loại lớp đạt là 0,5.")
        result = {"image": uploaded.name, "model": metadata["model"],
                  "predicted_label": predicted_label, "p_ok": p_ok,
                  "model_sha256": metadata["model_sha256"]}
        st.download_button("Tải kết quả JSON", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="ket_qua_phan_loai.json", mime="application/json")

st.caption("Nhóm 10 · Học phần Học sâu · ICTU")
