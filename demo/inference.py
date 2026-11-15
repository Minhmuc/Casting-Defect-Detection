import hashlib
import importlib
import json
import os
from pathlib import Path
import zipfile

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_DIR = Path(__file__).resolve().parent / "models"
CLASS_TITLES = ["Sản phẩm lỗi", "Sản phẩm đạt yêu cầu"]
MODEL_SPECS = {
    "cnn": {"title": "CNN (baseline)", "metadata": "metadata_cnn.json", "shape": (256, 256, 1)},
    "resnet18": {"title": "ResNet-18", "metadata": "metadata_resnet18.json", "shape": (224, 224, 3)},
    "mobilenetv2": {"title": "MobileNetV2", "metadata": "metadata.json", "shape": (224, 224, 3)},
}


class ResNetImageConverterForInference(keras.layers.Layer):
    """Nạp cấu hình xử lý ảnh của KerasHub bằng các phép Keras tương ứng.

    KerasHub 0.23 kiểm tra tensorflow-text cả ở lớp xử lý ảnh trên Windows.
    Thứ tự resize, nhân scale và cộng offset theo ImageConverter 0.23.0.
    https://github.com/keras-team/keras-hub/blob/v0.23.0/keras_hub/src/layers/preprocessing/image_converter.py
    """

    def __init__(self, image_size=None, scale=None, offset=None, interpolation="bilinear",
                 antialias=False, crop_to_aspect_ratio=True, pad_to_aspect_ratio=False,
                 bounding_box_format="yxyx", data_format=None, **kwargs):
        super().__init__(**kwargs)
        if data_format not in (None, "channels_last"):
            raise ValueError("Checkpoint của dự án sử dụng channels_last.")
        self.scale = scale
        self.offset = offset
        self.resizing = None
        if image_size is not None:
            self.resizing = keras.layers.Resizing(
                image_size[0], image_size[1], interpolation=interpolation, antialias=antialias,
                crop_to_aspect_ratio=crop_to_aspect_ratio, pad_to_aspect_ratio=pad_to_aspect_ratio,
                data_format="channels_last", dtype=self.dtype_policy,
            )

    def call(self, inputs):
        output = self.resizing(inputs) if self.resizing is not None else inputs
        if self.scale is not None:
            output = output * keras.ops.cast(np.asarray(self.scale), self.compute_dtype)
        if self.offset is not None:
            output = output + keras.ops.cast(np.asarray(self.offset), self.compute_dtype)
        return output


def read_metadata(model_key):
    spec = MODEL_SPECS[model_key]
    metadata = json.loads((MODEL_DIR / spec["metadata"]).read_text(encoding="utf-8"))
    preprocessing = metadata["preprocessing"]
    expected_transfer = [224, 224] if spec["shape"][2] == 3 else None
    if (metadata["model"] != model_key
            or metadata["classes"] != ["def_front", "ok_front"]
            or metadata["image_size"] != list(spec["shape"][:2])
            or metadata["channels"] != spec["shape"][2]
            or metadata["output_meaning"] != "p_ok"
            or metadata["input_range"] != [0, 255]
            or metadata["input_dtype"] != "float32"
            or metadata["threshold"] != 0.5
            or preprocessing["base_resize"] != [256, 256]
            or preprocessing.get("transfer_resize") != expected_transfer
            or preprocessing["normalization"] != "included in model"):
        raise ValueError("Metadata khác cấu hình thực nghiệm của mô hình.")
    if not metadata.get("verification", {}).get("labels_agree"):
        raise ValueError("Mô hình chưa hoàn tất kiểm chứng sau khi nạp trên CPU.")
    model_path = (MODEL_DIR / metadata["model_file"]).resolve()
    if model_path.parent != MODEL_DIR.resolve():
        raise ValueError("Đường dẫn mô hình trong metadata không hợp lệ.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Thiếu {model_path.name}. Chạy demo/prepare_models.py theo README.")
    return metadata


def load_classifier(model_file, expected_hash, model_key):
    model_path = Path(model_file)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Trọng số khác checkpoint đã kiểm chứng.")
    if model_key == "resnet18":
        # Đăng ký các lớp ResNetBackbone và ResNetImageConverter đã lưu trong tệp.
        importlib.import_module("keras_hub")
    with tf.device("/CPU:0"):
        if model_key == "resnet18":
            # Dùng tên lớp riêng để tránh registry KerasHub ghi đè custom_objects.
            with zipfile.ZipFile(model_path) as archive:
                config = json.loads(archive.read("config.json"))
            config.pop("compile_config", None)
            converters = [layer for layer in config["config"]["layers"]
                          if layer["class_name"] == "ResNetImageConverter"]
            if len(converters) != 1:
                raise ValueError("Checkpoint cần đúng một lớp ResNetImageConverter.")
            converters[0]["registered_name"] = "Casting>ResNetImageConverterForInference"
            model = keras.models.model_from_json(
                json.dumps(config),
                custom_objects={"Casting>ResNetImageConverterForInference": ResNetImageConverterForInference},
            )
            model.load_weights(model_path)
        else:
            model = keras.models.load_model(model_path, compile=False)
    model.jit_compile = False
    if (tuple(model.input_shape[1:]) != MODEL_SPECS[model_key]["shape"]
            or tuple(model.output_shape[1:]) != (1,)):
        raise ValueError("Kích thước đầu vào hoặc đầu ra khác cấu hình mô hình.")
    return model


def prepare_image(source, preprocessing):
    gray = ImageOps.exif_transpose(source).convert("L")
    height, width = preprocessing["base_resize"]
    resized = gray.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)[..., None]
    if preprocessing.get("transfer_resize"):
        with tf.device("/CPU:0"):
            array = tf.image.resize(array, preprocessing["transfer_resize"],
                                    method="bilinear", antialias=True)
            array = tf.image.grayscale_to_rgb(array).numpy()
    return gray, array[None, ...]
