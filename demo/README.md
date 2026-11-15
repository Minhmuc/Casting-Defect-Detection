# Demo phân loại và so sánh ba mô hình

Ứng dụng Streamlit chạy trên Windows/CPU, nhận một hoặc nhiều ảnh và cho chọn các mô hình cần sử dụng. Mặc định chọn cả ba để so sánh trên cùng ảnh.

| Mô hình | Vai trò | Checkpoint | Đầu vào |
| --- | --- | --- | --- |
| CNN | Baseline học từ đầu | Epoch 29 | 256 × 256 × 1 |
| ResNet-18 | Học chuyển giao, backbone cố định | Epoch 29 | 224 × 224 × 3 |
| MobileNetV2 | Học chuyển giao, backbone cố định | Epoch 21 | 224 × 224 × 3 |

Các checkpoint lấy từ lượt chính thức `casting_full_3models_01`, mỗi checkpoint có validation loss thấp nhất của mô hình tương ứng. MobileNetV2 có validation loss thấp nhất trong ba cấu hình ở báo cáo. Demo cho phép xem CNN, ResNet-18 và MobileNetV2 để trình bày phép so sánh.

## 1. Cài thư viện

Mở PowerShell tại thư mục gốc dự án. Dùng Python 3.11–3.13 bản 64 bit, ví dụ Python 3.12.

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r .\demo\requirements.txt
```

Nếu đã có môi trường `.venv` đầy đủ của dự án, có thể dùng `.\.venv\Scripts\python.exe` thay cho `.venv-demo` trong các lệnh.

## 2. Chuẩn bị mô hình

Thư mục `demo/models/` cần có các tệp sau.

```text
cnn.keras
metadata_cnn.json
resnet18.keras
metadata_resnet18.json
casting_classifier.keras
metadata.json
comparison_verification.json
```

Ba tệp `.keras` chứa checkpoint gốc. Metadata tương ứng ghi tên mô hình, đầu vào, quy ước nhãn, tiền xử lý và mã băm. `comparison_verification.json` ghi kết quả đối chiếu dự đoán trên CPU của cả ba mô hình.

Khi có bộ kết quả `casting_full_3models_01/` tại thư mục gốc, chạy lệnh chuẩn bị sau.

```powershell
.\.venv-demo\Scripts\python.exe .\demo\prepare_models.py
```

Lệnh này đọc các checkpoint và tệp `fit_*.json`, đối chiếu dự đoán của từng mô hình trên 5 ảnh lỗi và 5 ảnh đạt với `test_predictions_*.csv`, sau đó chép mô hình cùng metadata vào `demo/models/`. Lệnh chỉ nạp và suy luận từ checkpoint đã huấn luyện.

Nếu nhận gói demo đã chuẩn bị, đặt các tệp trong `models/` đúng vị trí rồi mở ứng dụng. Thư mục mô hình được bỏ qua bởi Git, nên bản clone cần tải kèm trọng số từ gói mã nguồn của nhóm. Gói `demo_bundle.zip` cũ từ notebook chứa MobileNetV2. Để có đủ ba mô hình, dùng bộ kết quả chính thức hoặc lấy toàn bộ `demo/models/` đã chuẩn bị.

## 3. Mở ứng dụng

Tại thư mục gốc dự án, chạy lệnh sau.

```powershell
.\.venv-demo\Scripts\python.exe -m streamlit run .\demo\app.py
```

Mở Local URL trong terminal, thường là `http://localhost:8501`. Giữ terminal mở trong lúc sử dụng. Nhấn Ctrl+C để dừng ứng dụng. Các lần sau chỉ cần chạy lại lệnh mở Streamlit.

## 4. Phân loại và so sánh

1. Trong **Mô hình sử dụng**, giữ cả ba mô hình để so sánh hoặc bỏ bớt để chạy riêng từng mô hình.
2. Chọn một hoặc nhiều ảnh JPG, JPEG, PNG, BMP rồi nhấn **Phân loại tất cả**.
3. Xem số ảnh, số mô hình và số lượt phân loại thành công. Với 2 ảnh và 3 mô hình, chương trình tạo **6 lượt phân loại**.
4. Bảng thống kê đếm dự đoán lỗi/đạt riêng cho từng mô hình. Mỗi ảnh có bảng đối chiếu CNN, ResNet-18, MobileNetV2, nhãn và điểm dự đoán.
5. Tải kết quả dưới dạng CSV hoặc JSON. Mỗi bản ghi tương ứng một cặp ảnh–mô hình. Dữ liệu có tên ảnh, mã mô hình, epoch checkpoint, mã băm, nhãn, điểm và trạng thái xử lý.

Điểm đạt là `p_ok`, điểm lỗi bằng `1 − p_ok`. Điểm nằm trong khoảng 0–1 và được hiển thị với 6 chữ số thập phân. Nhãn đạt được chọn khi điểm đạt đầy đủ ≥ 0,5. CSV/JSON giữ giá trị số trước khi làm tròn. Khi thay đổi mô hình hoặc danh sách ảnh, kết quả cũ được xóa để chờ phân loại lại.

Ảnh đạt tham chiếu nằm tại `demo/reference/cast_ok_0_45.jpeg`. Ảnh lỗi có thể chọn `casting_full_3models_01/demo_test/def_front/cast_def_0_1059.jpeg`. Ảnh của hai lớp nằm trong `demo_test/def_front/` và `demo_test/ok_front/`. Trên cùng ảnh đạt 45, ba mô hình có thể cùng dự đoán đạt nhưng cho điểm khác nhau. Điểm của một ảnh được đọc riêng với accuracy trên toàn bộ 715 ảnh test.

Một ảnh đọc lỗi được báo riêng theo từng mô hình. Các ảnh hợp lệ vẫn được xử lý. Số thứ tự giúp phân biệt các tệp có cùng tên.

## 5. Tiền xử lý và nạp ResNet-18 trên Windows

Tất cả ảnh được hiệu chỉnh EXIF, chuyển xám và resize Pillow bilinear về 256 × 256. CNN dùng trực tiếp một kênh. Hai mạng học chuyển giao tiếp tục resize TensorFlow bilinear có antialias về 224 × 224 rồi lặp kênh xám thành ba kênh. Chuẩn hóa nằm trong mô hình.

Tệp ResNet-18 chứa lớp `ResNetImageConverter` của KerasHub 0.23. Trên Windows, demo nạp cấu hình xử lý ảnh bằng các lớp Keras tương ứng, theo đúng thứ tự resize, nhân scale và cộng offset, rồi nạp trọng số từ checkpoint gốc. Phần này nằm trong `inference.py` và đã được đối chiếu với dự đoán Colab. Cách xử lý dựa trên mã nguồn và tài liệu chính thức. [ImageConverter 0.23.0](https://github.com/keras-team/keras-hub/blob/v0.23.0/keras_hub/src/layers/preprocessing/image_converter.py) · [ResNetImageConverter](https://keras.io/keras_hub/api/models/resnet/resnet_image_converter/)

`keras-hub` được cài để nạp backbone ResNet-18. Trên các nền tảng có gói TensorFlow Text tương ứng, requirements cài phiên bản phù hợp TensorFlow. Thông tin hỗ trợ nền tảng được công bố tại kho chính thức. [TensorFlow Text](https://github.com/tensorflow/text#installation)

## 6. Kiểm tra khi chuẩn bị bảo vệ

- Chạy cùng hai ảnh đạt/lỗi với cả ba mô hình, kiểm tra đủ 6 bản ghi.
- Chạy riêng từng mô hình để kiểm tra bộ chọn và số lượt phân loại.
- Thêm một tệp JPG hỏng để kiểm tra thông báo và kết quả của các ảnh hợp lệ.
- Tải CSV và JSON, đối chiếu số bản ghi, tên ảnh, mô hình, nhãn và điểm.
- Đổi mô hình hoặc bỏ một ảnh, kiểm tra kết quả cũ được xóa và chạy lại được.

Phép kiểm chứng checkpoint dùng 10 ảnh test cho mỗi mô hình. Các nhãn đều khớp dự đoán đã lưu, với dung sai tương đối 10⁻³ và tuyệt đối 10⁻⁴. Chi tiết được ghi tại `models/comparison_verification.json`. Đây là phép đối chiếu sau khi nạp mô hình. Kết quả đánh giá toàn bộ test nằm ở Chương 3 của báo cáo.

Ngày 15/11/2026, giao diện đã chạy thành công cả ba mô hình với hai ảnh tham chiếu, tạo và tải đủ 6 bản ghi CSV/JSON. Chế độ chạy riêng CNN cho 2 kết quả. Đổi lựa chọn đã xóa kết quả cũ. Phép thử thêm một tệp JPG hỏng cho 3 thông báo lỗi riêng, đồng thời giữ 6 lượt thành công của hai ảnh hợp lệ.

Nếu thiếu thư viện, cài lại bằng chính Python dùng để mở ứng dụng. Nếu thiếu mô hình hoặc mã băm không khớp, chuẩn bị lại theo mục 2 từ bộ kết quả chính thức. Khi chuyển máy, kiểm tra đủ mã `app.py`, `inference.py`, `style.css`, requirements, cấu hình `.streamlit/config.toml` và thư mục `models/`.

[README tổng hợp](../README.md) · [Notebook](../Casting.ipynb) · [Mã giao diện](app.py)
