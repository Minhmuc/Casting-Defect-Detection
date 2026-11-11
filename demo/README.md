# Chạy demo phân loại sản phẩm đúc trên Windows

Ứng dụng Streamlit nhận một ảnh và hiển thị nhãn **Sản phẩm lỗi** hoặc **Sản phẩm đạt yêu cầu**, kèm điểm dự đoán của hai lớp. Demo sử dụng **MobileNetV2 epoch 21** của lượt `casting_full_3models_01`, được chọn theo validation loss và lưu bằng định dạng `.keras`. Các phép dự đoán chạy trên CPU.

## 1. Chuẩn bị tệp mô hình

Trong thư mục dự án, cần có cấu trúc sau.

```text
Casting-Defect-Detection-CNN/
├── demo/
│   ├── app.py
│   ├── requirements.txt
│   ├── models/
│   │   ├── casting_classifier.keras
│   │   └── metadata.json
│   └── reference/
│       └── cast_ok_0_45.jpeg
└── Casting.ipynb
```

`casting_classifier.keras` chứa kiến trúc và trọng số. `metadata.json` ghi nhãn, quy tắc xử lý ảnh và mã băm của đúng checkpoint. Hai tệp cần đi cùng nhau từ gói mô hình đã kiểm chứng.

Nếu nhận dự án qua Git, lấy hai tệp từ `demo_bundle.zip` của lượt chính thức và giải nén vào `demo/models/`. Thư mục mô hình được bỏ qua trong cấu hình Git, nên cần kiểm tra bước này trên máy của từng thành viên. Trong bộ kết quả đã giải nén, các tệp tương ứng nằm ở `casting_full_3models_01/demo/`.

Khi có bộ kết quả tại thư mục gốc dự án, có thể sao chép bằng PowerShell.

```powershell
New-Item -ItemType Directory -Path .\demo\models -Force
Copy-Item -LiteralPath .\casting_full_3models_01\demo\casting_classifier.keras -Destination .\demo\models\casting_classifier.keras -Force
Copy-Item -LiteralPath .\casting_full_3models_01\demo\metadata.json -Destination .\demo\models\metadata.json -Force
```

## 2. Cài môi trường lần đầu

Dùng Python 3.11, 3.12 hoặc 3.13 bản 64 bit. Các lệnh dưới đây lấy Python 3.12 làm ví dụ. Mở PowerShell tại **thư mục gốc dự án**, nơi có `Casting.ipynb` và thư mục `demo`. Nếu terminal đang ở trong `demo`, chạy `cd ..` để về thư mục gốc.

Kiểm tra các phiên bản Python đang có.

```powershell
py -0p
```

Tạo môi trường và cài thư viện.

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r .\demo\requirements.txt
```

Nếu máy có Python 3.11 hoặc 3.13, thay `-3.12` bằng phiên bản đang cài. File requirements của demo cài TensorFlow, Keras, NumPy, Pillow và Streamlit. Lệnh gọi trực tiếp Python trong `.venv-demo` giúp dùng đúng môi trường đã tạo.

## 3. Mở và tắt giao diện

Từ thư mục gốc dự án, chạy lệnh sau.

```powershell
.\.venv-demo\Scripts\python.exe -m streamlit run .\demo\app.py
```

Mở địa chỉ **Local URL** do terminal hiển thị, thường là `http://localhost:8501`. Giữ terminal mở trong lúc dùng giao diện. Để dừng ứng dụng, nhấn **Ctrl+C** tại terminal.

Những lần sử dụng tiếp theo, mở PowerShell ở thư mục gốc và chạy lại lệnh Streamlit ở trên. Môi trường `.venv-demo` được dùng lại cho các lần mở sau.

## 4. Thử phân loại ảnh

1. Trong giao diện, chọn tệp JPG, JPEG, PNG hoặc BMP.
2. Nhấn **Phân loại ảnh**. Chương trình hiển thị nhãn và điểm dự đoán của lớp lỗi, lớp đạt.
3. Nhấn **Tải kết quả JSON** nếu muốn lưu tên ảnh, nhãn, điểm dự đoán và thông tin checkpoint.
4. Chọn ảnh tiếp theo rồi nhấn nút phân loại lại.

Ảnh đạt để đối chiếu nằm tại `demo/reference/cast_ok_0_45.jpeg`. Kết quả tham chiếu từ notebook là **Đạt**, điểm đạt hiển thị **99,82%**. Giá trị đã lưu trong bảng dự đoán test là `p_ok = 0,99819237`. [Ảnh tham chiếu](reference/cast_ok_0_45.jpeg) · [Kết quả từ notebook](reference/demo_cast_ok_0_45.png)

Để thử ảnh lỗi, chọn ảnh từ `casting_full_3models_01/demo_test/def_front/`. Ảnh đạt nằm trong `casting_full_3models_01/demo_test/ok_front/`. Nhãn thật của các ảnh này được xác định theo thư mục. Có thể thử ảnh tự chụp với điều kiện tương tự bộ dữ liệu và đối chiếu kết quả với tình trạng thực tế của sản phẩm.

Đầu ra `p_ok` là điểm lớp đạt. Chương trình gán nhãn đạt khi `p_ok ≥ 0,5` và nhãn lỗi khi thấp hơn ngưỡng. Điểm 99,82% thuộc một ảnh cụ thể. Accuracy 99,44% trong báo cáo được tính trên toàn bộ 715 ảnh test.

## 5. Chuẩn bị minh họa khi bảo vệ

Mở giao diện và thử một ảnh đạt, một ảnh lỗi trước buổi bảo vệ. Khi trình bày, giới thiệu ảnh đầu vào, nhấn phân loại rồi đọc nhãn và điểm dự đoán. Các tệp JSON tải về giúp đối chiếu kết quả với checkpoint.

Sau khi tự chạy thành công, chụp 1–2 ảnh giao diện có ảnh sản phẩm và kết quả để bổ sung vào báo cáo, gồm một trường hợp lỗi và một trường hợp đạt. Hình `reference/demo_cast_ok_0_45.png` hiện là output từ notebook.

## 6. Xử lý lỗi thường gặp

| Thông báo hoặc hiện tượng | Cách xử lý |
|---|---|
| Máy chưa nhận lệnh `py` hoặc phiên bản `-3.12` | Cài Python bản 64 bit, mở lại PowerShell và kiểm tra bằng `py -0p`. Chọn phiên bản có trên máy. |
| Không tìm thấy `.venv-demo\Scripts\python.exe` | Kiểm tra terminal đang ở thư mục gốc và đã tạo môi trường ở mục 2. |
| Không tìm thấy `app.py` hoặc `requirements.txt` | Kiểm tra thư mục hiện tại có thư mục `demo`, sau đó dùng đúng các lệnh ở mục 2–3. |
| `No module named streamlit` hoặc `tensorflow` | Cài lại theo mục 2 bằng Python trong `.venv-demo`, rồi mở ứng dụng bằng chính Python đó. |
| Giao diện báo chưa nạp được mô hình hoặc thiếu `metadata.json` | Kiểm tra hai tệp trong `demo/models/` theo mục 1. |
| Giao diện báo trọng số khác mô hình đã kiểm chứng | Sao chép lại cả mô hình và metadata từ cùng gói demo chính thức. |
| Không đọc được ảnh | Chọn lại một tệp ảnh hợp lệ thuộc các định dạng ở mục 4. |

Notebook đã kiểm chứng checkpoint sau khi nạp trên CPU với 10 ảnh validation, nhãn trùng 10/10 ảnh và sai lệch điểm lớn nhất khoảng 4,21 × 10⁻⁸. Giao diện local được kiểm tra tĩnh. Các bước chạy trên đây giúp từng thành viên xác nhận môi trường và kết quả thực tế trên máy của mình.

[README tổng hợp của nhóm](../README.md) · [Notebook thực nghiệm](../Casting.ipynb) · [Mã giao diện](app.py)
