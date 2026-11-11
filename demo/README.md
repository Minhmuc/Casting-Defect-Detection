# Demo phân loại sản phẩm đúc trên máy

Giao diện dùng MobileNetV2 của lần thực nghiệm casting_full_3models_01. Mô hình và metadata được đặt trong demo/models. Các tệp được sao chép từ gói đã kiểm chứng trên Colab.

## 1. Chuẩn bị môi trường

Mở PowerShell trong thư mục dự án. Dùng Python 3.11, 3.12 hoặc 3.13 bản 64 bit. Ví dụ sau chọn Python 3.12 đã cài trên máy.

```powershell
cd D:\Casting-Defect-Detection-CNN
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo\requirements.txt
```

Nếu cần xem các phiên bản Python đang có, chạy `py -0p`, sau đó thay `-3.12` bằng phiên bản phù hợp.

## 2. Mở giao diện

```powershell
.\.venv-demo\Scripts\python.exe -m streamlit run demo\app.py
```

Mở địa chỉ local do terminal hiển thị, thường là http://localhost:8501. Chọn ảnh, nhấn **Phân loại ảnh** và xem nhãn cùng xác suất. Nút tải JSON lưu kết quả để đối chiếu với notebook.

Ứng dụng được cấu hình suy luận trên CPU. TensorFlow 2.20 trên Windows có thể dùng CPU cho demo này. Giao diện triển khai quy trình ảnh xám 256 × 256, resize TensorFlow bilinear có antialias về 224 × 224 và lặp ba kênh, giống notebook. Chuẩn hóa [-1,1] đã có trong mô hình.

## 3. Tự kiểm tra trước khi bảo vệ

1. Chọn ảnh `demo/reference/cast_ok_0_45.jpeg`. Tệp này đã được sao chép từ đúng test/ok_front và đối chiếu mã băm với danh sách test.
2. Kết quả tham chiếu trong notebook là **Đạt**, xác suất đạt hiển thị 99,82%. Thư mục demo/reference lưu hình và văn bản kết quả đã trích từ notebook. Giá trị test của ảnh này trong test_predictions_mobilenetv2.csv là p_ok = 0,99819237.
3. Thử thêm một ảnh lỗi từ demo_test/def_front và quan sát kết quả. Nhãn thật được xác định theo thư mục của tập dữ liệu.
4. Tải JSON kết quả, kiểm tra tên mô hình là mobilenetv2 và mã băm khớp metadata.
5. Chụp giao diện sau khi bạn thực hiện nếu muốn bổ sung hình giao diện local vào báo cáo.

Notebook đã kiểm chứng nạp lại checkpoint trên CPU với 10 ảnh validation. Giao diện local mới được soạn và kiểm tra tĩnh; bạn thực hiện các bước trên để xác nhận môi trường cài đặt và hoạt động thực tế.

Tài liệu API dùng để xây giao diện có tại Streamlit và Keras. [Upload ảnh](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [lưu và nạp mô hình](https://keras.io/guides/serialization_and_saving/).
