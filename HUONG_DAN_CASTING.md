# Chạy thực nghiệm casting_data trên Google Colab

Notebook chính là `Casting.ipynb`. CNN Conv2D/MaxPooling học từ đầu; ResNet-18 và MobileNetV2 dùng backbone ImageNet cố định. Mỗi cấu hình chạy 30 epoch. Mô hình demo được chọn theo validation loss thấp nhất.

## 1. Chuẩn bị dữ liệu và môi trường

1. Trên Windows, nén thư mục `D:/Casting-Defect-Detection-CNN/data/casting_data` thành ZIP. Bên trong cần có train/def_front, train/ok_front, test/def_front và test/ok_front. ZIP có thể có thêm các cấp thư mục bọc ngoài.
2. Mở notebook mới trên Colab bằng **File → Upload notebook**.
3. Chọn **Runtime → Change runtime type → T4 GPU**, hoặc GPU khác được cấp. CPU vẫn chạy được nhưng lâu hơn.
4. Chạy hai ô ở mục 1 để gỡ ba gói Colab xung đột và cài phiên bản dùng trong notebook.
5. Chọn **Runtime → Restart session**, sau đó chạy từ mục 2 trở đi. Phiên bản yêu cầu gồm TensorFlow2.20.0, Keras3.12.0, KerasHub0.23.0 và NumPy2.2.6.
6. Mục 3 upload ZIP vừa nén. Kiểm tra đường dẫn được chọn chứa cả train và test.

Ô cài thư viện đã có trong notebook. Chuẩn bị notebook và ZIP dữ liệu là đủ cho quy trình Colab này. Bộ casting_data có 6.633 ảnh train và 715 ảnh test, kích thước 300 × 300. Thư mục casting_512x512 là bộ 1.300 ảnh dùng trong lần trước.

## 2. Chia tập và xử lý ảnh trùng

Giữ 715 ảnh test, đọc test trước để ưu tiên bảo toàn tập đánh giá. Loại các bản sao ở train trùng pixel với test hoặc với ảnh train đã giữ khỏi danh sách huấn luyện. Mã chỉ đọc tệp ảnh gốc và tạo bản sao ảnh demo trong thư mục kết quả.

Kiểm tra dữ liệu hiện tại phát hiện **64 ảnh đạt trong train trùng pixel với test**. Số lượng dự kiến sau kiểm tra là

| Tập | Số ảnh |
|---|---:|
| Train nguồn sau loại trùng | 6.569 |
| Train thực dùng | 5.583 |
| Validation, tách 15% từ train đã làm sạch | 986 |
| Test giữ nguyên | 715 |

Đối chiếu với output mục 4. Danh sách ảnh loại nằm trong excluded_train_duplicates.csv; danh sách ảnh thực dùng nằm trong split.json. Ảnh hỏng, trùng khác nhãn hoặc trùng trong test sẽ khiến notebook dừng để kiểm tra.

SHA256 chỉ phát hiện ảnh giống hệt nhau sau chuyển xám, trước resize. Các biến thể augmentation có thể gần giống nhau dù mã băm khác. Nguồn thiếu mã sản phẩm/nhóm ảnh để xác nhận tính độc lập theo sản phẩm; đây là giới hạn cần trình bày trong báo cáo.

Bộ nguồn đã có augmentation. Cách diễn đạt cho phương pháp của nhóm là “không bổ sung tăng cường dữ liệu trong pipeline huấn luyện”.

## 3. Thứ tự chạy

| Mục | Công việc và điểm kiểm tra |
|---|---|
| 2–3 | Cấu hình, thiết bị và upload ZIP |
| 4 | Kiểm tra ảnh, loại trùng ở train, chia validation, xem ảnh mẫu |
| 5 | Khai báo pipeline, hàm sao lưu và hàm huấn luyện |
| 6 | Xem cấu trúc và train CNN đủ 30 epoch |
| 7 | Xem cấu trúc và train ResNet-18 đủ 30 epoch |
| 8 | Xem cấu trúc và train MobileNetV2 đủ 30 epoch |
| 9 | Lưu lựa chọn theo validation loss, đánh giá ba checkpoint trên test |
| 10 | Tải bản sao kết quả, trọng số và ảnh test để chọn demo |
| 11 | Chuẩn bị mô hình đã chọn, kiểm chứng trên CPU |
| 12 | Upload ảnh, xem và lưu dự đoán |
| 13 | Tải gói demo và gói complete cuối |

Sau khi từng mô hình hoàn tất, notebook tạo gói after_cnn_*.zip, after_resnet18_*.zip hoặc after_mobilenetv2_*.zip và bắt đầu tải về. Nếu trình duyệt chặn tải, lấy tệp từ bảng **Files** trong outputs/. Các gói này chứa checkpoint cùng lịch sử và cấu hình để bảo toàn kết quả.

Lần chạy mặc định là `casting_full_3models_01`. Các cell train kiểm tra tệp đã tồn tại. Khi muốn bắt đầu một lượt huấn luyện khác, đổi RUN_NAME ở mục 2.

Nếu gặp lỗi XLA ResizeBicubic, kiểm tra dòng `candidate.jit_compile = False` sau khi nạp checkpoint ở mục 9. Chạy lại toàn bộ cell đánh giá sau khi sửa.

## 4. Demo nhanh và mô hình xuất

Sau mục 10, giải nén training_casting_full_3models_01.zip. Trong casting_full_3models_01/demo_test có def_front gồm 453 ảnh lỗi và ok_front gồm 262 ảnh đạt. Chọn ảnh tại đây để upload ở mục 12.

Mục 11 sao chép mô hình được chọn thành demo/casting_classifier.keras. Với CNN, đầu vào là ảnh xám 256 × 256 × 1. Với ResNet-18/MobileNetV2, ảnh qua bước xám 256 × 256, tiếp tục resize TensorFlow bilinear có antialias về 224 × 224 rồi lặp ba kênh. Metadata ghi rõ toàn bộ tiền xử lý để giao diện local triển khai đúng. Chuẩn hóa đặc thù đã được tích hợp trong graph.

Kiểm chứng dùng 10 ảnh validation, nạp mô hình trên CPU và so sánh xác suất, nhãn với pipeline đánh giá. Các điều kiện kiểm chứng cần đạt trước khi đóng gói.

Mục 12 trả nhãn và xác suất lớp đạt p_ok. Xác suất lỗi là 1-p_ok. Thử một ảnh lỗi và một ảnh đạt thuộc test. Mỗi lần chạy ghi demo_prediction.png và demo_prediction.json; tải kết quả về trước khi chạy ảnh tiếp theo nếu muốn giữ cả hai.

Gói demo_bundle.zip chứa

```text
casting_classifier.keras
metadata.json
verification.json
verification.csv
```

Nếu mô hình được chọn là ResNet-18, môi trường demo local cần KerasHub để nạp các lớp tương ứng. Có thể đối chiếu định dạng lưu và mô hình trong tài liệu chính thức. [Keras serialization](https://keras.io/guides/serialization_and_saving/), [ResNetBackbone](https://keras.io/keras_hub/api/models/resnet/resnet_backbone/), [ResNetImageConverter](https://keras.io/keras_hub/api/models/resnet/resnet_image_converter/).

## 5. Gửi lại output để hoàn thiện báo cáo

1. Casting.ipynb lưu kèm output sau khi chạy xong.
2. complete_casting_full_3models_01.zip ở mục 13.
3. Ảnh demo và JSON tương ứng nếu thử cả ảnh lỗi và ảnh đạt.

Gói complete gồm checkpoint ba mô hình, lịch sử, lựa chọn mô hình, manifest, thống kê loại trùng, bảng so sánh, hình thực nghiệm, ảnh test và gói demo. Chỉ số CSV ở miền 0–1, nhân 100 khi trình bày phần trăm.

Sau khi nhận output mới sẽ cập nhật Chương 2 về dữ liệu, loại trùng, mô hình, cấu hình và tiêu chí lựa chọn; Chương 3 về kết quả, đường học, so sánh và demo. Kết quả cũ thuộc bộ 1.300 ảnh được phân biệt với lần này.

## 6. Tiếp tục từ một bản sao kết quả

Giải nén gói sao lưu vào outputs/ để khôi phục outputs/casting_full_3models_01/. Chuẩn bị lại môi trường, upload đúng ZIP và chạy các ô dữ liệu để đối chiếu manifest. Chạy các cell định nghĩa hàm ở mục 5 rồi các bước đánh giá/demo theo nhu cầu. Ba cell gọi train_model ở mục 6–8 dành cho lượt huấn luyện mới.

Notebook đã được kiểm tra JSON và cú pháp bằng phân tích tĩnh. Các bước cài thư viện, tải trọng số, huấn luyện, đánh giá và kiểm chứng CPU do người học chạy trên Colab để tạo output chính thức.
