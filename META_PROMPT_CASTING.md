# Hoàn thiện thực nghiệm phân loại lỗi sản phẩm đúc

Hãy biên tập notebook học phần Học sâu của nhóm sinh viên ICTU tại `D:/Casting-Defect-Detection-CNN`. Đọc mã nguồn và dữ liệu thực tế, sau đó thực hiện các yêu cầu dưới đây.

## Phạm vi và đầu ra

- Cập nhật `Casting.ipynb` và `HUONG_DAN_CASTING.md`. Lưu bản sao notebook có output trước khi sửa.
- Người học tự cài thư viện, huấn luyện, đánh giá và demo trên Colab. Việc biên tập gồm đọc mã, kiểm tra cú pháp, đối chiếu dữ liệu và tệp kết quả.
- Các cell mới có output và execution count rỗng. Ghi rõ phần nào chưa được kiểm chứng trên Colab.
- Báo cáo được cập nhật sau khi người học cung cấp output mới. Phân biệt lần thực nghiệm trên 1.300 ảnh với lần dùng casting_data.

## 1. Dữ liệu và cách chia tập

Dùng `data/casting_data/casting_data`, gồm train/test, mỗi thư mục có def_front và ok_front. Bộ mới có 6.633 ảnh train, 715 ảnh test, kích thước 300 × 300. Hỗ trợ upload ZIP trực tiếp từ máy, tự tìm đúng cấp chứa đủ train/test và hai nhãn; hiển thị lựa chọn khi có nhiều bộ hợp lệ.

Giữ toàn bộ 715 ảnh test. Đọc test trước train, kiểm tra ảnh hỏng và SHA256 pixel ảnh xám trước resize. Loại bản sao ở train khi trùng với test hoặc ảnh train đã giữ. Nếu ảnh trùng khác nhãn hoặc trùng trong test thì dừng để kiểm tra. Lưu danh sách ảnh loại để đối chiếu.

Kiểm tra chỉ đọc trên dữ liệu hiện tại phát hiện 64 ảnh ok_front ở train trùng pixel với test. Sau loại trùng, tập train nguồn còn 6.569 ảnh. Tách 15% có phân tầng, seed42, được 5.583 ảnh train và 986 ảnh validation. Xác nhận lại số lượng bằng output khi người học chạy.

Lưu split.json, split_counts.csv, data_check.json, excluded_train_duplicates.csv và mã băm manifest. Kiểm tra giao ảnh giữa ba tập. Kiểm tra RUN_NAME/cấu hình trước khi ghi các thống kê. Dùng tên lần chạy mới `casting_full_3models_01`.

Bộ nguồn đã có augmentation. Pipeline của nhóm chỉ chuyển xám, resize và chuẩn hóa. Phân biệt augmentation sẵn có trong nguồn với augmentation bổ sung lúc huấn luyện. Nêu giới hạn SHA256 chỉ phát hiện ảnh giống hệt nhau; các biến thể gần giống do augmentation và việc thiếu mã nhóm ảnh/sản phẩm cần được ghi nhận.

Sao chép 715 ảnh test thực dùng vào `outputs/<RUN_NAME>/demo_test/def_front` và `ok_front` để chọn ảnh upload demo.

## 2. Ba mô hình

Giữ CNN baseline TensorFlow/Keras học từ đầu theo kiến trúc sau.

`Input(256,256,1) → Rescaling(1/255) → Conv2D(32,3,same,ReLU) → MaxPooling2D → Conv2D(64,3,same,ReLU) → MaxPooling2D → Conv2D(128,3,same,ReLU) → MaxPooling2D → GlobalAveragePooling2D → Dense(64,ReLU) → Dropout(0.3) → Dense(1,sigmoid)`.

Tổng tham số dự kiến 100.993, xác nhận bằng model.summary khi chạy.

ResNet-18 và MobileNetV2 dùng ImageNet, giữ cố định backbone và học đầu phân loại mới gồm Global Average Pooling, Dense64 ReLU, Dropout0,3, Dense1 sigmoid. ResNet-18 dùng KerasHub preset resnet_18_imagenet và ResNetImageConverter cùng preset. MobileNetV2 dùng Keras Applications, include_top=False, alpha=1.0, Rescaling về [-1,1] trong graph.

Pipeline đọc ảnh xám, resize PIL bilinear về 256 × 256. Với hai mô hình học chuyển giao, tiếp tục resize TensorFlow bilinear có antialias về 224 × 224 và lặp ba kênh. Chuẩn hóa đặc thù nằm trong mô hình.

Mỗi mô hình có cell summary và cell huấn luyện riêng. Cùng seed42, batch32, Adam learning rate0,001, binary cross-entropy, 30 epoch. Lưu checkpoint tốt nhất theo validation loss. Tắt JIT khi compile và sau load_model trước predict để hỗ trợ ResizeBicubic trên GPU.

## 3. Lựa chọn và đánh giá

Hoàn tất huấn luyện cả ba mô hình, xác minh checkpoint, manifest và đủ 30 epoch. Chọn mô hình demo theo validation loss thấp nhất của từng checkpoint. Lưu selection.json trước khi đánh giá test. Khi bằng nhau, dùng thứ tự CNN, ResNet-18, MobileNetV2 đã khai báo.

Đánh giá ba checkpoint trên cùng 715 ảnh test. Lưu classification report, confusion matrix, dự đoán từng ảnh, ảnh sai, learning curves và comparison.csv. Quy ước nhãn0 là lỗi, nhãn1 là đạt, sigmoid trả p_ok, ngưỡng0,5. Bỏ sót lỗi là ô [0,1] trong ma trận nhầm lẫn.

Tạo ZIP sao lưu kết quả và checkpoint sau từng mô hình. Gói complete cuối chứa cả trọng số, manifest, bảng, hình, ảnh test để demo và gói mô hình.

## 4. Xuất mô hình và demo

Dùng định dạng .keras, đóng gói checkpoint được chọn thành casting_classifier.keras. Metadata ghi mô hình, nhãn, ngưỡng, kích thước/kênh đầu vào, miền giá trị, toàn bộ tiền xử lý, mã băm và căn cứ lựa chọn.

Đối chiếu 10 ảnh validation giữa pipeline đánh giá và đường đọc ảnh demo sau khi nạp lại trên CPU. Kiểm tra xác suất hữu hạn, sai số trong dung sai và nhãn trùng. Demo phải tái tạo đầy đủ tiền xử lý theo mô hình được chọn.

Có cell upload một ảnh để hiển thị nhãn, xác suất, lưu PNG và JSON. Người học thử một ảnh lỗi và một ảnh đạt thuộc test. demo_bundle.zip gồm mô hình, metadata và kết quả kiểm chứng. Giao diện local được xây dựng sau khi xác nhận output.

## 5. Kiểm tra và bàn giao

Markdown/comment tiếng Việt ngắn gọn, theo cách sinh viên trình bày thực nghiệm. Mỗi cell tập trung một việc. Bảo toàn tệp người học tạo, kiểm tra trạng thái trước khi ghi đè.

Kiểm tra JSON, cú pháp, quan hệ giữa các cell, thứ tự lựa chọn trước đánh giá test, tiền xử lý và tên tệp đóng gói bằng phân tích tĩnh. Trả lại notebook, hướng dẫn từng bước, vị trí bản sao và danh sách output người học cần gửi. Khi viết báo cáo, giải thích khác biệt giữa học từ đầu và học chuyển giao trong phép so sánh.
