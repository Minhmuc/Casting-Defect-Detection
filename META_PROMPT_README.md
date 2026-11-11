# Meta-prompt viết README dự án cho Nhóm 10

Hãy viết `README.md` tại thư mục gốc dự án phân loại lỗi sản phẩm đúc của Nhóm 10, học phần Học sâu tại ICTU. Người đọc là các thành viên trong nhóm. Viết tiếng Việt từ mức nhập môn, giải thích đủ chi tiết để mỗi người có thể hiểu phương pháp, chạy lại quy trình và bảo vệ kết quả bằng lời của mình.

## Nguồn đối chiếu

- `Casting.ipynb` là notebook thực nghiệm chính. Kiểm tra trực tiếp luồng tiền xử lý, định nghĩa mô hình, cấu hình huấn luyện, điều kiện chọn checkpoint và output.
- `casting_full_3models_01/` chứa kết quả chính thức. Lấy cách chia tập từ `split_counts.csv`, chỉ số từ `comparison.csv` và `metrics_*.json`, diễn biến theo epoch từ `history_*.csv`, mô hình demo từ `selection.json` và kiểm chứng nạp lại từ `demo/verification.json`.
- `bao_cao/baocao_DL_theo_dangbaocao.docx` cung cấp bối cảnh và cách gọi đã dùng ở Chương 2–3. Dùng `CAU_HOI_VA_GIAI_THICH_BAO_VE.md` để đồng bộ cách giải thích cho người mới.
- `HUONG_DAN_CASTING.md`, `demo/README.md`, `demo/app.py` và `demo/requirements.txt` cung cấp quy trình chạy và thông tin demo local.

## Nội dung README

1. Nêu mục tiêu của đề tài, đầu vào, hai nhãn đầu ra, ý nghĩa của `p_ok`, và phạm vi đánh giá. Giải thích lựa chọn một CNN học từ đầu làm mốc và hai cấu hình học chuyển giao để so sánh.
2. Mô tả nguồn ảnh, cách phát hiện 64 ảnh trùng giữa train và test, lý do ưu tiên giữ test và cách thu được 5.583 ảnh train, 986 ảnh validation, 715 ảnh test. Giải thích vai trò của từng tập và các bước tiền xử lý cho mỗi cấu hình.
3. Giải thích CNN bằng luồng qua các lớp, kích thước đầu ra và chức năng chính của Conv2D, ReLU, MaxPooling, Global Average Pooling, Dense, Dropout và sigmoid.
4. Dùng thuật ngữ **transfer learning**, tiếng Việt là **học chuyển giao**. Định nghĩa backbone và đầu phân loại. Ghi rõ ResNet-18 dùng backbone `resnet_18_imagenet` của KerasHub, còn MobileNetV2 dùng backbone ImageNet của Keras Applications với `include_top=False`, `alpha=1.0`. Mô tả phần được giữ cố định, cách gọi với `training=False`, đầu phân loại mới và phần trọng số được cập nhật. Nêu kích thước đặc trưng và tổng số tham số để thấy sự khác nhau giữa ba cấu hình.
5. Giải thích batch, epoch, loss, accuracy, checkpoint và quá khớp ở mức nhập môn. Ghi cấu hình 30 epoch, batch 32, seed 42, Adam và ReduceLROnPlateau đúng theo notebook. Dùng ví dụ ngắn về các đỉnh validation của ba mô hình, bám những gì lịch sử đã ghi nhận.
6. Đưa bảng kết quả test, các epoch được chọn và validation loss. Giải thích số ảnh đúng, ảnh lỗi bị bỏ sót, ảnh đạt bị báo lỗi và lý do chọn MobileNetV2 epoch 21 theo validation. Nêu giới hạn khi diễn giải kết quả trên một lần chia và tập test cùng nguồn.
7. Hướng dẫn đầy đủ cách mở kết quả có sẵn, chạy notebook trên Colab, lấy gói kết quả, chuẩn bị checkpoint và chạy demo Streamlit trên Windows. Dẫn sang hướng dẫn chi tiết của repo tại các bước cần thao tác thêm. Nêu trạng thái kiểm chứng CPU và kết quả tham chiếu của ảnh demo.
8. Thêm bản đồ tệp và một đoạn trình bày ngắn để các thành viên tập bảo vệ. Dùng liên kết tương đối trong README để mở được trên máy của từng thành viên. Đặt nguồn chính thức gần nội dung cần dẫn.

## Kiểm tra bàn giao

- Đối chiếu số ảnh, epoch, checkpoint, kết quả test, kiến trúc và tiền xử lý với tệp thực nghiệm trước khi ghi README.
- Giữ phạm vi công việc ở tài liệu Markdown. Xác minh đường dẫn và cấu hình bằng cách đọc mã và output đã lưu. Thành viên tự thực hiện các lệnh chạy theo hướng dẫn khi cần tái lập kết quả.
- Bàn giao `README.md` và meta-prompt này ở thư mục gốc.
