# Hoàn thiện báo cáo Học sâu Nhóm 10

Đọc báo cáo D:/Casting-Defect-Detection-CNN/bao_cao/baocao_DL_theo_dangbaocao.docx, notebook Casting.ipynb và toàn bộ output casting_full_3models_01. Đối chiếu yeucau.docx và mẫu DA/KLTN ICTU tại D:/Crop-Recommendation-ML/nhom08/Template_DA_KLTN_ICTU cùng hướng dẫn sử dụng mẫu. Thực hiện các chỉnh sửa đã đủ căn cứ, lưu bản sao trước khi cập nhật báo cáo chính.

## Những lựa chọn người học đã xác nhận

- Cập nhật báo cáo chính, có bản sao dự phòng.
- Giữ phần lớn Chương 1. Tập trung Chương 2–3, mở đầu, kết luận và hình thức trình bày.
- CNN Conv2D/MaxPooling học từ đầu làm baseline. ResNet-18 và MobileNetV2 giữ cố định backbone ImageNet, học đầu phân loại mới.
- Cả ba thực nghiệm đủ 30 epoch. Bộ nguồn 7.348 ảnh 300 × 300 đã có augmentation; pipeline của nhóm không bổ sung augmentation.
- Giữ 715 ảnh test, loại 64 bản sao ở train. Sau làm sạch, train nguồn còn 6.569 ảnh; chia train thực dùng 5.583 và validation 986, seed42.
- Chọn mô hình demo theo validation loss tốt nhất trước bước test. Lần này MobileNetV2 được chọn tại epoch21.
- Minh họa demo trong báo cáo bằng ảnh đạt cast_ok_0_45.jpeg từ output notebook. Trích xuất đúng ảnh PNG trong notebook và kết quả văn bản, phân biệt với ảnh lỗi đang được lưu trong complete ZIP.
- Xây giao diện upload ảnh local đơn giản, dùng mô hình .keras đã xuất, kèm hướng dẫn để người học tự cài và chạy.

## Chỉnh sửa nội dung

1. Sửa mở đầu và phạm vi theo ba mô hình hiện tại, bỏ mục tiêu thực nghiệm có/không augmentation.
2. Chương 2 trình bày bài toán bằng lời và toán học, nguồn dữ liệu, kiểm tra/loại trùng, chia tập, kiến trúc thật và cấu hình học. CNN gồm Conv32/MaxPool, Conv64/MaxPool, Conv128/MaxPool, Global Average Pooling, Dense64 ReLU, Dropout0,3, Dense1 sigmoid; tổng 100.993 tham số. Giải thích các lớp trong ngữ cảnh mô hình. Bảng kiến trúc gọn, tổng tham số nêu trong nội dung.
3. Chương 3 thay toàn bộ số liệu, hình và nhận xét cũ bằng output mới. Dùng CSV/JSON, history và dự đoán từng ảnh để xác minh. Phân biệt checkpoint tốt nhất với epoch cuối. Phân tích đường học, các lỗi bỏ sót/báo nhầm, so sánh ba cấu hình và giới hạn.
4. Gộp các nội dung so sánh, phân tích và demo thành cấu trúc mạch lạc; bỏ mục thực nghiệm augmentation cũ. Tách đánh giá phân loại với kiểm chứng nạp lại mô hình.
5. Trình bày kết quả CPU có thật trên 10 ảnh validation. Giải thích định dạng .keras bằng thuật ngữ dễ hiểu. Nội dung ứng dụng local mô tả cách xây dựng; kết quả giao diện chỉ đưa vào sau khi người học tự chạy và xác nhận.
6. Hoàn thiện kết luận chung bằng các kết quả đã kiểm chứng. Nêu giới hạn một seed, nguồn dữ liệu đã tăng cường, kiểm tra trùng chính xác chưa phát hiện mọi biến thể gần giống, khác biệt học từ đầu/học chuyển giao và thiếu đánh giá trên nguồn ảnh độc lập.

## Hình thức và tính trung thực

Giữ mẫu ICTU, Times New Roman13, lề trái3/phải2/trên2/dưới2cm theo mẫu hiện có; số trang cuối trang theo yêu cầu người học. Giữ hai bảng phân công, sửa tên nhiệm vụ theo cách triển khai thực tế. Mục tiêu khoảng60–70 trang bằng nội dung có giá trị. Chương 1 giữ độ dài ít nhất18 trang nếu bố cục cho phép, không kéo giãn hoặc thêm trang trắng.

Hình/bảng đánh số theo chương, tên bảng phía trên và tên hình phía dưới. Dùng hình thực từ output; không tạo ảnh AI. Bảng kẻ đủ đường, lặp hàng tiêu đề khi cần; kiểm tra hình rõ chữ, công thức Word hiển thị đúng. Nguồn bên ngoài phải có căn cứ; dữ liệu/hình do nhóm tạo được giải thích trong nội dung, không thêm dòng nguồn nội bộ thừa. Giữ khoảng15 tài liệu tham khảo, cập nhật mục có nội dung cũ và kiểm tra trích dẫn.

Cập nhật mục lục, danh mục hình/bảng và tham chiếu. Render toàn bộ và kiểm tra từng trang trước khi ghi đè bản chính. Bảo toàn nội dung Chương 1 và tệp mẫu gốc. Báo lại các điểm còn cần người học tự chạy hoặc xác nhận.

## Giới hạn thao tác

Không chạy notebook, huấn luyện, suy luận, cài TensorFlow hoặc tự mở ứng dụng demo. Chỉ kiểm tra tĩnh mã, đối chiếu tệp dữ liệu và kết quả đã cung cấp. Được chạy công cụ soạn, render và kiểm tra tài liệu. Không tạo số liệu hoặc ảnh chụp giao diện giả.

## Bàn giao

Báo cáo DOCX chính đã cập nhật và bản sao trước sửa; meta-prompt này; mã giao diện local, gói mô hình/metadata và hướng dẫn tự chạy. Tóm tắt số trang thực tế, thay đổi chính và công việc cần người học thực hiện.
