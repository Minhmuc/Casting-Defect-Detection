# Chốt nội dung và bố cục báo cáo Học sâu Nhóm 10

Chỉnh trực tiếp `bao_cao/baocao_DL_theo_dangbaocao.docx`, lưu bản dự phòng trước khi chỉnh. Đọc bản hiện tại và yêu cầu giảng viên trong `yeucau.docx`, đối chiếu mã nguồn và kết quả chính thức.

## Phạm vi đã được người dùng xác nhận

- Đổi tên đề tài thành **Kiểm tra lỗi sản phẩm đúc bằng học sâu**, thống nhất trên bìa và các chỗ nhắc tên đề tài.
- Bổ sung một hình kiến trúc CNN từ đúng mã nguồn, có kích thước đầu ra. Giữ bảng kiến trúc hiện tại.
- Sửa bốn điểm trong lần rà soát cuối, gồm cách gọi mô hình đề xuất và mô hình demo, mục kiểm chứng CPU, lỗi tên thư mục kết quả và phần Kết luận.
- Rút các câu nhắc giới hạn bị lặp ở Chương 2–3. Giữ các giới hạn có ý nghĩa và diễn giải thận trọng cần thiết khi nhận xét thực nghiệm.

## Những điều cần giữ đúng

Báo cáo theo mẫu DA/KLTN ICTU tại `D:/Crop-Recommendation-ML/nhom08/Template_DA_KLTN_ICTU/` cùng hướng dẫn sử dụng mẫu. Dùng bản hiện tại làm căn cứ form, phông, lề, giãn dòng, bìa và số trang dưới chân trang. Giữ hai bảng phân công, nội dung Chương 1 và 15 tài liệu tham khảo.

Căn cứ kỹ thuật là `Casting.ipynb`, `casting_full_3models_01/` và `demo/models/comparison_verification.json`. CNN baseline có Rescaling, ba khối Conv2D/ReLU với 32, 64, 128 bộ lọc kèm MaxPooling2D, Global Average Pooling, Dense 64/ReLU, Dropout 0,3 và Dense 1/sigmoid. ResNet-18 và MobileNetV2 dùng backbone ImageNet cố định. Kích thước đầu vào tương ứng 256 × 256 × 1 và 224 × 224 × 3.

Giữ các số liệu huấn luyện, chia dữ liệu, epoch, accuracy, loss và ma trận nhầm lẫn. Công việc lần này là biên tập báo cáo từ kết quả đã có. Các tệp notebook, mã demo và checkpoint thuộc phạm vi tham chiếu.

## Chỉnh sửa cụ thể

1. Thống nhất MobileNetV2 là **mô hình đề xuất theo validation loss**. Ứng dụng demo hỗ trợ **cả ba mô hình** để đối chiếu trên cùng ảnh. Rà soát mở đầu, mục 2.1, 2.2.3, 2.4.4, 3.1.3, 3.3.1, 3.4.3 và Kết luận cùng các bảng liên quan để cách gọi nhất quán. Giữ nguyên quy tắc lựa chọn trước đánh giá test.
2. Mở rộng mục 3.5.2 và Bảng 3.7 cho cả ba checkpoint. Phân biệt rõ phép kiểm chứng MobileNetV2 trước đây trên 10 ảnh validation với phép đối chiếu mới trên 10 ảnh test cho mỗi mô hình. Đọc số đếm, dung sai và sai lệch lớn nhất từ các JSON thực tế. Phân biệt nhãn khớp đầu ra đã lưu với độ chính xác so với nhãn thật.
3. Sửa `casting_full_3 models_01` thành `casting_full_3models_01` trong báo cáo.
4. Rút phần Kết luận còn một trang thực tế. Giữ kết quả chính, mô hình đề xuất, sản phẩm demo, hạn chế và hướng phát triển. Gộp các ý lặp và sửa câu khó đọc.
5. Rút những câu nhắc lại về một seed, giới hạn dữ liệu, tính tối ưu hoặc độ trễ ở Chương 2–3 khi chúng đã được trình bày đầy đủ tại mục 3.5.4. Giữ phần giải thích các đỉnh validation, nhận xét quá khớp và giới hạn suy luận nguyên nhân khi thiếu bằng chứng.
6. Tạo Hình 2.2 bằng công cụ vẽ kỹ thuật từ `architecture_cnn.json`, đối chiếu với notebook. Hình phải thể hiện các tầng mạng, các khối Conv/Pool, số bộ lọc và kích thước tensor, đồng thời đủ rõ để in trong Word. Dùng sơ đồ kiến trúc mạng có căn cứ từ mã nguồn. Giới thiệu và giải thích hình trong nội dung, ghi nguồn nhóm tự xây dựng từ kiến trúc trong mã. Hình là minh họa kiến trúc, được vẽ bằng công cụ kỹ thuật thông thường.
7. Cập nhật mục lục, danh mục hình và bảng cùng các tham chiếu liên quan. Giữ cách đánh số phụ lục A.1 và C.1 của bản hiện tại.

## Kiểm tra và bàn giao

Render báo cáo và kiểm tra bìa với tên mới, hình CNN, bảng kiểm chứng, Kết luận một trang và phụ lục. Kiểm tra chữ, công thức, đường kẻ bảng và ngắt trang. Giữ Chương 1 ít nhất 18 trang thực tế, tổng báo cáo khoảng 60–70 trang bằng bố cục tự nhiên. Đối chiếu nội dung Chương 1 và các bảng kết quả cũ trước/sau. Kiểm tra nguồn chưa bị sửa trong lúc xử lý trước khi cập nhật bản chính.

Trả lại báo cáo đã cập nhật và tệp meta-prompt này, kèm tóm tắt ngắn các thay đổi và số trang thực tế. Nếu còn thông tin cần nhóm xác nhận thì liệt kê ở câu trả lời.
