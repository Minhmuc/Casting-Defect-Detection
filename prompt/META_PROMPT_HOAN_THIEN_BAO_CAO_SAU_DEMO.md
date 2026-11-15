# Hoàn thiện báo cáo sau khi bổ sung demo nhiều ảnh

Chỉnh trực tiếp `bao_cao/baocao_DL_theo_dangbaocao.docx`, lưu bản sao trước khi sửa. Đọc báo cáo hiện tại, yêu cầu giảng viên, mã demo và output thực nghiệm chính thức trước khi thực hiện.

## Phạm vi đã thống nhất

- Rà soát Chương 2–3, hoàn thiện mục 3.2, phần demo và phụ lục. Điều chỉnh mở đầu và kết luận nếu cần thống nhất với sản phẩm thực tế.
- Giữ nội dung Chương 1, bìa, hai bảng phân công và các kết quả huấn luyện đã được xác nhận.
- Bám mẫu ICTU tại `D:/Crop-Recommendation-ML/nhom08/Template_DA_KLTN_ICTU/` cùng hướng dẫn sử dụng. Giữ phông, lề, giãn dòng và số trang dưới chân trang của bản hiện tại.

## Căn cứ thực nghiệm

Notebook `Casting.ipynb` và bộ `casting_full_3models_01` là căn cứ chính thức. Đối chiếu CNN Conv2D/MaxPooling huấn luyện từ đầu, ResNet-18 và MobileNetV2 học chuyển giao với backbone ImageNet cố định. Không dùng bộ thử nghiệm cosine đã loại bỏ. Giữ nguyên số liệu, checkpoint MobileNetV2 epoch 21 và mô hình `.keras` phục vụ demo. Không huấn luyện lại.

## Công việc thực hiện

1. Rút phần lặp ở 3.2. Giữ kết quả đọc ảnh, loại 64 bản sao, bảng phân bố train/validation/test và nhận xét ngắn cần thiết. Chương 2 giải thích cách xử lý dữ liệu, Chương 3 trình bày kết quả thu được. Tránh viết lại toàn bộ số lượng trong bảng thành nhiều đoạn văn.
2. Rà soát tên chương, thuật ngữ, tham chiếu và các mô tả ứng dụng. Chỉnh đúng những câu đã lỗi thời so với mã hiện tại. Văn phong sinh viên học thuật, rõ ràng và tự nhiên.
3. Người dùng đã cho phép chạy demo trên máy để kiểm tra. Dùng môi trường hiện có, chạy Streamlit trên localhost và CPU. Thử một ảnh đạt, một ảnh lỗi cùng lúc, đối chiếu ảnh tham chiếu `cast_ok_0_45.jpeg`, kiểm tra CSV/JSON, thay đổi danh sách và khả năng tiếp tục khi có tệp ảnh hỏng. Ghi nhận kết quả thực tế, chụp giao diện và bổ sung vào báo cáo. Phân biệt ví dụ sử dụng giao diện với phép đánh giá trên 715 ảnh test. Giữ các giới hạn chưa đo như độ trễ dài hạn và dữ liệu ngoài nguồn.
4. Cập nhật phần demo về tải nhiều ảnh, nhãn và điểm dự đoán từng ảnh, thống kê số ảnh được dự đoán lỗi/đạt, tải CSV và JSON. Chỉ viết chức năng đã kiểm tra hoặc có căn cứ trực tiếp từ mã. Không tạo ảnh AI hay kết quả giả.
5. Phụ lục có liên kết Google Drive người dùng cung cấp, bảng giới thiệu các tệp chính và hướng dẫn chạy ngắn. Dùng đúng đường dẫn tương đối trong dự án. Liên kết mã nguồn là `https://drive.google.com/drive/folders/1otbkaT-wvJfNERf_CL0uapmz2M2ebSnR?usp=sharing`. Chỉ khẳng định nội dung Drive đã được kiểm tra khi thực sự kiểm tra được.
6. Cập nhật mục lục, danh mục hình/bảng, số hình/bảng và các tham chiếu. Tên bảng phía trên, tên hình phía dưới, đánh số theo chương. Các hình do nhóm chạy chương trình tạo ra dùng chú thích mô tả trực tiếp.

## Kiểm tra và bàn giao

Render báo cáo, xem bố cục các trang sau chỉnh sửa và đối chiếu toàn văn. Kiểm tra bảng có đủ đường kẻ, hình rõ chữ, tiêu đề đi cùng nội dung và số trang ở chân trang. Đối chiếu dữ liệu chính, kiểm tra Chương 1 được giữ nguyên nội dung. Cập nhật bản chính khi Word đã đóng và bảo đảm nguồn chưa bị sửa trong lúc xử lý.

Trả lại đường dẫn báo cáo và tệp meta-prompt, tóm tắt thay đổi, kết quả kiểm tra demo và những việc còn cần nhóm bổ sung nếu có.
