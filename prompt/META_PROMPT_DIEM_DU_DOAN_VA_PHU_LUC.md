# Cập nhật điểm dự đoán và hoàn thiện phụ lục báo cáo

Thực hiện trên dự án `D:/Casting-Defect-Detection-CNN`. Chỉnh báo cáo chính `bao_cao/baocao_DL_theo_dangbaocao.docx` và lưu bản dự phòng trước khi sửa. Dùng báo cáo hiện tại làm căn cứ nội dung, định dạng theo mẫu ICTU đã cung cấp.

## Các lựa chọn đã xác nhận

- Đổi demo và phần mô tả, ảnh giao diện trong báo cáo sang điểm dự đoán dạng thập phân có 6 chữ số sau dấu phẩy.
- Mở rộng phụ lục khoảng 2 trang, gồm mã nguồn, môi trường và hướng dẫn chạy, bảng các trường hợp đã kiểm tra demo.
- Gom các meta-prompt và prompt Markdown vào thư mục `prompt`. Giữ `yeucau.md` và `detai.md` ở vị trí hiện có nếu có. Từ các lần sau, lưu prompt mới trực tiếp trong `prompt`.

## Cách thực hiện

1. Đọc mã demo, metadata mô hình, số liệu và ảnh giao diện hiện có. Điểm đạt là `p_ok`, điểm lỗi bằng `1 - p_ok`. Ngưỡng quyết định là 0,5. Giữ nguyên trọng số, tiền xử lý và quy tắc dự đoán.
2. Hiển thị hai điểm bằng định dạng 6 chữ số thập phân. Dùng nhãn tiếng Việt dễ hiểu và một câu ngắn giải thích miền 0–1. CSV và JSON tiếp tục chứa giá trị số đầy đủ để đối chiếu.
3. Chạy demo trên CPU trong môi trường đã có. Thử ảnh đạt `cast_ok_0_45.jpeg` và ảnh lỗi `cast_def_0_1059.jpeg`, tải CSV/JSON để đối chiếu. Kiểm tra ảnh hỏng và việc xóa kết quả cũ khi danh sách ảnh thay đổi. Dùng ảnh chụp giao diện thực tế để cập nhật Hình 3.11. Các output notebook đã có được giữ đúng giá trị đã ghi nhận.
4. Trong báo cáo, giải thích ngắn ý nghĩa điểm dự đoán và cách dùng ngưỡng. Phân biệt điểm từng ảnh với accuracy trên tập test. Các chỉ số accuracy, precision, recall, F1 tiếp tục dùng đơn vị đang thống nhất trong báo cáo.
5. Phụ lục giữ liên kết Google Drive cùng bảng tệp chính. Bổ sung môi trường thực sự đã kiểm tra, cách đặt mô hình, lệnh chạy và cách sử dụng. Thêm bảng ngắn về các trường hợp đã thử và kết quả quan sát. Dẫn tới Hình 3.11 cho giao diện, tránh chèn lại hình giống nhau. Nội dung cần đủ để người chấm tải mã và chạy thử.
6. Giữ nội dung Chương 1 và dữ liệu thực nghiệm, không huấn luyện lại. Thông số môi trường được đọc từ môi trường đang dùng. Dữ liệu và kết quả phải có căn cứ từ tệp hoặc phép thử thực tế.
7. Gom các tệp prompt bằng thao tác di chuyển có kiểm tra đường dẫn và trùng tên. Tạo quy ước lưu prompt trong AGENTS.md. Rà soát các liên kết đang sử dụng và cập nhật liên kết bị ảnh hưởng, bảo toàn nội dung lịch sử của các prompt cũ.
8. Đồng bộ hướng dẫn demo và các mô tả liên quan trong README với định dạng điểm mới.

## Kiểm tra đầu ra

Cập nhật mục lục, danh mục hình và bảng. Render báo cáo để kiểm tra số trang, ngắt trang, hình và đường kẻ bảng. Phụ lục khoảng 2 trang, tổng báo cáo trong khoảng 60–70 trang nếu bố cục thực tế cho phép. Không giãn định dạng để đạt số trang. Đối chiếu nội dung Chương 1, số liệu trong các bảng cũ và mã băm mô hình trước/sau.

Trả lại liên kết báo cáo đã cập nhật, tệp meta-prompt trong `prompt`, tóm tắt kết quả chạy thử và việc còn cần nhóm bổ sung nếu có.
