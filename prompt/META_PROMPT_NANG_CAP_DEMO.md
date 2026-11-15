# Nâng cấp demo phân loại sản phẩm đúc cho Nhóm 10

Chỉnh trực tiếp giao diện Streamlit để thành viên ICTU có thể tải một hoặc nhiều ảnh, phân loại và trình bày kết quả khi bảo vệ môn Học sâu. Người dùng chọn giao diện nền sáng, xanh đậm, dễ đọc khi trình chiếu và xuất kết quả bằng CSV cùng JSON.

## Căn cứ

Đọc `demo/app.py`, `demo/requirements.txt`, `demo/README.md`, metadata và checkpoint trong `demo/models/`. Kết quả chính thức là `casting_full_3models_01`, MobileNetV2 epoch 21. Đối chiếu mô hình local với `selection.json` và metadata. Tham khảo API Streamlit tương thích phiên bản tối thiểu 1.50 khi dùng thêm tính năng.

## Công việc

- Tạo bố cục nền sáng, xanh đậm với vùng tải ảnh, nút “Phân loại tất cả”, thống kê tổng ảnh và nhãn dự đoán, lưới ảnh kết quả. Hiển thị rõ chữ “Sản phẩm lỗi” hoặc “Sản phẩm đạt yêu cầu” bên cạnh màu sắc.
- Cho phép xử lý một hoặc nhiều ảnh JPG, JPEG, PNG, BMP. Một ảnh hỏng cần có thông báo riêng để các ảnh hợp lệ còn lại vẫn được xử lý.
- Dùng checkpoint MobileNetV2 đã chọn, kiểm tra mã băm và tiền xử lý theo metadata. Áp dụng ngưỡng p_ok 0,5. Các thao tác suy luận dùng CPU như mã hiện có.
- Lưu kết quả trong phiên giao diện để tải CSV và JSON sau phân loại. Khi danh sách ảnh, nội dung ảnh hoặc checkpoint thay đổi, giao diện cần hiển thị kết quả tương ứng bộ ảnh hiện tại. Phân biệt được các ảnh trùng tên qua số thứ tự.
- Xuất tên ảnh, số thứ tự, nhãn, p_ok, điểm lỗi, tên mô hình và mã băm checkpoint. Ghi trạng thái và lỗi của các tệp gặp sự cố. Thống kê số sản phẩm lỗi/đạt dựa trên các ảnh xử lý thành công.
- Cập nhật hướng dẫn chạy demo và các bước thành viên tự kiểm tra trước khi chụp giao diện đưa vào báo cáo. Đồng bộ phần mô tả demo trong README tổng hợp khi cần.

## Phạm vi xác nhận của người dùng

Người dùng chọn sửa mã, kiểm tra tĩnh và tự chạy ứng dụng trên máy. Công việc kiểm tra gồm cú pháp Python, cấu hình, đường dẫn, tính nhất quán checkpoint và luồng trạng thái/xuất tệp qua đọc mã. Bàn giao hướng dẫn kiểm tra thực tế cho một ảnh, nhiều ảnh, ảnh hỏng, ảnh trùng tên, tải cả hai định dạng và thay đổi danh sách ảnh.

Đầu ra gồm mã demo cập nhật, cấu hình giao diện cần thiết, hướng dẫn sử dụng và meta-prompt này. Tóm tắt các thay đổi và nói rõ phạm vi kiểm tra khi bàn giao.
