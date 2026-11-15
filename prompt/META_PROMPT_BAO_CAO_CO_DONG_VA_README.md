# Hoàn thiện bản báo cáo cô đọng và README của Nhóm 10

## Yêu cầu đã xác nhận

Tạo `bao_cao/baocaocodong.docx` từ `bao_cao/baocao_DL_theo_dangbaocao.docx`. Bản nguồn là căn cứ đối chiếu. Cập nhật README tại gốc thành tài liệu nhập môn đầy đủ để các thành viên hiểu đề tài, phương pháp, kết quả và cách bảo vệ. Dùng bộ kết quả `casting_full_3models_01`.

Giữ định dạng ICTU, bìa, lời cảm ơn, hai bảng phân công, nội dung Chương 1, các số liệu thực nghiệm và kiến trúc đã huấn luyện. Thực hiện biên tập bằng dữ liệu hiện có, tập trung Chương 2–3. Lưu bản sao nguồn và README trước khi chỉnh.

## Công việc trong báo cáo

1. Thống nhất ký hiệu tại 2.1 cho ba mô hình. Định nghĩa chỉ số mô hình, đầu vào chung, phép biến đổi riêng, tham số được học và điểm sigmoid. Dùng công thức Word có thể chỉnh sửa. Nêu cách chọn checkpoint bằng validation loss.
2. Gộp 2.3.5 vào 2.2.4, đặt bảng chuẩn hóa đầu vào tại phần tiền xử lý. Giữ giải thích converter đặc thù trong phần ResNet-18. Cập nhật số bảng và các tham chiếu.
3. Giải thích ngắn ReLU và sigmoid ngay tại CNN sử dụng. Bổ sung hai hình kỹ thuật về basic residual block và inverted residual block, đối chiếu cấu hình đã lưu cùng mã nguồn KerasHub 0.23.0/Keras 3.12.0. Hình thể hiện đường nối tắt, phép cộng và kích thước. Ví dụ phải thuộc mạng thực tế, được ghi rõ là minh họa cấu tạo khối. Ghi căn cứ tài liệu [8], [11], [12], [13].
4. Thêm danh mục viết tắt phù hợp các thuật ngữ xuất hiện. Giữ kiến thức Chương 1 và bố cục của chương.
5. Rút nội dung lặp tại 3.1.3, 3.4 và phần phân tích ảnh sai. Giữ số đếm, ý nghĩa các loại sai, mức kết luận được bằng chứng hỗ trợ và giải thích thận trọng về các đỉnh validation.
6. Giữ hình so sánh ba mô hình trên ảnh `cast_ok_0_45.jpeg`, thay phần minh họa riêng MobileNetV2 trên cùng ảnh. Cập nhật số hình sau khi bỏ hình cũ và sửa mọi tham chiếu.
7. Mô tả ngắn cách nạp converter ResNet-18 trên Windows theo `demo/inference.py` và kết quả kiểm chứng CPU hiện có. Chuẩn hóa điểm p_ok trong bảng ảnh sai thành sáu chữ số thập phân, sử dụng giá trị gốc trong CSV.
8. Chuẩn hóa cách gọi độ chính xác, hàm mất mát, tập xác thực và thời gian huấn luyện. Dẫn liên kết trực tiếp tới nguồn hình scikit-learn trong tài liệu tham khảo. Giữ số lượng 15 tài liệu. Thông tin bài giảng lấy theo tài liệu có thể kiểm tra.
9. Giữ phụ lục mã nguồn, môi trường, cách chạy và kết quả kiểm tra demo. Đối chiếu liên kết Google Drive trong phạm vi có thể truy cập, mô tả đúng mức xác minh.

## Công việc trong README gốc

Trình bày từ mức nhập môn, gồm định hướng, dữ liệu và giới hạn kiểm tra trùng, CNN baseline, hai backbone ImageNet cố định, đầu phân loại được học, kiến trúc khối, tham số, chuẩn hóa, cấu hình huấn luyện, chọn checkpoint, cách đọc loss/accuracy và spike, quá khớp, ma trận nhầm lẫn, kết quả chính thức, quy tắc ngưỡng, cách chạy Colab và demo đủ ba mô hình. Phân biệt mô hình đề xuất với các mô hình được hỗ trợ trong demo bằng mô tả rõ vai trò từng phần. Giới thiệu bản `baocaocodong.docx` cùng báo cáo nguồn để nhóm đối chiếu. Tổng hợp các câu hỏi bảo vệ và cách giải thích ngắn, đúng căn cứ.

## Kiểm chứng đầu ra

- Đối chiếu bảng, số liệu, lịch sử, mã băm checkpoint và cấu hình trước/sau biên tập.
- Render bản DOCX, cập nhật mục lục và danh mục, kiểm tra hình, công thức, bảng, ngắt trang và số trang. Chương 1 giữ 22 trang như nguồn, toàn báo cáo trong khoảng 60–70 trang.
- Kiểm tra các liên kết nội bộ trong README và hướng dẫn chạy với tệp thực tế.
- Xác nhận mã băm báo cáo nguồn, notebook và các tệp kết quả được bảo toàn.
- Trả lại bản DOCX mới, README, meta-prompt, số trang thực tế và các nội dung còn cần nhóm xác nhận.
