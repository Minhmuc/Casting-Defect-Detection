# Meta-prompt: hoàn thiện báo cáo để duyệt in quyển (2026-11-25)

Đầu vào: `bao_cao/baocaocodong_v4.docx`. Đây là bản người dùng sửa tay lúc 11:23 ngày 25/11, đã đổi tên đề tài thành "KIỂM TRA LỖI SẢN PHẨM ĐÚC KIM LOẠI" và đổi tên Chương 3 thành "Kết quả thực nghiệm bài toán". Bản chép làm việc: `report_work/hoan_thien_in_20261125/v4_base.docx`. Báo cáo hiện có 47 trang đánh số Ả Rập.

## Yêu cầu đã chốt
- Báo cáo đạt **58–59 trang đánh số Ả Rập**, không tính bìa và các trang đánh số La Mã.
- **Chương 1 giữ nguyên.** Chỉ kéo dài **Chương 2 và Chương 3** bằng nội dung thật: phân tích kỹ hơn, thêm hình, thêm bảng. Không giãn cỡ chữ hay khoảng cách. Không thêm mã nguồn vào Phụ lục.
- **Tên hình và tên bảng đều in nghiêng.**
- **Tài liệu tham khảo:** giữ 15 nguồn hiện có, chuẩn hóa theo một kiểu trích dẫn. Bài nào hiện trích bản arXiv thì đổi sang bản đã xuất bản nếu kiểm được (tìm ở đâu cũng được, miễn trích đúng). Mọi URL và DOI phải là hyperlink bấm được.
- **Không** thêm câu "CNN theo VGG" vào báo cáo.
- Hình vẽ bằng script hoặc draw.io, không dùng ảnh AI tạo. Số liệu chỉ lấy từ output thật.

## Nội dung thêm
- **Chương 3:**
  - (A) hình phân bố pₒₖ của ảnh dự đoán đúng và sai;
  - (B) bảng FN/FP khi đổi ngưỡng, ghi rõ là phân tích bổ sung trên test, không dùng để chọn ngưỡng;
  - (C) bảng ảnh sai chéo giữa các mô hình;
  - (D) bảng chi phí: tham số, thời gian huấn luyện, dung lượng tệp mô hình; không đưa tốc độ suy luận vì chưa đo;
  - (E) hình validation loss của ba mô hình trên cùng trục;
  - (F) bảng chỉ số theo từng lớp.

  Mỗi hình/bảng có 1–2 đoạn phân tích. Có thể phân tích sâu thêm diễn biến huấn luyện, ảnh dự đoán sai và demo.
- **Chương 2:**
  - sơ đồ quy trình xử lý dữ liệu và sơ đồ quy trình thực nghiệm tổng thể (vẽ bằng draw.io);
  - bảng số ảnh qua từng bước xử lý;
  - phân tích thêm về dữ liệu (64 ảnh trùng, mức lệch lớp), là phần giảng viên hỏi nhiều.

## Chia việc
Bốn subagent chạy song song: hình Ch.3, sơ đồ Ch.2, viết nội dung, tài liệu tham khảo. Sau đó một agent ghép vào docx (chỉ một người ghi), render bằng Word, rồi một agent QA độc lập kiểm lại. Cuối cùng phiên chính kiểm, chép vào `bao_cao/` và cập nhật README.
