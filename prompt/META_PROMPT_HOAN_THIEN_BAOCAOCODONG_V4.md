# Meta-prompt: Rà soát và hoàn thiện baocaocodong_v3.docx thành baocaocodong_v4.docx

## Vai trò
Biên tập viên kỹ thuật cho báo cáo học phần Học sâu (ICTU, Nhóm 10, đề tài "Kiểm tra lỗi sản phẩm đúc bằng học sâu") ở giai đoạn về đích.
Đọc `AGENTS.md`, `README.md`, `prompt/PROMPT_BAN_GIAO_DU_AN_DL.md` trước khi sửa. Căn cứ số liệu: `casting_full_3models_01/`.

## Tệp
- Nguồn: `bao_cao/baocaocodong_v3.docx` (giữ nguyên, không ghi đè).
- Đích: `bao_cao/baocaocodong_v4.docx` + `bao_cao/baocaocodong_v4.pdf`.
- Tư liệu làm việc (bản trích văn bản, ảnh render, script) đặt trong `report_work/hoan_thien_v4_<ngày>/`.

## Phạm vi
- Chương 1 đã tạm ổn: chỉ sửa khi lỗi định dạng chung (khoảng trống, tên bảng) chạm tới.
- Trọng tâm: Chương 2, 3, bảng phân công, tài liệu tham khảo.
- Được phép viết lại, rút gọn đoạn yếu hoặc lặp ở Chương 2, 3; số trang có thể thay đổi.
- Giữ nguyên mọi số liệu thực nghiệm và mức kết luận; không thêm kết quả chưa có output.

## Việc bắt buộc (từ ghi chú của nhóm)
1. **Tên Chương 3:** đổi thành "CHƯƠNG 3. THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ"; cập nhật mục lục, Mở đầu (bố cục) và mọi câu nhắc tên chương.
2. **Nói rõ cấu trúc CNN (2.3.2):**
   - Bảng 2.4 thêm cột cấu hình lớp (kernel, stride, padding) và số tham số từng lớp: Conv1 320, Conv2 18.496, Conv3 73.856, Dense 64 8.256, Dense 1 65, tổng 100.993 (đối chiếu `model_summary_cnn.txt`).
   - Thêm đoạn đi qua luồng ảnh từng khối và công thức `(k × k × C_in + 1) × C_out`.
   - **Vẽ lại Hình 2.2 bằng draw.io** theo đúng cấu hình đã lưu. Lưu tệp nguồn `.drawio` cùng PNG xuất ra. Không dùng ảnh do AI tạo sinh.
3. **Hạn chế/xóa khoảng trống:** bỏ đoạn rỗng thừa (giữ đoạn chứa công thức OMML), xử lý khoảng trắng lớn cuối trang do ảnh/bảng bị đẩy sang trang sau (đặt lại vị trí, keep-with-next hợp lý, bỏ ngắt trang thừa). Kiểm tra bằng ảnh render từng trang.
4. **Căn đều tài liệu tham khảo:** đoạn trong mục TÀI LIỆU THAM KHẢO căn đều hai bên (justify), thụt treo cho số thứ tự nếu mẫu ICTU yêu cầu; URL dài không làm giãn chữ xấu.
5. **Bảng phân công:** mô tả cụ thể từng nhiệm vụ thêm 1–2 dòng, ví dụ "Huấn luyện CNN baseline, ResNet-18, MobileNetV2 trên Colab; chọn checkpoint theo validation loss". Giữ đúng mẫu 2 bảng; sửa lỗi dính tên "Nguyễn Quang MinhVũ Ngọc Quý" (xuống dòng hoặc dấu phẩy).

## Rà soát và đánh giá Chương 2, 3
- Câu lặp ý giữa các mục (ví dụ "dùng chung dữ liệu" nhắc nhiều lần, 3.3.1 nhắc lại 30 epoch); rút gọn.
- Câu tự mâu thuẫn hoặc diễn giải quá mức kết luận (ví dụ 3.3.2 "chưa có dấu hiệu quá khớp kéo dài"; 3.4.3 "ưu thế ... nói chung").
- Đoạn tham chiếu hình/bảng phải đứng gần hình/bảng; mỗi hình/bảng được nhắc trong văn bản.
- Thuật ngữ thống nhất: train, validation, test, baseline, checkpoint, loss, p_ok, "mã băm tệp checkpoint".
- Tên bảng/hình theo format "Bảng x.y: ..." / "Hình x.y: ...", đánh số liên tục, danh mục khớp.
- Công thức có ký hiệu được giải thích ngay sau công thức.
- Lập bảng đánh giá: mục, vấn đề, mức độ (phải sửa / nên sửa / ghi nhận), đã xử lý thế nào.

## Kiểm tra trước khi giao
- Mở bằng Word, cập nhật toàn bộ field (mục lục, danh mục hình, bảng), xuất PDF, render ảnh từng trang Chương 2, 3, bảng phân công, tài liệu tham khảo và xem lại.
- Đối chiếu số liệu: 7.348; 64; 5.583/986/715; 453/262; 94,83/98,74/99,44; 428/25/12/250; 446/7/2/260; 451/2/2/260; epoch 29/29/21; 100.993/11.219.009/2.340.033; 32.897/82.049.
- So sánh văn bản v3 và v4: liệt kê đoạn bị xóa hoặc viết lại.
- Định dạng giữ mẫu ICTU: A4, Times New Roman 13, lề trái 3 cm, phải/trên/dưới 2 cm, giãn dòng 1,5, số trang ở chân trang.
- Cập nhật README (trỏ sang v4 sau khi người dùng duyệt), ghi log vào TrueBrain `10-projects/casting-defect/log.md`.

## Báo cáo kết quả
Ngắn gọn, tiếng Việt: các thay đổi theo 5 mục bắt buộc, bảng đánh giá rà soát, số trang v3 → v4, những gì chưa xử lý và lý do.
