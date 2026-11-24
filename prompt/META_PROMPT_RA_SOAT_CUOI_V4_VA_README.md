# Meta-prompt: rà soát cuối báo cáo v4, văn phong AI và README bảo vệ

Ngày: 2026-11-24. Báo cáo: `bao_cao/baocaocodong_v4.docx` (58 trang). Thư mục làm việc: `report_work/ra_soat_cuoi_20261124/`.

## Yêu cầu của người dùng (đã làm rõ)
1. Rà soát báo cáo lần cuối. Rà xong thì **báo danh sách trong chat trước**, người dùng chọn mục, sau đó mới sửa.
2. Đánh giá văn phong AI (những gì AI hay viết), **chỉ ở Chương 2–3 và Kết luận**.
3. Viết lại `README.md` thành tài liệu **tự đủ**: thành viên nào đọc xong cũng bảo vệ được dự án. Cập nhật `HUONG_DAN_BAO_VE.md` và `CAU_HOI_VA_GIAI_THICH_BAO_VE.md` theo v4, rồi liên kết từ README.

## Chia việc (quy ước: luôn dùng subagent, không để hai agent ghi cùng một tệp)
- **Agent R (rà soát cuối, chỉ đọc):** đọc toàn bộ báo cáo. Kiểm tra tính nhất quán giữa các chương, số liệu Chương 3 và Kết luận so với `casting_full_3models_01/`, yêu cầu học phần (`yeucau.docx`), lỗi chữ và trình bày còn sót. Kết quả ghi vào `ra_soat_cuoi.md`.
- **Agent A (văn phong AI, chỉ đọc):** quét Chương 2, Chương 3 và Kết luận để tìm dấu hiệu văn AI. Chấm mức độ từng mục, liệt kê câu cụ thể kèm câu thay. Kết quả ghi vào `van_phong_ai.md`.
- **Agent D (tài liệu):** viết lại `README.md`, cập nhật hai tệp bảo vệ. Lấy số liệu từ output thật và báo cáo v4; dẫn theo số mục, không dẫn số trang. Không sửa docx.
- **Phiên chính:** tổng hợp, kiểm lại một số phát hiện, báo người dùng, sửa theo lựa chọn bằng script tái lập, render bằng Word (viết COM trực tiếp trong Start-Job), cho QA kiểm lại.

## Dấu hiệu văn AI cần soi
- Câu mở hoặc kết sáo rỗng ("Trong bối cảnh…", "Có thể thấy rằng…", "Tóm lại…", "đóng vai trò quan trọng", "không chỉ… mà còn…").
- Liệt kê ba ý theo quán tính; tính từ tâng bốc ("vượt trội", "đáng kể", "toàn diện", "mạnh mẽ") không có số liệu đi kèm.
- Lặp ý giữa các đoạn hoặc giữa chương với kết luận; câu dẫn nói lại điều bảng vừa trình bày.
- Từ nối dày đặc ("Bên cạnh đó", "Ngoài ra", "Đồng thời", "Hơn nữa"); câu cân đối quá đều; dấu hai chấm hoặc dấu chấm phẩy dùng thay động từ.
- Thuật ngữ dịch sát nghĩa tiếng Anh ("ngân sách", "đường ống"); diễn đạt bị động kiểu dịch máy.
- Khẳng định vượt quá bằng chứng: suy diễn nguyên nhân mà không có thí nghiệm kiểm chứng.
