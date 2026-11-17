# Meta-prompt: Sửa baocaocodong.docx thành baocaocodong_v2.docx

## Vai trò
Biên tập viên báo cáo học phần Học sâu (ICTU, Nhóm 10, đề tài "Kiểm tra lỗi sản phẩm đúc bằng học sâu").
Làm trên bản sao `bao_cao/baocaocodong_v2.docx`; không ghi đè `baocaocodong.docx`.

## Việc phải làm
1. **Chương 1 (phương án A):**
   - Bỏ 1.3.2 (thuộc tính dạng bảng), 1.3.3 (khoảng cách, Hình 1.7 k-means), 1.3.7 (mô tả bài toán); rút gọn 1.1.2.
   - Thêm mục "Mạng nơ-ron và mạng nơ-ron tích chập": perceptron/MLP, hàm kích hoạt, lan truyền xuôi/ngược, tích chập, pooling, cấu trúc CNN, học chuyển giao.
   - Nguồn chính: slide BÀI 02, 03, 04 của thầy (bao_cao/*.pptx). Hình cắt từ slide, ghi "Nguồn: Bài giảng Học sâu, Bài x [5]". Không dùng ảnh AI.
   - Chương 1 không chứa dữ liệu, mô hình hay bài toán cụ thể của đề tài. Tổng Chương 1 khoảng 18–20 trang.
2. **Thuật ngữ:** thống nhất tiếng Anh: train, validation, test, baseline, checkpoint, loss; lần đầu có giải thích tiếng Việt. "nơ-ron" có gạch nối.
3. **Sửa lỗi:**
   - Bỏ dòng Bảng A.1, Bảng C.1 trong danh mục bảng.
   - Bỏ câu nhắc Bảng A.1 ở Phụ lục, câu nhắc Bảng C.1 ở 3.5.3.
   - Bỏ tên tác giả dataset ở 2.2.1.
   - Chuẩn hoá tài liệu [2], [5].
   - Sửa dẫn chiếu "Chương 3" ở 1.4.6.
4. **Chương 3:**
   - Đổi tên thành "CHƯƠNG 3. KẾT QUẢ THỰC NGHIỆM".
   - Chuyển 3.1.2 (ma trận nhầm lẫn, công thức chỉ số) sang 2.4. Đánh số lại bảng.
   - Mỗi chương có đoạn kết ngắn 2–3 câu, cùng kiểu. 3.6 rút gọn, không chép KẾT LUẬN.
5. **Văn phong:** hạn chế câu phủ định rào đón ("không phải…", "chưa đủ…", "không đồng nghĩa…", "cần phân biệt…"). Viết câu khẳng định, ngắn. Giữ nguyên mọi số liệu và kết luận.
6. Mở đầu không chia mục con. Lời cảm ơn, phụ lục (link Drive) giữ nguyên.

## Kiểm tra trước khi giao
- Render PDF, cập nhật số trang mục lục và danh mục.
- Đánh số hình, bảng, mục liên tục; mọi dẫn chiếu khớp.
- Đối chiếu số liệu với bản gốc: 7.348; 5.583/986/715; 94,83/98,74/99,44; 428/25/12/250; 446/7/2/260; 451/2/2/260; epoch 29/29/21.
- Tài liệu tham khảo đều được trích, đánh số theo thứ tự xuất hiện.
- Đếm câu phủ định trước/sau.
