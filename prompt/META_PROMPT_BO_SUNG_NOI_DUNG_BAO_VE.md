# Meta-Prompt: Bổ sung và nâng cấp nội dung tài liệu ôn tập bảo vệ đề tài

## 1. Mục tiêu
Rà soát, chuẩn hóa và bổ sung toàn diện các nội dung phục vụ bảo vệ học phần Học sâu cho nhóm 10 tại các tệp tài liệu:
- `CAU_HOI_VA_GIAI_THICH_BAO_VE.md`
- `HUONG_DAN_BAO_VE.md`
- `README.md` (Mục 9 - Chuẩn bị khi bảo vệ)

Tập trung vào các câu hỏi chất vấn trọng tâm của hội đồng, các bẫy lý thuyết thường gặp và giải thích bám sát thực nghiệm `casting_full_3models_01`.

## 2. Các nội dung chuyên môn cần bổ sung
1. **So sánh tham số học được giữa MobileNetV2 và ResNet-18**:
   - Giải thích nghịch lý: MobileNetV2 có tổng tham số ít hơn nhiều (2,34 triệu vs 11,22 triệu), nhưng số lượng tham số được cập nhật ở đầu phân loại mới lại nhiều hơn đáng kể (82.049 vs 32.897).
   - Căn cứ: Bản đồ đặc trưng trước GAP của MobileNetV2 là 1.280 kênh ($1.280 \times 64 + 64 = 82.048$, cộng lớp Dense cuối là 82.049), trong khi ResNet-18 chỉ có 512 kênh ($512 \times 64 + 64 = 32.832$, cộng lớp Dense cuối là 32.897).
2. **Lặp 3 kênh cho ảnh xám ở 2 backbone học chuyển giao**:
   - Lý do kỹ thuật: Backbone ImageNet cố định cấu trúc nhận đầu vào 3 kênh RGB `(224, 224, 3)`.
   - Bản chất: Việc sao chép 3 kênh xám giống hệt nhau giữ nguyên vẹn nội dung hình học và cường độ sáng, không làm thay đổi thông tin bản chất, đồng thời cho phép tận dụng nguyên vẹn trọng số trích xuất đặc trưng của ImageNet mà không cần phá vỡ cấu trúc tầng Conv đầu tiên.
3. **Ý nghĩa công nghiệp của FN vs FP và bài toán điều chỉnh ngưỡng (Threshold Tuning)**:
   - Trong kiểm định đúc kim loại thực tế: Bỏ sót phôi lỗi (FN - False Negative) nghiêm trọng hơn rất nhiều so với báo nhầm phôi đạt thành lỗi (FP - False Positive).
   - Ngưỡng 0,5 là ngưỡng tự nhiên ($p_{ok} \ge 0,5 \Leftrightarrow p_{ok} \ge p_{def}$).
   - Đề xuất mở rộng trong công nghiệp: Giảm ngưỡng $p_{ok}$ (ví dụ xuống 0,35 - 0,40) nhằm đẩy Recall lớp lỗi tiệm cận 100%, triệt tiêu tối đa rủi ro đưa sản phẩm lỗi ra thị trường.
4. **Phân biệt kiến trúc khối: Basic Residual Block vs Inverted Residual Block**:
   - ResNet-18: Khối dư cơ bản (Basic Residual Block) gồm 2 tầng Conv 3x3 + BatchNorm + ReLU, cộng nhánh tắt `x + F(x)`.
   - MobileNetV2: Khối dư đảo ngược (Inverted Residual Block) theo cấu trúc Narrow $\rightarrow$ Wide $\rightarrow$ Narrow: Conv 1x1 mở rộng số kênh $\rightarrow$ Depthwise Conv 3x3 lọc từng kênh riêng biệt $\rightarrow$ Conv 1x1 Linear Bottleneck nén số kênh lại (không dùng hàm kích hoạt phi tuyến ở tầng nén) $\rightarrow$ Skip connection nối giữa các tầng thắt nút (bottlenecks).
5. **Chuẩn hóa đường dẫn tương đối**:
   - Chuyển các đường dẫn gán cứng ổ đĩa `D:/Casting-Defect-Detection-CNN/...` trong `CAU_HOI_VA_GIAI_THICH_BAO_VE.md` và `HUONG_DAN_BAO_VE.md` thành đường dẫn tương đối để hiển thị chính xác trên mọi môi trường và máy tính của các thành viên.

## 3. Quy tắc thực hiện
- Không làm thay đổi số liệu thực nghiệm gốc trong `casting_full_3models_01`.
- Giữ văn phong khoa học, trung thực, dễ hiểu cho sinh viên khi trả lời trực tiếp trước hội đồng.
- Bổ sung các câu hỏi làm rõ với người dùng để định hướng thêm về kịch bản bảo vệ thực tế.
