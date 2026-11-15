# Bổ sung hình kiến trúc hai mô hình học chuyển giao

Chỉnh báo cáo chính `bao_cao/baocao_DL_theo_dangbaocao.docx`, lưu bản sao dự phòng trước khi sửa. Mẫu trình bày theo DA/KLTN ICTU đã cung cấp và định dạng của báo cáo hiện tại.

## Phạm vi đã xác nhận

- Tạo hai hình riêng, đặt tại mục ResNet-18 và MobileNetV2 tương ứng.
- Mỗi hình thể hiện các nhóm khối chính, kích thước đầu ra, backbone ImageNet giữ cố định và đầu phân loại được huấn luyện.
- Giữ các bảng kiến trúc hiện tại. Rút những đoạn diễn giải đang trùng với hình và bảng.
- Cập nhật báo cáo chính sau khi kiểm tra, có bản dự phòng. Lưu meta-prompt này trong `prompt/`.

## Căn cứ kỹ thuật

Đọc `Casting.ipynb`, `casting_full_3models_01/architecture_resnet18.json`, `architecture_mobilenetv2.json`, hai tệp `model_summary_*.txt` và các bảng hiện có trong báo cáo.

ResNet-18 dùng preset `resnet_18_imagenet`, đầu vào 224 × 224 × 3, converter tương ứng, phần đầu Conv 7 × 7 và MaxPool, bốn nhóm basic block với 64/128/256/512 kênh, mỗi nhóm hai block. Tensor cuối backbone là 7 × 7 × 512.

MobileNetV2 dùng ImageNet, `include_top=False`, `alpha=1.0`, đầu vào 224 × 224 × 3 và chuẩn hóa x/127,5 − 1. Backbone có Conv đầu 32 kênh, các nhóm bottleneck 16/24/32/64/96/160/320 với số khối tương ứng 1/2/3/4/3/3/1, rồi Conv 1 × 1 ra 1280 kênh. Tensor cuối là 7 × 7 × 1280.

Cả hai backbone giữ `trainable=False` và được gọi với `training=False`. Đầu phân loại gồm Global Average Pooling → Dense(64, ReLU) → Dropout(0,3) → Dense(1, sigmoid). Chú thích cần thể hiện các lớp Dense được cập nhật trong huấn luyện.

## Cách thực hiện

1. Đối chiếu thông số với cấu hình đã lưu trước khi dựng hình. Dùng công cụ vẽ kỹ thuật thông thường để tạo sơ đồ kiến trúc mạng có độ phân giải tốt và chữ rõ khi in trong Word.
2. Hình thể hiện tensor, nhóm khối mạng và phần phân loại theo đúng trình tự. Số khối và kích thước đầu ra phải khớp mã nguồn. Các mặt phẳng hoặc nút biểu diễn rút gọn cần được ghi chú rõ.
3. Đặt Hình 2.3 tại mục ResNet-18 và Hình 2.4 tại mục MobileNetV2. Giới thiệu, giải thích ngắn hai hình trong nội dung và nêu căn cứ từ cấu hình chương trình. Giữ Hình 2.2 của CNN và các bảng kiến trúc.
4. Rút các câu liệt kê tầng và kích thước đã có đủ trong hình/bảng. Giữ lý do lựa chọn, đặc điểm residual/inverted residual, cách đóng băng backbone, tiền xử lý và tổng tham số của từng mô hình.
5. Giữ tên đề tài, Chương 1, các kết quả thực nghiệm, bảng phân công, nội dung Kết luận và phụ lục. Phần mềm, checkpoint và notebook là nguồn tham chiếu của lần biên tập này.
6. Cập nhật danh mục hình, mục lục và các tham chiếu liên quan. Tuân thủ phông chữ, lề, giãn dòng và đánh số trang của bản hiện tại. Tên hình phía dưới, hình và chú thích đi cùng nhau.

## Kiểm tra đầu ra

Render báo cáo, xem lại hai hình ở kích thước in thực tế, kiểm tra chữ, đường nối, ngắt trang và bảng. Bảo đảm Chương 1 giữ đủ số trang và tổng báo cáo trong khoảng 60–70 trang bằng bố cục tự nhiên. Đối chiếu các bảng kết quả, hình cũ và số liệu trước/sau. Cập nhật bản chính khi tệp đã đóng trong Word và nguồn chưa bị thay đổi trong lúc biên tập.

Trả lại báo cáo đã cập nhật, meta-prompt và tóm tắt ngắn về hai hình, các đoạn đã rút và số trang thực tế.
