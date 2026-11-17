# Kiểm tra lỗi sản phẩm đúc bằng học sâu

**Nhóm 10 · Học phần Học sâu · ICTU**

Dự án nhận ảnh mặt trước của sản phẩm đúc và phân loại toàn ảnh thành **lỗi** hoặc **đạt yêu cầu**. Nhóm xây dựng CNN baseline rồi so sánh với ResNet-18 và MobileNetV2 sử dụng **học chuyển giao** (*transfer learning*). Kết quả chính thức nằm trong `casting_full_3models_01`. MobileNetV2 tại epoch 21 là mô hình đề xuất theo validation loss thấp nhất, dự đoán đúng **711/715 ảnh test**. Giao diện demo cho chọn và so sánh **cả ba mô hình**.

> **Bản báo cáo chính thức để nộp và bảo vệ là [`bao_cao/baocaocodong_v3.docx`](bao_cao/baocaocodong_v3.docx)** (63 trang, kèm bản PDF cùng tên). Mọi số mục, số hình, số bảng trong README này đều theo bản v3. Các bản `baocaocodong.docx`, `baocaocodong_v2.docx` và các tệp `*_backup.docx` chỉ để đối chiếu, **không dùng để ôn**.
>
> **Cả nhóm đọc theo thứ tự:** mục 0 (bản đồ báo cáo) → mục 9 (chuẩn bị bảo vệ, câu hỏi khó và phép tính tay) → các mục 1–6 khi cần hiểu sâu một phần.

| Tài liệu | Dùng khi cần |
|---|---|
| [Báo cáo chính thức v3](bao_cao/baocaocodong_v3.docx) | Bản nộp và bản dùng khi bảo vệ |
| [Câu hỏi và giải thích bảo vệ](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) | Ôn cách đọc biểu đồ, giải thích spike và tập trả lời |
| [Casting.ipynb](Casting.ipynb) | Xem mã, cấu hình và output của lượt chính thức |
| [Hướng dẫn chạy Colab](HUONG_DAN_CASTING.md) và [hướng dẫn demo local](demo/README.md) | Làm theo các bước chi tiết trên máy của từng thành viên |

## 0. Bản đồ báo cáo v3 (để tra nhanh khi bị hỏi)

Khi thầy hỏi một chi tiết, hãy **chỉ đúng mục, hình hoặc bảng** trong báo cáo. Bảng dưới đây giúp tìm nhanh.

| Phần | Nội dung chính | Hình, bảng cần nhớ | Người phụ trách chính |
|---|---|---|---|
| Mở đầu | Lý do chọn đề tài, mục tiêu, phạm vi, bố cục ba chương | — | Cả nhóm |
| **Chương 1. Cơ sở lý thuyết** | 1.1 học máy và học sâu · 1.2 toán nền tảng (tensor, xác suất, entropy, đạo hàm) · 1.3 học từ dữ liệu · 1.4 hàm mục tiêu, tối ưu, quá khớp · **1.5 mạng nơ-ron và CNN** (nơ-ron, lan truyền xuôi/ngược, tích chập, pooling, học chuyển giao) · 1.6 kết luận | Hình 1.8–1.13 lấy từ slide Bài 01, 02, 04 của thầy; Bảng 1.1, 1.2 | Nông Hữu Thịnh |
| **Chương 2. Phương pháp và mô hình** | 2.1 phát biểu bài toán (nhãn 0 = lỗi, 1 = đạt; ngưỡng 0,5) · 2.2 dữ liệu, loại trùng SHA-256, chia tập, tiền xử lý · 2.3 kiến trúc CNN, ResNet-18, MobileNetV2 · 2.4 huấn luyện, **2.4.4 tiêu chí đánh giá**, **2.4.5 chọn mô hình theo validation loss** · 2.5 kết luận | Bảng 2.4 (CNN) · Bảng 2.5 (ResNet-18) · Bảng 2.6 (MobileNetV2) · Bảng 2.8 (thiết lập huấn luyện) · Bảng 2.9 (ma trận nhầm lẫn) · Hình 2.2–2.6 (sơ đồ mạng) | Nguyễn Quang Minh (2.1, 2.3, 2.4), Nông Hữu Thịnh (2.2) |
| **Chương 3. Kết quả thực nghiệm** | 3.1 môi trường · 3.2 số ảnh sau loại trùng · 3.3 diễn biến huấn luyện từng mô hình · 3.4 kết quả test, ma trận nhầm lẫn, so sánh · 3.5 ảnh sai, kiểm chứng sau khi nạp, demo, **3.5.4 hạn chế** · 3.6 kết luận | Bảng 3.1 (phân bố dữ liệu) · Bảng 3.2 (checkpoint, thời gian) · **Bảng 3.3 (accuracy, precision, recall, F1)** · Bảng 3.4 (TP/FN/FP/TN) · Bảng 3.5 (4 ảnh MobileNetV2 sai) · Bảng 3.6 (kiểm chứng sau nạp) · Hình 3.1–3.3 (đường học) · Hình 3.4–3.6 (ma trận) · **Hình 3.10 (demo ảnh đạt), Hình 3.11 (demo ảnh lỗi)** | Nguyễn Quang Minh (3.1–3.4), Vũ Ngọc Quý (3.4–3.5) |
| Kết luận, Tài liệu tham khảo, Phụ lục | Tóm tắt kết quả, hạn chế, hướng phát triển; 15 tài liệu; link Google Drive mã nguồn | — | Cả nhóm |

**Bảng phân công chỉ cho biết ai viết phần nào. Khi bảo vệ, thầy có thể hỏi bất kỳ ai về bất kỳ phần nào**, nên mỗi người phải nắm được mục 9 của README.

**Năm con số cả nhóm phải thuộc:**

| Nội dung | Số liệu |
|---|---|
| Dữ liệu | 7.348 ảnh nguồn → loại 64 ảnh trùng → **5.583 train / 986 validation / 715 test** (453 lỗi, 262 đạt) |
| Huấn luyện | 30 epoch, batch 32, Adam lr 0,001, seed 42, ngưỡng 0,5, chọn checkpoint theo **validation loss thấp nhất** |
| Accuracy test | CNN **94,83%** · ResNet-18 **98,74%** · MobileNetV2 **99,44%** |
| MobileNetV2 (mô hình đề xuất, epoch 21) | Đúng **711/715**, bỏ sót **2** ảnh lỗi (FN), báo nhầm **2** ảnh đạt (FP) |
| Số tham số (tổng / được học) | CNN 100.993 / 100.993 · ResNet-18 11.219.009 / 32.897 · MobileNetV2 2.340.033 / 82.049 |

## 1. Định hướng của đề tài

Nhóm muốn biết một CNN học từ đầu đạt kết quả ra sao trên bộ ảnh đúc, và việc dùng đặc trưng ảnh đã học trước có giúp phát hiện lỗi tốt hơn trong cùng điều kiện đánh giá hay không. Ba cấu hình dùng chung danh sách ảnh train, validation và test. Chúng khác nhau về kiến trúc, kích thước đầu vào và phần trọng số được học, nên kết quả phản ánh hiệu quả của từng cấu hình đầy đủ.

**Cả ba mô hình đều thuộc họ CNN.** Cụm “CNN baseline” chỉ cấu hình Conv2D/MaxPooling do nhóm lựa chọn làm mốc. ResNet-18 và MobileNetV2 là hai kiến trúc CNN đã được công bố. Phần nhóm thực hiện gồm chọn cấu hình baseline, xử lý dữ liệu, tổ chức so sánh, huấn luyện đầu phân loại, đánh giá và làm demo.

```text
Ảnh sản phẩm → Tiền xử lý → Mô hình → Điểm lớp đạt p_ok → Nhãn lỗi hoặc đạt
```

- **Nhãn 0, `def_front`** là sản phẩm lỗi.
- **Nhãn 1, `ok_front`** là sản phẩm đạt.
- `p_ok` từ **0,5** trở lên cho nhãn đạt. Dưới 0,5 cho nhãn lỗi. Giao diện hiển thị điểm lỗi bằng `1 - p_ok`.

Ngưỡng **0,5** được chọn để so sánh hai điểm lớp. Khi `p_ok ≥ 0,5`, điểm đạt lớn hơn hoặc bằng điểm lỗi. Với giá trị nhỏ hơn 0,5, điểm lỗi lớn hơn. Trường hợp bằng nhau được chương trình gán đạt. Mô hình học cách tính `p_ok` từ dữ liệu, còn ngưỡng là quy tắc quyết định của chương trình. Nếu muốn ưu tiên tránh bỏ sót sản phẩm lỗi, có thể khảo sát ngưỡng bằng validation và đánh giá trên một tập kiểm thử được giữ riêng.

`p_ok = 0,998192` là đầu ra sigmoid cho một ảnh. Việc đánh giá mức độ đáng tin của các điểm này cần phép hiệu chuẩn và dữ liệu tương ứng. Accuracy 99,44% là tỷ lệ dự đoán đúng trên 715 ảnh đã có nhãn.

Đầu ra là nhãn cho **toàn ảnh**. Phạm vi này phù hợp với dữ liệu đang có, vốn gắn nhãn lỗi hoặc đạt cho mỗi ảnh. Định vị khuyết tật là bài toán cần dữ liệu và đánh giá theo vùng ảnh.

## 2. Dữ liệu và tiền xử lý

Nhóm dùng phiên bản `casting_data` của bộ ảnh **Casting Product Image Data for Quality Inspection**. Phiên bản này có 7.348 ảnh kích thước 300 × 300, gồm 6.633 ảnh train nguồn và 715 ảnh test nguồn. [Nguồn dữ liệu Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

Nhóm đọc test trước để giữ tập kiểm thử do nguồn cung cấp. Khi tìm thấy **64 ảnh đạt trong train trùng pixel với test**, nhóm bỏ các bản sao phía train khỏi danh sách học. Cách kiểm tra dùng SHA-256 trên kích thước và pixel ảnh xám đã hiệu chỉnh hướng, trước khi đổi kích thước. Sau đó nhóm tách 15% từ 6.569 ảnh train còn lại để làm validation, chia theo tỷ lệ lớp với seed 42.

| Tập | Lỗi | Đạt | Tổng | Vai trò |
|---|---:|---:|---:|---|
| Train | 3.194 | 2.389 | **5.583** | Học và cập nhật trọng số |
| Validation | 564 | 422 | **986** | Theo dõi lượt học và chọn checkpoint |
| Test | 453 | 262 | **715** | Đánh giá sau khi đã chọn checkpoint |

Ảnh nguồn đã có các biến thể tăng cường dữ liệu. Pipeline của nhóm dùng các bước xác định như chuyển xám và resize, với `pipeline_augmentation=False`. Kiểm tra trùng pixel xử lý được bản sao chính xác. Các biến thể gần giống hoặc ảnh của cùng một sản phẩm vẫn là giới hạn vì bộ nguồn thiếu mã nhóm ảnh. [Danh sách ảnh bị loại](casting_full_3models_01/excluded_train_duplicates.csv) · [Thống kê chia tập](casting_full_3models_01/split_counts.csv)

### Đầu vào của từng mô hình

1. Đọc ảnh, hiệu chỉnh hướng EXIF, chuyển xám và resize bằng Pillow với nội suy bilinear về **256 × 256 × 1**.
2. CNN nhận đầu vào đó. Với hai backbone ImageNet, dùng TensorFlow để resize bilinear có antialias về **224 × 224**, rồi lặp kênh xám thành ba kênh. Ba kênh có cùng nội dung xám.
3. Mỗi mô hình thực hiện bước chuẩn hóa của mình bên trong mạng.

| Mô hình | Hình dạng đầu vào | Chuẩn hóa trong mạng |
|---|---|---|
| CNN | 256 × 256 × 1 | Chia giá trị pixel cho 255 |
| ResNet-18 | 224 × 224 × 3 | `ResNetImageConverter` từ cùng preset ImageNet, có scale và offset riêng theo kênh |
| MobileNetV2 | 224 × 224 × 3 | `x / 127,5 - 1`, đưa pixel từ khoảng 0–255 về khoảng −1 đến 1 |

## 3. CNN tự xây dựng

CNN học từ đầu. Nó học các **bộ lọc** quét qua những vùng nhỏ của ảnh để tạo ra các bản đồ đặc trưng. Một bản đồ đặc trưng ghi vị trí và mức độ mà bộ lọc phản ứng với mẫu hình nó đã học.

```text
Ảnh 256 × 256 × 1
  → Conv2D 32 bộ lọc + ReLU → MaxPooling → 128 × 128 × 32
  → Conv2D 64 bộ lọc + ReLU → MaxPooling → 64 × 64 × 64
  → Conv2D 128 bộ lọc + ReLU → MaxPooling → 32 × 32 × 128
  → Global Average Pooling → 128 giá trị
  → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid → p_ok
```

- **Conv2D** dùng bộ lọc 3 × 3. Padding `same` giữ chiều cao và chiều rộng sau tích chập. **ReLU** đưa giá trị âm về 0 và giữ giá trị dương để mạng học được các quan hệ phi tuyến.
- **MaxPooling** lấy giá trị lớn nhất trong từng vùng 2 × 2, giảm chiều cao và chiều rộng còn một nửa sau mỗi khối.
- **Global Average Pooling (GAP)** lấy trung bình trên mỗi bản đồ. Đầu ra có 128 số vì khối cuối có 128 bản đồ đặc trưng.
- **Dense 64** kết hợp các đặc trưng. **Dropout 0,3** tạm đặt ngẫu nhiên một phần giá trị về 0 khi huấn luyện. **Dense 1 và sigmoid** tạo điểm lớp đạt từ 0 đến 1.

CNN có **100.993 tham số được cập nhật**. Phần tìm đặc trưng ảnh và phần phân loại đều học từ ảnh đúc của nhóm. [Bảng cấu trúc CNN](casting_full_3models_01/model_summary_cnn.txt)

### “Tự xây dựng” được hiểu thế nào?

Nhóm ghép các lớp Keras và lựa chọn số bộ lọc, số khối, cách giảm kích thước, đầu phân loại và cấu hình học. Các thành phần Conv2D, MaxPooling, GAP, Dense và Dropout là những thành phần phổ biến. CNN baseline được huấn luyện từ khởi tạo ngẫu nhiên trên tập ảnh đúc đã chia.

Một tham số là một giá trị trọng số hoặc bias. Ví dụ, lớp Conv2D đầu có `(3 × 3 × 1 + 1) × 32 = 320` tham số. Số 1 trong ngoặc là bias của mỗi bộ lọc. Lớp Dense 64 nhận 128 giá trị từ GAP có `(128 + 1) × 64 = 8.256` tham số. Cộng các lớp có trọng số cho tổng **100.993**. Số tham số và kích thước đầu ra là hai thông tin riêng, có thể xem trong `model_summary_cnn.txt`.

### Hai hàm kích hoạt cần hiểu

- **ReLU** có công thức `max(0, z)`. Ví dụ minh họa, đầu vào −2 cho đầu ra 0, còn đầu vào 3 cho đầu ra 3. Trong baseline, ReLU nằm sau tích chập và tại Dense 64.
- **Sigmoid** có công thức `1 / (1 + exp(-z))`, tạo điểm lớp đạt ở đầu ra. Khi `z = 0`, sigmoid bằng 0,5. Khi z tăng, điểm tiến gần 1, khi z giảm mạnh, điểm tiến gần 0.

Chuẩn hóa, tích chập và các lớp tiếp theo tạo thành một chuỗi đã được học để chuyển ảnh thành điểm phân loại.

## 4. Hai mô hình học chuyển giao và phần backbone được giữ lại

**Backbone** là phần nhận ảnh và tạo bản đồ đặc trưng. **Đầu phân loại** dùng các đặc trưng đó để dự đoán nhãn. Trong học chuyển giao, nhóm lấy một backbone đã học trên ImageNet, giữ trọng số của nó cố định, rồi huấn luyện đầu phân loại cho hai lớp của bài toán ảnh đúc. [Học chuyển giao trong Keras](https://keras.io/guides/transfer_learning/)

| Mô hình | Backbone giữ lại | Đặc trưng trước GAP | Đầu phân loại của nhóm |
|---|---|---|---|
| ResNet-18 | `ResNetBackbone.from_preset("resnet_18_imagenet", load_weights=True)` từ KerasHub | 7 × 7 × 512 | GAP → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid |
| MobileNetV2 | `keras.applications.MobileNetV2(weights="imagenet", include_top=False, alpha=1.0)` | 7 × 7 × 1.280 | GAP → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid |

Ở cả hai cấu hình, backbone chạy với `trainable=False` và được gọi với `training=False`. Backbone vẫn tính đặc trưng cho từng ảnh. Hai lớp Dense mới nhận cập nhật trọng số, còn các lớp học trước, kể cả thống kê Batch Normalization, giữ trạng thái suy luận. Đây là học chuyển giao với **backbone cố định**. Fine-tuning là bước mở khóa một phần backbone để học tiếp trên dữ liệu mới. [Hướng dẫn về đóng băng lớp và Batch Normalization](https://keras.io/guides/transfer_learning/)

### ResNet-18 làm gì trong luồng ảnh?

ResNet-18 dùng các khối có **đường nối tắt** (*skip connection*). Khi kích thước phù hợp, khối cộng đầu vào với phần biến đổi do các lớp bên trong tạo ra. Cách nối này giúp học các mạng sâu dễ hơn. Trong dự án, ảnh đi qua converter của preset rồi qua backbone ResNet-18 để lấy 512 bản đồ đặc trưng. GAP tóm tắt chúng thành 512 số cho đầu phân loại. [Bài báo ResNet](https://arxiv.org/abs/1512.03385) · [Bảng cấu trúc ResNet-18 của nhóm](casting_full_3models_01/model_summary_resnet18.txt)

Ví dụ một basic block cùng kích thước trong nhóm 64 kênh có luồng sau. BN là Batch Normalization.

```text
Nhánh biến đổi F(x)
x → Conv 3×3, 64 → BN → ReLU → Conv 3×3, 64 → BN

Nhánh tắt giữ nguyên x
Đầu ra = ReLU(F(x) + x), kích thước 56 × 56 × 64
```

Hai lớp tích chập tạo phần biến đổi `F(x)`. Sau cộng và ReLU, đầu ra là `ReLU(F(x) + x)`. Khi khối đổi kích thước hoặc số kênh, nhánh tắt dùng phép chiếu để cộng được hai tensor. ResNet-18 đã học những trọng số này trong giai đoạn ImageNet. Trong lượt thực nghiệm của nhóm, chúng giữ cố định. [Triển khai basic block KerasHub 0.23.0](https://github.com/keras-team/keras-hub/blob/v0.23.0/keras_hub/src/models/resnet/resnet_backbone.py)

### MobileNetV2 làm gì trong luồng ảnh?

MobileNetV2 dùng các khối mở rộng số kênh, lọc theo từng kênh bằng **depthwise convolution**, rồi nén về số kênh đầu ra. Đây là cách tổ chức được thiết kế để giảm chi phí tính toán. `include_top=False` lấy phần trích đặc trưng của mạng ImageNet. Sau chuẩn hóa, backbone tạo 1.280 bản đồ đặc trưng và GAP tóm tắt chúng thành 1.280 số. [Bài báo MobileNetV2](https://arxiv.org/abs/1801.04381) · [Bảng cấu trúc MobileNetV2 của nhóm](casting_full_3models_01/model_summary_mobilenetv2.txt)

Ví dụ `block_7` trong mạng thực tế có đầu vào **14 × 14 × 64**, hệ số mở rộng 6 và stride 1.

```text
Conv 1×1 mở rộng 64 → 384 kênh → BN → ReLU6
  → Depthwise 3×3 trên từng kênh → BN → ReLU6
  → Conv 1×1 chiếu 384 → 64 kênh → BN
  → Cộng với đầu vào x
Đầu ra 14 × 14 × 64
```

ReLU6 giới hạn giá trị trong khoảng 0–6. Phép chiếu về 64 kênh giữ dạng tuyến tính trước khi cộng. Nhánh tắt được dùng khi stride bằng 1 và số kênh hai đầu bằng nhau. Khối đầu mạng dùng hệ số mở rộng 1, các khối tiếp theo sử dụng cấu hình mở rộng 6. [Triển khai MobileNetV2 Keras 3.12.0](https://github.com/keras-team/keras/blob/v3.12.0/keras/src/applications/mobilenet_v2.py)

| Mô hình | Tổng tham số | Tham số được cập nhật |
|---|---:|---:|
| CNN | 100.993 | 100.993 |
| ResNet-18 | 11.219.009 | 32.897 |
| MobileNetV2 | 2.340.033 | 82.049 |

MobileNetV2 có ít tham số tổng hơn ResNet-18, nhưng đầu Dense 64 của nó nhận 1.280 số thay vì 512 số, nên đầu phân loại có nhiều trọng số được học hơn. Thời gian huấn luyện còn gồm phần tính đặc trưng của backbone. Trên Tesla T4 trong lượt chính thức, thời gian fit xấp xỉ 300 giây với CNN, 991 giây với ResNet-18 và 557 giây với MobileNetV2. [Số tham số và thời gian fit](casting_full_3models_01/comparison.csv)

## 5. Huấn luyện, đọc biểu đồ và chọn checkpoint

Ba mô hình dùng **30 epoch**, **batch 32**, seed 42 và Adam với learning rate khởi đầu **0,001**. Một epoch là một lượt qua toàn bộ 5.583 ảnh train. Sau mỗi nhóm tối đa 32 ảnh, Adam điều chỉnh các trọng số được phép học. ReduceLROnPlateau giảm learning rate còn một nửa khi validation loss không cải thiện đủ trong 3 epoch, với mức sàn 0,000001. [Cấu hình trong notebook](Casting.ipynb)

**Accuracy** đếm tỷ lệ ảnh đúng theo ngưỡng 0,5. **Loss** là điểm phạt dùng trong học, ở đây là binary cross-entropy. Ví dụ, với một ảnh lỗi, nếu mô hình cho điểm đạt 0,60 hoặc 0,99 thì cả hai lần đều sai một ảnh. Loss phạt nặng hơn ở lần mô hình gán điểm 0,99 cho nhãn sai. Vì vậy, accuracy có thể thay đổi ít trong khi loss tăng rõ.

### Vì sao CNN lên chậm và vì sao biểu đồ có spike?

Sau epoch đầu, validation accuracy của CNN là **57,20%**, ResNet-18 là **95,44%** và MobileNetV2 là **97,26%**. CNN phải học cả đặc trưng và đầu phân loại từ đầu. Hai backbone ImageNet đã tạo được đặc trưng cho ảnh, nên đầu phân loại mới có thể học nhanh hơn. Khi nhận xét quá khớp, cần xem cả train và validation qua nhiều epoch.

Sau mỗi epoch, mô hình được đánh giá lại trên cùng **986 ảnh validation**. Trọng số vừa thay đổi có thể khiến một số ảnh đang được dự đoán đúng thành sai. Nếu chất lượng giảm ở một epoch rồi phục hồi ở epoch tiếp theo, đường loss xuất hiện một đỉnh ngắn.

| Mô hình | Epoch trước → đỉnh → sau | Số ảnh validation sai trên 986 ảnh |
|---|---|---|
| CNN | 8 → 9 → 10 | 98 → **216** → 82 |
| CNN | 17 → 18 → 19 | 67 → **281** → 57 |
| ResNet-18 | 18 → 19 → 20 | 15 → **18** → 15 |
| MobileNetV2 | 13 → 14 → 15 | 6 → **18** → 6 |

Các số trên cho thấy mức tụt và mức phục hồi. Lịch sử loss và accuracy theo epoch chưa xác định nguyên nhân riêng của mỗi đỉnh. Dấu hiệu quá khớp cần nhìn trên nhiều lượt. Với MobileNetV2, từ epoch 21 đến 30, train loss giảm từ 0,0119 xuống 0,0046, trong khi validation loss kết thúc cao hơn, từ 0,0150 lên 0,0179. Kết quả gợi ý quá khớp nhẹ về cuối lượt. [Biểu đồ CNN](casting_full_3models_01/learning_curves_cnn.png) · [Biểu đồ ResNet-18](casting_full_3models_01/learning_curves_resnet18.png) · [Biểu đồ MobileNetV2](casting_full_3models_01/learning_curves_mobilenetv2.png)

Đỉnh CNN ở epoch 18 đáng chú ý vì accuracy validation giảm từ **93,20% xuống 71,50%**, sau đó lên **94,22%** ở epoch 19. Số ảnh sai tại đỉnh là 281. Với ResNet-18 và MobileNetV2, số ảnh sai tăng ở các đỉnh được nêu nhỏ hơn nhiều. Khi trình bày, cần đọc đúng mức dao động của từng mạng.

Hai cơ chế đánh giá cũng giúp hiểu vì sao validation có thể tốt hơn train. Chỉ số train được tích lũy qua các batch trong lúc trọng số liên tục thay đổi và Dropout hoạt động. Validation dùng trọng số cuối epoch và tắt Dropout. Chênh lệch giữa hai đường cần được đọc trong điều kiện đó.

**Accuracy đạt 0,99 tự nó chưa đủ chẩn đoán quá khớp.** Cần xem khả năng cải thiện trên train, diễn biến validation và độ ổn định qua nhiều epoch. CNN có chất lượng thấp hơn hai cấu hình học trước, còn kết luận thiếu khớp cần thêm thực nghiệm về khả năng học và quá trình tối ưu.

### Vì sao chạy 30 epoch mà demo dùng epoch 21?

**Checkpoint** là bản lưu của mô hình tại một thời điểm trong quá trình huấn luyện. Nhóm chọn một checkpoint có validation loss nhỏ nhất cho từng mô hình. Sau đó, nhóm so sánh ba giá trị này để chọn mô hình đề xuất. Quyết định được lưu trong `selection.json` trước khi đánh giá trên 715 ảnh test. MobileNetV2 đạt validation loss thấp nhất ở **epoch 21**. Demo local hỗ trợ cả ba checkpoint đã chọn của từng mạng. [Kết quả lựa chọn](casting_full_3models_01/selection.json)

## 6. Kết quả chính thức và cách đọc

| Mô hình | Epoch chọn | Validation loss | Accuracy test | Ảnh đúng / 715 | Ảnh lỗi bị bỏ sót | Ảnh đạt bị báo lỗi |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 29 | 0,1210 | 94,83% | 678 | 25 | 12 |
| ResNet-18 | 29 | 0,0269 | 98,74% | 706 | 7 | 2 |
| MobileNetV2 | **21** | **0,0150** | **99,44%** | **711** | **2** | **2** |

Bảng trên gộp Bảng 3.2, 3.3 và 3.4 của báo cáo. Validation loss ghi 4 chữ số như trong báo cáo (giá trị đầy đủ trong `fit_*.json`: 0,121010 / 0,026946 / 0,015027).

Với MobileNetV2, trong 453 ảnh thật sự lỗi có 451 ảnh được phát hiện đúng, nên **recall lớp lỗi là 451/453**, khoảng 99,56%. Trong các ảnh bị báo lỗi, có 451 ảnh lỗi thật và 2 ảnh đạt bị báo nhầm, nên **precision lớp lỗi cũng là 451/453**. Accuracy lấy số đúng của cả hai lớp, gồm 451 ảnh lỗi và 260 ảnh đạt, chia cho 715.

Ma trận được đọc theo **hàng là nhãn thật, cột là nhãn dự đoán**. Ô hàng lỗi, cột đạt cho biết số sản phẩm lỗi bị cho qua. Hai ảnh lỗi MobileNetV2 bỏ sót gồm `cast_def_0_150.jpeg`, có điểm đạt 0,955552, và `cast_def_0_2102.jpeg`, có điểm đạt 0,502379 sau làm tròn sáu chữ số. Một ảnh được dự đoán đạt với điểm cao, ảnh kia nằm sát ngưỡng 0,5. [Bảng kết quả đầy đủ](casting_full_3models_01/comparison.csv) · [Ma trận MobileNetV2](casting_full_3models_01/metrics_mobilenetv2.json) · [Danh sách ảnh sai](casting_full_3models_01/errors_mobilenetv2.csv)

### TP, FN, FP, TN trong báo cáo

Nhóm chọn **lớp lỗi là lớp quan tâm**, dù nhãn số của nó là 0. Cách đặt tên dương tính được xác định theo lớp đang đánh giá.

| Ô trong ma trận | Ý nghĩa | MobileNetV2 |
|---|---|---:|
| TP, thật lỗi và dự đoán lỗi | Phát hiện đúng sản phẩm lỗi | 451 |
| FN, thật lỗi và dự đoán đạt | Bỏ sót sản phẩm lỗi | 2 |
| FP, thật đạt và dự đoán lỗi | Báo lỗi nhầm | 2 |
| TN, thật đạt và dự đoán đạt | Nhận đúng sản phẩm đạt | 260 |

`Precision = TP / (TP + FP)` trả lời trong số ảnh bị báo lỗi, có bao nhiêu ảnh lỗi thật. `Recall = TP / (TP + FN)` trả lời trong số ảnh lỗi thật, phát hiện được bao nhiêu. F1 kết hợp precision và recall bằng trung bình điều hòa. **Macro F1** là trung bình F1 của hai lớp, mỗi lớp có trọng số bằng nhau.

### Phạm vi của kết luận so sánh

Kết quả cho thấy hai cấu hình học chuyển giao đạt độ chính xác cao hơn baseline trên bộ kiểm thử này. Cả ba dùng cùng danh sách ảnh, số epoch và cách đánh giá. Khởi tạo ImageNet, độ phân giải và phần tham số được cập nhật khác nhau giữa các cấu hình. Muốn tách riêng tác động của từng yếu tố cần thiết kế thêm thực nghiệm đối chứng.

Các chỉ số trên mô tả một lần chạy với seed 42 và tập test cùng nguồn. Khả năng sử dụng với sản phẩm, camera hoặc ánh sáng khác cần được đánh giá trên dữ liệu tương ứng. Thời gian fit trên Colab đo cả lượt học và validation, còn tốc độ suy luận local là phép đo riêng.

## 7. Xem hoặc chạy lại trên Google Colab

Để đọc ngay, mở [notebook có output](Casting.ipynb) và xem các hình, bảng trong [thư mục kết quả chính thức](casting_full_3models_01/). Các tệp `history_*.csv` ghi loss và accuracy qua từng epoch. Cột `epoch` trong CSV bắt đầu từ 0, còn biểu đồ và báo cáo đếm từ 1.

Để tái lập thực nghiệm:

1. Nén phiên bản `data/casting_data` thành ZIP. Thư mục gốc dữ liệu trong ZIP cần chứa `train/def_front`, `train/ok_front`, `test/def_front` và `test/ok_front`.
2. Upload `Casting.ipynb` lên Colab và chọn GPU nếu có. Chạy các ô chuẩn bị thư viện ở mục 1, khởi động lại session rồi chạy từ mục 2. Notebook thiết lập TensorFlow 2.20.0, Keras 3.12.0 và KerasHub 0.23.0 cho lượt này.
3. Upload ZIP dữ liệu ở mục 3. Kiểm tra output mục 4 có 64 bản sao bị loại và ba số đếm 5.583 / 986 / 715.
4. Chạy các mục 5–8 để chuẩn bị hàm và huấn luyện từng mô hình. Notebook lưu checkpoint `.keras`, lịch sử và gói sao lưu sau từng mô hình. Khi chạy thêm một lượt trong cùng thư mục kết quả, đặt `RUN_NAME` mới để ghi kết quả vào thư mục riêng.
5. Chạy mục 9 để chọn mô hình và đánh giá test. Mục 10 tải gói kết quả huấn luyện. Các mục 11–13 chuẩn bị checkpoint demo, kiểm chứng khi nạp lại trên CPU, thử ảnh và tải `demo_bundle.zip` cùng gói `complete_*.zip`.

[Hướng dẫn Colab theo từng mục](HUONG_DAN_CASTING.md)

## 8. Chạy demo trên Windows

Demo Streamlit cho chọn **CNN epoch 29, ResNet-18 epoch 29 và MobileNetV2 epoch 21**, chạy riêng hoặc so sánh trên cùng ảnh bằng CPU. Chuẩn bị các checkpoint và metadata trong `demo/models/` bằng `demo/prepare_models.py` từ bộ kết quả `casting_full_3models_01`. Thư mục trọng số được bỏ qua bởi Git, nên cần tải kèm khi chuyển máy. MobileNetV2 có validation loss thấp nhất trong thực nghiệm.

Mở PowerShell tại thư mục gốc dự án. Ví dụ dưới đây dùng Python 3.12.

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo\requirements.txt
.\.venv-demo\Scripts\python.exe demo\prepare_models.py
.\.venv-demo\Scripts\python.exe -m streamlit run demo\app.py
```

Mở Local URL, chọn một hoặc cả ba mô hình rồi tải ảnh và nhấn **Phân loại tất cả**. Mỗi ảnh có bảng đối chiếu nhãn và điểm dự đoán của các mô hình đã chọn. Điểm hiển thị 6 chữ số thập phân. CSV/JSON chứa một bản ghi cho mỗi cặp ảnh–mô hình, nên 2 ảnh với 3 mô hình tạo 6 bản ghi. Đây là điểm của từng ảnh, còn accuracy 99,44% được tính trên toàn bộ 715 ảnh test.

Hai ảnh dùng để demo khi bảo vệ (đúng như Hình 3.10 và Hình 3.11 trong báo cáo):

| Ảnh | Mô hình | Kết quả | Điểm lỗi | Điểm đạt (p_ok) |
|---|---|---|---:|---:|
| `demo/reference/cast_ok_0_45.jpeg` (ảnh đạt, Hình 3.10) | CNN | Đạt | 0.151642 | 0.848358 |
| | ResNet-18 | Đạt | 0.000407 | 0.999593 |
| | MobileNetV2 | Đạt | 0.001808 | 0.998192 |
| `casting_full_3models_01/demo_test/def_front/cast_def_0_1059.jpeg` (ảnh lỗi, Hình 3.11) | CNN | Lỗi | 0.999961 | 0.000039 |
| | ResNet-18 | Lỗi | 0.996825 | 0.003175 |
| | MobileNetV2 | Lỗi | 0.999987 | 0.000013 |

Lượt chạy ảnh lỗi được thực hiện lại ngày 18/11/2026 bằng chính `demo/app.py`. Ảnh chụp giao diện và số liệu lưu tại `demo/reference/demo_cast_def_0_1059.png` và `.txt`. Nếu máy demo hỏng lúc bảo vệ, mở Hình 3.10 và 3.11 trong báo cáo để trình bày thay.

Notebook đã đối chiếu MobileNetV2 sau nạp trên CPU với 10 ảnh validation, có nhãn trùng **10/10 ảnh** và sai lệch điểm lớn nhất khoảng **4,21 × 10⁻⁸**. Khi chuẩn bị demo đủ ba mô hình, mỗi checkpoint tiếp tục được đối chiếu trên 10 ảnh test, gồm 5 ảnh lỗi và 5 ảnh đạt.

| Mô hình | Nhãn trùng dự đoán Colab | Sai lệch điểm lớn nhất |
|---|---:|---:|
| CNN | 10/10 | 8,2600 × 10⁻⁸ |
| ResNet-18 | 10/10 | 3,1825 × 10⁻⁶ |
| MobileNetV2 | 10/10 | 6,2771 × 10⁻⁷ |

Nhãn trùng trong phép kiểm chứng có nghĩa đầu ra sau nạp khớp dự đoán tham chiếu. Tỷ lệ đúng so với nhãn thật được tính trên toàn bộ 715 ảnh test. Sai lệch số thực nhỏ giữa GPU và CPU được kiểm tra với dung sai tương đối 10⁻³ và tuyệt đối 10⁻⁴. [Kiểm chứng ban đầu](casting_full_3models_01/demo/verification.json) · [Kiểm chứng cả ba mô hình](demo/models/comparison_verification.json)

Giao diện Streamlit đã được chạy trên Windows/CPU ngày 15/11/2026 với hai ảnh đạt/lỗi, tạo sáu kết quả và kiểm tra tải CSV/JSON, đổi lựa chọn, chạy riêng CNN và xử lý tệp ảnh hỏng. Thành viên có thể dùng ảnh tham chiếu để kiểm tra lại trên máy của mình. [Hướng dẫn demo](demo/README.md)

### Lưu ý khi chuyển máy

- Gói `complete_*.zip` chứa bộ kết quả và ba checkpoint. Gói `demo_bundle.zip` từ notebook chứa mô hình đề xuất. Để mở giao diện đủ ba mạng, dùng bộ kết quả chính thức với `demo/prepare_models.py` hoặc nhận đủ thư mục `demo/models/` đã chuẩn bị.
- Giữ cùng chuỗi tiền xử lý từ ảnh nguồn đến tensor đầu vào. Metadata mô tả bước chuyển xám, resize 256, resize 224 đối với mạng học trước và quy tắc chuẩn hóa trong mô hình.
- Trên Windows, `demo/inference.py` tái tạo converter ResNet-18 bằng các phép Keras theo cấu hình đã lưu, sau đó nạp trọng số checkpoint. Các kết quả đối chiếu ở trên là căn cứ kiểm tra cách nạp này.
- Môi trường demo đã kiểm tra gồm Python 3.12.14, TensorFlow 2.20.0, Keras 3.12.0, KerasHub 0.23.0, NumPy 2.2.6, Pillow 11.3.0 và Streamlit 1.67.0. Tệp requirements quy định các gói cài đặt. [Thư viện demo](demo/requirements.txt)

Ảnh dùng minh họa `cast_ok_0_45.jpeg` thuộc tập test. Khi trình bày, có thể mở hai ảnh tham chiếu đạt/lỗi, chọn cả ba mô hình, xem sáu kết quả rồi tải CSV hoặc JSON. Ảnh ngoài bộ dữ liệu có điều kiện chụp tương tự có thể dùng để khảo sát thêm, kèm thông tin về nhãn và nguồn ảnh.

## 9. Chuẩn bị khi bảo vệ

> Nhóm dùng CNN học từ đầu làm mốc và hai backbone ImageNet là ResNet-18, MobileNetV2 để học chuyển giao. Với hai backbone này, nhóm giữ cố định phần trích đặc trưng và huấn luyện đầu phân loại mới. Cả ba cấu hình dùng cùng cách chia train, validation và test. Nhóm chọn MobileNetV2 ở epoch 21 theo validation loss. Trên 715 ảnh test, mô hình đúng 711 ảnh, còn bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.

Mỗi thành viên nên tập giải thích được cách chia tập, đường đi của ảnh qua mô hình, sự khác nhau giữa loss và accuracy, và cách đọc số ảnh sai từ ma trận. Khi được hỏi vì sao loss có đỉnh, chỉ vào số ảnh sai ở lượt trước, tại đỉnh và lượt sau. Khi được hỏi kết quả có áp dụng ngoài nhà máy hay không, nêu rõ điều kiện của tập test và phạm vi phân loại toàn ảnh. [Câu hỏi và câu trả lời mẫu](CAU_HOI_VA_GIAI_THICH_BAO_VE.md)

### Những câu cần trả lời được

| Câu hỏi | Cách giải thích bám thực nghiệm |
|---|---|
| Vì sao dùng ba mô hình này? | CNN là mốc học từ đầu, ResNet-18 và MobileNetV2 đại diện hai cấu hình học chuyển giao có cách tổ chức khối và quy mô khác nhau. |
| Bộ dữ liệu bao nhiêu ảnh thật sự được dùng? | 7.348 ảnh nguồn, loại 64 bản trùng chính xác phía train, còn 7.284 ảnh chia thành 5.583 train, 986 validation và 715 test. |
| Loại trùng đã xử lý được những gì? | SHA-256 kiểm tra kích thước và pixel ảnh xám trước resize. Quan hệ giữa các biến thể xoay hoặc đổi sáng còn cần thông tin ảnh gốc để xác định. |
| Vì sao dùng ảnh xám? | Ảnh nguồn mô tả bề mặt và cấu hình dùng thông tin cường độ sáng. Hai backbone nhận ba bản sao của cùng kênh xám để đáp ứng hình dạng đầu vào. |
| Vì sao resize hai lần? | Biểu diễn chung dùng Pillow về 256, sau đó hai mạng học trước dùng TensorFlow về 224. Demo tái tạo đúng chuỗi mà mô hình đã nhận trong thực nghiệm. |
| CNN tự xây có đóng góp gì? | Nhóm chọn và ghép cấu hình baseline, chuẩn bị dữ liệu, huấn luyện, đối chiếu, đánh giá và triển khai demo bằng các lớp CNN phổ biến. |
| Backbone giữ lại những gì? | Giữ phần trích xuất và trọng số ImageNet. Hai lớp Dense mới được học, còn backbone và thống kê Batch Normalization giữ cố định. |
| Vì sao accuracy lên nhanh với hai mạng học trước? | Chúng đã có bộ trích đặc trưng ImageNet, phần học mới tập trung vào đầu phân loại. |
| Spike có chứng minh overfitting? | Cần đọc nhiều epoch. Báo cáo ghi nhận mức giảm và phục hồi ở từng mạng, đồng thời giữ nguyên mức chưa xác định của nguyên nhân cụ thể. |
| Vì sao chọn MobileNetV2 epoch 21? | Đây là checkpoint có validation loss thấp nhất theo tiêu chí đã đặt và được lưu trước bước đánh giá test. |
| Precision, recall lớp lỗi có ý nghĩa gì? | Precision xét các ảnh bị báo lỗi, recall xét các ảnh thực sự lỗi. Với MobileNetV2, cả hai bằng 451/453, khoảng 99,56%. |
| Vì sao MobileNetV2 ít tham số tổng nhưng tham số học lại nhiều hơn ResNet-18? | MobileNetV2 tạo ra 1.280 kênh đặc trưng trước GAP (so với 512 của ResNet-18), nên lớp Dense 64 của nó nhận nhiều đầu vào hơn và học 82.049 tham số (so với 32.897 tham số của ResNet-18). |
| Khối Inverted Residual của MobileNetV2 khác gì Residual của ResNet-18? | ResNet-18 dùng **basic block**: hai Conv 3×3 **giữ nguyên số kênh** (ví dụ 64 → 64), cộng nối tắt rồi ReLU (Hình 2.3). MobileNetV2 làm ngược kiểu "mỏng → dày → mỏng": Conv 1×1 **mở rộng** kênh 6 lần (64 → 384), Depthwise 3×3 lọc riêng từng kênh cho nhẹ, Conv 1×1 **nén tuyến tính** về 64 kênh (không ReLU), rồi nối tắt giữa hai đầu ít kênh (Hình 2.5). *Lưu ý: kiểu "dày → mỏng → dày" là bottleneck của ResNet-50 trở lên, không phải ResNet-18.* |
| Điểm đạt cao có bảo đảm đúng? | Có ảnh lỗi `cast_def_0_150.jpeg` nhận điểm đạt 0,955552. Điểm sigmoid cần được đọc cùng dự đoán sai và phạm vi kiểm chứng. |
| Trong nhà máy thực tế, FN hay FP nguy hiểm hơn và nhóm chỉnh thế nào? | FN (bỏ sót lỗi) nguy hiểm hơn vì sản phẩm hỏng lọt ra ngoài; FP chỉ tốn công kiểm tra lại. Quy tắc là **p_ok ≥ ngưỡng thì đạt**, nên muốn bớt bỏ sót phải **TĂNG ngưỡng** (khó công nhận "đạt" hơn), đổi lại FP tăng. **Tuyệt đối không nói "hạ ngưỡng"**, vì hạ ngưỡng làm bỏ sót nhiều hơn. Ngưỡng mới phải chọn trên **validation**, không chọn trên test. Xem phần "Câu hỏi về ngưỡng" ở mục 9. |
| Có áp dụng ngoài nhà máy ngay được? | Kết quả hiện có gắn với tập test cùng nguồn. Ảnh độc lập theo sản phẩm, camera và ánh sáng là dữ liệu cần bổ sung để đánh giá phạm vi sử dụng rộng hơn. |
| Nhóm đã so sánh tốc độ chạy mỗi ảnh chưa? | Các thời gian 300,36 / 990,97 / 557,11 giây là thời gian huấn luyện và validation trên T4. Độ trễ dự đoán là phép đo cần tổ chức riêng. |

### Câu hỏi khó cần chuẩn bị kỹ

Đây là những câu giảng viên hỏi kỹ thường dùng để kiểm tra nhóm có thật sự hiểu bài không. Mỗi câu có ý chính cần nói và mục trong báo cáo để chỉ vào.

1. **"99,44% có cao bất thường không? Có rò rỉ dữ liệu không?"**
   **Chủ động thừa nhận, đừng chối.** Bộ dữ liệu nguồn đã có sẵn ảnh tăng cường (xoay, đổi sáng) và không có mã sản phẩm gốc, nên train và test có thể chứa biến thể của cùng một ảnh. SHA-256 chỉ bắt được **64 ảnh trùng y hệt**. Vì vậy 99,44% có thể lạc quan hơn thực tế; nhóm đã ghi rõ ở **mục 3.5.4** và phần Kết luận. Cả ba mô hình chịu cùng điều kiện nên **phép so sánh giữa chúng vẫn công bằng**. Hướng khắc phục là chia tập theo nhóm ảnh gốc và thêm ảnh độc lập.
2. **"Sao chọn mô hình theo validation loss mà không theo accuracy, cũng không theo test?"**
   Accuracy của nhiều epoch gần bằng nhau, còn loss phản ánh cả mức tự tin nên phân biệt tốt hơn. Chọn theo test là dùng test để quyết định, khiến kết quả test không còn khách quan. Nhóm ghi lại quyết định **trước** khi chạy test (mục 2.4.5).
3. **"Sao validation accuracy của CNN có lúc cao hơn train?"**
   Dropout chỉ bật khi train và tắt khi đánh giá validation. Chỉ số train lại là trung bình trong lúc trọng số đang thay đổi, còn validation dùng trọng số cuối epoch (mục 3.3.2).
4. **"ResNet-18 chỉ học 32.897 tham số mà sao chạy lâu nhất (991 giây)?"**
   Mỗi batch vẫn phải **lan truyền xuôi qua toàn bộ backbone 11,2 triệu tham số** để lấy đặc trưng. MobileNetV2 dùng depthwise convolution nên phần tính xuôi nhẹ hơn. Thời gian phụ thuộc lượng phép tính, không phụ thuộc số tham số được học.
5. **"Sao đóng băng backbone mà không fine-tune?"**
   Đóng băng nhanh hơn, ít rủi ro quá khớp với dữ liệu nhỏ, và đã đạt 99,44%. Fine-tuning là hướng phát triển (mục 3.5.4). **Không nhận là đã làm fine-tuning.**
6. **"Ảnh xám mà mạng ImageNet cần 3 kênh thì sao?"**
   Lặp kênh xám thành 3 kênh giống nhau (mục 2.2.4).
7. **"Sao dùng Global Average Pooling mà không dùng Flatten?"**
   Flatten tạo 32 × 32 × 128 = 131.072 đầu vào, khiến Dense 64 cần 131.072 × 64 + 64 = **8.388.672 tham số**. GAP chỉ còn 128 đầu vào nên Dense 64 có **8.256 tham số**. Gọn hơn khoảng 1.000 lần và ít quá khớp hơn (mục 2.3.2).
8. **"Sao đầu ra là 1 nút sigmoid mà không phải softmax 2 nút?"**
   Bài toán có 2 lớp. Một nút sigmoid cho p_ok, còn điểm lỗi bằng 1 − p_ok, tương đương softmax 2 nút nhưng ít tham số hơn và đi cùng binary cross-entropy.
9. **"Ảnh `cast_def_0_150.jpeg` là ảnh lỗi mà MobileNetV2 cho p_ok 0,9556, tại sao?"**
   Nói thật là **nhóm chưa xác định được nguyên nhân**. ResNet-18 cũng đoán sai ảnh này (p_ok 0,9799), nên đây là ảnh khó chung. Muốn biết mô hình nhìn vào đâu cần thêm chú giải vùng lỗi hoặc công cụ giải thích như Grad-CAM; **nhóm chưa làm phần này** (mục 3.5.1). Không tự bịa nguyên nhân.
10. **"Đỉnh nhọn trên biểu đồ loss có phải lỗi code không?"**
   Không phải. Mỗi đỉnh xuất hiện một epoch rồi phục hồi ngay ở epoch sau. Chỉ vào số ảnh sai cụ thể, ví dụ CNN epoch 17 → 18 → 19: 67 → 281 → 57 ảnh sai trên 986 (mục 3.3).
11. **"Tập test lệch lớp (453 lỗi, 262 đạt) thì accuracy còn ý nghĩa không?"**
   Vì vậy báo cáo có thêm precision, recall, F1 của lớp lỗi, macro F1 và ma trận nhầm lẫn, không chỉ dựa vào accuracy (Bảng 3.3, 3.4).
12. **"Nhóm tự làm gì hay chỉ chép trên Kaggle?"**
   Nhóm tự chọn cấu hình CNN baseline, viết bước kiểm tra trùng bằng SHA-256, chia tập, huấn luyện ba mô hình trong cùng điều kiện, đánh giá, phân tích ảnh sai và làm demo so sánh ba mô hình. Các lớp mạng dùng thư viện Keras.

### Câu hỏi về ngưỡng (hay bị hỏi và dễ trả lời sai)

Quy tắc của nhóm là **p_ok ≥ ngưỡng thì kết luận đạt**, còn lại là lỗi.

- **Tăng ngưỡng** thì khó được công nhận "đạt" hơn, nên **bớt bỏ sót lỗi (FN giảm)** nhưng **báo nhầm nhiều hơn (FP tăng)**.
- **Hạ ngưỡng** thì ngược lại, **bỏ sót nhiều hơn**. Vì vậy **không bao giờ nói "hạ ngưỡng để giảm bỏ sót"**.

Minh họa bằng dự đoán của MobileNetV2 trên 715 ảnh test (chỉ để thấy xu hướng; khi làm thật phải chọn ngưỡng trên **validation**, sau đó mới đánh giá test):

| Ngưỡng p_ok | FN (bỏ sót lỗi) | FP (báo nhầm) |
|---:|---:|---:|
| 0,50 (đang dùng) | 2 | 2 |
| 0,60 | 1 | 2 |
| 0,80 | 1 | 3 |
| 0,90 | 1 | 6 |
| 0,96 | 0 | 9 |

Ảnh `cast_def_0_150.jpeg` có p_ok 0,9556, nên phải đẩy ngưỡng lên khoảng 0,96 mới bắt được nó, trong khi số ảnh báo nhầm tăng từ 2 lên 9. Đó là **đánh đổi**; mức ngưỡng phụ thuộc chi phí của nhà máy.

### Phép tính tay có thể bị yêu cầu viết lên bảng

| Nội dung | Công thức | Ví dụ trong đề tài |
|---|---|---|
| Kích thước đầu ra Conv | o = ⌊(n + 2p − k) / s⌋ + 1 | Ảnh 5×5, bộ lọc 3×3, p = 0, s = 1 → 3×3 (Hình 1.11). Padding same, s = 1 → giữ 256×256 |
| MaxPooling 2×2, stride 2 | chia đôi mỗi chiều | 256 → 128 → 64 → 32 |
| Số tham số một lớp Conv | (k × k × C_in + 1) × C_out | Conv1: (3×3×1 + 1) × 32 = **320**; Conv2: (3×3×32 + 1) × 64 = **18.496**; Conv3: (3×3×64 + 1) × 128 = **73.856** |
| Số tham số lớp Dense | (số đầu vào + 1) × số nơ-ron | Dense 64: (128 + 1) × 64 = **8.256**; Dense 1: 64 + 1 = **65**. Tổng CNN = **100.993** |
| Đầu phân loại hai mạng học trước | (đặc trưng + 1) × 64 + 65 | ResNet-18: (512 + 1) × 64 + 65 = **32.897**; MobileNetV2: (1.280 + 1) × 64 + 65 = **82.049** |
| Sigmoid | σ(z) = 1 / (1 + e^(−z)); σ′ = σ(1 − σ) | σ(0) = 0,5; đạo hàm lớn nhất là **0,25** tại z = 0, nên qua nhiều tầng sigmoid gradient nhỏ dần (gradient biến mất) |
| Binary cross-entropy (1 ảnh) | L = −[y·ln p + (1 − y)·ln(1 − p)] | Ảnh đạt (y = 1): p_ok = 0,9 → L ≈ **0,105**; p_ok = 0,1 → L ≈ **2,303**. Đoán sai mà tự tin thì bị phạt rất nặng |
| Accuracy, precision, recall | (TP + TN) / tổng; TP / (TP + FP); TP / (TP + FN) | MobileNetV2: (451 + 260) / 715 = **99,44%**; 451 / 453 = **99,56%** cho cả precision và recall lớp lỗi |
| Một ảnh test bằng bao nhiêu % | 1 / 715 | ≈ **0,14 điểm phần trăm** |
| Chia tập | 6.633 − 64 = 6.569; lấy 15% | 986 validation, 5.583 train |
| Số bước mỗi epoch | ⌈số ảnh train / batch⌉ | ⌈5.583 / 32⌉ = **175 bước** |

### Thuật ngữ trong báo cáo, mỗi người phải giải thích được trong một câu

| Thuật ngữ | Giải thích một câu |
|---|---|
| SHA-256 (mã băm) | "Dấu vân tay" của ảnh: hai ảnh giống hệt từng pixel thì có cùng mã, dùng để tìm ảnh trùng |
| EXIF | Thông tin đi kèm ảnh, trong đó có hướng xoay; nhóm xoay ảnh đúng chiều trước khi xử lý |
| Stratify (chia phân tầng) | Chia validation sao cho tỷ lệ lỗi/đạt giống tập train |
| Checkpoint | Bản lưu trọng số mô hình tại một epoch |
| Baseline | Mô hình mốc để so sánh; ở đây là CNN tự xây |
| Backbone | Phần trích đặc trưng của mạng học trước trên ImageNet |
| Đóng băng (`trainable=False`) | Giữ nguyên trọng số backbone, chỉ học đầu Dense mới |
| `training=False` | Gọi backbone ở chế độ suy luận để Batch Normalization giữ thống kê cũ |
| ReduceLROnPlateau | Validation loss không cải thiện 3 epoch thì giảm tốc độ học một nửa |
| TerminateOnNaN | Dừng huấn luyện nếu loss bị lỗi số (NaN) |
| Batch Normalization | Chuẩn hóa đầu ra của một tầng để học ổn định hơn |
| Dropout 0,3 | Khi train, tạm tắt ngẫu nhiên 30% nơ-ron để giảm quá khớp; khi dự đoán thì không tắt |
| Global Average Pooling | Lấy trung bình mỗi bản đồ đặc trưng thành một số |
| ReLU6 | ReLU nhưng chặn trên ở 6, dùng trong MobileNetV2 |
| Depthwise convolution | Mỗi kênh dùng một bộ lọc riêng, ít phép tính hơn Conv thường |
| ResNetImageConverter | Lớp tiền xử lý đi kèm ResNet-18 trong KerasHub: resize, nhân scale, cộng offset |
| Hiệu chuẩn (calibration) | Kiểm tra xem "90% tự tin" có thật sự đúng 90% không; nhóm **chưa làm** |
| Macro F1 | Trung bình F1 của lớp lỗi và lớp đạt, coi hai lớp quan trọng như nhau |

### Quy tắc khi trả lời

- **Ngắn, có số, có mục.** Ví dụ: "Dạ, MobileNetV2 bỏ sót 2 ảnh lỗi, ở Bảng 3.4 ạ."
- **Không biết thì nói thật**: "Phần này nhóm chưa kiểm chứng, đã ghi ở mục hạn chế 3.5.4." Không bịa.
- **Không nói những câu tuyệt đối**: "chắc chắn đúng", "không hề quá khớp", "dùng ngay trong nhà máy được", "triệt tiêu hoàn toàn lỗi".
- **Không nhận những việc chưa làm**: fine-tuning, Grad-CAM, chạy nhiều seed, đo độ trễ từng ảnh, hiệu chuẩn xác suất.
- **Cả nhóm dùng cùng một câu tóm tắt 30 giây** (khung trích dẫn đầu mục 9) để không nói lệch nhau.

### Checklist trước buổi bảo vệ

- [ ] Mỗi người đọc lại mục 0 và mục 9 của README, tự trả lời to 12 câu hỏi khó ở trên.
- [ ] Chạy thử demo trên laptop sẽ mang đi (lần đầu nạp mô hình hơi lâu); để sẵn ảnh `cast_ok_0_45.jpeg` và `cast_def_0_1059.jpeg`.
- [ ] Mở sẵn file PDF báo cáo v3; nếu demo lỗi thì chỉ vào Hình 3.10 và 3.11.
- [ ] Mở link Google Drive ở cửa sổ ẩn danh để chắc người khác xem được.
- [ ] Sạc đầy laptop, có mặt sớm.

### Đọc báo cáo và trình bày theo thứ tự

1. Chương 2 giải thích bài toán, nhãn, dữ liệu, đầu vào, ba cấu hình và cách chọn checkpoint.
2. Chương 3 bắt đầu từ số lượng ảnh thực dùng, trình bày kết quả huấn luyện, so sánh test và phân tích các ảnh sai.
3. Khi nói về một mô hình, nêu phần được học, checkpoint sử dụng và kết quả trên test. Khi nói về spike, chỉ vào epoch cụ thể cùng số ảnh sai trước và sau.
4. Mở demo đủ ba mạng trên cùng hai ảnh, đọc nhãn và điểm theo từng mạng, rồi xuất kết quả.

## 10. Các tệp dùng đối chiếu và gói nộp

| Tệp hoặc thư mục | Nội dung cần biết |
|---|---|
| `Casting.ipynb` | Mã và output của lượt thực nghiệm chính thức |
| `casting_full_3models_01/config.json`, `split.json` | Cấu hình và danh sách ảnh của ba tập |
| `history_*.csv`, `fit_*.json`, `selection.json` | Diễn biến huấn luyện, checkpoint và quyết định lựa chọn |
| `architecture_*.json`, `model_summary_*.txt` | Cấu trúc mạng và số tham số |
| `test_predictions_*.csv`, `metrics_*.json`, `errors_*.csv` | Dự đoán từng ảnh, chỉ số và ảnh sai |
| `demo/app.py`, `inference.py`, `prepare_models.py`, `style.css` | Giao diện, nạp mạng, chuẩn bị checkpoint và cách trình bày |
| `demo/models/` | Ba checkpoint `.keras`, metadata và kết quả kiểm chứng |
| `.streamlit/config.toml` | Cấu hình giao diện khi chạy từ gốc dự án |
| `casting_full_3models_01_decay/` | Dữ liệu của lượt thử nghiệm lịch tốc độ học khác, được lưu để tham khảo |

Các bảng số liệu trong báo cáo và README lấy từ **`casting_full_3models_01`**. Model summary mô tả tầng, kích thước và tham số từ cấu hình mạng. Các hình đường học và ma trận sử dụng output đã lưu của cùng lượt.

Mã nguồn nộp cho giảng viên được dẫn bằng liên kết trong phụ lục. Trước khi nộp, nhóm mở liên kết ở cửa sổ riêng tư để xác nhận quyền xem và kiểm tra gói có notebook, mã demo, requirements, cấu hình, ba checkpoint cùng metadata, kết quả chính thức và hướng dẫn chạy. Quyền truy cập và danh sách tệp trên Drive cần được nhóm xác nhận từ liên kết chia sẻ. [Thư mục Google Drive](https://drive.google.com/drive/folders/1otbkaT-wvJfNERf_CL0uapmz2M2ebSnR?usp=sharing)

Việc phát triển tiếp có thể tập trung vào chia tập theo nhóm ảnh gốc khi có mã nhóm, chạy nhiều seed, khảo sát ngưỡng trên validation, fine-tuning và đo thời gian suy luận. Các kết quả mới cần đi kèm cấu hình và bộ output riêng để đối chiếu.

Các prompt và meta-prompt của dự án được lưu trong thư mục [prompt](prompt/). Prompt mới tiếp tục được lưu tại thư mục này theo quy ước trong AGENTS.md.
