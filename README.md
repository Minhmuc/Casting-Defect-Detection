# Phân loại lỗi sản phẩm đúc bằng CNN và học chuyển giao

**Nhóm 10 · Học phần Học sâu · ICTU**

Dự án nhận một ảnh mặt trước của sản phẩm đúc và phân loại ảnh thành **lỗi** hoặc **đạt yêu cầu**. Nhóm dùng CNN tự xây dựng làm mốc so sánh, sau đó thử hai mô hình **học chuyển giao** (*transfer learning*) là ResNet-18 và MobileNetV2. Kết quả chính thức nằm trong `casting_full_3models_01`. MobileNetV2 tại epoch 21 được chọn cho demo và dự đoán đúng **711/715 ảnh test**.

| Tài liệu | Dùng khi cần |
|---|---|
| [Báo cáo Nhóm 10](bao_cao/baocao_DL_theo_dangbaocao.docx) | Xem phương pháp ở Chương 2 và kết quả ở Chương 3 |
| [Câu hỏi và giải thích bảo vệ](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) | Ôn cách đọc biểu đồ, giải thích spike và tập trả lời |
| [Casting.ipynb](Casting.ipynb) | Xem mã, cấu hình và output của lượt chính thức |
| [Hướng dẫn chạy Colab](HUONG_DAN_CASTING.md) và [hướng dẫn demo local](demo/README.md) | Làm theo các bước chi tiết trên máy của từng thành viên |

## 1. Định hướng của đề tài

Nhóm muốn biết một CNN học từ đầu đạt kết quả ra sao trên bộ ảnh đúc, và việc dùng đặc trưng ảnh đã học trước có giúp phát hiện lỗi tốt hơn trong cùng điều kiện đánh giá hay không. Ba cấu hình dùng chung danh sách ảnh train, validation và test. Chúng khác nhau về kiến trúc, kích thước đầu vào và phần trọng số được học, nên kết quả phản ánh hiệu quả của từng cấu hình đầy đủ.

```text
Ảnh sản phẩm → Tiền xử lý → Mô hình → Điểm lớp đạt p_ok → Nhãn lỗi hoặc đạt
```

- **Nhãn 0, `def_front`** là sản phẩm lỗi.
- **Nhãn 1, `ok_front`** là sản phẩm đạt.
- `p_ok` từ **0,5** trở lên cho nhãn đạt. Dưới 0,5 cho nhãn lỗi. Giao diện hiển thị điểm lỗi bằng `1 - p_ok`.

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

1. Đọc ảnh, hiệu chỉnh hướng, chuyển xám và resize bằng Pillow về **256 × 256 × 1**.
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

## 4. Hai mô hình học chuyển giao và phần backbone được giữ lại

**Backbone** là phần nhận ảnh và tạo bản đồ đặc trưng. **Đầu phân loại** dùng các đặc trưng đó để dự đoán nhãn. Trong học chuyển giao, nhóm lấy một backbone đã học trên ImageNet, giữ trọng số của nó cố định, rồi huấn luyện đầu phân loại cho hai lớp của bài toán ảnh đúc. [Học chuyển giao trong Keras](https://keras.io/guides/transfer_learning/)

| Mô hình | Backbone giữ lại | Đặc trưng trước GAP | Đầu phân loại của nhóm |
|---|---|---|---|
| ResNet-18 | `ResNetBackbone.from_preset("resnet_18_imagenet", load_weights=True)` từ KerasHub | 7 × 7 × 512 | GAP → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid |
| MobileNetV2 | `keras.applications.MobileNetV2(weights="imagenet", include_top=False, alpha=1.0)` | 7 × 7 × 1.280 | GAP → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid |

Ở cả hai cấu hình, backbone chạy với `trainable=False` và được gọi với `training=False`. Backbone vẫn tính đặc trưng cho từng ảnh. Hai lớp Dense mới nhận cập nhật trọng số, còn các lớp học trước, kể cả thống kê Batch Normalization, giữ trạng thái suy luận. Đây là học chuyển giao với **backbone cố định**. Fine-tuning là bước mở khóa một phần backbone để học tiếp trên dữ liệu mới. [Hướng dẫn về đóng băng lớp và Batch Normalization](https://keras.io/guides/transfer_learning/)

### ResNet-18 làm gì trong luồng ảnh?

ResNet-18 dùng các khối có **đường nối tắt** (*skip connection*). Khi kích thước phù hợp, khối cộng đầu vào với phần biến đổi do các lớp bên trong tạo ra. Cách nối này giúp học các mạng sâu dễ hơn. Trong dự án, ảnh đi qua converter của preset rồi qua backbone ResNet-18 để lấy 512 bản đồ đặc trưng. GAP tóm tắt chúng thành 512 số cho đầu phân loại. [Bài báo ResNet](https://arxiv.org/abs/1512.03385) · [Bảng cấu trúc ResNet-18 của nhóm](casting_full_3models_01/model_summary_resnet18.txt)

### MobileNetV2 làm gì trong luồng ảnh?

MobileNetV2 dùng các khối mở rộng số kênh, lọc theo từng kênh bằng **depthwise convolution**, rồi nén về số kênh đầu ra. Đây là cách tổ chức được thiết kế để giảm chi phí tính toán. `include_top=False` lấy phần trích đặc trưng của mạng ImageNet. Sau chuẩn hóa, backbone tạo 1.280 bản đồ đặc trưng và GAP tóm tắt chúng thành 1.280 số. [Bài báo MobileNetV2](https://arxiv.org/abs/1801.04381) · [Bảng cấu trúc MobileNetV2 của nhóm](casting_full_3models_01/model_summary_mobilenetv2.txt)

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

Hãy hình dung mô hình làm lại cùng một đề sau mỗi lượt học. Trọng số vừa thay đổi có thể làm một số câu đang đúng thành sai. Nếu số sai tăng ở một lượt rồi giảm ở lượt tiếp theo, trên biểu đồ loss có thể xuất hiện một đỉnh ngắn.

| Mô hình | Epoch trước → đỉnh → sau | Số ảnh validation sai trên 986 ảnh |
|---|---|---|
| CNN | 8 → 9 → 10 | 98 → **216** → 82 |
| CNN | 17 → 18 → 19 | 67 → **281** → 57 |
| ResNet-18 | 18 → 19 → 20 | 15 → **18** → 15 |
| MobileNetV2 | 13 → 14 → 15 | 6 → **18** → 6 |

Các số trên cho thấy mức tụt và mức phục hồi. Lịch sử loss và accuracy theo epoch chưa xác định nguyên nhân riêng của mỗi đỉnh. Dấu hiệu quá khớp cần nhìn trên nhiều lượt. Với MobileNetV2, từ epoch 21 đến 30, train loss giảm từ 0,0119 xuống 0,0046, trong khi validation loss kết thúc cao hơn, từ 0,0150 lên 0,0179. Kết quả gợi ý quá khớp nhẹ về cuối lượt. [Biểu đồ CNN](casting_full_3models_01/learning_curves_cnn.png) · [Biểu đồ ResNet-18](casting_full_3models_01/learning_curves_resnet18.png) · [Biểu đồ MobileNetV2](casting_full_3models_01/learning_curves_mobilenetv2.png)

### Vì sao chạy 30 epoch mà demo dùng epoch 21?

**Checkpoint** là bản lưu của mô hình tại một thời điểm trong quá trình huấn luyện. Nhóm chọn một checkpoint có validation loss nhỏ nhất cho từng mô hình. Sau đó, nhóm so sánh ba giá trị validation loss để chọn mô hình demo. Quyết định được lưu trong `selection.json` trước khi đánh giá trên 715 ảnh test. MobileNetV2 đạt validation loss thấp nhất ở **epoch 21**. [Kết quả lựa chọn](casting_full_3models_01/selection.json)

## 6. Kết quả chính thức và cách đọc

| Mô hình | Epoch chọn | Validation loss | Accuracy test | Ảnh đúng / 715 | Ảnh lỗi bị bỏ sót | Ảnh đạt bị báo lỗi |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 29 | 0,121010 | 94,83% | 678 | 25 | 12 |
| ResNet-18 | 29 | 0,026946 | 98,74% | 706 | 7 | 2 |
| MobileNetV2 | **21** | **0,015027** | **99,44%** | **711** | **2** | **2** |

Với MobileNetV2, trong 453 ảnh thật sự lỗi có 451 ảnh được phát hiện đúng, nên **recall lớp lỗi là 451/453**, khoảng 99,56%. Trong các ảnh bị báo lỗi, có 451 ảnh lỗi thật và 2 ảnh đạt bị báo nhầm, nên **precision lớp lỗi cũng là 451/453**. Accuracy lấy số đúng của cả hai lớp, gồm 451 ảnh lỗi và 260 ảnh đạt, chia cho 715.

Khi bảo vệ, hãy đọc ma trận theo **hàng là nhãn thật, cột là nhãn dự đoán**. Ô hàng lỗi, cột đạt cho biết số sản phẩm lỗi bị cho qua. Hai ảnh lỗi MobileNetV2 bỏ sót gồm `cast_def_0_150.jpeg`, có điểm đạt khoảng 0,9556, và `cast_def_0_2102.jpeg`, có điểm đạt khoảng 0,5024. Một ảnh được dự đoán đạt với điểm cao, ảnh kia nằm sát ngưỡng 0,5. [Bảng kết quả đầy đủ](casting_full_3models_01/comparison.csv) · [Ma trận MobileNetV2](casting_full_3models_01/metrics_mobilenetv2.json) · [Danh sách ảnh sai](casting_full_3models_01/errors_mobilenetv2.csv)

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

Demo Streamlit nạp **MobileNetV2 epoch 21** và dự đoán trên CPU. Kiểm tra `demo/models/` có `casting_classifier.keras` và `metadata.json` của lượt chính thức. Với bản clone Git, lấy hai tệp từ gói `demo_bundle.zip` của lượt đó hoặc từ `casting_full_3models_01/demo/`, vì thư mục mô hình local được cấu hình bỏ qua khi theo dõi bằng Git.

Mở PowerShell tại thư mục gốc dự án. Ví dụ dưới đây dùng Python 3.12.

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo\requirements.txt
.\.venv-demo\Scripts\python.exe -m streamlit run demo\app.py
```

Mở địa chỉ local do terminal hiển thị, chọn ảnh và nhấn **Phân loại ảnh**. Ảnh `demo/reference/cast_ok_0_45.jpeg` có kết quả tham chiếu trong notebook là **Đạt**, với điểm đạt hiển thị **99,82%**. Đây là điểm của một ảnh. Accuracy 99,44% là tỷ lệ đúng trên toàn bộ 715 ảnh test.

Notebook đã đối chiếu checkpoint sau khi nạp lại trên CPU với 10 ảnh validation. Nhãn trùng **10/10 ảnh**, sai lệch điểm dự đoán lớn nhất khoảng **4,21 × 10⁻⁸**. Giao diện Streamlit được kiểm tra tĩnh. Thành viên có thể dùng ảnh tham chiếu và tệp JSON tải từ giao diện để xác nhận trên máy của mình. [Kiểm chứng CPU](casting_full_3models_01/demo/verification.json) · [Ảnh minh họa từ notebook](demo/reference/demo_cast_ok_0_45.png) · [Hướng dẫn demo](demo/README.md)

## 9. Chuẩn bị khi bảo vệ

> Nhóm dùng CNN học từ đầu làm mốc và hai backbone ImageNet là ResNet-18, MobileNetV2 để học chuyển giao. Với hai backbone này, nhóm giữ cố định phần trích đặc trưng và huấn luyện đầu phân loại mới. Cả ba cấu hình dùng cùng cách chia train, validation và test. Nhóm chọn MobileNetV2 ở epoch 21 theo validation loss. Trên 715 ảnh test, mô hình đúng 711 ảnh, còn bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.

Mỗi thành viên nên tập giải thích được cách chia tập, đường đi của ảnh qua mô hình, sự khác nhau giữa loss và accuracy, và cách đọc số ảnh sai từ ma trận. Khi được hỏi vì sao loss có đỉnh, chỉ vào số ảnh sai ở lượt trước, tại đỉnh và lượt sau. Khi được hỏi kết quả có áp dụng ngoài nhà máy hay không, nêu rõ điều kiện của tập test và phạm vi phân loại toàn ảnh. [Câu hỏi và câu trả lời mẫu](CAU_HOI_VA_GIAI_THICH_BAO_VE.md)
