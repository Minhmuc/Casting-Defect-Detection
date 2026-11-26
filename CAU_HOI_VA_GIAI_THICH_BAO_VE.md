# Câu hỏi và giải thích bảo vệ đề tài "Kiểm tra lỗi sản phẩm đúc kim loại" (môn Học sâu)

Tài liệu đọc thêm cho [README](README.md). Bám **báo cáo v4 bản in 25/11/2026** ([bao_cao/baocaocodong_v4.docx](bao_cao/baocaocodong_v4.docx), 58 trang Ả Rập) và bộ kết quả [`casting_full_3models_01`](casting_full_3models_01/). Các chỗ "mục", "Bảng", "Hình" dưới đây là số trong bản in; bản này đã đánh số lại Hình/Bảng Chương 2–3 so với bản trước.

## 1. Đọc kết quả huấn luyện ba mô hình (mục 3.3)

### Một epoch là gì? Biểu đồ đang đo gì?

Một **epoch** là một lượt mô hình học qua toàn bộ 5.583 ảnh train. Mỗi lần mô hình xử lý một nhóm tối đa 32 ảnh, gọi là **batch**, rồi điều chỉnh trọng số. Mỗi epoch có ⌈5.583 / 32⌉ = 175 lần cập nhật. Cuối epoch, chương trình đo kết quả trên 986 ảnh validation. Cả ba mô hình chạy 30 epoch.

Ở Hình 3.1–3.3, trục ngang là epoch. Đường xanh là train, đường cam là validation. Mỗi hình có hai biểu đồ.

- **Accuracy** (tỷ lệ đúng) trả lời "bao nhiêu ảnh được phân loại đúng?". Accuracy 0,95 là 95%.
- **Loss** là giá trị hàm mất mát binary cross-entropy. Loss càng nhỏ thì mô hình càng gán điểm cao cho nhãn thật.

Ví dụ, ảnh thật sự đạt. Hai dự đoán cho điểm đạt 0,51 và 0,99 đều đúng ở ngưỡng 0,5. Dự đoán 0,99 có loss thấp hơn. Vì vậy accuracy có thể gần như đứng yên trong khi loss thay đổi.

### Ba mô hình học như thế nào?

| Mô hình | Phần được học | Val accuracy sau epoch 1 | Epoch được lưu | Val loss tại epoch lưu | Val accuracy tại epoch lưu |
|---|---|---:|---:|---:|---:|
| CNN | Toàn bộ mạng, từ khởi tạo ngẫu nhiên | 57,20% | 29 | 0,1210 | 95,84% |
| ResNet-18 | Đầu phân loại mới | 95,44% | 29 | 0,0269 | 98,78% |
| MobileNetV2 | Đầu phân loại mới | 97,26% | 21 | 0,0150 | 99,49% |

(Bảng 3.2. Giá trị đầy đủ trong `fit_*.json`: 0,121010 / 0,026946 / 0,015027.)

**Vì sao CNN lên chậm, còn hai mô hình kia đạt hơn 90% ngay epoch đầu?**

> CNN phải học cả cách tìm đặc trưng trong ảnh lẫn cách phân loại. ResNet-18 và MobileNetV2 đã có phần trích xuất đặc trưng học trên ImageNet. Nhóm giữ phần đó cố định và chỉ dạy đầu phân loại mới phân biệt lỗi và đạt.

Có thể hình dung backbone chuyển ảnh thành một dãy số mô tả ảnh, còn đầu phân loại đọc dãy số đó để ra nhãn. Cách tận dụng phần đã học trước gọi là **học chuyển giao**.

- **CNN.** Ba epoch đầu, validation accuracy đứng ở 57,20% = 564/986, đúng bằng tỷ lệ ảnh lỗi trong validation: mô hình đang đoán mọi ảnh là lỗi. Từ epoch 4 lên 85,19%, đến epoch 29 đạt 95,84%. Đỉnh dao động rõ ở epoch 9 và 18. [Biểu đồ CNN](casting_full_3models_01/learning_curves_cnn.png)
- **ResNet-18.** Validation accuracy 95,44% sau epoch đầu và 98,78% tại epoch 29. Epoch 30 có accuracy nhỉnh hơn (98,88%) nhưng validation loss tăng từ 0,0269 lên 0,0295. Nhóm giữ epoch 29 theo tiêu chí loss. [Biểu đồ ResNet-18](casting_full_3models_01/learning_curves_resnet18.png)
- **MobileNetV2.** Tại epoch 21, validation loss thấp nhất là 0,0150, accuracy 99,49%. Đây là checkpoint dùng cho test và demo. [Biểu đồ MobileNetV2](casting_full_3models_01/learning_curves_mobilenetv2.png)

### Accuracy lên 0,99 có phải quá khớp?

> Em xem kết quả đó thuộc tập nào và theo dõi nhiều epoch. Nếu train tiếp tục tốt lên còn validation dần xấu đi, đó là dấu hiệu quá khớp. Accuracy 99% chỉ cho biết tỷ lệ đúng trên tập đang xem.

Với **MobileNetV2**, từ epoch 21 đến 30, train loss giảm từ **0,0119 xuống 0,0046**, còn validation loss không thấp hơn 0,0150 và kết thúc ở **0,0179**. Validation accuracy giảm từ 99,49% xuống 99,39%, tức thêm một ảnh sai trên 986. Báo cáo gọi đây là dấu hiệu quá khớp nhẹ cuối lượt (mục 3.3.4). Backbone cố định nhưng đầu phân loại vẫn có thể quá khớp.

Với **CNN**, ngoài các đỉnh tạm thời, validation loss giảm cùng xu hướng train loss và cuối lượt vẫn thấp hơn train loss, nên lịch sử chưa cho thấy quá khớp (mục 3.3.2). Với **ResNet-18**, từ epoch 20 có khoảng cách nhỏ giữa train và validation, validation loss dao động 0,027–0,030. **Thiếu khớp** là khi mô hình còn kém ngay trên train. CNN điểm thấp hơn hai mô hình học chuyển giao chưa đủ để kết luận thiếu khớp.

### Validation spike là gì và vì sao xuất hiện?

**Validation spike** là đỉnh tăng đột ngột trên đường validation loss. Để giải thích, đọc loss và số ảnh sai ở ba epoch liên tiếp.

| Mô hình | Epoch trước → đỉnh → sau | Validation loss | Số ảnh sai trên 986 |
|---|---|---|---|
| CNN | 8 → 9 → 10 | 0,2704 → **0,4488** → 0,2405 | 98 → **216** → 82 |
| CNN | 17 → 18 → 19 | 0,1910 → **0,5188** → 0,1722 | 67 → **281** → 57 |
| ResNet-18 | 18 → 19 → 20 | 0,0302 → **0,0455** → 0,0279 | 15 → **18** → 15 |
| MobileNetV2 | 13 → 14 → 15 | 0,0171 → **0,0696** → 0,0204 | 6 → **18** → 6 |

**Tại sao cùng 986 ảnh mà kết quả thay đổi?** Mỗi batch train làm thay đổi trọng số. Cuối epoch, mô hình dùng trọng số vừa học để dự đoán trên cùng tập validation. Việc cập nhật dựa trên ảnh train, nên ở một số epoch, dự đoán validation có thể tạm thời kém đi. Ví dụ minh họa: một ảnh lỗi được cho điểm đạt 0,45 ở epoch trước và 0,55 ở epoch sau; điểm vượt ngưỡng 0,5 nên nhãn chuyển từ lỗi sang đạt. Báo cáo (mục 3.3.2) ghi các đỉnh này có thể liên quan đến thứ tự batch xáo trộn và bước cập nhật của Adam; nhóm chưa chạy thêm seed để kiểm tra.

**Vì sao loss đôi khi tăng mạnh dù accuracy giảm ít?** Loss còn tính độ tự tin. Với một ảnh lỗi, mô hình cho điểm đạt 0,60 hay 0,99 thì đều là một ảnh sai, nhưng 0,99 bị phạt nặng hơn nhiều. CNN ở epoch 18 có số ảnh sai tăng mạnh nên cả loss và accuracy đều biến động rõ. MobileNetV2 ở epoch 14 có accuracy giảm khoảng 1,22 điểm phần trăm trong khi loss tăng khoảng 4 lần. Lịch sử chỉ ghi chỉ số tổng hợp theo epoch, nên nhóm chưa xác định được đóng góp của từng ảnh hay nguyên nhân cụ thể của từng đỉnh.

#### CNN có spike ở epoch 9 và 18

> CNN có hai đỉnh rõ ở epoch 9 và 18. Ở đỉnh lớn nhất, accuracy validation giảm từ 93,20% xuống 71,50%, số ảnh sai tăng từ 67 lên 281, rồi còn 57 ở epoch 19. Vì nhiều ảnh đổi sang nhãn sai, cả accuracy và loss đều biến động mạnh. Nhìn qua nhiều epoch, kết quả validation vẫn cải thiện và nhóm giữ checkpoint epoch 29.

#### ResNet-18 có spike ở epoch 19

> Từ epoch 18 sang 19, validation loss tăng từ 0,0302 lên 0,0455. Số ảnh sai chỉ tăng từ 15 lên 18, rồi về 15 ở epoch 20. Loss còn tính điểm mô hình gán cho nhãn thật, nên loss đổi rõ dù số ảnh sai chỉ tăng ít. Sau đỉnh, loss giảm về 0,0279 và nhóm giữ checkpoint epoch 29.

#### MobileNetV2 có spike ở epoch 14

> Từ epoch 13 sang 14, validation loss tăng từ 0,0171 lên 0,0696, số ảnh sai tăng từ 6 lên 18. Sang epoch 15, mô hình lại chỉ sai 6 ảnh. Sau đó validation tiếp tục cải thiện và đạt loss thấp nhất ở epoch 21. Khi nói quá khớp nhẹ, em nhìn đoạn sau epoch 21, khi train loss giảm tiếp còn validation loss kết thúc cao hơn.

Nếu được hỏi chính xác vì sao từng đỉnh xảy ra: trọng số đổi sau mỗi epoch nên dự đoán trên cùng tập validation cũng đổi. Lịch sử cho biết mức dao động và sự phục hồi, còn nguyên nhân riêng của từng đỉnh chưa được xác định.

[Lịch sử CNN](casting_full_3models_01/history_cnn.csv) · [Lịch sử ResNet-18](casting_full_3models_01/history_resnet18.csv) · [Lịch sử MobileNetV2](casting_full_3models_01/history_mobilenetv2.csv) (cột `epoch` đếm từ 0)

### Đặt ba đường validation loss cạnh nhau thì thấy gì? (Hình 3.4, mục 3.3.5)

> Hình 3.4 vẽ validation loss ba mô hình trên cùng trục, thang logarit. Sau epoch 1, MobileNetV2 0,0923, ResNet-18 0,2464, CNN 0,6838, gần ln 2 ≈ 0,693 là mức của bộ phân loại gán 0,5 cho mọi ảnh. Thứ tự MobileNetV2 < ResNet-18 < CNN giữ ở 28 trên 30 epoch, trừ epoch 14 và 17. Vì vậy việc chọn MobileNetV2 không phụ thuộc một epoch riêng lẻ.

Validation loss thấp hơn train loss ở 26/30 epoch với CNN, nhưng chỉ 11 với ResNet-18 và 4 với MobileNetV2. Ở epoch 30, validation loss MobileNetV2 gấp khoảng 3,9 lần train loss: đầu phân loại khớp train tốt hơn mà validation gần như không giảm thêm.

### Vì sao validation đôi lúc tốt hơn train?

Train được tính trong lúc trọng số đang thay đổi và Dropout đang bật. Validation dùng trọng số cuối epoch và tắt Dropout (mục 3.3.2). [Keras về cách huấn luyện và đánh giá](https://keras.io/api/models/model_training_apis/)

### Vì sao chạy 30 epoch nhưng dùng epoch 21?

> 30 epoch là số lượt học nhóm đặt chung cho ba mô hình. Sau mỗi epoch, ModelCheckpoint giữ bản có validation loss thấp nhất. MobileNetV2 có loss thấp nhất ở epoch 21, nên nhóm dùng bản đó để kiểm tra trên test và chạy demo.

Có hai quyết định. Trước hết chọn **một checkpoint cho mỗi mô hình** (theo validation loss của chính nó). Sau đó so validation loss của ba checkpoint để chọn **mô hình đề xuất**. MobileNetV2 có giá trị thấp nhất (mục 2.4.5, 3.3.1). [Bảng kết quả](casting_full_3models_01/comparison.csv) · [Kết quả chọn](casting_full_3models_01/selection.json)

### Vì sao không dùng early stopping?

> Nhóm tắt early stopping để lưu đủ lịch sử 30 epoch cho cả ba mô hình. ModelCheckpoint theo validation loss đã giữ bản tốt nhất, tức đã làm vai trò chọn điểm dừng. TerminateOnNaN chỉ dừng khi loss bị lỗi số (mục 2.4.3).

### Vì sao chọn theo validation loss mà không theo validation accuracy?

> Accuracy trên 986 ảnh chỉ đổi theo bước khoảng 0,1 điểm phần trăm và dễ bằng nhau. Loss thay đổi liên tục theo xác suất nên phân biệt tốt hơn (mục 2.4.5).

Ví dụ trong output: MobileNetV2 có validation accuracy cao nhất ở epoch 25 (99,59%), nhưng validation loss ở đó là 0,0155, vẫn cao hơn 0,0150 của epoch 21.

## 2. Dữ liệu và cách chia tập (mục 2.2, 3.2)

### Dữ liệu lấy từ đâu? Nhóm làm gì với dữ liệu?

> Nhóm dùng bộ Casting Product Image Data for Quality Inspection trên Kaggle, hai nhãn lỗi và đạt, thư mục `casting_data` có 7.348 ảnh 300 × 300: 6.633 ảnh train và 715 ảnh test. Nhóm tìm thấy 64 ảnh đạt trong train trùng pixel với ảnh test, nên bỏ các bản ở train. Phần train còn lại 6.569 ảnh được chia thành 5.583 ảnh train và 986 ảnh validation. Tập test giữ nguyên 715 ảnh.

| Tập (Bảng 3.1) | Lỗi | Đạt | Tổng | Vai trò (Bảng 2.3) |
|---|---:|---:|---:|---|
| Train | 3.194 | 2.389 | 5.583 | Cập nhật trọng số |
| Validation | 564 | 422 | 986 | Giảm learning rate, chọn checkpoint và mô hình |
| Test | 453 | 262 | 715 | Đánh giá cấu hình đã chọn |

Chuỗi xử lý vẽ ở **Hình 2.1**; số ảnh qua từng bước kiểm tra ở **Bảng 2.2** (7.348 nguồn, 0 ảnh hỏng, 64 ảnh đạt trùng bị loại, 6.569 train sạch với 57,21% ảnh lỗi). Loại trùng làm trước khi chia validation, nên validation không chứa ảnh trùng test.

### Vì sao loại ảnh trùng? Vì sao giữ test gốc? Seed 42 dùng để làm gì?

Nếu cùng một ảnh nằm ở cả train và test, kết quả test cao lên vì mô hình gặp lại ảnh đã học. Nhóm băm SHA-256 trên kích thước và pixel ảnh xám trước resize để tìm bản sao chính xác. So pixel thay vì so tên tệp, vì ảnh cùng tên ở train và test có thể khác nội dung: còn 80 cặp như vậy (44 lỗi, 36 đạt), so theo tên sẽ loại nhầm 80 ảnh train (mục 2.2.2).

Nhóm đọc test trước train và loại bản ở phía train, để tập test do nguồn cung cấp giữ nguyên 715 ảnh. Sau xử lý, ba tập không còn ảnh trùng pixel (mục 3.2). Ảnh gần giống do xoay, đổi sáng, hoặc nhiều ảnh của cùng một sản phẩm vẫn chưa xác định được khi thiếu thông tin ảnh gốc (mục 3.5.4).

Seed 42 giúp tái lập cách chia. Chia phân tầng (stratify) giữ tỷ lệ lỗi/đạt ở train và validation cùng khoảng 57,2%. Test giữ tỷ lệ nguồn, khoảng 63,4% ảnh lỗi. Nguồn đã có ảnh tăng cường; nhóm không thêm augmentation. [Thống kê sau chia tập](casting_full_3models_01/split_counts.csv)

### Vì sao dùng ảnh xám? Vì sao hai mô hình học trước cần ba kênh?

> Ảnh nguồn lưu ba kênh nhưng ba kênh bằng nhau, tức nội dung đã là ảnh xám. Chuyển về một kênh không mất thông tin và giảm tính toán (mục 2.2.1).
> CNN nhận trực tiếp ảnh xám 256 × 256 × 1. ResNet-18 và MobileNetV2 học trên ImageNet với đầu vào 224 × 224 × 3, nên nhóm resize về 224 và lặp kênh xám ba lần để khớp đầu vào backbone. Lặp kênh không thêm thông tin mới; thu nhỏ về 224 có thể làm mờ vết lỗi rất nhỏ (mục 2.2.4).

## 3. CNN hoạt động thế nào? (mục 2.3.2)

### Em giải thích kiến trúc CNN của nhóm được không?

> CNN nhận ảnh xám, đưa pixel về 0–1. Ba tầng tích chập 32, 64, 128 bộ lọc tìm đặc trưng. Sau mỗi tầng, MaxPooling thu nhỏ bản đồ đặc trưng một nửa. Cuối mạng, Global Average Pooling gom 128 bản đồ thành 128 số, rồi Dense 64, Dropout 0,3 và Dense 1 sigmoid cho điểm lớp đạt. Tổng 100.993 tham số, học toàn bộ từ đầu (Bảng 2.5, Hình 2.3).

| Thành phần | Giải thích nhập môn |
|---|---|
| Conv2D | Bộ lọc 3 × 3 quét qua ảnh để tìm các kiểu đặc trưng. Nhóm dùng 32, 64, 128 bộ lọc. |
| ReLU | Giữ giá trị dương, đưa giá trị âm về 0, giúp mạng học quan hệ phi tuyến. |
| MaxPooling | Giữ giá trị lớn nhất trong ô 2 × 2. Ba lần pooling: 256 → 128 → 64 → 32. |
| Global Average Pooling | Lấy trung bình mỗi bản đồ đặc trưng: 128 bản đồ → 128 số. |
| Dense | Kết hợp các đặc trưng để phân loại. |
| Dropout | Khi học, tạm tắt ngẫu nhiên 30% giá trị để mạng bớt phụ thuộc vài đặc trưng. Khi dự đoán thì tắt. |
| Sigmoid | Đưa đầu ra về 0–1, làm điểm lớp đạt `p_ok` (công thức 2.4). |

### Mô hình học từ nhãn như thế nào?

> Ảnh đi qua mô hình để tạo dự đoán. Hàm mất mát so dự đoán với nhãn thật. Gradient cho biết đổi mỗi trọng số thì loss thay đổi thế nào. Adam dùng gradient để điều chỉnh trọng số qua từng batch (mục 2.4.2).

**Learning rate** là độ lớn bước cập nhật. Lượt chính dùng Adam với learning rate khởi đầu **0,001**, giá trị mặc định do tác giả Adam đề xuất. Khi validation loss không cải thiện trong 3 epoch, ReduceLROnPlateau giảm learning rate còn một nửa, thấp nhất 10⁻⁶. Lịch sử không ghi learning rate theo epoch. [Notebook và cấu hình huấn luyện](Casting.ipynb)

### Vì sao MobileNetV2 ít tham số tổng hơn ResNet-18 nhưng tham số được cập nhật lại nhiều hơn?

> MobileNetV2 có khoảng 2,34 triệu tham số, nhẹ hơn 11,22 triệu của ResNet-18, nhờ tích chập tách biệt theo chiều sâu.
> Nhưng trước GAP, MobileNetV2 có **1.280 kênh đặc trưng**, ResNet-18 chỉ có **512 kênh**.
> Dense 64 của MobileNetV2 nhận 1.280 đầu vào: (1.280 + 1) × 64 = 82.048, cộng 65 của Dense cuối thành **82.049**. Với ResNet-18: (512 + 1) × 64 = 32.832, cộng 65 thành **32.897** (Bảng 2.8).

### Khối inverted residual của MobileNetV2 khác khối residual của ResNet-18 thế nào?

> **ResNet-18 dùng basic block** (Hình 2.4): hai Conv 3 × 3 kèm Batch Normalization, **giữ nguyên số kênh** (ví dụ 64 → 64), cộng với đầu vào qua đường nối tắt rồi ReLU: y = ReLU(F(x) + x). Đường tắt giúp gradient truyền thẳng về tầng trước, giảm gradient biến mất. Khi đổi số kênh hoặc độ phân giải, đường tắt dùng phép chiếu.
> **MobileNetV2 dùng inverted residual** (Hình 2.6, ví dụ block_7), theo hướng mỏng → dày → mỏng:
> 1. Conv 1 × 1 mở rộng số kênh 6 lần (64 → 384).
> 2. Depthwise Conv 3 × 3 lọc riêng từng kênh. Cùng với bước 1 × 1, tích chập tách biệt cần ít phép tính hơn Conv 3 × 3 thường khoảng 8–9 lần.
> 3. Conv 1 × 1 nén về 64 kênh, **không dùng ReLU** (linear bottleneck), để tránh mất thông tin khi nén về ít kênh.
> 4. Nối tắt giữa hai đầu ít kênh, vì hai đầu cùng kích thước.
>
> Gọi là "ngược" vì ngược với bottleneck của ResNet-50 trở lên (dày → mỏng → dày). **ResNet-18 không dùng bottleneck đó**, mà dùng basic block giữ nguyên kênh.

### Ba mô hình có được so sánh trong cùng điều kiện?

> Cả ba dùng chung danh sách ảnh train, validation, test, cùng 30 epoch, batch 32, Adam, ngưỡng 0,5 và cách đánh giá. Khác nhau ở kiến trúc, nguồn khởi tạo (ngẫu nhiên hay ImageNet), độ phân giải (256 hay 224) và phần tham số được học. Kết quả là so sánh ba cấu hình hoàn chỉnh; tác động riêng của từng yếu tố cần thí nghiệm khác để đo (mục 2.4.1, 3.4.3). Bảng 2.10 liệt kê đủ: nửa trên là yếu tố giữ chung, nửa dưới là yếu tố khác (phần trích đặc trưng, đầu vào, khởi tạo, 128/512/1.280 đặc trưng vào đầu phân loại, 0/11.186.112/2.257.984 tham số cố định).

## 4. Đọc kết quả test và ma trận nhầm lẫn (mục 3.4)

### Kết quả trên 715 ảnh test là gì?

| Mô hình | Ảnh đúng | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi | Macro F1 | Bỏ sót (FN) | Báo nhầm (FP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 678 | 94,83% | 97,27% | 94,48% | 95,86% | 94,48% | 25 | 12 |
| ResNet-18 | 706 | 98,74% | 99,55% | 98,45% | 99,00% | 98,65% | 7 | 2 |
| MobileNetV2 | 711 | 99,44% | 99,56% | 99,56% | 99,56% | 99,40% | 2 | 2 |

(Bảng 3.3 và 3.5.)

> MobileNetV2 dự đoán đúng 711 trên 715 ảnh. Trong 4 ảnh sai, 2 ảnh lỗi bị dự đoán đạt và 2 ảnh đạt bị báo lỗi. So với ResNet-18, MobileNetV2 bỏ sót ít hơn 5 ảnh với cùng số báo nhầm; so với CNN, bỏ sót ít hơn 23 ảnh và báo nhầm ít hơn 10 ảnh.

### Còn lớp đạt thì sao? (Bảng 3.4)

> CNN có precision lớp đạt chỉ 90,91%: 275 ảnh được dự đoán đạt có 25 ảnh lỗi. MobileNetV2 có FN = FP = 2 nhưng F1 đạt 99,24% thấp hơn F1 lỗi 99,56%, vì lớp đạt chỉ 262 ảnh nên mỗi ảnh sai chiếm tỷ lệ lớn hơn. Cả ba mô hình có F1 đạt < F1 lỗi, nên macro F1 thấp hơn F1 lỗi.

### MobileNetV2 hơn ResNet-18 5 ảnh. Đủ để nói tốt hơn không? (mục 3.4.3)

> Chưa đủ. Chỉ 11 ảnh hai mô hình bất đồng: ResNet-18 sai riêng 8, MobileNetV2 sai riêng 3. Nếu hai mô hình ngang nhau, mỗi ảnh rơi về một bên với xác suất 0,5; xác suất lệch đến mức này (tính cả hai phía) khoảng 0,23. So với CNN thì rõ: 35 so với 7 và 36 so với 3, xác suất đều nhỏ hơn 10⁻⁴.

Nếu thầy hỏi **"phép tính này là gì?"**: "Dạ đây là kiểm định McNemar dạng chính xác (kiểm định nhị thức trên các ảnh hai mô hình bất đồng); báo cáo mô tả bằng lời, không nêu tên."

### Chi phí của ba cấu hình thế nào? (Bảng 3.6)

> Thời gian 30 epoch: 300,36 / 990,97 / 557,11 giây; tệp .keras 1,26 / 45,35 / 10,61 MB; ảnh test sai 37 / 9 / 4. ResNet-18 học ít tham số nhất nhưng chậm nhất, khoảng 1,78 lần MobileNetV2, vì backbone 11.186.112 tham số cố định vẫn chạy xuôi mỗi batch. Nhóm chưa đo thời gian dự đoán một ảnh.

### Precision và recall khác nhau ở câu hỏi nào?

Trong bài này, **lỗi** là lớp nhóm quan tâm (mục 2.4.4). Với MobileNetV2, có 453 ảnh thật sự lỗi và mô hình phát hiện đúng 451 ảnh.

- **Recall lỗi** = TP / (TP + FN): "trong số ảnh thật sự lỗi, phát hiện được bao nhiêu?". 451 / 453.
- **Precision lỗi** = TP / (TP + FP): "trong số ảnh bị báo lỗi, bao nhiêu ảnh lỗi thật?". Cũng 451 / 453, vì mô hình báo đúng 451 ảnh lỗi và báo nhầm 2 ảnh đạt.
- **F1** gộp precision và recall bằng trung bình điều hòa. **Macro F1** lấy trung bình F1 của hai lớp để mỗi lớp có trọng số ngang nhau.

Công thức (2.5): accuracy và precision; công thức (2.6): recall và F1.

Nếu giảng viên chỉ vào ma trận nhầm lẫn của MobileNetV2 (Hình 3.7), đọc **hàng là nhãn thật, cột là nhãn dự đoán**:

| Nhãn thật | Dự đoán lỗi | Dự đoán đạt |
|---|---:|---:|
| Lỗi | 451 phát hiện đúng (TP) | 2 bỏ sót (FN) |
| Đạt | 2 báo lỗi nhầm (FP) | 260 nhận đúng (TN) |

Trong kiểm tra chất lượng, cần chú ý 2 ảnh lỗi bị dự đoán đạt, vì đó là lỗi bị bỏ sót. [Ma trận và chỉ số MobileNetV2](casting_full_3models_01/metrics_mobilenetv2.json)

### Test lệch lớp (453 lỗi, 262 đạt) thì accuracy còn ý nghĩa không?

> Một mô hình đoán toàn "lỗi" đã đạt khoảng 63,4%. Vì vậy báo cáo dùng thêm precision, recall, F1 lớp lỗi, macro F1 và ma trận nhầm lẫn, không chỉ dựa vào accuracy (mục 2.4.4, 3.2).

### Bỏ sót lỗi (FN) hay báo nhầm (FP) nguy hiểm hơn? Nếu áp dụng thực tế, nhóm chỉnh gì?

> **FN** nghiêm trọng hơn: sản phẩm lỗi bị cho qua khâu kiểm tra. **FP** làm sản phẩm đạt bị đưa đi kiểm tra lại, tốn thêm công (mục 3.4.2).
>
> Nhóm dùng ngưỡng 0,5, tức chọn lớp có điểm lớn hơn. Mục 3.4.4 và Bảng 3.7 cho thấy đánh đổi khi đổi ngưỡng; đây là phân tích bổ sung trên test, không dùng để chọn ngưỡng.
> - Quy tắc là **p_ok ≥ ngưỡng thì đạt**. Muốn khó công nhận "đạt" hơn thì phải **TĂNG** ngưỡng, không phải hạ. Hạ ngưỡng làm **bỏ sót nhiều hơn**.
> - Tăng ngưỡng giảm FN, tăng recall lớp lỗi, đổi lại FP tăng.
> - Ngưỡng phải chọn trên **validation** theo chi phí của nhà máy, rồi mới báo FN, FP trên test. Lượt chạy chỉ lưu dự đoán 10 ảnh validation, nên nhóm chưa làm bước này.

Số liệu Bảng 3.7 cần nhớ: tại 0,5 là FN/FP = 25/12 (CNN), 7/2 (ResNet-18), 2/2 (MobileNetV2). MobileNetV2 giữ FN = 1 suốt 0,6–0,95 vì `cast_def_0_150.jpeg` có p_ok 0,9556; tại 0,96 thì FN = 0, FP lên 9. CNN chỉ đạt FN = 0 từ t = 0,96 (trong các ngưỡng của bảng), khi đó báo nhầm 229/262 ≈ 87,4% ảnh đạt. ResNet-18 cần ngưỡng > 0,9878 (p_ok của `cast_def_0_2177.jpeg`); tại 0,99 báo nhầm 36 ảnh.

### Vì sao ảnh `cast_def_0_150.jpeg` bị đoán đạt với điểm cao?

> Nhóm chưa xác định được. Cả ba mô hình đều đoán đạt (p_ok 0,7087 ở CNN, 0,9799 ở ResNet-18, 0,9556 ở MobileNetV2); đây là ảnh test duy nhất cả ba cùng sai, và nhìn bằng mắt nhóm không thấy vết lỗi rõ. Dữ liệu không có chú giải vị trí lỗi; muốn tìm hiểu thêm cần công cụ giải thích như Grad-CAM, nhóm chưa làm (mục 3.5.1, Bảng 3.8).

### Ba mô hình có sai cùng những ảnh không? (Bảng 3.9)

> Gần như không. Chỉ `cast_def_0_150.jpeg` sai ở cả ba; chỉ `cast_def_0_1591.jpeg` sai ở hai mô hình (CNN và ResNet-18). 47 ảnh sai ở ít nhất một mô hình, 45 ảnh chỉ sai ở một mô hình, 668 ảnh đúng ở cả ba. 16 ảnh FP là 16 ảnh khác nhau. Phần lớn ảnh sai gắn với từng mô hình.

### Hình 3.11 (phân bố p_ok) nói gì?

> MobileNetV2 tách hai lớp rõ nhất: 448/453 ảnh lỗi có p_ok < 0,05, 253/262 ảnh đạt có p_ok > 0,95, chỉ 7 ảnh trong khoảng 0,2–0,8. CNN có 76 ảnh trong khoảng đó, 28 ảnh sai. Mọi ảnh có độ tự tin ≥ 0,99 đều đúng, nhưng vẫn có ảnh sai dù độ tự tin ≥ 0,9 (6 / 3 / 1). Điểm chưa hiệu chuẩn nên không đọc là xác suất thật.

### Nhóm ảnh `new__0_*` là gì?

> Chỉ nói số liệu: test có 37 ảnh lỗi tên dạng `new__0_*`, mẫu tên chỉ có trong test. CNN bỏ sót 7/37 (18,9%), so với 18/416 ≈ 4,3% ở các ảnh lỗi còn lại; ResNet-18 và MobileNetV2 không sai ảnh nào trong nhóm. Nhóm không suy đoán nguyên nhân.

### Kết quả này nói được gì về dùng ngoài thực tế?

> 99,44% là kết quả trên 715 ảnh test cùng nguồn, một seed. Nguồn có sẵn ảnh tăng cường và không có mã sản phẩm gốc, nên biến thể gần giống có thể nằm ở cả train và test; kết quả có thể lạc quan so với ảnh mới. Nếu đổi sản phẩm, camera hoặc ánh sáng, cần ảnh từ điều kiện mới để đánh giá lại. Mô hình trả nhãn cho toàn ảnh, không định vị khuyết tật (mục 3.5.4).

## 5. Demo và cách giải thích điểm dự đoán (mục 3.5.2, 3.5.3)

### Demo lấy mô hình ở đâu? Dự đoán bằng cách nào?

> Demo dùng ba checkpoint đã chọn: CNN epoch 29, ResNet-18 epoch 29, MobileNetV2 epoch 21, lưu dạng `.keras` kèm metadata. Khi có ảnh mới, chương trình xử lý ảnh giống lúc đánh giá: EXIF → xám → 256 × 256; với hai mạng học trước thì resize 224 và lặp ba kênh. Mô hình trả `p_ok`. Từ 0,5 trở lên là đạt, dưới 0,5 là lỗi. Người dùng chọn một hoặc cả ba mô hình để so sánh trên cùng ảnh, và tải kết quả CSV/JSON.

Quy ước **lỗi = 0, đạt = 1** là mã nhãn để lưu dữ liệu. Khi tính precision, recall, nhóm vẫn chọn lỗi làm lớp cần theo dõi.

### Kết quả demo trên hai ảnh mẫu (Hình 3.12, 3.13)

| Ảnh | CNN p_ok | ResNet-18 p_ok | MobileNetV2 p_ok | Nhãn |
|---|---:|---:|---:|---|
| cast_ok_0_45.jpeg | 0,848358 | 0,999593 | 0,998192 | Đạt (cả ba) |
| cast_def_0_1059.jpeg | 0,000039 | 0,003175 | 0,000013 | Lỗi (cả ba) |

Cùng một ảnh, ba mô hình cho cùng nhãn nhưng mức điểm khác nhau. Nhãn trùng với kết quả đã lưu khi đánh giá test.

### Vì sao ảnh demo hiện 99,82%, còn kết quả test là 99,44%?

> 99,82% (0,998192) là điểm lớp đạt MobileNetV2 cho riêng ảnh `cast_ok_0_45.jpeg`. 99,44% là tỷ lệ đúng: 711 ảnh đúng trên 715 ảnh test. Hai số đo hai thứ khác nhau. Điểm sigmoid cũng chưa được hiệu chuẩn, nên 99,82% không có nghĩa là "chắc chắn đúng 99,82%".

### Kiểm chứng 10 ảnh trên CPU có ý nghĩa gì?

> Nhóm nạp lại tệp mô hình trên CPU và so với dự đoán đã lưu từ Colab (Bảng 3.10). MobileNetV2 được đối chiếu trên 10 ảnh validation; cả ba mô hình được đối chiếu trên 10 ảnh test (5 lỗi, 5 đạt). Cả bốn phép đều trùng 10/10 nhãn; sai lệch điểm lớn nhất là 3,1825 × 10⁻⁶ (ResNet-18), trong dung sai tương đối 10⁻³ và tuyệt đối 10⁻⁴. Phép này cho biết tệp nạp lại cho cùng kết quả, không phải đo độ chính xác.

[Output minh họa trong notebook](Casting.ipynb) · [Kiểm chứng MobileNetV2](casting_full_3models_01/demo/verification.json) · [Kiểm chứng ba mô hình](demo/models/comparison_verification.json)

## 6. Bài nói mẫu khi tới phần thực nghiệm

> Nhóm phân loại ảnh sản phẩm đúc thành hai lớp lỗi và đạt. Sau khi loại 64 ảnh trùng giữa train và test, nhóm dùng 5.583 ảnh để huấn luyện, 986 ảnh để chọn checkpoint và 715 ảnh để kiểm tra.
>
> Nhóm dùng CNN học từ đầu làm mốc. ResNet-18 và MobileNetV2 giữ cố định backbone ImageNet, chỉ học đầu phân loại, nên đạt tỷ lệ đúng cao sớm hơn. Cả ba chạy 30 epoch, chọn checkpoint theo validation loss thấp nhất.
>
> MobileNetV2 được chọn ở epoch 21. Cuối lượt, train loss tiếp tục giảm trong khi validation loss cao hơn một chút, gợi ý quá khớp nhẹ. Trên tập test, mô hình đúng 711 trong 715 ảnh, bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt. Demo cho so sánh cả ba mô hình trên cùng ảnh.

Nếu chỉ có thời gian ôn vài câu, hãy thử nói lại bằng lời của mình:

1. Train, validation và test dùng để làm gì?
2. Vì sao phải kiểm ảnh trùng, và SHA-256 bắt được loại trùng nào?
3. Vì sao CNN học chậm hơn hai mô hình học chuyển giao?
4. Dựa vào đâu để nhận xét quá khớp khi accuracy đã lên 99%?
5. Vì sao chạy 30 epoch mà dùng epoch 21?
6. MobileNetV2 đúng bao nhiêu ảnh, bỏ sót bao nhiêu ảnh lỗi?
7. Muốn bớt bỏ sót thì tăng hay hạ ngưỡng? Vì sao Bảng 3.7 không dùng để chọn ngưỡng?
8. Vì sao chênh 5 ảnh giữa MobileNetV2 và ResNet-18 chưa đủ khẳng định?

Bộ 43 câu hỏi đầy đủ kèm câu trả lời ngắn: [README mục 6](README.md#6-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi) (câu 32–43 là nội dung mới của bản in).
