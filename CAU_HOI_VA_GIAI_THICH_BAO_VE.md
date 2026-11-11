# Câu hỏi và giải thích bảo vệ môn Học sâu

## 1. Đọc kết quả huấn luyện ba mô hình

### Một epoch là gì? Biểu đồ đang đo gì?

Một **epoch** là một lượt mô hình học qua toàn bộ 5.583 ảnh train. Mỗi lần mô hình xử lý một nhóm tối đa 32 ảnh, gọi là **batch**, rồi điều chỉnh các trọng số để dự đoán tốt hơn. Cuối epoch, chương trình đo kết quả trên 986 ảnh validation. Cả ba mô hình chạy 30 epoch.

Ở mục 3.3 của báo cáo, trục ngang là epoch. Đường xanh là train, đường cam là validation. Mỗi hình có hai biểu đồ.

- **Accuracy**, hay tỷ lệ đúng, trả lời câu hỏi “Bao nhiêu ảnh được phân loại đúng?”. Accuracy 0,95 tương ứng 95%.
- **Loss** là giá trị của hàm mất mát. Nó cho biết dự đoán lệch khỏi nhãn thật đến mức nào. Loss càng nhỏ thì mô hình càng gán điểm phù hợp cho nhãn thật.

Ví dụ, ảnh thật sự đạt. Hai dự đoán cho điểm đạt 0,51 và 0,99 đều được tính đúng ở ngưỡng 0,5. Dự đoán 0,99 cho loss thấp hơn vì mô hình gán điểm cao hơn cho nhãn thật. Vì vậy, có thể thấy accuracy gần như đứng yên trong khi loss thay đổi.

### Ba mô hình học như thế nào?

| Mô hình | Phần được học | Accuracy validation sau epoch 1 | Epoch được lưu | Validation loss tại epoch được lưu |
|---|---|---:|---:|---:|
| CNN | Toàn bộ mạng, bắt đầu từ trọng số khởi tạo | 57,20% | 29 | 0,12101 |
| ResNet-18 | Đầu phân loại mới | 95,44% | 29 | 0,02695 |
| MobileNetV2 | Đầu phân loại mới | 97,26% | 21 | 0,01503 |

**Vì sao CNN lên chậm, còn hai mô hình kia đạt hơn 90% sớm?**

> CNN phải học cách tìm đặc trưng trong ảnh và cách dùng chúng để phân loại. ResNet-18 và MobileNetV2 đã có phần trích xuất đặc trưng học trước trên ImageNet. Nhóm giữ phần đó cố định và chỉ dạy đầu phân loại mới cách phân biệt sản phẩm lỗi và đạt.

Có thể hình dung phần trích xuất đặc trưng chuyển ảnh thành một nhóm con số mô tả ảnh. Đầu phân loại đọc các con số đó để đưa ra nhãn. Cách tận dụng phần đã học trước gọi là **học chuyển giao**.

- **CNN.** Tỷ lệ đúng trên validation tăng từ 57,20% sau epoch đầu lên 95,84% tại epoch 29. Các đỉnh dao động thấy rõ ở epoch 9 và 18. [Biểu đồ CNN](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_cnn.png)
- **ResNet-18.** Tỷ lệ đúng trên validation đạt 95,44% sau epoch đầu và 98,78% tại epoch 29. Epoch 30 có tỷ lệ đúng nhỉnh hơn, 98,88%, nhưng validation loss tăng từ 0,02695 lên 0,02946. Nhóm lưu epoch 29 theo tiêu chí loss. [Biểu đồ ResNet-18](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_resnet18.png)
- **MobileNetV2.** Tại epoch 21, validation loss đạt mức thấp nhất là 0,01503 và tỷ lệ đúng là 99,49%. Đây là checkpoint được dùng để đánh giá test và chạy demo. [Biểu đồ MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_mobilenetv2.png)

### Accuracy lên 0,99 có phải quá khớp?

> Em xem kết quả đó thuộc tập nào và theo dõi nhiều epoch. Nếu mô hình tiếp tục dự đoán tốt hơn trên train, còn kết quả validation dần xấu đi, đó là dấu hiệu quá khớp. Accuracy 99% chỉ cho biết tỷ lệ đúng trên tập đang xem.

Với **MobileNetV2**, từ epoch 21 đến 30, train loss giảm từ **0,0119 xuống 0,0046**, còn validation loss kết thúc cao hơn, từ **0,0150 lên 0,0179**. Validation accuracy giảm từ 99,49% xuống 99,39%, tương ứng thêm một ảnh sai trên 986 ảnh. Kết quả gợi ý quá khớp nhẹ về cuối lượt. Ngay cả khi phần trích xuất đặc trưng được giữ cố định, đầu phân loại vẫn có thể quá khớp.

Với **CNN** và **ResNet-18**, đường validation nhìn chung cải thiện trong quá trình học. Có lúc chỉ số dao động, nhưng biểu đồ chưa cho thấy suy giảm kéo dài ở cuối lượt. **Thiếu khớp** là tình huống mô hình còn dự đoán kém ngay trên tập train. Việc CNN đạt điểm thấp hơn hai mô hình học chuyển giao chưa đủ để kết luận thiếu khớp.

### Validation spike là gì và vì sao xuất hiện?

**Validation spike** là một đỉnh tăng đột ngột trên đường validation loss. Để giải thích một đỉnh, hãy đọc loss và số ảnh sai ở ba epoch liên tiếp.

| Mô hình | Epoch trước → đỉnh → sau | Validation loss | Số ảnh sai trên 986 ảnh |
|---|---|---|---|
| CNN | 8 → 9 → 10 | 0,2704 → **0,4488** → 0,2405 | 98 → **216** → 82 |
| CNN | 17 → 18 → 19 | 0,1910 → **0,5188** → 0,1722 | 67 → **281** → 57 |
| ResNet-18 | 18 → 19 → 20 | 0,0302 → **0,0455** → 0,0279 | 15 → **18** → 15 |
| MobileNetV2 | 13 → 14 → 15 | 0,0171 → **0,0696** → 0,0204 | 6 → **18** → 6 |

**Tại sao cùng 986 ảnh mà kết quả thay đổi?** Mỗi batch train làm thay đổi trọng số. Cuối epoch, mô hình dùng trọng số vừa học để dự đoán trên cùng tập validation. Việc cập nhật dựa trên ảnh train, nên ở một số epoch, các dự đoán trên validation có thể tạm thời kém đi. Chẳng hạn, một ảnh lỗi có thể được mô hình cho điểm đạt 0,45 ở epoch trước và 0,55 ở epoch sau. Điểm đã vượt ngưỡng 0,5, khiến nhãn chuyển từ lỗi sang đạt. Đây là ví dụ minh họa cách một ảnh có thể đổi nhãn qua hai epoch.

**Vì sao loss đôi khi tăng mạnh dù accuracy giảm ít?** Loss còn tính mức độ tự tin của dự đoán. Với một ảnh thật sự lỗi, nếu mô hình cho điểm đạt 0,60 rồi tăng lên 0,99 thì cả hai lần đều bị tính sai một ảnh. Lần sau cho loss lớn hơn vì mô hình gán điểm rất thấp cho nhãn thật. Với CNN ở epoch 18, số ảnh sai tăng mạnh nên cả loss lẫn accuracy biến động rõ. Với MobileNetV2 ở epoch 14, accuracy giảm khoảng 1,22 điểm phần trăm trong khi loss tăng khoảng 4 lần. Lịch sử hiện có ghi chỉ số tổng hợp theo epoch, nên nhóm chưa xác định được mức đóng góp của từng ảnh hay nguyên nhân cụ thể của đỉnh.

#### CNN có spike ở epoch 9 và 18

> CNN có hai đỉnh rõ ở epoch 9 và 18. Ở đỉnh lớn nhất, số ảnh sai tăng từ 67 lên 281, rồi giảm còn 57 ở epoch tiếp theo. Vì có nhiều ảnh bị đổi sang nhãn sai ở epoch 18, cả accuracy và loss đều thay đổi mạnh. Nhìn qua nhiều epoch, kết quả validation vẫn cải thiện và nhóm giữ checkpoint epoch 29.

#### ResNet-18 có spike ở epoch 19

> Từ epoch 18 sang 19, validation loss tăng từ 0,0302 lên 0,0455. Số ảnh sai chỉ tăng từ 15 lên 18, rồi về 15 ở epoch 20. Loss còn tính điểm mô hình gán cho nhãn thật, nên loss có thể thay đổi rõ dù số ảnh dự đoán sai chỉ tăng ít. Sau đỉnh, loss giảm về 0,0279 và nhóm giữ checkpoint epoch 29.

#### MobileNetV2 có spike ở epoch 14

> Từ epoch 13 sang 14, validation loss tăng từ 0,0171 lên 0,0696, trong khi số ảnh sai tăng từ 6 lên 18. Sang epoch 15, mô hình lại chỉ sai 6 ảnh. Sau đó validation tiếp tục cải thiện và đạt loss thấp nhất ở epoch 21. Khi nhận xét quá khớp nhẹ, em nhìn đoạn cuối sau epoch 21, khi train loss tiếp tục giảm còn validation loss kết thúc cao hơn.

Nếu được hỏi chính xác vì sao từng đỉnh xảy ra, em có thể nói rằng trọng số thay đổi sau mỗi lượt huấn luyện nên dự đoán trên cùng tập validation cũng thay đổi. Lịch sử lưu loss và accuracy theo epoch, nên nhóm đọc được mức dao động và sự phục hồi, còn nguyên nhân cụ thể của từng đỉnh chưa được xác định.

[Lịch sử CNN](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_cnn.csv) · [Lịch sử ResNet-18](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_resnet18.csv) · [Lịch sử MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_mobilenetv2.csv)

### Vì sao validation đôi lúc tốt hơn train?

Khi validation tốt hơn train ở một số epoch, có thể giải thích bằng cách đo. Train được tính trong lúc trọng số đang thay đổi và Dropout đang bật. Validation dùng trọng số cuối epoch và tắt Dropout. [Keras về cách huấn luyện và đánh giá](https://keras.io/api/models/model_training_apis/)

### Vì sao chạy 30 epoch nhưng dùng epoch 21?

> 30 epoch là số lượt học nhóm đặt cho cả ba mô hình. Sau mỗi lượt, nhóm kiểm tra validation và giữ bản mô hình có loss thấp nhất. Bản lưu đó gọi là checkpoint. MobileNetV2 cho loss thấp nhất ở epoch 21, nên nhóm dùng bản lưu ở đó để kiểm tra trên test.

Có hai quyết định cần nhớ. Trước hết chọn **một checkpoint cho mỗi mô hình**. Sau đó so sánh validation loss của ba checkpoint để chọn mô hình dùng cho demo. MobileNetV2 có giá trị thấp nhất trong lần thực nghiệm này. [Bảng kết quả](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/comparison.csv) · [Kết quả chọn checkpoint](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/selection.json)

## 2. Dữ liệu và cách chia tập

### Dữ liệu lấy từ đâu? Nhóm làm gì với dữ liệu?

> Nhóm dùng bộ ảnh sản phẩm đúc trên Kaggle, gồm hai nhãn là lỗi và đạt. Bộ dữ liệu ban đầu có 6.633 ảnh trong thư mục train và 715 ảnh test. Nhóm tìm thấy 64 ảnh train trùng chính xác với ảnh test, nên bỏ các bản sao đó khỏi danh sách huấn luyện. Sau đó nhóm chia phần train còn lại thành 5.583 ảnh train và 986 ảnh validation. Tập test vẫn có 715 ảnh.

| Tập | Số ảnh | Vai trò |
|---|---:|---|
| Train | 5.583 | Dùng ảnh và nhãn để cập nhật trọng số |
| Validation | 986 | Theo dõi huấn luyện và chọn checkpoint |
| Test | 715 | Đo kết quả của checkpoint đã chọn |

### Vì sao loại ảnh trùng? Seed 42 dùng để làm gì?

Nếu cùng một ảnh xuất hiện trong train và test, kết quả test có thể tăng vì mô hình gặp lại ảnh từng học. Nhóm so sánh pixel để loại bản sao chính xác. Những ảnh gần giống do biến đổi hoặc nhiều ảnh chụp cùng một sản phẩm vẫn khó xác định khi thiếu thông tin ảnh gốc.

Nhóm dùng seed 42 để có thể tái lập cách chia. Cách chia theo tỷ lệ lớp giúp train và validation có tỷ lệ ảnh lỗi, ảnh đạt gần với tập đem chia. Dữ liệu nguồn đã có ảnh tăng cường. Trong lượt này, nhóm dùng trực tiếp các ảnh đã chia và áp dụng các bước tiền xử lý đầu vào. [Thống kê sau chia tập](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/split_counts.csv)

### Vì sao dùng ảnh xám? Vì sao hai mô hình cần ba kênh?

> Nhóm đưa ảnh về cùng kích thước để mô hình có đầu vào thống nhất. CNN nhận ảnh xám 256 × 256 với một kênh. ResNet-18 và MobileNetV2 nhận đầu vào 224 × 224 với ba kênh, nên nhóm đổi kích thước rồi lặp cùng kênh xám ba lần để phù hợp đầu vào của mô hình học trước.

## 3. CNN hoạt động thế nào?

### Em giải thích kiến trúc CNN của nhóm được không?

> CNN nhận ảnh xám và đưa giá trị điểm ảnh về khoảng 0 đến 1. Ba lớp tích chập tìm đặc trưng trên ảnh. Sau mỗi lớp, pooling thu nhỏ các bản đồ đặc trưng. Cuối mạng, các đặc trưng được gom lại rồi đưa vào đầu phân loại để dự đoán sản phẩm lỗi hay đạt.

| Thành phần | Giải thích nhập môn |
|---|---|
| Conv2D | Dùng các bộ lọc nhỏ quét qua ảnh để tìm các kiểu đặc trưng. Nhóm dùng lần lượt 32, 64 và 128 bộ lọc. |
| ReLU | Giữ giá trị dương và đưa giá trị âm về 0, giúp mạng học các quan hệ phức tạp hơn. |
| MaxPooling | Giữ giá trị lớn nhất trong từng vùng nhỏ. Ba lần pooling giảm chiều cao và chiều rộng từ 256 xuống 128, 64 rồi 32. |
| Global Average Pooling | Lấy trung bình mỗi bản đồ đặc trưng. Sau lớp này, 128 bản đồ được tóm tắt thành 128 số. |
| Dense | Kết hợp các đặc trưng để phân loại. |
| Dropout | Tạm tắt ngẫu nhiên một phần giá trị trong lúc học để mạng bớt phụ thuộc vào một vài đặc trưng. |
| Sigmoid | Đưa đầu ra về khoảng 0 đến 1 để nhóm dùng làm điểm dự đoán lớp đạt. |

### Mô hình học từ nhãn như thế nào?

> Ảnh đi qua mô hình để tạo dự đoán. Hàm mất mát so dự đoán với nhãn thật. Gradient cho biết thay đổi các trọng số sẽ tác động đến loss thế nào. Adam dùng thông tin đó để điều chỉnh trọng số qua từng batch.

**Learning rate** điều chỉnh độ lớn của bước cập nhật. Lượt chính dùng Adam với learning rate khởi đầu **0,001**. Khi validation loss không cải thiện đủ trong 3 epoch, ReduceLROnPlateau giảm learning rate còn một nửa. Cơ chế lưu checkpoint theo validation loss hoạt động sau mỗi epoch. [Notebook và cấu hình huấn luyện](D:/Casting-Defect-Detection-CNN/Casting.ipynb)

### Ba mô hình có được so sánh trong cùng điều kiện?

> Cả ba dùng chung danh sách ảnh train, validation và test, cùng 30 epoch và batch size 32. CNN học từ đầu, còn hai mô hình kia tận dụng phần đã học trên ImageNet. Kết quả là so sánh ba cấu hình hoàn chỉnh. Tác động riêng của kiến trúc và trọng số học trước cần một thí nghiệm riêng để đo.

## 4. Đọc kết quả test và ma trận nhầm lẫn

### Kết quả trên 715 ảnh test là gì?

| Mô hình | Ảnh đúng | Accuracy test | Ảnh lỗi bị bỏ sót | Ảnh đạt bị báo lỗi |
|---|---:|---:|---:|---:|
| CNN | 678 | 94,83% | 25 | 12 |
| ResNet-18 | 706 | 98,74% | 7 | 2 |
| MobileNetV2 | 711 | 99,44% | 2 | 2 |

> MobileNetV2 dự đoán đúng 711 trên 715 ảnh. Trong 4 ảnh sai, 2 ảnh thật sự lỗi bị dự đoán đạt và 2 ảnh thật sự đạt bị báo lỗi.

### Precision và recall khác nhau ở câu hỏi nào?

Trong bài này, **lỗi** là lớp nhóm quan tâm. Với MobileNetV2, có 453 ảnh thật sự lỗi và mô hình phát hiện đúng 451 ảnh.

- **Recall lỗi** hỏi “Trong số ảnh thật sự lỗi, phát hiện được bao nhiêu?”. Kết quả là 451 trên 453.
- **Precision lỗi** hỏi “Trong số ảnh bị mô hình báo lỗi, bao nhiêu ảnh thực sự lỗi?”. Kết quả cũng là 451 trên 453, vì mô hình báo đúng 451 ảnh lỗi và báo nhầm 2 ảnh đạt.
- **F1** gộp precision và recall vào một chỉ số. **Macro F1** lấy trung bình F1 của hai lớp để mỗi lớp có trọng số ngang nhau.

Nếu giảng viên chỉ vào ma trận nhầm lẫn, hãy đọc **hàng là nhãn thật, cột là nhãn dự đoán**. Với lớp lỗi là lớp cần phát hiện, bốn ô của MobileNetV2 được đọc như sau.

| Nhãn thật | Dự đoán lỗi | Dự đoán đạt |
|---|---:|---:|
| Lỗi | 451 phát hiện đúng (TP) | 2 bỏ sót (FN) |
| Đạt | 2 báo lỗi nhầm (FP) | 260 nhận đúng ảnh đạt (TN) |

Trong kiểm tra chất lượng, cần chú ý 2 ảnh lỗi bị dự đoán đạt vì đó là lỗi bị bỏ sót. [Ma trận và chỉ số MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/metrics_mobilenetv2.json)

### Kết quả này nói được gì về dùng ngoài thực tế?

> 99,44% là kết quả trên 715 ảnh test cùng nguồn dữ liệu. Nếu thay sản phẩm, camera hoặc ánh sáng, nhóm cần ảnh từ điều kiện mới để đánh giá lại. Mô hình hiện trả nhãn cho toàn ảnh. Bài toán xác định vị trí khuyết tật cần dữ liệu và cách đánh giá theo vùng lỗi.

## 5. Demo và cách giải thích điểm dự đoán

### Demo lấy mô hình ở đâu? Dự đoán bằng cách nào?

> Nhóm lưu MobileNetV2 ở epoch 21 dưới dạng .keras. Khi có ảnh mới, chương trình xử lý ảnh giống lúc đánh giá rồi đưa vào mô hình đã lưu. Mô hình trả điểm p_ok cho lớp đạt. Từ 0,5 trở lên, chương trình gán nhãn đạt. Dưới 0,5, chương trình gán nhãn lỗi.

Quy ước của nhóm là **lỗi = 0, đạt = 1**. Mã nhãn dùng để lưu dữ liệu. Khi tính recall hay precision cho lỗi, nhóm vẫn chọn lỗi làm lớp cần theo dõi.

### Vì sao trên ảnh demo hiện 99,82%, còn kết quả test là 99,44%?

> Với ảnh cast_ok_0_45.jpeg trong notebook, 99,82% là điểm dự đoán lớp đạt của riêng ảnh đó. Accuracy 99,44% được tính từ 711 ảnh đúng trên toàn bộ 715 ảnh test.

### Kiểm chứng 10 ảnh trên CPU có ý nghĩa gì?

> Nhóm nạp lại tệp mô hình trên CPU và đối chiếu với kết quả dự đoán đã lưu trên 10 ảnh validation. Cả 10 ảnh cho cùng nhãn. Phép kiểm tra cho biết tệp mô hình sau khi nạp lại giữ được kết quả dự đoán trên các ảnh được đối chiếu.

[Output minh họa trong notebook](D:/Casting-Defect-Detection-CNN/Casting.ipynb) · [Kết quả kiểm chứng CPU](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/demo/verification.json)

## 6. Bài nói mẫu khi tới phần thực nghiệm

> Nhóm phân loại ảnh sản phẩm đúc thành hai lớp lỗi và đạt. Sau khi loại ảnh trùng giữa train và test, nhóm dùng 5.583 ảnh để huấn luyện, 986 ảnh để chọn checkpoint và 715 ảnh để kiểm tra.
>
> Nhóm dùng CNN học từ đầu làm mốc so sánh. ResNet-18 và MobileNetV2 học đầu phân loại trên các đặc trưng đã có từ ImageNet, nên đạt tỷ lệ đúng cao sớm hơn. Cả ba chạy 30 epoch, và nhóm chọn checkpoint theo validation loss thấp nhất.
>
> MobileNetV2 được chọn ở epoch 21. Cuối lượt, loss trên train tiếp tục giảm trong khi validation loss cao hơn một chút, gợi ý quá khớp nhẹ. Trên tập test, mô hình đúng 711 trong 715 ảnh và còn bỏ sót 2 ảnh lỗi. Nhóm dùng chính checkpoint epoch 21 cho demo.

Nếu chỉ có thời gian ôn vài câu, hãy thử nói lại bằng lời của mình.

1. Train, validation và test dùng để làm gì?
2. Vì sao CNN học chậm hơn hai mô hình học chuyển giao?
3. Dựa vào đâu để nhận xét quá khớp khi accuracy đã lên 99%?
4. Vì sao chạy 30 epoch mà demo dùng epoch 21?
5. MobileNetV2 dự đoán đúng bao nhiêu ảnh và bỏ sót bao nhiêu ảnh lỗi?
