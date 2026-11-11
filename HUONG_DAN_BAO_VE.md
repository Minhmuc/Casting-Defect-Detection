# Hướng dẫn hiểu và bảo vệ đề tài phân loại lỗi sản phẩm đúc

Tài liệu này bám báo cáo và kết quả trong `casting_full_3models_01`. Các câu nói mẫu dùng để luyện diễn đạt. Khi trình bày, cần hiểu ý và nói lại bằng lời của mình.

## 1. Nói rõ nhóm đã làm gì

> Nhóm nghiên cứu bài toán phân loại ảnh sản phẩm đúc thành hai lớp lỗi và đạt yêu cầu. Nhóm xây dựng một CNN đơn giản làm mô hình cơ sở, sau đó so sánh với ResNet-18 và MobileNetV2 sử dụng trọng số ImageNet. Ba cấu hình được đánh giá trên cùng tập test. MobileNetV2 được chọn theo validation loss và được lưu để sử dụng trong chương trình nhận ảnh.

Đầu vào là ảnh của một sản phẩm. Đầu ra là nhãn lỗi hoặc đạt. Đây là bài toán phân loại toàn ảnh. Output hiện có chưa xác định vị trí hay loại khuyết tật cụ thể.

Phần nhóm thực hiện gồm kiểm tra dữ liệu trùng, chia train/validation, xây dựng CNN và đầu phân loại cho hai mô hình học trước, huấn luyện ba cấu hình, chọn checkpoint, đánh giá, phân tích ảnh sai và chuẩn bị mô hình cho demo. Dữ liệu và các kiến trúc ResNet-18, MobileNetV2 có nguồn bên ngoài.

## 2. Giải thích Chương 2

### 2.1. Phát biểu bài toán và đầu ra

Với mỗi ảnh x, mô hình tạo một số p_ok trong khoảng 0 đến 1. Nhãn quy ước là 0 cho ảnh lỗi và 1 cho ảnh đạt. Với ngưỡng 0,5, p_ok từ 0,5 trở lên được nhận là đạt, còn dưới ngưỡng được nhận là lỗi.

Ví dụ minh họa, p_ok = 0,8 sẽ cho nhãn đạt. Điểm đầu ra cao vẫn có thể đi kèm dự đoán sai, như trường hợp thực tế ở mục 3.5.1. Mức 99,82% trên ảnh demo cũng khác với accuracy 99,44% tính trên 715 ảnh test.

**Câu nói mẫu**

> Đầu ra sigmoid biểu diễn điểm xác suất của lớp đạt. Nhóm áp dụng cùng ngưỡng 0,5 cho cả ba mô hình để tạo nhãn và so sánh.

### 2.2. Thu thập và tiền xử lý dữ liệu

Nhớ ba vai trò của các tập.

- Train cung cấp ảnh và nhãn để cập nhật trọng số.
- Validation theo dõi quá trình học, chọn checkpoint và chọn mô hình demo.
- Test đánh giá các checkpoint sau khi lựa chọn đã được xác định.

Nguồn có 6.633 ảnh train và 715 ảnh test. Nhóm tìm thấy 64 ảnh train trùng pixel với test, loại những bản sao đó khỏi danh sách train. Còn 6.569 ảnh, nhóm tách 15% làm validation có giữ tỷ lệ hai lớp, seed 42.

| Tập | Ảnh lỗi | Ảnh đạt | Tổng |
|---|---:|---:|---:|
| Train thực dùng | 3.194 | 2.389 | 5.583 |
| Validation | 564 | 422 | 986 |
| Test | 453 | 262 | 715 |

**Câu nói mẫu**

> Nhóm giữ nguyên tập test của nguồn. Sau khi loại 64 bản sao khỏi train, nhóm tách validation để theo dõi huấn luyện. Cả ba mô hình dùng chung danh sách ảnh ở mỗi tập.

Khi được hỏi về độc lập dữ liệu, trả lời đúng phạm vi đã kiểm tra. Sau bước xử lý, các tập không còn ảnh trùng pixel theo phép kiểm tra của nhóm. Bộ dữ liệu nguồn đã có augmentation, nên kiểm tra đó chưa xác định hết ảnh biến đổi gần giống hoặc ảnh cùng một sản phẩm.

Ảnh được chuyển xám và đưa về 256 × 256 cho CNN. Với hai backbone học trước, ảnh tiếp tục được resize về 224 × 224 rồi lặp thành ba kênh. Lặp kênh giúp khớp định dạng đầu vào, lượng thông tin màu vẫn tương ứng ảnh xám. Chuẩn hóa nằm trong mô hình và phù hợp từng backbone.

**Vì sao dùng ảnh xám, resize?**

> Ảnh của nguồn chủ yếu biểu diễn hình dạng và bề mặt sản phẩm. Nhóm dùng ảnh xám và kích thước cố định để thống nhất đầu vào và giảm chi phí tính toán. Trong phạm vi thực nghiệm hiện tại, nhóm chưa so sánh ảnh màu với ảnh xám hoặc nhiều kích thước để xác định phương án tối ưu.

Nguồn đã có augmentation. Pipeline hiện tại không thêm augmentation, vì vậy báo cáo này so sánh ba cấu hình mô hình. Kết quả chưa đo riêng tác dụng của augmentation.

### 2.3. Mô hình sử dụng

Cả ba mô hình thuộc họ mạng tích chập. Tên CNN trong bảng kết quả là tên viết gọn của mạng đơn giản nhóm xây dựng làm baseline, tức mốc cơ sở để so sánh.

**CNN học từ đầu**

Chuỗi lớp cần giải thích được bằng lời.

1. Rescaling đưa giá trị điểm ảnh từ 0–255 về 0–1.
2. Ba lớp Conv2D dùng 32, 64 và 128 bộ lọc, kernel 3 × 3, học các mẫu ảnh. Số kênh đặc trưng tăng qua các khối.
3. ReLU tạo tính phi tuyến, giúp biểu diễn quan hệ phức tạp hơn phép biến đổi tuyến tính.
4. MaxPooling giảm kích thước không gian từ 256 xuống 128, 64 rồi 32 theo mỗi chiều. Nó giữ giá trị lớn trong từng vùng nhỏ.
5. Global Average Pooling lấy trung bình mỗi bản đồ đặc trưng. Với đầu ra 32 × 32 × 128, ta thu được 128 giá trị.
6. Dense(64) phối hợp các đặc trưng để phân loại.
7. Dropout 0,3 ngẫu nhiên đặt một phần đầu vào lớp về 0 trong lúc huấn luyện, giúp hạn chế phụ thuộc quá nhiều vào một số thành phần. Khi suy luận, Dropout được tắt.
8. Dense(1) với sigmoid tạo p_ok.

Mạng có tổng 100.993 tham số và toàn bộ được huấn luyện. Kiến trúc dùng các lớp CNN thông dụng, nhóm lựa chọn và ghép các lớp cho bài toán. Phần đóng góp nên trình bày ở quy trình thực nghiệm và kết quả, tránh nhận là phát minh một kiến trúc mới.

**ResNet-18 và MobileNetV2**

Backbone là phần trích xuất đặc trưng từ ảnh. Hai backbone đã được học trước trên ImageNet và được giữ cố định. Nhóm thêm GAP → Dense(64) → Dropout 0,3 → Dense(1) sigmoid để học phân loại ảnh sản phẩm đúc.

ResNet-18 dùng các kết nối tắt, cộng đầu vào của khối với kết quả biến đổi trong khối. MobileNetV2 dùng các khối tích chập được thiết kế tiết kiệm tính toán, với tích chập theo từng kênh và khối inverted residual. Khi trình bày, cần gắn mô tả đó với hình kiến trúc trong Chương 2.

| Cấu hình | Tổng tham số | Tham số được huấn luyện |
|---|---:|---:|
| CNN | 100.993 | 100.993 |
| ResNet-18 | 11.219.009 | 32.897 |
| MobileNetV2 | 2.340.033 | 82.049 |

**Câu nói mẫu**

> Với CNN, nhóm học toàn bộ trọng số từ đầu. Với hai mô hình học trước, nhóm giữ cố định phần trích xuất đặc trưng và chỉ học đầu phân loại mới. Vì vậy, kết quả so sánh phản ánh cả kiến trúc, trọng số học trước và cách tiền xử lý của từng cấu hình.

Đóng băng giữ nguyên trọng số backbone, còn phép tính đặc trưng vẫn được thực hiện cho các ảnh. Điều này giúp giải thích tại sao ResNet-18 có ít tham số được học hơn CNN nhưng thời gian fit vẫn dài hơn trong lượt chạy.

### 2.4. Huấn luyện và lựa chọn

- Một epoch tương ứng một lượt đi qua tập train. Batch 32 chia tập thành các nhóm ảnh để cập nhật trọng số nhiều lần trong mỗi epoch.
- Adam điều chỉnh trọng số dựa trên gradient. Tốc độ học ban đầu là 0,001.
- Loss binary cross-entropy đo mức phù hợp giữa xác suất dự đoán và nhãn thật. Loss thấp hơn thường thể hiện dự đoán xác suất phù hợp hơn trên cùng tập và cùng hàm mất mát.
- Accuracy đếm tỷ lệ ảnh được gán đúng nhãn theo ngưỡng.
- ModelCheckpoint lưu trạng thái có validation loss nhỏ nhất. Trạng thái đó được gọi là checkpoint tốt nhất theo tiêu chí đã chọn.
- ReduceLROnPlateau giảm tốc độ học khi validation loss không cải thiện đủ trong 3 epoch. Tất cả cấu hình vẫn chạy đủ 30 epoch.

**Vì sao epoch 30 chưa chắc tốt nhất?**

> Các lần cập nhật sau có thể tiếp tục giảm loss train nhưng không cải thiện validation. Nhóm chọn theo validation loss nhỏ nhất, nên CNN và ResNet-18 dùng epoch 29, MobileNetV2 dùng epoch 21.

**Vì sao chọn bằng validation rồi mới xem test?**

> Validation phục vụ quyết định trong quá trình xây dựng mô hình. Tập test được giữ để đánh giá sau lựa chọn, giúp hạn chế việc điều chỉnh mô hình theo chính tập dùng báo cáo kết quả cuối.

Mục lưu và sử dụng mô hình đã được gộp vào 3.5.2–3.5.3 để trình bày cùng kết quả kiểm chứng và demo.

## 3. Đọc biểu đồ huấn luyện ở mục 3.3

Ở mỗi hình, biểu đồ bên trái là loss, bên phải là accuracy. Màu xanh là train, màu cam là validation. Trục ngang là epoch. Trong CSV, epoch bắt đầu từ 0, khi trình bày đã cộng 1 để tương ứng các epoch 1–30 trên hình.

Khi giải thích một đường cong, nói theo ba ý.

1. Xu hướng chung có cải thiện không.
2. Có dao động hoặc khoảng cách train/validation đáng chú ý ở đâu.
3. Checkpoint nào được chọn và điều đó liên quan thế nào tới kết quả cuối.

“Diễn biến học” là cách mô tả dùng được. Một cụm từ riêng lẻ không xác định được văn bản do ai viết. Tên mới “Kết quả huấn luyện…” cụ thể hơn với nội dung mỗi mục. Phần bảo vệ cần chỉ được các mốc trên hình và giải thích bằng số liệu.

### 3.3.2. CNN

![Đường cong huấn luyện CNN](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_cnn.png)

Đầu lượt, accuracy khoảng 57% và loss khoảng 0,69. Đến epoch 29, train accuracy 94,36%, validation accuracy 95,84%. Cả hai tập đều cải thiện đáng kể, dù validation có hai đỉnh rõ.

| Epoch trình bày | Validation loss | Validation accuracy | Số ảnh sai trên 986 ảnh |
|---|---:|---:|---:|
| 8 | 0,2704 | 90,06% | 98 |
| 9 | 0,4488 | 78,09% | 216 |
| 10 | 0,2405 | 91,68% | 82 |
| 17 | 0,1910 | 93,20% | 67 |
| 18 | 0,5188 | 71,50% | 281 |
| 19 | 0,1722 | 94,22% | 57 |

**Câu nói mẫu**

> CNN học được tín hiệu phân loại vì loss của cả train và validation nhìn chung giảm. Tại epoch 18, validation suy giảm rõ nhưng phục hồi ngay ở epoch 19. Nhóm xem đây là dao động của kết quả ở các trạng thái trọng số khác nhau. Xu hướng chưa cho thấy validation suy giảm kéo dài trong khi train liên tục tốt lên.

Đỉnh epoch 18 đi kèm nhiều ảnh đổi sang dự đoán sai. Giải thích chỉ bằng “một vài ảnh khó làm loss tăng” sẽ chưa mô tả đủ mức giảm accuracy tại đỉnh này.

### 3.3.3. ResNet-18

![Đường cong huấn luyện ResNet-18](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_resnet18.png)

ResNet-18 đạt validation accuracy 95,44% ngay epoch 1. Backbone học trước đã cung cấp đặc trưng hữu ích, đầu phân loại thích nghi nhanh hơn CNN học từ đầu trong cấu hình này.

Tại epoch 19, validation loss tăng lên 0,0455 rồi giảm về 0,0279 ở epoch 20. Accuracy tương ứng chỉ giảm từ 98,48% ở epoch 18 xuống 98,17%, rồi trở lại 98,48%. Đỉnh dao động về tỷ lệ đúng nhỏ hơn CNN.

Tại epoch 29 được chọn, train loss 0,0143, validation loss 0,0269, accuracy tương ứng 99,62% và 98,78%. Có khoảng cách train/validation, còn validation vẫn duy trì kết quả cao ở cuối lượt.

Một chi tiết để giải thích loss và accuracy là epoch 30 có validation accuracy 98,88%, cao hơn epoch 29, nhưng loss cũng tăng lên 0,0295. Vì tiêu chí chọn là loss, nhóm vẫn dùng epoch 29.

### 3.3.4. MobileNetV2

![Đường cong huấn luyện MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/learning_curves_mobilenetv2.png)

| Epoch trình bày | Validation loss | Validation accuracy | Số ảnh sai trên 986 ảnh |
|---|---:|---:|---:|
| 13 | 0,01705 | 99,39% | 6 |
| 14 | 0,06959 | 98,17% | 18 |
| 15 | 0,02044 | 99,39% | 6 |

Loss ở epoch 14 tăng khoảng 4 lần, trong khi accuracy giảm khoảng 1,22 điểm phần trăm. Hai chỉ số thay đổi với mức độ khác nhau vì loss còn xét xác suất.

**Câu nói mẫu**

> MobileNetV2 có đỉnh validation ở epoch 14, với số ảnh sai tăng từ 6 lên 18 rồi trở về 6. Sau đó mô hình tiếp tục cải thiện và đạt validation loss thấp nhất ở epoch 21. Đỉnh đơn lẻ đó khác với xu hướng cuối lượt, khi loss train tiếp tục giảm còn validation gần như chững lại và tăng nhẹ.

Epoch 21 có train loss 0,0119 và validation loss 0,0150. Đến epoch 30, train loss còn 0,0046 nhưng validation loss là 0,0179. Đây là dấu hiệu gợi ý quá khớp nhẹ cuối lượt. Mô hình dùng cho test và demo vẫn là checkpoint epoch 21.

## 4. Trả lời vì sao validation có mũi nhọn

### Điều đã xác nhận từ code và output

Tập validation giữ cố định 986 ảnh, không cập nhật trọng số từ loss validation. Sau mỗi epoch, mô hình được đánh giá bằng trọng số vừa học xong. Những trọng số này thay đổi theo quá trình học trên train, vì vậy dự đoán validation cũng có thể thay đổi.

Code chỉ shuffle train và không bổ sung augmentation trong pipeline. CNN hiện tại không có Batch Normalization. Với hai backbone học trước, code giữ cố định backbone và gọi ở chế độ training=False.

### Vì sao loss có thể nhọn hơn accuracy

Với một ảnh, nếu gọi q là xác suất mô hình gán cho nhãn thật, loss binary cross-entropy có dạng −ln(q). Ví dụ số dưới đây chỉ minh họa công thức.

- q=0,9 thì loss khoảng 0,105.
- q=0,01 thì loss khoảng 4,605.

Sai với xác suất quá cao có thể đóng góp loss rất lớn. Accuracy chỉ đếm nhãn đúng/sai theo ngưỡng, nên ít phản ánh mức thay đổi của xác suất khi nhãn vẫn giữ nguyên. Loss của tập là trung bình các đóng góp của từng ảnh. [Tài liệu BinaryCrossentropy](https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class)

### Nguyên nhân nào đã biết, nguyên nhân nào còn là giả thuyết

Output xác nhận kết quả suy giảm tạm thời tại các mốc đã nêu. Quá trình cập nhật Adam theo từng batch, thứ tự train được xáo trộn và tốc độ học có thể liên quan đến dao động trạng thái trọng số. Lịch sử tổng hợp hiện tại chưa chứng minh riêng yếu tố nào gây ra từng đỉnh. CSV cũng chưa lưu xác suất validation của từng ảnh tại những epoch đó.

**Câu trả lời ngắn khi giảng viên hỏi**

> Tại epoch đó, mô hình sau cập nhật dự đoán kém hơn trên cùng tập validation nên loss tăng và accuracy giảm. Ở epoch sau, kết quả phục hồi. Nhóm đã kiểm tra tập validation cố định và không có augmentation ngẫu nhiên trong đường đánh giá. Quá trình cập nhật trọng số có thể gây dao động, nhưng output hiện có chưa xác định chính xác nguyên nhân của đỉnh. Nhóm chọn checkpoint theo validation loss thấp nhất của toàn lượt.

### Vì sao validation đôi lúc tốt hơn train

Trong huấn luyện, Dropout hoạt động và trọng số được cập nhật qua từng batch. Chỉ số train tích lũy trong quá trình đó. Khi đánh giá validation, Dropout được tắt và dùng trọng số tại cuối epoch. Hai chế độ đo khác nhau có thể tạo ra validation loss thấp hơn train. [Tài liệu Model.fit](https://keras.io/api/models/model_training_apis/), [Tài liệu Dropout](https://keras.io/api/layers/regularization_layers/dropout/)

Dropout được tắt ở validation. Khi giải thích đỉnh, cần nói về sự thay đổi kết quả của mô hình giữa các epoch, tránh gán đỉnh cho Dropout ngẫu nhiên ngay trong lần đánh giá validation.

### Phân biệt quá khớp và thiếu khớp

- Quá khớp thường được nhận qua xu hướng train cải thiện còn validation xấu dần hoặc khoảng cách tăng kéo dài. Với MobileNetV2, dấu hiệu cuối lượt ở mức nhẹ.
- Thiếu khớp cần bằng chứng mô hình chưa học đủ quy luật, thường cả train và validation còn kém. CNN thấp hơn hai mô hình học trước chưa đủ để kết luận riêng về thiếu khớp.
- Một đỉnh rồi phục hồi cho biết dao động tạm thời. Cần đọc cả xu hướng nhiều epoch trước khi nhận xét mức khái quát hóa.

## 5. Giải thích bảng so sánh và ma trận nhầm lẫn ở mục 3.4

| Cấu hình | Accuracy test | Số ảnh đúng /715 | Ảnh lỗi bị bỏ sót | Ảnh đạt bị báo lỗi |
|---|---:|---:|---:|---:|
| CNN | 94,83% | 678 | 25 | 12 |
| ResNet-18 | 98,74% | 706 | 7 | 2 |
| MobileNetV2 | 99,44% | 711 | 2 | 2 |

**Câu nói mẫu**

> Trên cùng 715 ảnh test, MobileNetV2 phân loại đúng 711 ảnh, gồm 451 ảnh lỗi và 260 ảnh đạt. Mô hình còn bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt. So với CNN cơ sở, số ảnh lỗi bị bỏ sót giảm từ 25 xuống 2 trong lần thực nghiệm này.

Ma trận nhầm lẫn có hàng là nhãn thật, cột là nhãn dự đoán. Thứ tự đều là lỗi rồi đạt. Với MobileNetV2, ma trận là [[451, 2], [2, 260]]. Chỉ được ô 2 ở hàng lỗi, cột đạt để giải thích sản phẩm lỗi bị cho qua.

Khi coi lớp lỗi là lớp quan tâm.

- Precision lỗi =451/(451+2)=99,56%. Trong những ảnh bị báo lỗi, bao nhiêu ảnh thật sự lỗi.
- Recall lỗi =451/(451+2)=99,56%. Trong 453 ảnh thực sự lỗi, mô hình tìm ra bao nhiêu.
- F1 kết hợp precision và recall bằng trung bình điều hòa. Với hai giá trị bằng nhau trong kết quả này, F1 lỗi cũng 99,56%.
- Accuracy =(451+260)/715=99,44%. Đây là tỷ lệ đúng chung trên cả hai lớp.

Hai mẫu số của precision và recall tình cờ cùng 453 do số bỏ sót và báo nhầm đều 2. Vai trò của hai chỉ số vẫn khác nhau.

**Vì sao quan tâm recall lỗi?**

> Với kiểm tra chất lượng, bỏ sót làm sản phẩm lỗi bị nhận là đạt. Recall lớp lỗi giúp theo dõi loại sai này. Nhóm cũng báo cáo số báo nhầm vì nó liên quan việc loại nhầm sản phẩm đạt.

**Có thể nói MobileNetV2 tốt nhất không?**

> Trong ba cấu hình và lần chia dữ liệu này, MobileNetV2 có validation loss thấp nhất và accuracy test cao nhất. Kết luận gắn với bộ ảnh cùng nguồn và một seed đã thực nghiệm.

MobileNetV2 được chọn theo validation trước khi xem test. Thời gian fit 557,11 giây thấp hơn ResNet-18 990,97 giây nhưng cao hơn CNN 300,36 giây trong lần đo T4. Các thời gian này bao gồm train và validation. Tốc độ nhận một ảnh trên máy cá nhân cần được đo riêng nếu muốn công bố.

## 6. Giải thích phần ảnh sai và demo ở mục 3.5

Hai ảnh lỗi MobileNetV2 bỏ sót là cast_def_0_150.jpeg và cast_def_0_2102.jpeg. Ảnh thứ nhất có p_ok khoảng 0,9556, ảnh thứ hai khoảng 0,5024. Điều này cho thấy có dự đoán sai cả khi điểm đạt cao và khi điểm gần ngưỡng 0,5. Bản thân điểm sigmoid chưa giải thích vùng nào trên ảnh dẫn đến quyết định.

**Lưu mô hình để làm gì?**

> Sau khi huấn luyện, nhóm lưu kiến trúc và trọng số bằng định dạng .keras. Chương trình demo nạp tệp đó để suy luận với ảnh mới. Quy tắc nhãn và tiền xử lý được lưu kèm để bảo đảm sử dụng đúng mô hình đã đánh giá.

Mô hình chọn là MobileNetV2 epoch 21, tệp `casting_classifier.keras`. Kiểm chứng CPU đã có output trên 10 ảnh validation, 10 nhãn trùng nhau, sai lệch xác suất lớn nhất khoảng 4,21 × 10⁻⁸. Đây là đối chiếu tính nhất quán sau nạp lại. Accuracy 99,44% thuộc phép đánh giá 715 ảnh test riêng.

**Trình tự giải thích demo**

1. Chọn ảnh và nói rõ nguồn ảnh.
2. Chương trình thực hiện cùng chuỗi tiền xử lý như đường đánh giá.
3. Mô hình đã lưu tạo p_ok, chương trình áp dụng ngưỡng 0,5.
4. Đọc nhãn, điểm xác suất và phân biệt chúng với accuracy toàn tập.

Ảnh minh họa của báo cáo là cast_ok_0_45.jpeg từ output đã lưu, dự đoán đạt 99,82%. Ảnh cùng tên trong test có xác suất khớp kết quả hiển thị. Một ảnh minh họa cung cấp ví dụ về cách sử dụng mô hình. Bằng chứng định lượng nằm ở bảng đánh giá toàn tập.

Giao diện local hiện có code trong `demo/app.py`. Kết quả nạp trên CPU có trong gói output, còn ảnh minh họa báo cáo lấy từ chương trình thực nghiệm. Khi bảo vệ, chỉ mô tả việc chạy giao diện local theo đúng những bước nhóm đã tự chạy và xác nhận. Hướng dẫn khởi động nằm trong [demo/README.md](D:/Casting-Defect-Detection-CNN/demo/README.md).

## 7. Trình tự nói khi bảo vệ

Thứ tự dưới đây có thể trình bày khoảng 8–10 phút và điều chỉnh theo thời lượng giảng viên cho phép.

1. Khoảng 1 phút nêu bài toán, đầu vào/đầu ra, mục tiêu và phần nhóm thực hiện.
2. Khoảng 1–2 phút nói nguồn dữ liệu, 64 ảnh trùng và cách chia 5.583/986/715.
3. Khoảng 2 phút giải thích CNN cơ sở rồi cách dùng hai backbone học trước. Nhấn mạnh phần nào được học và phần nào cố định.
4. Khoảng 2 phút đọc các đường huấn luyện, checkpoint 29/29/21 và đỉnh validation tiêu biểu.
5. Khoảng 1–2 phút đọc bảng test, tập trung 711/715 đúng và 2 ảnh lỗi còn bỏ sót của MobileNetV2.
6. Khoảng 1 phút minh họa suy luận, nêu giới hạn dữ liệu cùng nguồn và kết luận.

Chương 1 cung cấp kiến thức nền để trả lời. Khi trình bày chính, thời gian nên tập trung vào cách nhóm làm và bằng chứng ở Chương 2–3.

**Lời kết mẫu**

> Nhóm đã xây dựng và đánh giá ba cấu hình cho phân loại ảnh sản phẩm đúc. MobileNetV2 được chọn theo validation loss và đạt 99,44% accuracy trên tập test, còn bỏ sót 2 ảnh lỗi. Mô hình đã được lưu và có kết quả kiểm chứng sau nạp trên CPU. Phạm vi kết luận hiện tại là bộ dữ liệu cùng nguồn và cấu hình đã thực nghiệm.

## 8. Tự kiểm tra trước khi trình bày

- Giải thích được vì sao có ba tập và test được dùng ở bước nào.
- Vẽ hoặc nói lại đúng chuỗi lớp CNN và nhãn p_ok.
- Phân biệt tổng tham số với tham số được học.
- Chỉ đúng đỉnh CNN epoch 18 và MobileNetV2 epoch 14 trên hình, đọc được số liệu trước/sau.
- Giải thích được tại sao checkpoint MobileNetV2 thuộc epoch 21 dù chạy đủ 30 epoch.
- Từ ma trận [[451, 2], [2, 260]], tính được số đúng, số bỏ sót và số báo nhầm.
- Phân biệt xác suất của một ảnh, accuracy toàn tập và kiểm chứng sau nạp mô hình.
- Nêu được điều nhóm đã đo và điều chỉ đang là giả thuyết về nguyên nhân validation tăng.

## 9. Mở output nào để đối chiếu

- [Notebook có output](D:/Casting-Defect-Detection-CNN/Casting.ipynb) để kiểm tra các lớp mạng, pipeline, callbacks và ảnh minh họa.
- [Lịch sử CNN](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_cnn.csv), [ResNet-18](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_resnet18.csv), [MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/history_mobilenetv2.csv) để kiểm tra từng epoch.
- [Bảng so sánh](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/comparison.csv) và [ma trận MobileNetV2](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/metrics_mobilenetv2.json) để kiểm tra các chỉ số test.
- [Quyết định chọn mô hình](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/selection.json) và [kiểm chứng CPU](D:/Casting-Defect-Detection-CNN/casting_full_3models_01/demo/verification.json) để kiểm tra checkpoint và độ nhất quán sau nạp.

Các số ảnh sai theo epoch trong hướng dẫn được tính từ 986 × (1 − val_accuracy), làm tròn về số nguyên. Đây là số tổng hợp, chưa cho biết danh tính từng ảnh sai ở epoch tương ứng.
