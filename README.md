# Kiểm tra lỗi sản phẩm đúc kim loại

**Nhóm 10 · Học phần Học sâu · Trường ĐH Công nghệ Thông tin và Truyền thông (ICTU)**
Thành viên: Nguyễn Quang Minh, Nông Hữu Thịnh, Vũ Ngọc Quý · GVHD: TS. Ngô Hữu Huy

> README này viết để **một thành viên bất kỳ đọc xong có thể bảo vệ cả dự án**, kể cả phần mình không trực tiếp làm.
> Báo cáo chính thức là **[bao_cao/baocaocodong_v4.docx](bao_cao/baocaocodong_v4.docx)** (bản PDF: [baocaocodong_v4.pdf](bao_cao/baocaocodong_v4.pdf)), **bản in 25/11/2026**: 69 tờ PDF gồm bìa, 10 trang La Mã (i–x) và **58 trang** đánh số Ả Rập. Bản v3 và các bản cũ hơn chỉ để đối chiếu, **không dùng để ôn**.
> README dẫn theo **số mục, số hình, số bảng** của bản in (ví dụ "mục 2.4.5", "Bảng 3.3"). Số trang ghi chủ yếu ở [mục 8.1](#81-báo-cáo-gồm-những-gì) để lật nhanh khi bảo vệ. Bản in đã đánh số lại Hình 2.x, 3.x và Bảng 2.x, 3.x so với bản trước; đừng dùng số hình, số bảng ghi tay từ bản cũ.
> Mọi con số trong README lấy từ thư mục kết quả chính thức [`casting_full_3models_01/`](casting_full_3models_01/) và khớp với báo cáo v4.

## Mục lục

1. [Tóm tắt trong 5 dòng](#1-tóm-tắt-trong-5-dòng)
2. [Ôn KT3 – Chương 2 (đọc trước buổi báo cáo)](#2-ôn-kt3--chương-2-đọc-trước-buổi-báo-cáo)
3. [Cấu trúc thư mục và cách chạy](#3-cấu-trúc-thư-mục-và-cách-chạy)
4. [Hiểu dự án trong 15 phút](#4-hiểu-dự-án-trong-15-phút)
5. [Bảng số liệu chính](#5-bảng-số-liệu-chính)
6. [Ôn bảo vệ: câu hỏi hội đồng hay hỏi](#6-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi)
7. [Ai trình bày phần nào](#7-ai-trình-bày-phần-nào)
8. [Bản đồ báo cáo v4 và phép tính tay](#8-bản-đồ-báo-cáo-v4-và-phép-tính-tay)
9. [Liên kết và checklist trước buổi bảo vệ](#9-liên-kết-và-checklist-trước-buổi-bảo-vệ)

---

## 1. Tóm tắt trong 5 dòng

<!-- Nguồn: data_check.json, split_counts.csv, comparison.csv, selection.json, metrics_mobilenetv2.json; báo cáo mục 3.2, Bảng 3.2, 3.3, 3.5 -->

1. **Bài toán:** nhận ảnh mặt trước một sản phẩm đúc (cánh bơm), phân loại cả ảnh thành **lỗi** (nhãn 0) hoặc **đạt** (nhãn 1).
2. **Dữ liệu:** bộ *Casting Product Image Data for Quality Inspection* (Kaggle), 7.348 ảnh. Loại 64 ảnh train trùng pixel với test, còn 7.284 ảnh, chia **5.583 train / 986 validation / 715 test**.
3. **Ba mô hình:** CNN baseline tự xây, học từ đầu; **ResNet-18** và **MobileNetV2** học chuyển giao từ ImageNet, đóng băng backbone, chỉ học đầu phân loại mới.
4. **Kết quả trên 715 ảnh test:** CNN **94,83%** · ResNet-18 **98,74%** · MobileNetV2 **99,44%** (đúng 711/715, bỏ sót 2 ảnh lỗi, báo nhầm 2 ảnh đạt).
5. **Mô hình được chọn:** **MobileNetV2 ở epoch 21**, vì có validation loss thấp nhất (0,0150). Quyết định này được ghi vào `selection.json` **trước** khi chạy test (mục 2.4.5). Demo Streamlit cho chạy riêng hoặc so sánh cả ba mô hình.

**Câu tóm tắt 30 giây (cả nhóm nói giống nhau):**

> Nhóm phân loại ảnh sản phẩm đúc thành lỗi hoặc đạt. Sau khi loại 64 ảnh trùng giữa train và test, nhóm có 5.583 ảnh train, 986 ảnh validation và 715 ảnh test. Nhóm dùng một CNN học từ đầu làm mốc, so với ResNet-18 và MobileNetV2 dùng trọng số ImageNet, đóng băng backbone và chỉ học đầu phân loại. Mô hình được chọn theo validation loss là MobileNetV2 ở epoch 21. Trên 715 ảnh test, mô hình đúng 711 ảnh, bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.

---

## 2. Ôn KT3 – Chương 2 (đọc trước buổi báo cáo)

<!-- Nguồn: báo cáo v4 bản in 25/11/2026, Chương 2 (trích toàn văn baocaocodong_v4.docx; bảng đánh số cũ → mới ở report_work/hoan_thien_in_20261125/nhat_ky_ghep.md), công thức (2.1)–(2.6) trích từ baocaocodong_v4.docx; config.json, fit_*.json, selection.json, model_summary_*.txt, metrics_*.json; report_work/ra_soat_chuong2_20261124/doi_chieu_so_lieu.md. Ví dụ số không có trong báo cáo được ghi rõ là "ví dụ minh họa". -->

Mục này dành cho buổi KT3: thầy hỏi **Chương 2 "Phương pháp và mô hình"**, hỏi bất kỳ ai. Mục tiêu là **hiểu**, không học thuộc. Mỗi mục báo cáo có ba phần: *Mục này nói gì* → *Hiểu từ gốc* (khái niệm nền, có số của dự án) → *Thầy dễ hỏi* (kèm đáp án ngắn).

> **Lưu ý về số mục.** Trong mục này, "2.1", "2.2.3", "2.4.5"… là **số mục của báo cáo**, không phải số mục của README. Câu hỏi đánh số **K1, K2…** để không lẫn với 43 câu ở [mục 6](#6-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi).
> Số liệu kết quả (accuracy, ma trận nhầm lẫn, epoch được chọn) là **kết quả Chương 3**. Ở đây chỉ dùng làm ví dụ, khi nói phải ghi rõ "kết quả ở Chương 3".

### A. Chương 2 trong 1 phút

Nói được 7 câu này là nắm được khung Chương 2:

1. **(2.1)** Bài toán là phân loại nhị phân cả ảnh sản phẩm đúc: nhãn 0 là lỗi, nhãn 1 là đạt. Mô hình cho điểm pₒₖ của lớp đạt, pₒₖ ≥ 0,5 thì dự đoán đạt.
2. **(2.2)** Dữ liệu là 7.348 ảnh Kaggle. Nhóm đọc ảnh xám, băm SHA-256 theo pixel để loại ảnh train trùng test, giữ nguyên 715 ảnh test.
3. **(2.2)** Phần train sạch được chia 85/15 thành train và validation, có stratify, seed 42. Ảnh resize về 256 × 256; mỗi mô hình có phép chuẩn hóa riêng nằm trong mô hình.
4. **(2.3)** Ba mô hình: CNN baseline tự cấu hình, học từ đầu 100.993 tham số; ResNet-18 và MobileNetV2 dùng trọng số ImageNet, đóng băng backbone, chỉ học đầu Dense mới (32.897 và 82.049 tham số).
5. **(2.4.1–2.4.3)** Cả ba học 30 epoch, batch 32, Adam với tốc độ học 0,001, hàm mất mát binary cross-entropy. Checkpoint là epoch có validation loss thấp nhất; không dùng dừng sớm.
6. **(2.4.4)** Đánh giá coi lớp lỗi là lớp quan tâm: TP, FN, FP, TN, rồi accuracy, precision, recall, F1 lớp lỗi và macro F1, ngưỡng cố định 0,5.
7. **(2.4.5)** Mô hình đề xuất được chọn theo validation loss nhỏ nhất và được ghi lại **trước** khi chạy test; sau đó cả ba checkpoint mới được đánh giá trên cùng 715 ảnh test.

### A2. Dữ liệu – phần thầy hỏi nhiều nhất (ôn kỹ phần này trước)

<!-- Số liệu kiểm ngày 25/11/2026 từ casting_full_3models_01/data_check.json, split_counts.csv, excluded_train_duplicates.csv, Casting.ipynb (cell nạp dữ liệu), và băm lại toàn bộ 7.348 ảnh trong data/casting_data. -->

**Ưu tiên ôn:** phần dữ liệu (A2 này + K5–K19, K63–K64, K68–K70 ở phần B) **kỹ**; phần mô hình chỉ cần nắm mức "nói được ý chính" (xem cuối A2).

#### Con số phải thuộc

Sáu dòng đầu chính là **Bảng 2.2** (số ảnh qua các bước kiểm tra, mục 2.2.3); ba dòng train/validation/test là **Bảng 3.1**.

| Bước | Lỗi (`def_front`, nhãn 0) | Đạt (`ok_front`, nhãn 1) | Tổng |
|---|---:|---:|---:|
| Thư mục train của nguồn | 3.758 | 2.875 | 6.633 |
| Thư mục test của nguồn | 453 | 262 | 715 |
| **Tổng ảnh nguồn** | 4.211 | 3.137 | **7.348** |
| Ảnh train trùng test, bị loại | 0 | 64 | 64 |
| Train sạch (trước khi tách validation) | 3.758 | 2.811 | 6.569 |
| → **Train** (85% train sạch) | 3.194 | 2.389 | **5.583** |
| → **Validation** (15% train sạch) | 564 | 422 | **986** |
| **Test** (giữ nguyên) | 453 | 262 | **715** |
| **Tổng ảnh thực dùng** | 4.211 | 3.073 | **7.284** |

- Tỷ lệ ảnh lỗi: train **57,2%**, validation **57,2%** (nhờ stratify), test **63,4%** (giữ theo nguồn).
- Trên tổng 7.284 ảnh: train ≈ 76,6%, validation ≈ 13,5%, test ≈ 9,8%. "85/15" chỉ là tỷ lệ chia **bên trong** 6.569 ảnh train sạch.
- Ảnh nguồn: JPEG **300 × 300**, lưu 3 kênh nhưng R = G = B (nội dung xám). Kiểm cả 7.348 ảnh: **0 ảnh hỏng**, tất cả cùng 300 × 300.

#### Dữ liệu đi qua những bước nào (theo notebook)

Báo cáo vẽ chuỗi này ở **Hình 2.1 "Quy trình xử lý dữ liệu"** (đầu mục 2.2). Cách đọc hình: ảnh nguồn → kiểm tra đọc được và kích thước → băm SHA-256, loại bản train trùng test → chia train/validation (test giữ nguyên) → resize 256 × 256 dùng chung → nhánh riêng từng mô hình (CNN giữ 256 × 256 × 1; hai mạng học trước đổi 224 × 224, lặp 3 kênh) → chuẩn hóa trong mô hình. Điểm cần nói: **loại trùng đặt trước khi chia validation**, nên không ảnh nào trùng test lọt vào validation và ảnh hưởng việc chọn checkpoint.

1. Đọc ảnh bằng Pillow, xoay theo EXIF nếu có, chuyển về ảnh xám một kênh (chế độ `L`).
2. Tính **SHA-256 trên pixel xám** (kèm kích thước ảnh), **trước** khi resize. Đọc test trước, train sau; ảnh train nào trùng mã với ảnh test thì bị loại khỏi train.
3. Resize về **256 × 256** (bilinear), lưu thành mảng số nguyên 0–255, kích thước 256 × 256 × 1. Cả 7.284 ảnh nằm trong RAM, khoảng 477 MB.
4. Tách validation: `train_test_split(test_size=0,15, stratify=nhãn, random_state=42)`.
5. Khi huấn luyện: `tf.data` xáo trộn toàn bộ train **mỗi epoch**, chia batch 32 (175 bước/epoch). Riêng ResNet-18 và MobileNetV2: resize tiếp về 224 × 224 và lặp kênh xám thành 3 kênh.
6. Chuẩn hóa nằm **bên trong mô hình**: CNN chia 255; MobileNetV2 đưa về −1…1; ResNet-18 trừ trung bình, chia độ lệch chuẩn của ImageNet. Nhờ vậy demo chỉ cần đưa ảnh 0–255 vào là đúng.
7. **Không** thêm tăng cường dữ liệu (augmentation) trong lúc huấn luyện, vì bộ nguồn đã có sẵn các biến thể tăng cường.

#### 64 ảnh trùng: đặc điểm cần nói được

- Cả 64 đều là **ảnh đạt**, mỗi cặp **cùng tên tệp** ở `train/ok_front` và `test/ok_front`, pixel giống hệt.
- Không có ảnh trùng **bên trong** train hay **bên trong** test (đã băm lại cả 7.348 ảnh để kiểm).
- Nếu không loại: mô hình đã "thấy" 64 ảnh test lúc học → accuracy test cao giả tạo (rò rỉ dữ liệu). 64/262 ≈ 24,4% số ảnh đạt của test, 64/715 ≈ 9,0% cả tập test (mục 2.2.2).
- Loại 64 ảnh chỉ mất 0,96% train nguồn; tỷ lệ ảnh lỗi train tăng từ 56,66% lên 57,21%.
- **Không thể so theo tên tệp:** giữa train sạch và test còn **80 cặp cùng tên nhưng khác nội dung** (44 cặp lỗi, 36 cặp đạt). So theo tên thì 80 ảnh train hợp lệ này bị loại nhầm.

#### Câu hỏi dữ liệu (DL1–DL22)

**DL1. Dữ liệu lấy ở đâu? Có công khai không?**
Bộ Casting Product Image Data for Quality Inspection trên Kaggle, công khai, có trong tài liệu tham khảo [12]. Nhóm dùng thư mục `casting_data` (7.348 ảnh 300 × 300, đã chia sẵn train/test).

**DL2. Ảnh chụp cái gì?**
Mặt trước cánh bơm đúc, chụp từ trên xuống, sản phẩm nằm giữa ảnh trên nền xám. Mỗi ảnh một sản phẩm.

**DL3. Lỗi gồm những loại nào?**
Nhãn chỉ có **lỗi / đạt**, không ghi loại lỗi. Nhìn ảnh lỗi thấy bavia, vết mẻ, khe nứt ở mép (Hình 2.2). Vì không có nhãn loại lỗi và vị trí lỗi, bài toán là phân loại nhị phân cả ảnh.

**DL4. Sao nguồn 7.348 ảnh mà chỉ dùng 7.284?**
Loại 64 ảnh train trùng pixel với ảnh test. 7.348 − 64 = 7.284.

**DL5. Kiểm trùng bằng cách nào? Sao không so tên tệp?**
Băm SHA-256 trên pixel. Hai ảnh có cùng mã băm thì coi là giống hệt. So tên tệp không đủ: hai tệp khác tên vẫn có thể cùng ảnh, cùng tên cũng có thể khác ảnh. Trong dữ liệu này, 64 cặp trùng cùng tên, nhưng còn 80 cặp cùng tên mà khác nội dung (mục 2.2.2); so theo tên sẽ loại nhầm 80 ảnh train.

**DL6. Sao băm trên pixel xám, trước resize, mà không băm cả tệp JPEG?**
Băm cả tệp thì hai tệp cùng ảnh nhưng khác metadata sẽ ra mã khác. Băm pixel thì chỉ so nội dung ảnh. Làm trước resize để kết quả không phụ thuộc phép nội suy.

**DL7. Sao loại bản ở train mà không loại bản ở test?**
Giữ test nguyên vẹn để đánh giá đúng trên tập test gốc của bộ dữ liệu. Train mất 64 ảnh trên 6.633 (≈ 1%), không đáng kể.

**DL8. Loại trùng xong là hết rò rỉ chưa?**
Chưa chắc. SHA-256 chỉ bắt ảnh **y hệt**. Bộ nguồn có sẵn ảnh tăng cường (xoay, đổi sáng…) và không có mã ảnh gốc, nên biến thể của cùng một sản phẩm vẫn có thể nằm ở cả train và test. Vì vậy 99,44% có thể lạc quan với ảnh mới (mục 3.5.4). Hướng làm: dùng băm cảm nhận (perceptual hash) hoặc nhóm ảnh theo sản phẩm gốc. Nhóm **chưa làm**.

**DL9. Rò rỉ dữ liệu là gì? Nói một câu.**
Thông tin của tập test lọt vào lúc huấn luyện, làm điểm test cao hơn thực tế.

**DL10. Có kiểm ảnh hỏng, ảnh sai kích thước không?**
Có. Kiểm cả 7.348 ảnh: 0 ảnh không đọc được, tất cả 300 × 300 (mục 3.2). Gặp ảnh không đọc được thì chương trình dừng và báo lỗi, không bỏ qua im lặng.

**DL11. Dữ liệu có cân bằng không?**
Lệch nhẹ: train/validation khoảng 57,2% ảnh lỗi, test 63,4%. Vì lệch nên ngoài accuracy, nhóm dùng precision, recall, F1 lớp lỗi, macro F1 và ma trận nhầm lẫn. Không dùng class weight vì lệch không lớn.

**DL12. Sao test lệch lớp hơn train?**
Do bộ nguồn chia sẵn như vậy (453 lỗi / 262 đạt). Nhóm giữ nguyên, không tự chọn lại test. Lưu ý thêm: việc loại 64 ảnh đạt ở train làm train lệch thêm một chút (từ 56,7% lên 57,2% ảnh lỗi).

**DL13. Sao không gộp hết rồi chia lại 70/15/15?**
Nhóm giữ tập test có sẵn của bộ dữ liệu để không tự chọn lại tập đánh giá. Validation được tách từ phần train còn lại. Nhóm chưa thử cách chia khác.

**DL14. Validation để làm gì? Sao không lấy từ test?**
Validation dùng để chọn checkpoint, giảm tốc độ học (ReduceLROnPlateau) và chọn mô hình đề xuất. Nếu làm những việc đó bằng test thì test không còn khách quan. Test chỉ dùng một lần ở cuối.

**DL15. Stratify là gì? Cho ví dụ.**
Chia sao cho tỷ lệ lớp ở mỗi phần giống nhau. Train sạch có 57,2% ảnh lỗi → validation 564/986 = 57,2% ảnh lỗi, train 3.194/5.583 = 57,2%.

**DL16. Vì sao validation 15%?**
Chọn theo thông lệ, nhóm chưa so sánh tỷ lệ khác. 986 ảnh đủ để đánh giá ổn định (mỗi ảnh ≈ 0,1 điểm phần trăm) mà không bớt quá nhiều ảnh huấn luyện.

**DL17. Ảnh có màu không? Sao dùng 1 kênh?**
Tệp lưu 3 kênh nhưng R = G = B, tức ảnh xám. Dùng 1 kênh giảm 3 lần bộ nhớ, không mất thông tin. Hai backbone cần 3 kênh nên lặp kênh xám thành 3.

**DL18. Có tăng cường dữ liệu không?**
Không thêm trong lúc huấn luyện. Bộ nguồn **đã có sẵn** biến thể tăng cường. Thêm nữa thì có thể chồng lên; nhóm chưa khảo sát. (Đừng nói "có augmentation".)

**DL19. Vì sao resize 256 × 256?**
Nhóm chưa khảo sát kích thước khác. Chỉ nói được: nhỏ hơn 300 nên nhẹ hơn, vẫn đủ chi tiết; 256 chia hết cho 8 nên ba lần gộp 2 × 2 ra 32 × 32 gọn. Đây là nhận xét, không phải kết quả thử nghiệm.

**DL20. Chuẩn hóa là gì? Sao mỗi mô hình một kiểu?**
Đưa giá trị pixel 0–255 về khoảng nhỏ để huấn luyện ổn định. CNN học từ đầu nên chia 255 là đủ. Hai backbone phải dùng **đúng cách chuẩn hóa lúc chúng được huấn luyện trên ImageNet**, nếu không đặc trưng sẽ lệch. Ví dụ pixel 128: CNN → 0,502; MobileNetV2 → 0,004.

**DL21. Nhãn được gán thế nào?**
Theo tên thư mục: `def_front` → 0 (lỗi), `ok_front` → 1 (đạt). Mô hình xuất pₒₖ = xác suất lớp đạt; pₒₖ ≥ 0,5 là đạt.

**DL22. Thư mục `casting_512x512` (1.300 ảnh) có dùng không? Ảnh nhà máy khác thì sao?**
Không dùng. Bộ Kaggle có thêm 1.300 ảnh 512 × 512 (781 lỗi, 519 đạt), nhưng nhóm dùng `casting_data` vì nhiều ảnh hơn và đã chia sẵn train/test; chưa thử bản 512. Với ảnh từ nhà máy khác (ánh sáng, góc chụp khác), kết quả hiện tại **chưa đảm bảo**; cần thu thêm ảnh và đánh giá lại.

#### Phần mô hình: chỉ cần nói được ý chính

Nếu bị hỏi mô hình, nắm 3 câu mô tả và 8 câu hỏi dưới đây là đủ:

- **CNN baseline:** 3 khối Conv 3 × 3 + MaxPooling (32, 64, 128 bộ lọc) → GAP → Dense 64 → Dropout 0,3 → Dense 1 sigmoid; học từ đầu 100.993 tham số.
- **ResNet-18 / MobileNetV2:** dùng trọng số học sẵn trên ImageNet, **đóng băng** phần backbone, chỉ học đầu phân loại mới (32.897 / 82.049 tham số).
- **Huấn luyện:** Adam, tốc độ học 0,001, BCE, 30 epoch, batch 32; giữ epoch có validation loss thấp nhất; chọn MobileNetV2 trước khi chạy test.

Câu nên đọc: K20 (vì sao chọn hai mạng), K21 (vì sao đóng băng), K23 (tính tham số Conv), K25 (GAP và Flatten), K31 (tính 32.897), K38 (vì sao BCE), K45 (precision và recall), K47 (chọn trước test), K71 (Bảng 2.10: giống và khác giữa ba cấu hình).

### B. Đi qua từng mục của Chương 2

#### Mục 2.1. Phát biểu bài toán

**Mục này nói gì.** Định nghĩa bài toán bằng ký hiệu toán: dữ liệu, nhãn, mô hình, điểm đầu ra và quy tắc ra nhãn. Nêu rõ phạm vi: chỉ phân loại cả ảnh, không định vị lỗi. Viết bài toán huấn luyện thành "tìm tham số làm nhỏ loss trên train" và nói nhóm giữ checkpoint theo validation loss.

**Hiểu từ gốc.**

- **Phân loại nhị phân** là chọn 1 trong 2 lớp. Ở đây: lỗi (nhãn 0, thư mục `def_front`) hoặc đạt (nhãn 1, `ok_front`). "Lớp" là **nhãn**; đừng lẫn với "**tầng**" của mạng.
- **Dữ liệu** D = {(Iᵢ, yᵢ)}, i = 1…N. Iᵢ là ảnh thứ i, yᵢ ∈ {0, 1} là nhãn của nó. T là phép tiền xử lý chung (xoay theo EXIF, chuyển xám, resize bilinear về 256 × 256 × 1); xᵢ = T(Iᵢ), giá trị pixel 0–255.
- **Công thức (2.1): pᵢ,ₘ = fₘ(Pₘ(xᵢ); θₘ).** Đọc thành lời: "điểm của ảnh i do mô hình m cho ra bằng mạng fₘ, áp lên ảnh xᵢ đã qua bước riêng Pₘ, với bộ tham số θₘ".
  - m là tên mô hình: CNN, ResNet-18 hoặc MobileNetV2.
  - Pₘ là bước riêng: CNN giữ nguyên 256 × 256 × 1; hai mạng học trước thì resize về 224 × 224 và lặp kênh xám thành 3 kênh.
  - fₘ gồm bước chuẩn hóa và mạng. Backbone ImageNet nằm trong fₘ nhưng **cố định**.
  - θₘ là phần **được học**: toàn bộ CNN, hoặc chỉ hai tầng Dense mới.
  - **Nói gọn:** mô hình m nhận ảnh đã xử lý, chạy qua mạng với bộ trọng số θ, cho ra một số pₒₖ từ 0 đến 1 là điểm của lớp đạt.
- **pᵢ,ₘ** (viết ngắn **pₒₖ**, trong code/CSV là `p_ok`) là điểm của lớp **đạt**. Điểm lớp lỗi = 1 − pₒₖ. Ví dụ ảnh có pₒₖ = 0,9 → điểm lỗi 0,1 → dự đoán đạt.
- **Ngưỡng 0,5:** pₒₖ ≥ 0,5 → đạt; pₒₖ < 0,5 → lỗi. Vì hai điểm cộng lại bằng 1, ngưỡng 0,5 đúng bằng "chọn lớp có điểm lớn hơn". Bằng nhau (đúng 0,5) thì gán đạt, do phép so sánh là "≥".
- **Công thức (2.2): θ*ₘ = arg min Ltrain,m(θₘ).** "θ*ₘ là bộ tham số làm cho loss trên train nhỏ nhất". Đây là đích **lý tưởng**.
- **Công thức (2.3):** Ltrain,m = −(1/Ntr) · Σ [yᵢ·ln(pᵢ,ₘ) + (1 − yᵢ)·ln(1 − pᵢ,ₘ)], lấy tổng trên Ntr = 5.583 ảnh train. Ảnh đạt (y = 1) chỉ còn số hạng −ln(pₒₖ); ảnh lỗi (y = 0) chỉ còn −ln(1 − pₒₖ). Chi tiết ở [mục 2.4.2](#mục-242-hàm-mất-mát-và-quá-trình-cập-nhật).
- **θ̂ₘ** (mũ) là trạng thái **thực tế được lưu**: epoch có validation loss thấp nhất trong 30 epoch. Adam chỉ tìm θ*ₘ gần đúng, và nhóm cố ý không lấy trạng thái có train loss nhỏ nhất để hạn chế quá khớp. Nói gọn: θ* là đích lý tưởng trên giấy; θ̂ là bản trọng số nhóm thật sự lưu, ở epoch có validation loss thấp nhất.

**Thầy dễ hỏi.**

**K1. pₒₖ = 0,3 thì mô hình dự đoán gì? Điểm lỗi bao nhiêu?**
Dự đoán lỗi, vì 0,3 < 0,5. Điểm lỗi = 1 − 0,3 = 0,7. Chú ý pₒₖ là điểm lớp **đạt**, không phải xác suất lỗi.

**K2. Vì sao không định vị vết lỗi trên ảnh?**
Dữ liệu chỉ có nhãn cấp ảnh (cả ảnh lỗi hay đạt), không có khung hay mặt nạ vùng lỗi. Không có nhãn vị trí thì không huấn luyện định vị được, nên đề tài giới hạn ở phân loại ảnh (mục 2.1).

**K3. θ*ₘ và θ̂ₘ khác nhau thế nào?**
θ*ₘ là nghiệm lý tưởng làm train loss nhỏ nhất (2.2). θ̂ₘ là checkpoint nhóm thật sự lưu: epoch có validation loss thấp nhất. Ví dụ MobileNetV2 lưu epoch 21 (validation loss thấp nhất, kết quả Chương 3), trong khi train loss thấp nhất lại ở epoch 30.

**K4. Vì sao không so với kết quả của các bài báo khác?**
Các công trình [10, 11] dùng cách chia, phiên bản dữ liệu và cấu hình khác. So số với họ không cùng điều kiện, nên nhóm chỉ so ba mô hình tự chạy trên cùng dữ liệu và cùng quy tắc đánh giá.

#### Mục 2.2.1. Nguồn dữ liệu và quy ước nhãn

**Mục này nói gì.** Dữ liệu là bộ *Casting Product Image Data for Quality Inspection* [12], thư mục `casting_data`: 7.348 ảnh JPEG 300 × 300 (train 6.633, test 715). Bảng 2.1 quy ước nhãn 0 = lỗi, 1 = đạt. Bộ nguồn đã có sẵn ảnh tăng cường. Hình 2.1 (đầu mục 2.2) là sơ đồ xử lý dữ liệu; Hình 2.2 là ảnh mẫu hai lớp sau tiền xử lý.

**Hiểu từ gốc.**

- Ảnh lưu **ba kênh nhưng ba kênh bằng nhau**, tức là nội dung xám. Đọc về một kênh không mất thông tin.
- **Nhãn cấp ảnh:** "lỗi" nghĩa là sản phẩm có lỗi ở đâu đó; nhãn không nói vị trí hay loại khuyết tật (bavia, mẻ, nứt…).
- Thứ tự nhãn đến từ code: `CLASS_NAMES = ["def_front", "ok_front"]`, chỉ số 0 là lỗi, 1 là đạt.
- **Ảnh tăng cường có sẵn** (xoay, đổi sáng…) là nguyên nhân của rủi ro rò rỉ "gần giống" ở [mục 2.2.2](#mục-222-kiểm-tra-chất-lượng-và-xử-lý-ảnh-trùng).

**Thầy dễ hỏi.**

**K5. Lỗi là nhãn 0, vậy "lớp quan tâm" là lớp nào?**
Vẫn là lớp lỗi. Mã nhãn 0 hay 1 chỉ là quy ước lưu trữ. Khi tính precision, recall, nhóm chủ động coi lớp lỗi là lớp "dương" (mục 2.4.4).

**K6. Bộ dữ liệu có cân bằng không?**
Không hoàn toàn. Thư mục train có 3.758 lỗi / 2.875 đạt, test có 453 lỗi / 262 đạt. Vì lệch lớp nên ngoài accuracy nhóm dùng thêm F1 và macro F1.

**K7. Ảnh có màu không? Sao lại dùng ảnh xám?**
Ảnh JPEG lưu ba kênh nhưng R = G = B, tức là đã là ảnh xám. Nhóm chuyển về một kênh cho gọn; hai backbone cần ba kênh thì lặp lại kênh xám (mục 2.2.4).

#### Mục 2.2.2. Kiểm tra chất lượng và xử lý ảnh trùng

**Mục này nói gì.** Ảnh được đọc bằng Pillow, xoay theo EXIF, chuyển xám. Chương trình kiểm tra ảnh đọc được, kích thước và số lượng. Ảnh trùng được phát hiện bằng SHA-256 trên pixel; ưu tiên giữ tập test, loại bản ở train. Mục này cũng nói rõ giới hạn: chỉ bắt được bản sao y hệt.

**Hiểu từ gốc.**

- **SHA-256** là hàm băm: biến một dãy byte bất kỳ thành một mã 64 ký tự hex. Hai dữ liệu giống hệt nhau → cùng mã. Khác dù chỉ 1 pixel → mã khác hẳn. Nhóm băm *kích thước ảnh + byte pixel ảnh xám*, **trước** khi resize. (Hai ảnh khác nhau trùng mã về lý thuyết có thể xảy ra, nhưng xác suất cực nhỏ, thực tế bỏ qua.)
- **Rò rỉ dữ liệu (data leakage)** là khi thông tin của tập đánh giá lọt vào lúc huấn luyện. Nếu ảnh X có ở cả train và test, mô hình đã "thấy đáp án" của câu X; điểm test cao hơn thực lực.
- **Quy trình:** đọc **test trước**, ghi mã băm. Đọc train sau: ảnh train trùng mã với ảnh đã thấy (cùng nhãn) thì bỏ bản train. Trùng **khác nhãn** hoặc trùng **ngay trong test** thì chương trình dừng để xem lại.
- **Kết quả** (bản in nêu ngay ở mục 2.2.2 và Bảng 2.2): 64 ảnh đạt trong train trùng test, bị loại; còn 7.348 − 64 = **7.284** ảnh. Cả 64 cặp cùng tên, cùng lớp đạt; không có cặp trùng nằm trọn trong train hay trọn trong test, không có cặp khác nhãn.
- **Tên tệp không thay được mã băm:** còn 80 cặp cùng tên nhưng khác nội dung (44 lỗi, 36 đạt).
- **Vì sao băm trước resize?** Hai ảnh gốc hơi khác nhau có thể thành giống hệt sau khi thu nhỏ. Băm ảnh gốc tránh loại nhầm.
- **Giới hạn:** ảnh xoay hay đổi sáng từ cùng ảnh gốc cho mã khác, SHA-256 không bắt được (mục 3.5.4).

**Thầy dễ hỏi.**

**K8. SHA-256 là gì, vì sao không so tên tệp?**
Là hàm băm cho "dấu vân tay" của dữ liệu; giống hệt từng byte thì cùng mã. Tên tệp không tin được: ảnh cùng tên ở train và test có thể khác nội dung, ảnh khác tên có thể giống hệt. Hai ảnh khác nhau trùng mã SHA-256 về lý thuyết có thể xảy ra, nhưng xác suất cực nhỏ, thực tế bỏ qua.

**K9. Vì sao loại bản ở train mà giữ bản ở test?**
Test là tập đánh giá do nguồn cung cấp. Nhóm không muốn tự thay đổi "đề thi", nên giữ đủ 715 ảnh test và bỏ bản trùng ở phía train.

**K10. Kiểm trùng xong thì hết rò rỉ chưa?**
Chưa chắc. SHA-256 chỉ loại bản sao y hệt. Nguồn có ảnh tăng cường và không có mã sản phẩm gốc, nên biến thể gần giống của cùng một ảnh vẫn có thể nằm ở cả train và test (mục 2.2.2, 3.5.4). Hướng làm: so độ giống bằng băm cảm nhận (perceptual hash) hoặc đặc trưng CNN, nhóm **chưa khảo sát**.

**K11. Nếu gặp ảnh hỏng thì sao?**
Chương trình gom danh sách ảnh không đọc được và dừng trước khi huấn luyện. Lượt chính thức không có ảnh hỏng nào trong 7.348 ảnh.

#### Mục 2.2.3. Phân chia train, validation và test

**Mục này nói gì.** Bảng 2.2 tổng hợp số ảnh qua các bước kiểm tra, kèm tỷ lệ ảnh lỗi. Giữ nguyên 715 ảnh test. Train sạch được chia bằng `train_test_split(test_size=0,15, stratify=nhãn, random_state=42)` thành train và validation. Bảng 2.3 nêu vai trò từng tập. Ba mô hình dùng chung danh sách chia và cùng thứ tự ảnh test. Cuối mục bàn hệ quả của lệch lớp lên việc chọn thước đo.

**Hiểu từ gốc.**

- **Ba tập, ba vai trò** (giống học – thi thử – thi thật):

| Tập | Số ảnh (Chương 3) | Dùng để | Có được nhìn khi ra quyết định không? |
|---|---:|---|---|
| Train | 5.583 | Cập nhật trọng số (học) | Có, mô hình học trên đó |
| Validation | 986 | Giảm tốc độ học, chọn checkpoint, chọn mô hình | Có, dùng để chọn |
| Test | 715 | Đánh giá một lần sau khi đã chọn | **Không** |

- **Tính tay:** train sạch = 6.633 − 64 = 6.569. Validation = 15% ≈ 986 (làm tròn lên). Train = 6.569 − 986 = 5.583.
- **Stratify** (chia phân tầng) là chia sao cho tỷ lệ lỗi/đạt hai phần giống nhau. Train 3.194/5.583 ≈ 57,2% lỗi; validation 564/986 ≈ 57,2% lỗi. Không stratify thì validation có thể lệch lớp do may rủi.
- **random_state = 42** cố định cách xáo, chạy lại ra đúng cùng một cách chia.
- Test giữ tỷ lệ của nguồn: 453/715 ≈ 63,4% lỗi, khác train một chút.
- **Bảng 2.2 đọc thế nào:** sáu dòng là train nguồn 6.633 (56,66% lỗi), test nguồn 715 (63,36%), tổng 7.348 (57,31%), ảnh không đọc được 0, ảnh train trùng test 64 (toàn ảnh đạt, 0,00% lỗi), train sạch 6.569 (57,21%).
- **Lệch lớp và thước đo:** train sạch lỗi : đạt ≈ 1,34 : 1, test ≈ 1,73 : 1. Bộ phân loại luôn trả lời "lỗi" đã được 57,20% accuracy trên validation và 63,36% trên test, nên accuracy phải đọc so với hai mốc này. Recall mỗi lớp không phụ thuộc tỷ lệ lớp; precision lớp lỗi thì có. Lệch ở train không lớn nên không dùng class weight.

**Thầy dễ hỏi.**

**K12. Validation khác test thế nào? Sao không dùng test để chọn luôn?**
Validation được dùng để **ra quyết định** (chọn checkpoint, chọn mô hình, giảm tốc độ học). Test chỉ dùng để **đo** sau cùng. Nếu dùng test để chọn thì test thành một phần của việc xây mô hình, điểm test không còn khách quan.

**K13. Stratify để làm gì? Cho ví dụ số.**
Giữ tỷ lệ hai lớp như nhau ở train và validation: cả hai khoảng 57,2% ảnh lỗi (3.194/5.583 và 564/986). Nhờ vậy validation có cùng tỉ lệ lớp với train.

**K14. Vì sao 15%?**
Nhóm **chưa khảo sát** tỷ lệ khác. 15% là mức phổ biến, cho 986 ảnh validation, đủ để một ảnh chỉ ứng với khoảng 0,1 điểm phần trăm accuracy, trong khi train vẫn giữ 5.583 ảnh.

**K15. Ba mô hình có cùng tập test không?**
Có. Ba mô hình dùng chung một danh sách chia (cùng mã băm danh sách trong `fit_*.json`) và cùng thứ tự 715 ảnh test, nên so sánh từng ảnh được.

**K68. Hình 2.1 (sơ đồ xử lý dữ liệu) nói gì? Vì sao loại trùng trước khi chia validation?**
Hình 2.1 là chuỗi: kiểm tra đọc ảnh và kích thước → băm SHA-256, loại bản train trùng test → chia train/validation, test giữ nguyên → resize 256 × 256 chung → bước riêng từng mô hình (224 × 224, lặp kênh, chuẩn hóa). Loại trùng làm trước, nên không ảnh nào trùng test nằm trong validation. Nếu không, validation cũng "nhìn thấy" ảnh test và ảnh hưởng việc chọn checkpoint.

**K69. Bảng 2.2 có những dòng nào? Đọc giúp tỷ lệ ảnh lỗi.**
Train nguồn 3.758 lỗi / 2.875 đạt / 6.633 (56,66%); test nguồn 453 / 262 / 715 (63,36%); tổng 7.348 (57,31%); ảnh không đọc được 0; ảnh train trùng test bị loại 0 / 64 / 64; train sạch 3.758 / 2.811 / 6.569 (57,21%). Loại 64 ảnh đạt nên tỷ lệ lỗi ở train nhích từ 56,66% lên 57,21%.

**K70. Một mô hình luôn đoán "lỗi" được bao nhiêu? Ý nghĩa?**
57,20% trên validation (564/986) và 63,36% trên test (453/715). Đây là mốc tối thiểu để đọc accuracy. Vì vậy báo cáo xem thêm chỉ số theo lớp và macro F1 (mục 2.2.3, 2.4.4).

#### Mục 2.2.4. Biểu diễn và chuẩn hóa đầu vào

**Mục này nói gì.** Mọi ảnh được resize bilinear về 256 × 256 × 1 (uint8). CNN nhận thẳng ảnh này. ResNet-18 và MobileNetV2 resize tiếp về 224 × 224 (TensorFlow, bilinear, antialias) và lặp thành 3 kênh. Mỗi mô hình có phép chuẩn hóa riêng, **đặt trong mô hình** (Bảng 2.4). Nhóm không thêm augmentation.

**Hiểu từ gốc.**

- **Resize** là đổi kích thước ảnh. **Bilinear** (song tuyến tính): theo định nghĩa, pixel mới là trung bình có trọng số của 4 pixel gần nhất. Khi thu nhỏ, Pillow và `tf.image.resize(antialias=True)` nới rộng vùng lấy trung bình để ảnh mượt, bớt răng cưa (**antialias**).
- **Chuẩn hóa** là đưa pixel 0–255 về một miền nhỏ, ổn định cho việc học. Ví dụ một pixel xám giá trị 128:

| Mô hình | Phép chuẩn hóa | Pixel 128 thành | Miền giá trị |
|---|---|---|---|
| CNN | x/255 | 0,502 | 0 đến 1 |
| ResNet-18 | (x/255 − μc)/σc, μ = (0,485; 0,456; 0,406), σ = (0,229; 0,224; 0,225) | R ≈ 0,074; G ≈ 0,205; B ≈ 0,427 | khoảng −2,1 đến 2,6 |
| MobileNetV2 | x/127,5 − 1 | ≈ 0,004 | −1 đến 1 |

- **Vì sao mỗi backbone cần chuẩn hóa riêng?** Trọng số ImageNet được học trên ảnh đã chuẩn hóa **theo đúng một cách**. ResNet-18 (KerasHub) học với mean/std ImageNet; MobileNetV2 (Keras) học với miền −1 đến 1. Đưa ảnh vào sai miền thì các bộ lọc và thống kê Batch Normalization đã đóng băng nhận giá trị lệch so với lúc học, đặc trưng sẽ sai. CNN học từ đầu nên tự chọn cách đơn giản x/255. Nói gọn: backbone đã quen nhìn ảnh theo một thang số; đưa ảnh sai thang thì nó "nhìn lệch", dù trọng số vẫn đúng.
- **Đặt chuẩn hóa trong mô hình** (tầng Rescaling của CNN, converter của ResNet-18, Rescaling của MobileNetV2): quy tắc được lưu cùng tệp `.keras`, demo nạp mô hình là có đúng bước chuẩn hóa.
- **Converter của ResNet-18** (`preprocessing_resnet18.json`) làm hai việc: resize về 224 × 224 (bicubic, nhưng ảnh vào đã đúng 224 × 224 nên ảnh không đổi), rồi nhân scale và cộng offset từng kênh, tương đương (x/255 − μc)/σc.
- **Lặp kênh xám thành 3 kênh** chỉ để khớp hình dạng đầu vào 3 kênh; không thêm thông tin màu.
- **Giới hạn:** thu nhỏ 300 → 256 → 224 có thể làm mờ vết lỗi rất nhỏ.

**Thầy dễ hỏi.**

**K16. Vì sao hai mạng học chuyển giao dùng 224 × 224 × 3?**
Đó là kích thước và số kênh chuẩn khi hai backbone được học trên ImageNet. Nhóm đổi ảnh về đúng dạng này; kênh xám được lặp ba lần.

**K17. Vì sao không resize thẳng ảnh gốc về 224?**
Để mọi mô hình bắt đầu từ cùng một biểu diễn 256 × 256. Demo lặp lại đúng chuỗi 256 → 224 bằng `tf.image.resize`; resize thẳng bằng thư viện khác có thể làm số lệch đi so với lúc huấn luyện. Converter ResNet-18 có ghi resize bicubic, nhưng ảnh vào đã đúng 224 × 224 nên bước này không đổi ảnh.

**K18. Có dùng augmentation (lật, xoay ngẫu nhiên) không?**
Không thêm. Bộ nguồn đã có sẵn biến thể tăng cường (`pipeline_augmentation = false`, `source_augmentation = true` trong `config.json`). Ba mô hình học trên cùng tập train cố định. Nhóm **chưa khảo sát** tác dụng của việc thêm augmentation.

**K19. Đổi MobileNetV2 sang x/255 thì sao?**
Nhóm **chưa thử**. Về nguyên tắc, backbone đóng băng sẽ nhận miền 0–1 thay vì −1 đến 1 như lúc học, đặc trưng bị lệch và kết quả thường kém đi. Vì vậy nhóm dùng đúng phép chuẩn hóa đi kèm từng backbone.

**K63. Backbone toàn tích chập nhận được 256 × 256, sao vẫn đổi về 224?**
224 × 224 là kích thước lúc hai backbone học trên ImageNet; converter của KerasHub cũng đặt 224 (`preprocessing_resnet18.json`). Giữ đúng kích thước này để đặc trưng giống lúc học nhất. Nhóm **chưa thử** đưa 256 vào backbone.

**K64. Sao lặp kênh xám thành 3 kênh mà không sửa tầng Conv đầu thành 1 kênh?**
Lặp kênh giữ nguyên được trọng số ImageNet mà không phải sửa kiến trúc. Vì ba kênh giống nhau, về toán Conv đầu tương đương một Conv 1 kênh có trọng số bằng tổng (có hệ số chuẩn hóa) trọng số của ba kênh. Đổi lại, Conv đầu tính nhiều hơn cần thiết khoảng 3 lần. Nhóm **chưa thử** cách sửa tầng đầu.

#### Mục 2.3.1. Vai trò của ba mô hình trong thực nghiệm

**Mục này nói gì.** CNN là mốc (baseline), học toàn bộ từ đầu. ResNet-18 và MobileNetV2 là hai backbone ImageNet, đóng băng, chỉ học đầu phân loại. Mục này nêu lý do chọn và nói rõ phép so sánh khác nhau ở những điểm nào.

**Hiểu từ gốc.**

- **Baseline** là mô hình mốc, đơn giản, để biết các mô hình khác hơn được bao nhiêu.
- **Học chuyển giao:** dùng lại một mạng đã học trên tập lớn (ImageNet-1k, khoảng 1,28 triệu ảnh, 1.000 lớp) cho bài toán mới. **Backbone** là phần trích đặc trưng: cho bản đồ đặc trưng 7 × 7 × 512 (MobileNetV2: 7 × 7 × 1.280); tầng GAP trong đầu mới của nhóm biến nó thành véc tơ 512 (1.280) số.
- **Đóng băng** = không cập nhật trọng số backbone; chỉ học phần đầu Dense mới. Khác với **fine-tuning** (mở khóa một phần backbone để học tiếp) – nhóm **không làm**.
- **Cái gì giống, cái gì khác giữa ba cấu hình** (mục 2.3.1 nêu gọn; bảng đầy đủ là **Bảng 2.10** ở mục 2.4.1):

| Giống nhau | Khác nhau |
|---|---|
| Dữ liệu, danh sách chia, 30 epoch, batch 32, Adam 0,001, BCE, ngưỡng 0,5, thước đo | Kiến trúc; khởi tạo (ngẫu nhiên / ImageNet); độ phân giải (256 / 224); phần tham số được học (toàn mạng / đầu Dense) |

**Thầy dễ hỏi.**

**K20. Vì sao chọn ResNet-18 và MobileNetV2?**
ResNet-18 là mạng nhỏ nhất họ ResNet, đại diện kiến trúc khối dư. MobileNetV2 là mạng nhẹ, hợp với demo chạy CPU. Cả hai có sẵn trọng số ImageNet trong hệ Keras (KerasHub và Keras Applications).

**K21. Vì sao đóng băng mà không fine-tune?**
Train chỉ khoảng 5.600 ảnh; mục tiêu là đo đặc trưng ImageNet giúp được bao nhiêu với chi phí thấp. Học ít tham số cũng giảm rủi ro quá khớp. Fine-tuning để lại cho thực nghiệm sau, nhóm chưa có số.

**K22. So sánh ba mô hình có công bằng không?**
Công bằng về dữ liệu, thiết lập huấn luyện và cách đánh giá. Nhưng chúng khác nhau cùng lúc ở kiến trúc, khởi tạo, độ phân giải và phần được học, nên chỉ kết luận cho **cả cấu hình**, chưa tách được yếu tố nào tạo ra chênh lệch (Bảng 2.10).

#### Mục 2.3.2. Kiến trúc CNN baseline

**Mục này nói gì.** CNN dạng Sequential: 3 khối [Conv2D 3 × 3 + ReLU → MaxPooling 2 × 2], số bộ lọc 32 → 64 → 128; rồi GAP → Dense 64 ReLU → Dropout 0,3 → Dense 1 sigmoid. Bảng 2.5 ghi kích thước và số tham số từng tầng, tổng 100.993, tất cả được học. Hình 2.3 vẽ luồng ảnh qua các tầng. Công thức (2.4) là sigmoid.

**Hiểu từ gốc.**

- **Tích chập 3 × 3:** một bộ lọc là một ô 3 × 3 trọng số. Nó trượt trên ảnh; ở mỗi vị trí, nhân từng cặp pixel–trọng số, cộng lại, cộng bias → một số. Số đó lớn khi vùng ảnh giống mẫu của bộ lọc.
  *Ví dụ minh họa* (dùng pixel 0–255 cho dễ nhìn): bộ lọc cột [−1, 0, 1] lặp 3 hàng, đặt lên vùng có 2 cột tối (10) và 1 cột sáng (200) → 3 × (200 − 10) = 570, phản ứng mạnh với **cạnh dọc**. Đặt lên vùng đều màu → 0. Trọng số bộ lọc **do mạng tự học**, không do nhóm đặt tay.
- **Số kênh:** mỗi bộ lọc cho ra một bản đồ đặc trưng. 32 bộ lọc → 32 kênh. Tầng sau, mỗi bộ lọc phải "nhìn" đủ 32 kênh, nên có 3 × 3 × 32 trọng số.
- **Stride** là bước trượt. Conv dùng stride 1 (trượt từng pixel). **Padding same** là đệm viền 0 (p = 1 với bộ lọc 3 × 3) để đầu ra giữ nguyên cỡ: o = ⌊(256 + 2·1 − 3)/1⌋ + 1 = 256.
- **ReLU:** max(0, z). Giữ phần dương, cắt phần âm, giúp mạng học quan hệ phi tuyến.
- **MaxPooling 2 × 2, stride 2:** lấy giá trị lớn nhất mỗi ô 2 × 2, chiều cao và rộng giảm một nửa: 256 → 128 → 64 → 32. Không có tham số.
- Mỗi lần ảnh nhỏ đi một nửa thì số bộ lọc tăng gấp đôi, nên chi phí mỗi tầng gần như không tăng: Conv thứ hai 128 × 128 × (3·3·32) × 64 ≈ 302 triệu phép nhân–cộng; Conv thứ ba 64 × 64 × (3·3·64) × 128 cũng ≈ 302 triệu.
- **GAP so với Flatten** (đầu ra trước đó là 32 × 32 × 128):

| | Số đầu vào cho Dense 64 | Tham số Dense 64 |
|---|---:|---:|
| Flatten (trải phẳng) | 32 × 32 × 128 = 131.072 | (131.072 + 1) × 64 = **8.388.672** |
| GAP (trung bình mỗi kênh) | 128 | (128 + 1) × 64 = **8.256** |

  GAP ít hơn khoảng 1.000 lần tham số, giảm rủi ro quá khớp; đổi lại mất thông tin vị trí.
- **Dense** (kết nối đầy đủ): mỗi nơ-ron nối với mọi đầu vào. **Dropout 0,3:** khi huấn luyện, mỗi giá trị trong 64 giá trị của từng ảnh bị đặt về 0 với xác suất 0,3 (trung bình khoảng 30%, không phải đúng 30%; mỗi ảnh và mỗi bước một mẫu khác). Các giá trị còn lại được nhân với 1/0,7 để tổng trung bình không đổi. Khi dự đoán, Dropout tắt hẳn và không cần chỉnh gì thêm.
- **Sigmoid (2.4): pₒₖ = σ(z) = 1/(1 + e^(−z)).** z là đầu ra tuyến tính của nơ-ron cuối. z = 0 → 0,5; z = 2 → 0,881; z = −2 → 0,119. Nhờ vậy đầu ra luôn nằm trong 0–1.
- **Tự tính số tham số** – công thức: Conv = (k × k × kênh vào + 1) × số bộ lọc; Dense = (số đầu vào + 1) × số nơ-ron. "+1" là bias.

| Bước | Tầng | Phép tính | Kết quả |
|---|---|---|---:|
| 1 | Conv2D 32 (vào 1 kênh xám) | (3 × 3 × 1 + 1) × 32 = 10 × 32 | 320 |
| 2 | Conv2D 64 (vào 32 kênh) | (3 × 3 × 32 + 1) × 64 = 289 × 64 | 18.496 |
| 3 | Conv2D 128 (vào 64 kênh) | (3 × 3 × 64 + 1) × 128 = 577 × 128 | 73.856 |
| 4 | Dense 64 (vào 128 số sau GAP) | (128 + 1) × 64 | 8.256 |
| 5 | Dense 1 (vào 64 số) | (64 + 1) × 1 | 65 |
| | Rescaling, MaxPooling, GAP, Dropout | không có trọng số | 0 |
| | **Tổng** | 320 + 18.496 + 73.856 + 8.256 + 65 | **100.993** |

**Thầy dễ hỏi.**

**K23. Tính số tham số tầng Conv thứ hai. Vì sao là 32 mà không phải 1?**
(3 × 3 × 32 + 1) × 64 = 18.496. Đầu vào tầng này có 32 kênh (do tầng trước có 32 bộ lọc), nên mỗi bộ lọc có 3 × 3 × 32 trọng số cộng 1 bias.

**K24. Padding same, stride 1 thì đầu ra Conv bao nhiêu? Sau MaxPooling?**
Giữ nguyên: 256 × 256 → 256 × 256 (số kênh = số bộ lọc). MaxPooling 2 × 2, stride 2 chia đôi: 256 → 128. Dùng công thức (1.11): o = ⌊(n + 2p − k)/s⌋ + 1.

**K25. Vì sao GAP mà không Flatten?**
GAP cần 8.256 tham số cho Dense 64, Flatten cần 8.388.672. Ít tham số hơn thì nhẹ hơn và ít quá khớp hơn; đổi lại mất thông tin vị trí, chấp nhận được vì bài toán chỉ phân loại cả ảnh.

**K26. Dropout 0,3 hoạt động lúc nào? Nó ảnh hưởng gì đến số đo?**
Chỉ khi huấn luyện. Accuracy/loss train được đo khi Dropout bật và là trung bình qua các batch trong lúc trọng số còn thay đổi; validation đo khi Dropout tắt, bằng trọng số cuối epoch (mục 3.3.2). Vì vậy validation đôi khi tốt hơn train là bình thường (ví dụ CNN, Chương 3).

**K27. CNN "tự xây" có phải kiến trúc mới không?**
Không. Nhóm tự chọn số tầng, số bộ lọc, đầu phân loại từ các tầng Keras phổ biến, và học từ đầu. Đó là mô hình mốc, không nhận là kiến trúc mới.

**K60. Sao đầu ra là 1 nơ-ron sigmoid mà không phải 2 nơ-ron softmax?**
Với hai lớp, hai cách tương đương: softmax hai lớp cho p(đạt) = σ(z_đạt − z_lỗi), tức một sigmoid của hiệu hai điểm. Một nơ-ron sigmoid đi với BCE thì gọn hơn, ít tham số hơn ở tầng cuối ((64 + 1) × 1 = 65 thay vì (64 + 1) × 2 = 130) và cho thẳng pₒₖ.

#### Mục 2.3.3. ResNet-18 với backbone cố định

**Mục này nói gì.** ResNet-18 dùng **basic block** có đường nối tắt (Hình 2.4). Nhóm dùng preset `resnet_18_imagenet` của KerasHub và converter chuẩn hóa ImageNet. Hình 2.5 vẽ các nhóm khối và đầu phân loại mới. Backbone đóng băng; chỉ học hai tầng Dense mới: 32.897 trên tổng 11.219.009 tham số (Bảng 2.6).

**Hiểu từ gốc.**

- **Basic block:** nhánh F(x) gồm hai Conv 3 × 3 kèm Batch Normalization. Đầu ra **y = ReLU(F(x) + x)**: cộng đầu vào x (đường nối tắt) với F(x) rồi mới ReLU. Khối chỉ phải học **phần chênh lệch** F(x) so với x.
- **Vì sao giúp gradient?** Trước ReLU, y = F(x) + x, nên đạo hàm theo x là F′(x) + **1**. Số "+1" là đường đi thẳng cho gradient. *Ví dụ minh họa:* nếu mỗi khối chỉ cho qua hệ số 0,5, qua 8 khối còn 0,5⁸ ≈ 0,004. Có nối tắt thì hệ số mỗi khối là khoảng 1 + F′ thay vì chỉ F′, nên tích qua nhiều khối không co về 0 nhanh như vậy. Điều này giảm hiện tượng gradient biến mất (mục 1.5.2). Nói gọn: đường tắt giống một đường cao tốc cho gradient chạy thẳng về các tầng đầu, nên tín hiệu học không bị mờ dần qua nhiều tầng.
- Khi đổi số kênh hoặc độ phân giải (64 → 128…), nhánh tắt dùng **phép chiếu** để hai tensor cùng kích thước mới cộng được.
- **Cấu trúc** (Bảng 2.6): Conv 7 × 7 stride 2 → MaxPool → 4 nhóm, mỗi nhóm **2 basic block**, kênh 64/128/256/512 → đầu ra 7 × 7 × 512 → GAP → 512 số → Dense 64 → Dropout 0,3 → Dense 1.
- **Đóng băng:** `trainable=False` (không cập nhật trọng số; trong Keras, BN bị đặt `trainable=False` cũng tự chạy chế độ suy luận) **và** gọi với `training=False` (Batch Normalization dùng trung bình, phương sai đã lưu từ ImageNet, không tính lại theo batch và không cập nhật; giữ được chế độ này kể cả khi sau này mở khóa để fine-tune). Nói gọn: `trainable=False` khóa trọng số; `training=False` bắt BN dùng số trung bình cũ của ImageNet; nhóm khóa cả hai. Backbone **vẫn chạy xuôi** ở mỗi batch để tạo 512 đặc trưng.
- **Tính tham số được học:** Dense 64: (512 + 1) × 64 = 32.832; Dense 1: (64 + 1) × 1 = 65; tổng **32.897**.

**Thầy dễ hỏi.**

**K28. Viết công thức khối dư, giải thích vì sao học dễ hơn.**
y = ReLU(F(x) + x). Mạng chỉ cần học phần sai khác F(x); nếu khối không cần thay đổi gì thì F(x) ≈ 0 là đủ. Đường nối tắt cho gradient đi thẳng về tầng trước, giảm gradient biến mất.

**K29. ResNet-18 dùng bottleneck 1 × 1 – 3 × 3 – 1 × 1 phải không?**
**Không.** ResNet-18 dùng **basic block** hai Conv 3 × 3. Bottleneck 1 × 1 – 3 × 3 – 1 × 1 là của ResNet-50 trở lên.

**K30. trainable=False và training=False khác nhau thế nào?**
`trainable=False`: trọng số (kể cả γ, β của BN) không được cập nhật. `training=False`: các tầng như BN, Dropout chạy ở chế độ suy luận; BN dùng trung bình, phương sai đã lưu từ ImageNet. Trong Keras, BN bị đặt `trainable=False` cũng tự chạy chế độ suy luận. Nhóm vẫn gọi `training=False` theo khuyến nghị của Keras, để BN giữ nguyên chế độ này kể cả khi sau này mở khóa để fine-tune. Kết quả là backbone giữ nguyên hoàn toàn.

**K31. Vì sao chỉ học 32.897 tham số? Tính thử.**
Chỉ hai tầng Dense mới được học: (512 + 1) × 64 + (64 + 1) × 1 = 32.832 + 65 = 32.897. Còn lại hơn 11,18 triệu tham số backbone cố định.

**K61. Batch Normalization là gì?**
BN chuẩn hóa đầu ra mỗi kênh về trung bình 0, phương sai 1 theo batch, rồi nhân γ, cộng β (hai tham số học). Lúc suy luận, BN dùng trung bình, phương sai trượt đã lưu. Trong hai backbone của nhóm, BN đóng băng và luôn dùng thống kê ImageNet. CNN baseline của nhóm **không** có BN (`model_summary_cnn.txt`).

**K62. Vì sao các Conv trong backbone không có bias?**
Vì ngay sau Conv là BN, mà β của BN đã đóng vai bias. Ví dụ Conv 3 × 3 đầu tiên trong khối dư của ResNet-18 (64 kênh vào, 64 kênh ra) có 3 × 3 × 64 × 64 = 36.864 tham số, không có "+1" (`model_summary_resnet18.txt`). Nhánh tắt chiếu 64 → 128 là Conv 1 × 1: 64 × 128 = 8.192 tham số.

#### Mục 2.3.4. MobileNetV2 với backbone cố định

**Mục này nói gì.** MobileNetV2 dùng **inverted residual** và **linear bottleneck** (Hình 2.6, ví dụ `block_7`). Nhóm dùng Keras Applications, `weights="imagenet"`, `include_top=False`, `alpha=1.0`. Hình 2.7 vẽ kiến trúc dùng trong đề tài. Backbone đóng băng, chỉ học 82.049 trên 2.340.033 tham số (Bảng 2.7).

**Hiểu từ gốc.**

- **Tích chập tách biệt theo chiều sâu (depthwise separable)** = 2 bước: (1) **depthwise 3 × 3**: mỗi kênh có bộ lọc 3 × 3 riêng, không trộn kênh; (2) **pointwise 1 × 1**: trộn các kênh.
- **Vì sao rẻ hơn ~8–9 lần?** *Ví dụ minh họa*, 384 kênh vào, 384 kênh ra, tính cho một vị trí đầu ra:

| Cách | Số phép nhân |
|---|---:|
| Conv 3 × 3 thường | 3 × 3 × 384 × 384 = 1.327.104 |
| Depthwise 3 × 3 + pointwise 1 × 1 | 3 × 3 × 384 + 384 × 384 = 3.456 + 147.456 = 150.912 |
| Tỷ lệ | ≈ 8,8 lần |

  Tổng quát: chi phí tách biệt / chi phí thường = 1/(số kênh ra) + 1/(k × k). Với k = 3 và 384 kênh ra: 1/384 + 1/9 ≈ 0,114, tức rẻ hơn khoảng 1/0,114 ≈ 8,8 lần. Kênh càng nhiều thì càng gần 9 lần.
  Nói gọn: Conv thường làm hai việc một lúc (lọc theo vùng và trộn kênh). Tách thành hai bước nhỏ thì chỉ tốn khoảng 1/9 công.
- **Inverted residual (block_7, trong `model_summary_mobilenetv2.txt`):** Conv 1 × 1 **mở rộng** 64 → 384 kênh → depthwise 3 × 3 trên 384 kênh → Conv 1 × 1 **chiếu** về 64 kênh → cộng với đầu vào. "Ngược" vì hẹp – rộng – hẹp, ngược với khối bottleneck của ResNet-50 (rộng – hẹp – rộng). ResNet-18 của nhóm dùng basic block nên không có bottleneck.
- **Linear bottleneck:** bước chiếu về 64 kênh **không có ReLU**. Khi ép thông tin vào ít kênh, nếu còn cắt bỏ số âm (ReLU) thì mất nhiều quá, nên bước ép để tuyến tính, không dùng ReLU. Các bước mở rộng và depthwise dùng **ReLU6** (ReLU chặn trên ở 6).
- **Tính tham số được học:** đặc trưng trước GAP là 7 × 7 × 1.280 → GAP → 1.280 số. Dense 64: (1.280 + 1) × 64 = 81.984; Dense 1: 65; tổng **82.049**.

**Thầy dễ hỏi.**

**K32. Depthwise separable rẻ hơn vì sao? Tính một ví dụ.**
Conv thường vừa lọc không gian vừa trộn kênh cùng lúc. Tách làm hai bước thì rẻ hơn: với 384 kênh, 1.327.104 so với 150.912 phép nhân mỗi vị trí, khoảng 8,8 lần.

**K33. "Inverted" và "linear bottleneck" nghĩa là gì?**
Inverted: mở rộng kênh ở giữa (64 → 384 → 64), hẹp ở hai đầu, ngược với khối bottleneck của ResNet-50 (rộng – hẹp – rộng). ResNet-18 của nhóm dùng basic block nên không có bottleneck. Linear bottleneck: phép chiếu về ít kênh không dùng ReLU để không mất thông tin.

**K34. MobileNetV2 tổng tham số ít hơn ResNet-18 sao lại học nhiều tham số hơn?**
Vì đầu Dense mới nhận 1.280 đặc trưng (MobileNetV2) so với 512 (ResNet-18). (1.280 + 1) × 64 + 65 = 82.049 lớn hơn (512 + 1) × 64 + 65 = 32.897.

**K35. Đóng băng mà ResNet-18 vẫn huấn luyện lâu hơn MobileNetV2?**
Đúng (990,97 s so với 557,11 s, một lượt trên T4, Chương 3). Backbone đóng băng vẫn phải chạy xuôi cho mọi ảnh ở mọi epoch, cả train lẫn validation. Backbone ResNet-18 nặng hơn MobileNetV2 nhiều về phép tính (MobileNetV2 được thiết kế bằng tích chập tách biệt để nhẹ). Vì vậy thời gian chủ yếu do phần chạy xuôi quyết định; số tham số được học (32.897 so với 82.049) ảnh hưởng rất ít. Đây chỉ là một lượt đo; thời gian còn phụ thuộc bước resize, đọc ghi bộ nhớ, mức tận dụng GPU.

#### Mục 2.4.1. Thiết kế ba cấu hình thực nghiệm

**Mục này nói gì.** Bảng 2.8 so ba cấu hình (khởi tạo, phần được học, tổng tham số, tham số được học). Bảng 2.9 là thiết lập chung: 30 epoch, batch 32, Adam 0,001, BCE, seed 42, ReduceLROnPlateau, checkpoint theo validation loss, tắt dừng sớm, ngưỡng 0,5. Hình 2.8 là quy trình thực nghiệm tổng thể. Bảng 2.10 tách yếu tố giữ chung với yếu tố khác nhau giữa ba cấu hình. Thiết lập được chốt trước khi xem test.

**Hiểu từ gốc.**

| Mô hình | Khởi tạo | Phần được học | Tổng tham số | Tham số được học |
|---|---|---|---:|---:|
| CNN | Từ đầu | Toàn mạng | 100.993 | 100.993 |
| ResNet-18 | ImageNet | Đầu Dense mới | 11.219.009 | 32.897 |
| MobileNetV2 | ImageNet | Đầu Dense mới | 2.340.033 | 82.049 |

- **Epoch** = một lượt qua hết 5.583 ảnh train. **Batch 32** = mỗi lần cập nhật dùng 32 ảnh, nên mỗi epoch có ⌈5.583/32⌉ = 175 lần cập nhật.
- Train được **xáo trộn** khi tạo batch; validation và test giữ thứ tự cố định để ghép đúng xác suất với đường dẫn ảnh.
- **Seed 42** cố định phép chia tập (chạy lại ra đúng danh sách cũ), cách khởi tạo trọng số và thứ tự xáo trộn. Trên GPU, nhóm không bật chế độ tính toán tất định, nên chạy lại có thể cho số lệch nhỏ. Vì vậy nhóm đã lưu checkpoint và mã băm để kiểm lại đúng kết quả đã báo cáo.
- **Hình 2.8 (quy trình thực nghiệm):** một danh sách chia tập dùng chung → huấn luyện ba cấu hình → chọn mô hình chỉ bằng validation loss, ghi `selection.json` → đánh giá cả ba trên test → nạp lại trên CPU, so 10 ảnh test (MobileNetV2 thêm 10 ảnh validation) → đưa vào demo. Mã băm SHA-256 của danh sách chia được ghi trong kết quả mỗi lượt và trùng nhau ở cả ba lượt.
- **Bảng 2.10, nửa trên (giữ chung):** danh sách ảnh train/validation/test (Bảng 2.2, 3.1); thiết lập huấn luyện (Bảng 2.9); cách chọn checkpoint, ngưỡng, thước đo (mục 2.4.3–2.4.5); cấu trúc đầu phân loại GAP → Dense 64 ReLU → Dropout 0,3 → Dense 1 sigmoid.
- **Bảng 2.10, nửa dưới (khác nhau):**

| Yếu tố | CNN | ResNet-18 | MobileNetV2 |
|---|---|---|---|
| Phần trích xuất đặc trưng | 3 khối Conv2D – MaxPooling2D | khối dư | inverted residual |
| Đầu vào mạng | 256 × 256 × 1 | 224 × 224 × 3, lặp kênh xám | 224 × 224 × 3, lặp kênh xám |
| Khởi tạo phần trích xuất | ngẫu nhiên, học từ đầu | ImageNet, cố định | ImageNet, cố định |
| Số đặc trưng vào đầu phân loại | 128 | 512 | 1.280 |
| Số tham số cố định | 0 | 11.186.112 | 2.257.984 |

**Thầy dễ hỏi.**

**K36. Vì sao 30 epoch, batch 32?**
Nhóm **chưa khảo sát** các giá trị khác. 30 epoch và batch 32 là thiết lập phổ biến; ba lượt trên Tesla T4 mất tổng khoảng 31 phút (300 + 991 + 557 giây, Chương 3). CNN và ResNet-18 đều có validation loss thấp nhất ở epoch 29, gần cuối lượt, nên có thể vẫn còn cải thiện nếu học thêm. MobileNetV2 dừng cải thiện từ epoch 21. Muốn kết luận phải chạy thêm epoch.

**K37. "Chốt trước khi đánh giá test" nghĩa là gì?**
Mọi thiết lập và quy tắc chọn được quyết định trên train/validation. Nhóm không quay lại sửa thiết lập sau khi thấy điểm test, để test vẫn là phép đo khách quan.

**K66. Chạy lại với seed 42 có ra đúng 99,44% không?**
Không chắc. Phép chia tập thì lặp lại y hệt. Huấn luyện trên GPU không bật chế độ tất định, nên có thể lệch nhỏ. Kết quả báo cáo được kiểm bằng checkpoint đã lưu và mã băm SHA-256 của checkpoint (`checkpoint_sha256` trong `fit_*.json`).

**K71. Bảng 2.10 cho thấy điều gì? Thiết kế này trả lời được câu hỏi nào, không trả lời được câu nào?**
Nửa trên là phần giữ chung: dữ liệu, thiết lập huấn luyện, cách chọn và đánh giá, cấu trúc đầu phân loại. Nửa dưới là phần khác nhau: phần trích đặc trưng, độ phân giải, khởi tạo, số đặc trưng vào đầu phân loại, số tham số cố định. Thiết kế trả lời được "cấu hình nào tốt hơn trong cùng điều kiện dữ liệu và huấn luyện". Nó **không** tách được đóng góp riêng của trọng số ImageNet, kiến trúc hay độ phân giải. Muốn tách phải chạy thêm lượt chỉ đổi một yếu tố, ví dụ CNN ở 224 × 224, hoặc MobileNetV2 khởi tạo ngẫu nhiên (mục 2.4.1).

**K72. Đầu phân loại ba mô hình giống nhau, vậy số tham số học khác nhau ở đâu ra?**
Từ số đặc trưng đưa vào Dense 64: 128 (CNN), 512 (ResNet-18), 1.280 (MobileNetV2). Riêng CNN còn học cả phần trích đặc trưng; hai mạng kia giữ cố định 11.186.112 và 2.257.984 tham số backbone (Bảng 2.10).

#### Mục 2.4.2. Hàm mất mát và quá trình cập nhật

**Mục này nói gì.** BCE là log-hợp lý âm của phân phối Bernoulli, hợp với một nơ-ron sigmoid. Loss và accuracy đo hai khía cạnh khác nhau. Adam với tốc độ học 0,001 cập nhật phần tham số được phép học. Không dùng class weight, L1/L2.

**Hiểu từ gốc.**

- **BCE cho một ảnh:** L = −[y·ln(pₒₖ) + (1 − y)·ln(1 − pₒₖ)]. Ảnh đạt: L = −ln(pₒₖ). Ảnh lỗi: L = −ln(1 − pₒₖ).
- **Phạt mạnh khi tự tin mà sai.** Ví dụ một ảnh **lỗi** (y = 0):

| pₒₖ mô hình cho | Dự đoán | Loss = −ln(1 − pₒₖ) |
|---:|---|---:|
| 0,01 | Lỗi, rất chắc | 0,010 |
| 0,10 | Lỗi | 0,105 |
| 0,50 | Đạt (sát ngưỡng) | 0,693 |
| 0,60 | Đạt, sai | 0,916 |
| 0,99 | Đạt, sai mà rất tự tin | 4,605 |

  Hai dòng 0,60 và 0,99 đều là "sai 1 ảnh" với accuracy, nhưng loss chênh gần 5 lần. Vì vậy loss có thể nhảy mạnh khi accuracy chỉ giảm chút ít.
- **Vòng cập nhật mỗi batch:** lan truyền xuôi → tính loss → lan truyền ngược tính gradient → Adam cập nhật trọng số. CNN cập nhật mọi tầng có trọng số; hai mạng học chuyển giao chỉ cập nhật hai tầng Dense mới.
- **Adam trong 2 câu:** Adam nhớ trung bình trượt của gradient (moment bậc một, giống quán tính) và của bình phương gradient (moment bậc hai, đo độ lớn). Bước cập nhật của mỗi tham số ≈ tốc độ học × (moment một / căn moment hai), nên tham số có gradient lớn không bị nhảy quá xa, tham số có gradient nhỏ vẫn được cập nhật. Nói gọn: Adam nhớ hướng đi gần đây (để đi đều) và độ lớn gradient gần đây (để tự chỉnh cỡ bước cho từng trọng số).
- **Tốc độ học 0,001** là "cỡ bước" mỗi lần cập nhật; 0,001 là giá trị mặc định do tác giả Adam đề xuất [7]. Quá lớn thì loss dao động, quá nhỏ thì học chậm.

**Thầy dễ hỏi.**

**K38. Vì sao dùng BCE?**
Đầu ra là một nơ-ron sigmoid cho xác suất lớp đạt, nhãn là 0/1. BCE chính là log-hợp lý âm của phân phối Bernoulli, nên cực tiểu BCE tương đương làm cho mô hình gán xác suất cao cho nhãn đúng.

**K39. Tính loss khi ảnh đạt được cho pₒₖ = 0,9 và 0,1.**
−ln(0,9) ≈ 0,105 và −ln(0,1) ≈ 2,303. Gán xác suất thấp cho nhãn đúng bị phạt nặng gấp khoảng 22 lần.

**K40. Có dùng class weight không? Lệch lớp thì sao?**
Không. Train lệch nhẹ (khoảng 57,2% lỗi). Nhóm để class weight và L1/L2 cho thực nghiệm sau (mục 2.4.2), và bù lại bằng cách đánh giá thêm F1, macro F1. Hướng làm: thử class_weight hoặc chọn ngưỡng trên validation; nhóm **chưa khảo sát**.

**K65. Trọng số ban đầu và tham số Adam là gì?**
Báo cáo không ghi chi tiết này; nhóm dùng mặc định của Keras. CNN và hai tầng Dense mới khởi tạo Glorot uniform, bias 0, cố định bằng seed 42. Adam: β₁ = 0,9; β₂ = 0,999; ε = 10⁻⁷; tốc độ học khởi đầu 0,001.

#### Mục 2.4.3. Theo dõi và lưu checkpoint

**Mục này nói gì.** Bốn callback: ModelCheckpoint (lưu bản có validation loss thấp nhất), CSVLogger (ghi loss, accuracy train/validation mỗi epoch), ReduceLROnPlateau (giảm tốc độ học), TerminateOnNaN (dừng nếu loss lỗi số). Dừng sớm bị tắt. Mỗi lượt lưu epoch chọn, thời gian và mã băm checkpoint.

**Hiểu từ gốc.**

- **Checkpoint** là bản lưu trọng số tại một epoch. Lưu theo **validation loss** vì đó là ảnh mô hình không học; train loss gần như luôn giảm, kể cả khi mô hình bắt đầu học thuộc train (quá khớp).
- **ReduceLROnPlateau** (hệ số 0,5; patience 3; tối thiểu 10⁻⁶): validation loss không cải thiện 3 epoch liền thì tốc độ học giảm một nửa, ví dụ 0,001 → 0,0005 → 0,00025…, không thấp hơn 0,000001. "Không cải thiện" nghĩa là validation loss không giảm quá ngưỡng nhỏ mặc định 10⁻⁴ của Keras. Lưu ý: lịch sử không ghi tốc độ học, nên **không biết** nó giảm ở epoch nào. Nói gọn: nếu 3 epoch liền validation loss không giảm thì bước học giảm một nửa, giống đi chậm lại khi gần đích.
- **Vì sao tắt dừng sớm?** Để có đủ lịch sử 30 epoch cho cả ba mô hình, so đường học trên cùng trục. ModelCheckpoint đã làm việc "chọn điểm dừng" thay cho dừng sớm.

**Thầy dễ hỏi.**

**K41. Không dừng sớm thì có bị quá khớp không?**
Có thể quá khớp ở cuối lượt, nhưng bản được dùng là checkpoint có validation loss thấp nhất, không phải bản epoch 30. Ví dụ MobileNetV2 dùng epoch 21 (Chương 3).

**K42. Tốc độ học giảm ở epoch nào?**
Nhóm không xác định được. `history_*.csv` chỉ có các cột epoch, loss, accuracy, val_loss, val_accuracy, không có cột tốc độ học. Báo cáo chỉ ghi quy tắc giảm (hệ số 0,5, patience 3, tối thiểu 10⁻⁶), không ghi epoch giảm thực tế. Lượt sau nhóm sẽ ghi thêm tốc độ học vào lịch sử.

**K67. "Epoch 21" đếm từ 0 hay từ 1? Mỗi epoch có bao nhiêu bước?**
Đếm từ 1: `fit_*.json` ghi `best_epoch` = chỉ số trong `history_*.csv` (đếm từ 0) cộng 1. Ví dụ MobileNetV2: dòng epoch = 20 trong `history_mobilenetv2.csv` là epoch 21. Mỗi epoch có ⌈5.583/32⌉ = 175 bước cập nhật (bước cuối chỉ có 15 ảnh); 30 epoch là 5.250 bước.

#### Mục 2.4.4. Tiêu chí đánh giá

**Mục này nói gì.** Lớp lỗi (nhãn 0) là lớp quan tâm. Bảng 2.11 đặt TP, FN, FP, TN trong ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán, thứ tự lỗi rồi đạt). Công thức (2.5), (2.6) cho accuracy, precision, recall, F1. Dùng accuracy và macro F1 cho chung hai lớp, theo dõi precision, recall, F1 lớp lỗi. Ngưỡng cố định 0,5.

**Hiểu từ gốc.**

| Nhãn thật \ Dự đoán | Lỗi | Đạt |
|---|---|---|
| **Lỗi** | **TP**: lỗi, bắt đúng | **FN**: lỗi bị cho là đạt (**bỏ sót**) |
| **Đạt** | **FP**: đạt bị báo lỗi (**báo nhầm**) | **TN**: đạt, nhận đúng |

- Mẹo nhớ: chữ thứ hai (P/N) là **dự đoán** (P = báo lỗi, N = cho đạt); chữ thứ nhất (T/F) là dự đoán **đúng hay sai**. FN = "cho đạt, mà sai" = bỏ sót lỗi.
- **(2.5)** Accuracy = (TP + TN)/(TP + TN + FP + FN); Precision = TP/(TP + FP): trong các ảnh bị báo lỗi, bao nhiêu ảnh lỗi thật.
- **(2.6)** Recall = TP/(TP + FN): trong các ảnh lỗi thật, bắt được bao nhiêu. F1 = 2 × Precision × Recall/(Precision + Recall): trung bình điều hòa, thấp nếu một trong hai thấp.
- **Macro F1** = (F1 lỗi + F1 đạt)/2, trong đó F1 đạt tính lại khi coi lớp đạt là lớp quan tâm. Hai lớp nặng như nhau.
- **Ví dụ số – ma trận thật của CNN trên test (kết quả Chương 3, Bảng 3.5; chỉ số theo từng lớp ở Bảng 3.4): TP = 428, FN = 25, FP = 12, TN = 250.**

| Chỉ số | Phép tính | Kết quả |
|---|---|---:|
| Accuracy | (428 + 250)/715 = 678/715 | 94,83% |
| Precision lỗi | 428/(428 + 12) = 428/440 | 97,27% |
| Recall lỗi | 428/(428 + 25) = 428/453 | 94,48% |
| F1 lỗi | 2 × 0,9727 × 0,9448/(0,9727 + 0,9448) | 95,86% |
| Precision đạt | TN/(TN + FN) = 250/275 | 90,91% |
| Recall đạt | TN/(TN + FP) = 250/262 | 95,42% |
| F1 đạt | 2 × 0,9091 × 0,9542/(0,9091 + 0,9542) | 93,11% |
| Macro F1 | (95,86 + 93,11)/2 | 94,48% |

  Recall lỗi và macro F1 của CNN cùng làm tròn ra 94,48% là **trùng hợp**, hai số tính khác nhau. CNN có recall lỗi thấp hơn precision vì bỏ sót (25) nhiều hơn báo nhầm (12).

**Thầy dễ hỏi.**

**K43. FN trong bài này là gì? Vì sao đáng chú ý?**
FN là sản phẩm lỗi bị mô hình cho là đạt, tức là lọt lỗi ra ngoài. Trong kiểm tra chất lượng, lỗi này thường tốn kém hơn FP (FP thường chỉ làm mất công kiểm lại), nhưng báo cáo không định lượng chi phí; báo cáo thống kê riêng FN và FP (mục 2.4.4) và nêu hệ quả của từng loại ở mục 3.4.2. Recall lớp lỗi theo dõi trực tiếp FN.

**K44. Accuracy đã cao sao còn cần precision, recall, macro F1?**
Test có khoảng 63,4% ảnh lỗi; mô hình đoán toàn "lỗi" đã được 63,4% accuracy mà vô dụng: F1 lớp đạt = 0, nên macro F1 chỉ khoảng 38,8% (F1 lỗi = 906/1.168 ≈ 77,6%). Các chỉ số theo lớp cho biết sai kiểu nào, macro F1 cho biết cả lớp ít ảnh hơn có tốt không.

**K45. Precision và recall khác nhau thế nào? Cho ví dụ.**
Precision nhìn **cột dự đoán lỗi**: báo lỗi 440 ảnh, đúng 428 → 97,27%. Recall nhìn **hàng lỗi thật**: có 453 ảnh lỗi, bắt được 428 → 94,48% (CNN, Chương 3).

**K46. Muốn giảm bỏ sót thì đổi ngưỡng thế nào?**
Quy tắc là pₒₖ ≥ ngưỡng thì đạt, nên muốn bớt bỏ sót phải **tăng** ngưỡng (khó được gọi là đạt hơn); đổi lại báo nhầm tăng. Báo cáo cố định 0,5; ngưỡng mới phải chọn trên validation. Mức đánh đổi trên test được đếm ở Bảng 3.7, mục 3.4.4 (Chương 3), xem [mục 8.3](#83-ảnh-hưởng-của-ngưỡng-bảng-37).

#### Mục 2.4.5. Lựa chọn mô hình trước đánh giá test

**Mục này nói gì.** So validation loss của checkpoint tốt nhất ở ba mô hình, chọn mô hình có giá trị nhỏ nhất. Quyết định (tên mô hình, epoch, validation loss) được ghi lại **trước** khi chạy test. Sau đó cả ba checkpoint được đánh giá trên cùng 715 ảnh test, lưu dự đoán từng ảnh.

**Hiểu từ gốc.**

- **Vì sao chọn trước khi xem test?** Nếu xem test rồi mới chọn, ta đang "chọn theo đề thi", điểm test sẽ lạc quan. Căn cứ để kiểm là **mã trong notebook**: ô chọn mô hình chỉ đọc validation loss từ `fit_*.json`, lấy giá trị nhỏ nhất và ghi `selection.json`; ô sau đó mới nạp checkpoint và chạy test. Ô chọn không dùng dữ liệu test. (Riêng tệp `selection.json` không có dấu thời gian, nên tự nó không chứng minh được thứ tự.)
- **Vì sao theo validation loss, không theo accuracy?** Trên 986 ảnh, accuracy chỉ đổi theo bước 1/986 ≈ 0,1 điểm phần trăm và dễ bằng nhau giữa các cấu hình. Loss thay đổi liên tục theo xác suất, phân biệt tinh hơn. Nói gọn: accuracy chỉ đếm đúng/sai nên nhảy từng nấc 0,1%; loss còn tính mô hình chắc chắn đến đâu, nên so được hai mô hình cùng accuracy.
- **Kết quả lựa chọn (Chương 3, `selection.json`):** CNN 0,1210 · ResNet-18 0,0269 · **MobileNetV2 0,0150 (epoch 21)** → chọn MobileNetV2.

**Thầy dễ hỏi.**

**K47. Nếu chọn theo test thì sao? Có ai kiểm được là nhóm chọn trước không?**
Chọn theo test làm điểm test lạc quan. Trong notebook, ô chọn mô hình chỉ đọc validation loss từ `fit_*.json`, lấy giá trị nhỏ nhất và ghi `selection.json` (tên mô hình, epoch, validation loss, mã băm checkpoint). Chỉ ô sau đó mới nạp checkpoint và chạy test, đồng thời kiểm lại mã băm checkpoint. Thầy có thể xem thứ tự các ô và mã chọn để kiểm.

**K48. Nếu mô hình được chọn lại kém nhất trên test thì sao?**
Vẫn báo cáo đúng như vậy; quy tắc chọn đã chốt trước. Thực tế MobileNetV2 cũng cao nhất trên test (99,44%, Chương 3), nhưng đó là kết quả, không phải lý do chọn.

#### Mục 2.5. Kết luận chương

**Mục này nói gì.** Chương 2 đã xác định bài toán, cách xử lý dữ liệu, ba mô hình và quy trình huấn luyện, lựa chọn, đánh giá. Ba quyết định ảnh hưởng trực tiếp đến cách đọc Chương 3: loại 64 ảnh train trùng test trước khi tách validation; chọn checkpoint và mô hình theo validation loss trước khi chạy test; giữ ngưỡng 0,5 cho cả ba mô hình. Giới hạn: kiểm trùng chỉ bắt ảnh giống hệt; thiết kế so sánh không tách được từng yếu tố (mục 2.4.1).

**Hiểu từ gốc.** Chương 2 là "kế hoạch thí nghiệm", Chương 3 là "kết quả thí nghiệm". Khi được hỏi số liệu kết quả, nói rõ "ở Chương 3" để thầy thấy nhóm phân biệt phương pháp với kết quả.

**Thầy dễ hỏi.**

**K49. Tóm tắt Chương 2 trong 3 câu.**
Bài toán phân loại ảnh đúc lỗi/đạt bằng điểm pₒₖ và ngưỡng 0,5. Dữ liệu được kiểm trùng SHA-256, chia train/validation/test và chuẩn hóa riêng cho từng mô hình. Ba mô hình (CNN từ đầu, ResNet-18 và MobileNetV2 đóng băng) học cùng thiết lập, chọn theo validation loss trước khi đánh giá test.

**K50. Số ảnh trùng và số ảnh mỗi tập nằm ở chương nào?**
Bản in đưa số ảnh qua các bước kiểm tra vào Chương 2 (Bảng 2.2: 7.348 nguồn, 64 ảnh trùng, 6.569 train sạch) vì chúng quyết định cách chia. Phân bố sau khi chia (5.583/986/715) nằm ở Chương 3 (Bảng 3.1). Mục 3.2 dẫn lại Bảng 2.2 chứ không lặp số.

### C. Bẫy hay gặp (dễ nói sai)

| Dễ nói sai | Nói đúng |
|---|---|
| "Lớp Conv thứ hai", "mô hình có 11 lớp" | Trong báo cáo, **tầng** là thành phần của mạng; **lớp** là nhãn (lỗi/đạt) |
| "pₒₖ là xác suất lỗi" | pₒₖ là điểm lớp **đạt**; điểm lỗi = 1 − pₒₖ |
| "Lớp lỗi là nhãn 1" / "lớp dương là nhãn 1" | Lỗi là nhãn **0**, nhưng là **lớp quan tâm** khi tính TP, precision, recall |
| "FN là báo nhầm" | FN = lỗi bị cho là đạt (**bỏ sót**); FP = đạt bị báo lỗi (**báo nhầm**) |
| "ResNet-18 dùng bottleneck 1 × 1 – 3 × 3 – 1 × 1" | ResNet-18 dùng **basic block** hai Conv 3 × 3; bottleneck là của ResNet-50 trở lên |
| "Nhóm đã fine-tune" | Backbone **đóng băng**, chỉ học đầu Dense mới; fine-tuning **chưa làm** |
| "Nhóm có dùng augmentation" | Nhóm **không thêm**; ảnh tăng cường có sẵn trong bộ nguồn |
| "Chọn MobileNetV2 vì test cao nhất" | Chọn theo **validation loss** nhỏ nhất, ghi lại **trước** khi chạy test |
| "Dữ liệu có 7.284 ảnh" / "loại 64 ảnh hỏng" | Nguồn 7.348 ảnh; loại **64 ảnh trùng** (không phải ảnh hỏng), còn 7.284 |
| "Đóng băng nên backbone không chạy, huấn luyện nhanh" | Backbone vẫn chạy xuôi mỗi batch, chỉ không cập nhật trọng số |
| "SHA-256 loại hết rò rỉ" | Chỉ loại bản sao y hệt; biến thể gần giống có thể còn |
| "Muốn bớt bỏ sót thì hạ ngưỡng" | Phải **tăng** ngưỡng pₒₖ, đổi lại báo nhầm tăng |
| "Theo Bảng 3.7, nhóm chọn ngưỡng 0,96" | Bảng 3.7 là **phân tích bổ sung trên test**, không dùng để chọn ngưỡng; ngưỡng vẫn là 0,5 |
| "MobileNetV2 chắc chắn tốt hơn ResNet-18" | Chênh 5 ảnh, xác suất ≈ 0,23 nên **chưa đủ khẳng định**; hơn CNN thì rõ (xác suất < 10⁻⁴) |
| "Số liệu dữ liệu chỉ có ở Chương 3" | Số ảnh qua các bước kiểm tra ở **Bảng 2.2** (Chương 2); phân bố sau chia ở Bảng 3.1 |

### D. Câu hỏi khó / mở rộng

**K51. Có thể còn rò rỉ do biến thể gần giống không? Ảnh hưởng thế nào?**
Có thể. Nguồn đã có ảnh tăng cường, không có mã sản phẩm gốc, SHA-256 chỉ bắt bản y hệt. Nếu ảnh xoay của cùng một sản phẩm nằm ở cả train và test thì điểm test có thể lạc quan so với ảnh mới tại nhà máy. Ba mô hình dùng cùng danh sách chia, nên chịu cùng một rủi ro dữ liệu. Tuy vậy, mức lạc quan có thể khác giữa các mô hình, nên nhóm không khẳng định thứ hạng sẽ giữ nguyên trên ảnh mới. Mục 3.5.4 ghi rõ 99,44% chỉ phản ánh phạm vi bộ dữ liệu này. Hướng làm: chia theo nhóm sản phẩm nếu có thông tin, hoặc lọc ảnh gần giống.

**K52. CNN dùng 256 × 256 × 1, hai mạng kia dùng 224 × 224 × 3 và trọng số ImageNet. So vậy có cùng điều kiện không?**
Không hoàn toàn. Ba cấu hình chung dữ liệu, thiết lập huấn luyện, ngưỡng và thước đo, nhưng khác cùng lúc kiến trúc, độ phân giải, khởi tạo và phần được học. Nhóm chỉ kết luận cho cả cấu hình; muốn tách yếu tố cần thí nghiệm loại trừ (ví dụ CNN ở 224, hoặc ResNet-18 khởi tạo ngẫu nhiên), nhóm **chưa làm**.

**K53. Chỉ chạy một seed thì chênh 5 ảnh giữa MobileNetV2 và ResNet-18 có đáng tin không?**
Chưa đủ để khẳng định (mục 3.4.3, kết quả Chương 3). Chỉ 11 ảnh có một mô hình sai còn mô hình kia đúng: ResNet-18 sai riêng 8, MobileNetV2 sai riêng 3. Nếu hai mô hình ngang nhau, xác suất lệch đến mức này là khoảng 0,23, quá lớn để kết luận. Ngoài ra chỉ có một seed. Hướng làm: chạy 3–5 seed và báo trung bình ± độ lệch. Câu đầy đủ ở [câu 39, mục 6](#g-nội-dung-mới-của-bản-in).

**K54. Vì sao 256 × 256, 30 epoch, batch 32, tốc độ học 0,001?**
Nhóm **không khảo sát** siêu tham số (mục 3.5.4). Nhận xét thêm: 256 nhỏ hơn ảnh gốc 300 và chia đôi gọn qua ba MaxPooling (256 → 128 → 64 → 32); đây là lý do thuận tiện, không phải kết quả thử nghiệm. 0,001 là tốc độ học mặc định đề xuất cho Adam [7]. 30 epoch, batch 32 là thiết lập phổ biến. Muốn tối ưu cần tìm kiếm trên validation.

**K55. Vì sao không fine-tune? Fine-tune thì làm thế nào?**
Mục tiêu là đo đóng góp của đặc trưng ImageNet với chi phí thấp, và train chỉ khoảng 5.600 ảnh. Cách làm sau này: huấn luyện đầu Dense như hiện tại, rồi mở khóa vài khối cuối của backbone, học tiếp với tốc độ học nhỏ hơn (ví dụ 10⁻⁵), vẫn giữ Batch Normalization ở chế độ suy luận. Nhóm chưa có số.

**K56. Vì sao không dùng class weight?**
Train chỉ lệch nhẹ (57,2% lỗi) và nhóm chọn thiết lập tối giản. Class weight nhân loss của lớp ít ảnh lên để mô hình chú ý hơn; nhóm để lại cho thực nghiệm sau và theo dõi macro F1 để thấy lớp ít ảnh có bị bỏ quên không.

**K57. Sao không chọn ngưỡng khác 0,5?**
Báo cáo cố định 0,5 cho cả ba mô hình để so sánh cùng quy tắc. Chọn ngưỡng theo chi phí bỏ sót/báo nhầm là hướng phát triển, phải làm trên validation, không làm trên test. Lượt chạy này chỉ lưu dự đoán 10 ảnh validation nên chưa chọn lại được. Bảng 3.7 chỉ minh họa đánh đổi trên test (xem [mục 8.3](#83-ảnh-hưởng-của-ngưỡng-bảng-37)).

**K58. Làm sao biết mô hình nhìn vào vết lỗi chứ không nhìn nền?**
Nhóm **chưa kiểm tra**. Có thể dùng Grad-CAM: lấy gradient của điểm đầu ra theo bản đồ đặc trưng tầng tích chập cuối, tô lên ảnh vùng ảnh hưởng mạnh. Dữ liệu không có chú giải vị trí lỗi nên cũng chưa đo được định lượng.

**K59. Điểm pₒₖ = 0,96 có nghĩa đúng 96% không?**
Không. Điểm sigmoid **chưa được hiệu chuẩn**. Ví dụ ảnh lỗi `cast_def_0_150.jpeg` bị MobileNetV2 cho pₒₖ ≈ 0,9556 (Chương 3). Muốn hiểu điểm như xác suất thật cần hiệu chuẩn trên validation, nhóm chưa làm.

### E. Phân vai nhanh cho buổi sáng (gợi ý)

Theo bảng phân công trong báo cáo (xem thêm [mục 7](#7-ai-trình-bày-phần-nào)). Đây là **gợi ý**, nhóm tự chốt lại.

| Phần Chương 2 | Người nói chính | Người đỡ khi bị hỏi | Ôn kỹ nhất |
|---|---|---|---|
| 2.1 Phát biểu bài toán | Minh | Thịnh | (2.1)–(2.3), pₒₖ, ngưỡng 0,5 |
| 2.2 Dữ liệu (2.2.1–2.2.4) | **Thịnh** | Minh | Hình 2.1, SHA-256, 64 ảnh trùng, stratify, Bảng 2.2, 2.3, 5.583/986/715 |
| 2.3 Ba mô hình | **Minh** | Quý | Tự tính 100.993, 32.897, 82.049; khối dư; depthwise |
| 2.4.1–2.4.3, 2.4.5 Huấn luyện, chọn mô hình | **Minh** | Quý | Hình 2.8, Bảng 2.10, BCE, Adam, ReduceLROnPlateau, checkpoint, chọn trước test |
| 2.4.4 Tiêu chí đánh giá | **Quý** | Minh | TP/FN/FP/TN, precision, recall, F1, macro F1 |
| Demo | **Quý** | Minh | Chuỗi tiền xử lý trong demo khớp 2.2.4, ngưỡng 0,5 |
| 2.5 Kết luận chương | Minh hoặc Quý | | Câu K49 |

**Mẹo khi bị hỏi phần không phải mình:**

1. **Bắt đầu từ khung 7 câu ở mục A**, rồi đi vào số. Ai cũng phải nói được mục A.
2. **Câu hỏi tính tay thì viết công thức trước, thay số sau**, ví dụ "(k × k × kênh vào + 1) × số bộ lọc = (3 × 3 × 32 + 1) × 64 = 18.496". Viết từng bước, sai còn được sửa.
3. **Câu "vì sao chọn X" mà nhóm không khảo sát:** nói thật "nhóm chưa khảo sát", nêu lý do hợp lý đã ghi trong báo cáo, rồi nêu cách sẽ thử. Không bịa số.
4. **Phân biệt phương pháp và kết quả:** nói "theo mục 2.x" khi là cách làm, "ở Chương 3" khi là con số kết quả.
5. **Dừng ở chỗ mình chắc**, rồi mời bạn phụ trách bổ sung; đừng đoán tiếp để lấp khoảng trống.

Ôn thêm: [Hiểu dự án trong 15 phút](#4-hiểu-dự-án-trong-15-phút) · [Bảng số liệu chính](#5-bảng-số-liệu-chính) · [43 câu ôn bảo vệ](#6-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi) · [Phép tính tay](#82-phép-tính-tay-có-thể-được-yêu-cầu-viết-lên-bảng).

---

## 3. Cấu trúc thư mục và cách chạy

### 3.1. Những gì quan trọng trong thư mục

```text
Casting-Defect-Detection-CNN/
├── bao_cao/baocaocodong_v4.docx, .pdf   Báo cáo chính thức (bản v3 trong cùng thư mục là bản cũ)
├── Casting.ipynb                        Notebook huấn luyện + đánh giá, có sẵn output của lượt chính thức
├── casting_full_3models_01/             Kết quả chính thức (mọi số liệu trong báo cáo lấy từ đây)
│   ├── config.json, environment.json    Cấu hình chạy và phiên bản thư viện
│   ├── data_check.json, split.json      Kiểm tra ảnh, danh sách ảnh trùng, danh sách chia tập
│   ├── split_counts.csv                 Số ảnh mỗi tập theo lớp
│   ├── excluded_train_duplicates.csv    64 ảnh train bị loại vì trùng test
│   ├── history_*.csv, fit_*.json        Loss/accuracy từng epoch, checkpoint được chọn, thời gian
│   ├── selection.json                   Quyết định chọn mô hình (ghi trước khi test)
│   ├── metrics_*.json, comparison.csv   Chỉ số test, ma trận nhầm lẫn, bảng so sánh
│   ├── test_predictions_*.csv           Điểm p_ok của từng ảnh test
│   ├── errors_*.csv                     Danh sách ảnh dự đoán sai
│   ├── architecture_*.json, model_summary_*.txt   Cấu trúc mạng và số tham số
│   ├── learning_curves_*.png, confusion_matrix_*.png   Hình dùng trong Chương 3
│   ├── cnn.keras, resnet18.keras, mobilenetv2.keras    Ba checkpoint đã chọn
│   └── demo_test/                       Ảnh test mẫu để thử demo
├── demo/                                Ứng dụng Streamlit chạy trên máy
│   ├── app.py, inference.py, style.css  Giao diện và phần nạp mô hình
│   ├── prepare_models.py                Chép checkpoint + metadata vào demo/models/ và kiểm chứng
│   ├── requirements.txt                 Thư viện cho demo
│   ├── models/                          Checkpoint cho demo (bị Git bỏ qua, phải tải kèm)
│   ├── reference/                       Ảnh tham chiếu và ảnh chụp giao diện (bị Git bỏ qua, chỉ có trên máy/Drive)
│   └── README.md                        Hướng dẫn demo chi tiết
├── .streamlit/config.toml               Giao diện Streamlit khi chạy từ gốc dự án
├── requirements.txt                     Môi trường đầy đủ (notebook + demo)
├── HUONG_DAN_CASTING.md                 Hướng dẫn chạy notebook trên Colab theo từng mục
├── HUONG_DAN_BAO_VE.md                  Đọc thêm: giải thích chi tiết từng phần để trình bày
├── CAU_HOI_VA_GIAI_THICH_BAO_VE.md      Đọc thêm: câu hỏi và cách trả lời theo chủ đề
└── prompt/                              Prompt và meta-prompt của dự án
```

`casting_full_3models_01_decay/` là một lượt thử lịch learning rate khác. **Báo cáo không dùng lượt này.** Không lấy số từ đó khi bảo vệ.

### 3.2. Môi trường

| Nơi chạy | Phiên bản đã dùng | Nguồn |
|---|---|---|
| Huấn luyện (Google Colab, GPU Tesla T4) | Python 3.13.16, TensorFlow 2.20.0, Keras 3.12.0, KerasHub 0.23.0, NumPy 2.2.6 | `environment.json`, mục 3.1 |
| Demo trên Windows (CPU) | Python 3.12.14, TensorFlow 2.20.0, Keras 3.12.0, KerasHub 0.23.0, NumPy 2.2.6, Pillow 11.3.0, Streamlit 1.67.0 | `.venv` của dự án, `demo/requirements.txt` |

<!-- Phiên bản demo kiểm bằng .venv/Scripts/python.exe ngày 24/11/2026 -->

### 3.3. Chạy lại notebook trên Google Colab

Muốn **xem** kết quả thì chỉ cần mở [Casting.ipynb](Casting.ipynb): notebook đã lưu output. Muốn **chạy lại**:

1. Nén thư mục `data/casting_data` thành ZIP. Trong ZIP phải có `train/def_front`, `train/ok_front`, `test/def_front`, `test/ok_front`.
2. Upload `Casting.ipynb` lên Colab, chọn GPU. Chạy mục 1 (cài thư viện), **khởi động lại session**, rồi chạy từ mục 2.
3. Ở mục 3, upload ZIP dữ liệu. Ở mục 4, kiểm tra output báo **64 ảnh trùng bị loại** và ba số **5.583 / 986 / 715**.
4. Chạy mục 5–8 để huấn luyện CNN, ResNet-18, MobileNetV2. Mỗi mô hình lưu checkpoint `.keras`, lịch sử và gói sao lưu. Nếu chạy thêm lượt mới, đặt `RUN_NAME` khác để không ghi đè.
5. Chạy mục 9 để chọn mô hình và đánh giá test. Mục 10 tải kết quả. Mục 11–13 chuẩn bị mô hình demo, kiểm chứng trên CPU, thử một ảnh và tải gói `demo_bundle.zip`, `complete_*.zip`.

Chi tiết từng bước: [HUONG_DAN_CASTING.md](HUONG_DAN_CASTING.md).

Lưu ý: notebook không bật chế độ tính toán tất định trên GPU, nên chạy lại có thể ra số lệch nhỏ dù cùng seed 42. Số trong báo cáo là của lượt `casting_full_3models_01`.

<a id="24-chạy-demo-streamlit-trên-máy-windows"></a>

### 3.4. Chạy demo Streamlit trên máy Windows

Mở PowerShell **tại thư mục gốc dự án**. Ví dụ dùng Python 3.12 bản 64 bit. Các lệnh gọi thẳng `python.exe` trong môi trường ảo nên không cần "activate".

```powershell
# 1. Tạo môi trường và cài thư viện (chỉ làm lần đầu)
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r .\demo\requirements.txt

# 2. Chuẩn bị ba mô hình vào demo\models\ (cần thư mục casting_full_3models_01 ở gốc dự án)
.\.venv-demo\Scripts\python.exe .\demo\prepare_models.py

# 3. Mở giao diện
.\.venv-demo\Scripts\python.exe -m streamlit run .\demo\app.py
```

Nếu máy đã có `.venv` đầy đủ của dự án, thay `.venv-demo` bằng `.venv` trong các lệnh.

**Cách dùng khi bảo vệ:**

1. Mở địa chỉ Local URL trong terminal (thường là `http://localhost:8501`). Giữ terminal mở.
2. Ở ô **Mô hình sử dụng**, giữ cả ba mô hình (mặc định) hoặc bỏ bớt.
3. Chọn ảnh rồi bấm **Phân loại tất cả**. Ảnh nên dùng:
   - Ảnh đạt: `casting_full_3models_01/demo_test/ok_front/cast_ok_0_45.jpeg` (cùng tệp với `demo/reference/cast_ok_0_45.jpeg`)
   - Ảnh lỗi: `casting_full_3models_01/demo_test/def_front/cast_def_0_1059.jpeg`
4. Đọc nhãn và điểm của từng mô hình. 2 ảnh × 3 mô hình = **6 kết quả**. Có thể tải CSV hoặc JSON (mỗi dòng là một cặp ảnh–mô hình).

Kết quả đúng phải như Hình 3.12 và 3.13 của báo cáo:

<!-- Nguồn: báo cáo mục 3.5.3; demo/reference/demo_cast_def_0_1059.txt -->

| Ảnh | Mô hình | Nhãn | Điểm đạt p_ok | Điểm lỗi = 1 − p_ok |
|---|---|---|---:|---:|
| `cast_ok_0_45.jpeg` (Hình 3.12) | CNN | Đạt | 0,848358 | 0,151642 |
| | ResNet-18 | Đạt | 0,999593 | 0,000407 |
| | MobileNetV2 | Đạt | 0,998192 | 0,001808 |
| `cast_def_0_1059.jpeg` (Hình 3.13) | CNN | Lỗi | 0,000039 | 0,999961 |
| | ResNet-18 | Lỗi | 0,003175 | 0,996825 |
| | MobileNetV2 | Lỗi | 0,000013 | 0,999987 |

Nếu máy demo hỏng khi bảo vệ: mở Hình 3.12 và 3.13 trong báo cáo, hoặc ảnh chụp giao diện `demo/reference/demo_cast_ok_0_45.png` và `demo/reference/demo_cast_def_0_1059.png`.

### 3.5. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `py -3.12` báo không tìm thấy Python | Máy chưa có Python 3.12 | Cài Python 3.11–3.13 bản 64 bit, hoặc dùng `.venv` có sẵn của dự án |
| `ModuleNotFoundError` khi chạy | Cài thư viện bằng một Python, chạy bằng Python khác | Cài lại bằng đúng `.\.venv-demo\Scripts\python.exe -m pip install -r .\demo\requirements.txt` |
| Giao diện báo "Thiếu cnn.keras. Chạy demo/prepare_models.py theo README." | Thư mục `demo/models/` trống. Thư mục này bị Git bỏ qua nên bản clone không có | Chạy `prepare_models.py` (bước 2), hoặc chép nguyên thư mục `demo/models/` từ gói của nhóm trên Drive |
| `prepare_models.py` lỗi không tìm thấy tệp | Thiếu `casting_full_3models_01/` ở gốc dự án | Giải nén `casting_full_3models_01.zip` ra gốc dự án rồi chạy lại |
| "Trọng số khác checkpoint đã kiểm chứng." | Tệp `.keras` trong `demo/models/` không khớp mã băm trong metadata | Chạy lại `prepare_models.py` từ bộ kết quả chính thức |
| "dự đoán CPU khác kết quả tham chiếu" khi chuẩn bị | Phiên bản thư viện khác bản đã kiểm | Cài đúng phiên bản trong `demo/requirements.txt` |
| Lần đầu bấm phân loại rất lâu | Nạp ba mô hình TensorFlow lần đầu | Chờ; các lần sau nhanh hơn. Nên chạy thử một lần trước giờ bảo vệ |
| Cổng 8501 đang bận | Đã có Streamlit khác chạy | Tắt terminal cũ (Ctrl+C) hoặc thêm `--server.port 8502` vào lệnh |

Chi tiết hơn: [demo/README.md](demo/README.md).

---

## 4. Hiểu dự án trong 15 phút

### 4.1. Bài toán

- **Đầu vào:** một ảnh mặt trước sản phẩm đúc.
- **Đầu ra:** một số `p_ok` trong khoảng 0–1, là điểm của lớp **đạt**. Điểm lỗi = `1 − p_ok`.
- **Quy tắc ra nhãn:** `p_ok ≥ 0,5` → **đạt**; `p_ok < 0,5` → **lỗi**. Ngưỡng 0,5 nghĩa là chọn lớp có điểm lớn hơn. Bằng nhau thì gán đạt (mục 2.1).
- **Nhãn:** 0 = `def_front` = lỗi; 1 = `ok_front` = đạt (Bảng 2.1).
- **Phạm vi:** phân loại **cả ảnh**. Dữ liệu chỉ có nhãn cấp ảnh, nên mô hình **không chỉ ra vị trí lỗi**.

Công thức (2.1) trong báo cáo viết gọn: `p = f(P(x); θ)`. Trong đó `x` là ảnh đã tiền xử lý, `P` là bước riêng của từng mô hình, `f` là mạng, `θ` là phần tham số được học.

### 4.2. Dữ liệu: 7.348 ảnh → 7.284 ảnh dùng được

<!-- Nguồn: data_check.json (original_counts, dimensions, bad_images, excluded 64), split_counts.csv; báo cáo mục 2.2, 3.2, Hình 2.1, Bảng 2.2, 3.1 -->

- Nguồn: thư mục `casting_data`, **7.348 ảnh JPEG 300 × 300**. Thư mục train có 6.633 ảnh (3.758 lỗi, 2.875 đạt), thư mục test có 715 ảnh (453 lỗi, 262 đạt).
- Ảnh lưu ba kênh nhưng ba kênh bằng nhau (nội dung xám). Khi đọc, nhóm chuyển về ảnh xám một kênh.
- Bộ nguồn **đã có sẵn ảnh tăng cường** (xoay, đổi sáng...). Nhóm **không thêm** augmentation (`pipeline_augmentation = false` trong `config.json`).
- Kiểm tra toàn bộ 7.348 ảnh: **không có ảnh hỏng**.
- Toàn bộ chuỗi xử lý vẽ ở **Hình 2.1**; số ảnh qua từng bước ở **Bảng 2.2**.

**Kiểm trùng bằng SHA-256.** SHA-256 là hàm băm: hai ảnh giống hệt từng pixel cho cùng một mã. Nhóm băm *kích thước + pixel của ảnh xám, trước khi resize* (mục 2.2.2).

- **Vì sao phải kiểm trùng?** Nếu một ảnh vừa nằm trong train vừa nằm trong test, mô hình "đã thấy đáp án". Accuracy test sẽ cao giả tạo. Đây là rò rỉ dữ liệu (data leakage).
- **Vì sao so pixel, không so tên tệp?** Ảnh cùng tên ở train và test có thể khác nội dung (còn 80 cặp như vậy), và ảnh khác tên có thể giống hệt nhau.
- **Vì sao băm trước khi resize?** Hai ảnh gốc khác nhau có thể trở nên giống nhau sau khi thu nhỏ. Băm trước resize tránh loại nhầm.
- **Kết quả:** **64 ảnh đạt** trong train trùng pixel với ảnh test cùng lớp. Nhóm **loại bản ở train**, giữ bản ở test. Không có trùng nào khác. Danh sách: [excluded_train_duplicates.csv](casting_full_3models_01/excluded_train_duplicates.csv).
- **Giới hạn:** SHA-256 chỉ bắt **bản sao y hệt**. Ảnh đã xoay hoặc đổi sáng từ cùng một ảnh gốc có mã khác, nên không bị phát hiện (mục 3.5.4).

**Vì sao giữ nguyên tập test gốc?** Test là "đề thi" do nguồn cung cấp. Nhóm không chọn lại ảnh test, chỉ loại bản trùng ở phía train. Nhờ vậy tập đánh giá giữ đủ 715 ảnh và không bị nhóm can thiệp. Chương trình đọc test trước train để ưu tiên giữ test.

**Chia validation.** Từ 6.633 − 64 = **6.569** ảnh train sạch, tách 15% làm validation bằng `train_test_split(test_size=0.15, stratify=nhãn, random_state=42)`. "Stratify" là giữ tỷ lệ lỗi/đạt giống nhau ở hai phần.

| Tập | Lỗi | Đạt | Tổng | Dùng để |
|---|---:|---:|---:|---|
| Train | 3.194 | 2.389 | **5.583** | Cập nhật trọng số |
| Validation | 564 | 422 | **986** | Giảm learning rate, chọn checkpoint, chọn mô hình |
| Test | 453 | 262 | **715** | Đánh giá một lần, sau khi đã chọn |
| Tổng | 4.211 | 3.073 | 7.284 | |

Tỷ lệ ảnh lỗi: train và validation khoảng **57,2%**, test khoảng **63,4%** (test giữ tỷ lệ của nguồn). Vì lệch lớp, báo cáo không chỉ dùng accuracy mà còn dùng precision, recall, F1, macro F1 và ma trận nhầm lẫn.

Sau khi chia, đối chiếu mã băm giữa ba tập không còn ảnh trùng pixel (mục 3.2).

### 4.3. Tiền xử lý cho từng mô hình

<!-- Nguồn: báo cáo mục 2.2.4, Bảng 2.4; doi_chieu_so_lieu.md mục 2 -->

Bước chung cho mọi ảnh: đọc bằng Pillow → xoay đúng chiều theo EXIF → chuyển xám → resize bilinear về **256 × 256 × 1** (giá trị 0–255).

| Mô hình | Bước riêng | Đầu vào mạng | Chuẩn hóa (nằm trong mô hình) |
|---|---|---|---|
| CNN | Không | 256 × 256 × 1 | `x / 255` (về 0–1) |
| ResNet-18 | Resize TensorFlow bilinear có antialias về 224 × 224, lặp kênh xám thành 3 kênh | 224 × 224 × 3 | `(x/255 − μ)/σ` theo ImageNet, μ = (0,485; 0,456; 0,406), σ = (0,229; 0,224; 0,225) |
| MobileNetV2 | Giống ResNet-18 | 224 × 224 × 3 | `x / 127,5 − 1` (về −1 đến 1) |

- **Vì sao 224 × 224 × 3?** Hai backbone học trên ImageNet với ảnh màu 3 kênh cỡ 224. Lặp kênh xám thành 3 kênh giống nhau để khớp hình dạng đầu vào. Cách này không thêm thông tin màu.
- **Vì sao đặt chuẩn hóa trong mô hình?** Để quy tắc chuẩn hóa được lưu cùng tệp `.keras`. Demo nạp mô hình là có luôn bước chuẩn hóa đúng.
- **Vì sao resize hai lần (256 rồi 224)?** Để mọi mô hình bắt đầu từ cùng một biểu diễn 256. Demo lặp lại đúng chuỗi này. Resize thẳng ảnh gốc về 224 bằng thư viện khác có thể làm số lệch đi.
- **Giới hạn:** thu nhỏ ảnh có thể làm mờ vết lỗi rất nhỏ.

### 4.4. CNN baseline: từng tầng và số tham số

<!-- Nguồn: Bảng 2.5, Hình 2.3, model_summary_cnn.txt, fit_cnn.json (100993/100993) -->

"Baseline" là mô hình mốc để so sánh. Nhóm tự chọn số tầng, số bộ lọc và đầu phân loại, ghép từ các tầng Keras có sẵn, rồi **học toàn bộ từ khởi tạo ngẫu nhiên**. Đây không phải kiến trúc mới.

| Tầng | Cấu hình | Đầu ra | Số tham số | Cách tính |
|---|---|---|---:|---|
| Rescaling | x/255 | 256 × 256 × 1 | 0 | |
| Conv2D | 32 bộ lọc 3 × 3, stride 1, padding same, ReLU | 256 × 256 × 32 | 320 | (3·3·1 + 1) × 32 |
| MaxPooling2D | 2 × 2, stride 2 | 128 × 128 × 32 | 0 | |
| Conv2D | 64 bộ lọc 3 × 3, ReLU | 128 × 128 × 64 | 18.496 | (3·3·32 + 1) × 64 |
| MaxPooling2D | 2 × 2 | 64 × 64 × 64 | 0 | |
| Conv2D | 128 bộ lọc 3 × 3, ReLU | 64 × 64 × 128 | 73.856 | (3·3·64 + 1) × 128 |
| MaxPooling2D | 2 × 2 | 32 × 32 × 128 | 0 | |
| Global Average Pooling | trung bình từng kênh | 128 | 0 | |
| Dense | 64 nơ-ron, ReLU | 64 | 8.256 | (128 + 1) × 64 |
| Dropout | tỷ lệ 0,3 | 64 | 0 | |
| Dense | 1 nơ-ron, sigmoid → `p_ok` | 1 | 65 | (64 + 1) × 1 |
| **Tổng** | | | **100.993** | toàn bộ được học |

Giải thích từng thành phần bằng lời:

- **Conv2D:** các bộ lọc 3 × 3 trượt trên ảnh, mỗi bộ lọc tìm một kiểu mẫu (cạnh, vết, kết cấu). Padding `same` giữ nguyên chiều cao, chiều rộng. Số tham số một tầng Conv = (k × k × số kênh vào + 1) × số bộ lọc. Số "+1" là bias của mỗi bộ lọc.
- **ReLU:** `max(0, z)`, giữ phần dương, đưa phần âm về 0. Nhờ đó mạng học được quan hệ phi tuyến.
- **MaxPooling 2 × 2:** lấy giá trị lớn nhất trong mỗi ô 2 × 2, chiều cao và chiều rộng giảm một nửa: 256 → 128 → 64 → 32. Không có tham số.
- Mỗi lần kích thước giảm một nửa, số bộ lọc tăng gấp đôi (32 → 64 → 128).
- **Global Average Pooling (GAP):** lấy trung bình mỗi bản đồ 32 × 32 thành một số, nên 128 bản đồ → 128 số. Nếu dùng Flatten, Dense 64 sẽ cần (131.072 + 1) × 64 = **8.388.672** tham số thay vì **8.256**.
- **Dropout 0,3:** khi huấn luyện, tắt ngẫu nhiên 30% giá trị để mạng không dựa vào vài đặc trưng. Khi dự đoán thì không tắt.
- **Sigmoid:** `σ(z) = 1 / (1 + e^(−z))` (công thức 2.4), đưa đầu ra về 0–1. `σ(0) = 0,5`.
- Chỉ có **5 tầng có tham số**: 3 Conv2D và 2 Dense (Conv: 320 + 18.496 + 73.856 = 92.672; Dense: 8.256 + 65 = 8.321; cộng lại 100.993).

### 4.5. Học chuyển giao: ResNet-18 và MobileNetV2

<!-- Nguồn: mục 2.3.1, 2.3.3, 2.3.4, Bảng 2.6–2.8, Hình 2.4–2.7; fit_resnet18.json, fit_mobilenetv2.json -->

**Học chuyển giao (transfer learning)** là dùng lại một mạng đã học trên tập lớn (ImageNet, khoảng một triệu ảnh nhiều loại vật thể) cho bài toán mới.

- **Backbone** là phần trích đặc trưng của mạng đã học sẵn. Nó biến ảnh thành một dãy số mô tả ảnh.
- **Đầu phân loại mới** của nhóm: `GAP → Dense 64 + ReLU → Dropout 0,3 → Dense 1 + sigmoid`, giống phần cuối của CNN.
- **Đóng băng backbone** nghĩa là đặt `trainable=False` và gọi với `training=False`. Trọng số backbone **không đổi** trong suốt 30 epoch, kể cả thống kê Batch Normalization. Chỉ hai tầng Dense mới được học.
- Backbone đóng băng **vẫn tính toán** cho mọi ảnh ở mọi batch. Đóng băng chỉ có nghĩa là không cập nhật trọng số.
- Đây là kiểu **trích đặc trưng cố định**, **không phải fine-tuning**. Fine-tuning là mở khóa một phần backbone để học tiếp; nhóm **chưa làm**.

**Vì sao đóng băng?** (mục 2.3.1) Tập train chỉ khoảng 5.600 ảnh. Mục tiêu là đo xem đặc trưng ImageNet giúp được bao nhiêu với chi phí huấn luyện thấp. Học ít tham số cũng giảm rủi ro quá khớp. Fine-tuning để lại cho thực nghiệm sau.

| | ResNet-18 | MobileNetV2 |
|---|---|---|
| Nguồn | KerasHub, preset `resnet_18_imagenet` | Keras Applications, `weights="imagenet"`, `include_top=False`, `alpha=1.0` |
| Ý tưởng chính | Khối dư (residual) có đường nối tắt: đầu ra = ReLU(F(x) + x) (Hình 2.4) | Khối inverted residual: mở rộng kênh → depthwise 3 × 3 → nén tuyến tính (Hình 2.6) |
| Đặc trưng trước GAP | 7 × 7 × **512** | 7 × 7 × **1.280** |
| Tổng tham số | 11.219.009 | 2.340.033 |
| Tham số được học | **32.897** = (512 + 1) × 64 + 65 | **82.049** = (1.280 + 1) × 64 + 65 |

**ResNet-18 (basic block):** hai Conv 3 × 3 giữ nguyên số kênh (ví dụ 64 → 64), kèm Batch Normalization, tạo `F(x)`; cộng với đầu vào `x` qua đường nối tắt rồi ReLU. Đường nối tắt giúp gradient đi thẳng về các tầng trước, giảm hiện tượng gradient biến mất. Khi đổi số kênh hoặc độ phân giải, đường tắt dùng phép chiếu (Conv 1 × 1). Bốn nhóm, mỗi nhóm 2 basic block, số kênh 64/128/256/512 (Hình 2.5, Bảng 2.6).

**MobileNetV2 (inverted residual, ví dụ block_7):** Conv 1 × 1 mở rộng 64 → 384 kênh → depthwise 3 × 3 lọc riêng từng kênh → Conv 1 × 1 chiếu về 64 kênh, **không ReLU** (linear bottleneck) → cộng với đầu vào. Gọi là "ngược" vì mở rộng ở giữa, hẹp ở hai đầu. Tích chập tách biệt (depthwise + 1 × 1) cần ít phép tính hơn Conv 3 × 3 thường khoảng 8–9 lần. Hàm kích hoạt là ReLU6 (ReLU chặn trên ở 6).

**Vì sao MobileNetV2 ít tham số tổng hơn mà tham số học lại nhiều hơn ResNet-18?** Vì đặc trưng trước GAP của MobileNetV2 có 1.280 kênh, của ResNet-18 chỉ 512. Dense 64 của MobileNetV2 nhận nhiều đầu vào hơn.

**Vì sao chọn hai mạng này?** ResNet-18 là mạng nhỏ nhất họ ResNet, đại diện kiến trúc khối dư. MobileNetV2 là mạng nhẹ, hợp với demo chạy CPU. Cả hai có sẵn trọng số ImageNet trong hệ Keras (mục 2.3.1).

### 4.6. Huấn luyện

<!-- Nguồn: Bảng 2.9, mục 2.4.2–2.4.3, config.json -->

| Thiết lập | Giá trị | Ghi chú dễ hiểu |
|---|---|---|
| Số epoch | 30 cho mỗi mô hình | 1 epoch = học qua hết 5.583 ảnh train |
| Batch size | 32 | ⌈5.583 / 32⌉ = 175 lần cập nhật mỗi epoch |
| Bộ tối ưu | Adam, learning rate khởi đầu 0,001 | 0,001 là giá trị mặc định do tác giả Adam đề xuất |
| Hàm mất mát | Binary cross-entropy (BCE) | Công thức (2.3) |
| Seed | 42 | Để tái lập cách chia và khởi tạo |
| Giảm learning rate | ReduceLROnPlateau: hệ số 0,5, patience 3, tối thiểu 10⁻⁶ | Validation loss không cải thiện 3 epoch thì giảm LR một nửa |
| Checkpoint | ModelCheckpoint theo **validation loss thấp nhất** | Giữ bản tốt nhất trong 30 epoch |
| Early stopping | **Tắt**; chỉ có TerminateOnNaN (dừng nếu loss lỗi số) | Xem giải thích dưới |
| Ngưỡng | `p_ok ≥ 0,5` → đạt | Giống nhau cho cả ba mô hình |
| Không dùng | class weight, L1/L2, augmentation trong pipeline, fine-tuning | Mục 2.4.2 |

- **Adam:** mỗi batch, mạng dự đoán (lan truyền xuôi), tính loss, tính gradient (lan truyền ngược), rồi Adam cập nhật trọng số. Adam tự điều chỉnh bước cho từng tham số dựa trên trung bình trượt của gradient và của bình phương gradient.
- **BCE:** với một ảnh, `L = −[y·ln(p) + (1 − y)·ln(1 − p)]`. Gán xác suất thấp cho nhãn đúng thì bị phạt rất nặng. Ví dụ ảnh đạt, `p_ok = 0,9` → L ≈ 0,105; `p_ok = 0,1` → L ≈ 2,303.
- **Vì sao checkpoint theo validation loss, không theo train loss?** Train loss luôn có xu hướng giảm, kể cả khi mô hình bắt đầu học thuộc train. Validation là ảnh mô hình không học, nên phản ánh khả năng khái quát.
- **Vì sao không dùng early stopping?** (mục 2.4.3) Để lưu đủ lịch sử 30 epoch cho cả ba mô hình, dễ so sánh đường học. ModelCheckpoint đã giữ bản tốt nhất, tức là đã làm vai trò "chọn điểm dừng".
- Thời gian huấn luyện (gồm train và validation, không gồm chuẩn bị dữ liệu, tải trọng số): CNN **300,36 s**, ResNet-18 **990,97 s**, MobileNetV2 **557,11 s** trên Tesla T4.
- ResNet-18 học ít tham số nhất mà chạy lâu nhất, vì mỗi batch vẫn phải tính xuôi qua backbone 11,2 triệu tham số.

### 4.7. Diễn biến huấn luyện và checkpoint được chọn

<!-- Nguồn: history_*.csv (epoch trong CSV đếm từ 0, ở đây +1), fit_*.json; Bảng 3.2, Hình 3.1–3.4, mục 3.3.2–3.3.5 -->

| Mô hình | Val acc sau epoch 1 | Epoch chọn | Val loss | Val acc tại epoch chọn |
|---|---:|---:|---:|---:|
| CNN | 57,20% | 29 | 0,1210 | 95,84% |
| ResNet-18 | 95,44% | 29 | 0,0269 | 98,78% |
| MobileNetV2 | 97,26% | **21** | **0,0150** | 99,49% |

- **CNN** đứng ở 57,20% trong 3 epoch đầu. 57,20% = 564/986 = tỷ lệ ảnh lỗi trong validation, tức là mô hình đang đoán tất cả là lỗi. Từ epoch 4 mới tách được hai lớp (85,19%). CNN phải học cả cách trích đặc trưng lẫn cách phân loại.
- **ResNet-18 và MobileNetV2** cao ngay từ epoch 1 vì backbone ImageNet đã cho đặc trưng tốt; chỉ cần học đầu phân loại.
- **Vì sao chạy 30 epoch mà dùng epoch 21?** MobileNetV2 có validation loss thấp nhất ở epoch 21. Sau đó train loss giảm tiếp (0,0119 → 0,0046 ở epoch 30) nhưng validation loss không thấp hơn và kết thúc ở 0,0179. Đó là **dấu hiệu quá khớp nhẹ** cuối lượt, nên dùng bản epoch 21.
- CNN và ResNet-18 có checkpoint ở epoch 29, sát cuối lượt. Với CNN, báo cáo nói lịch sử chưa cho thấy quá khớp; muốn biết CNN còn tăng được không cần thử thêm epoch (mục 3.3.2).

**Đỉnh nhọn (spike) trên đường validation loss.** Số ảnh sai = 986 × (1 − val_accuracy), làm tròn:

| Mô hình | Epoch trước → đỉnh → sau | Validation loss | Số ảnh validation sai / 986 |
|---|---|---|---|
| CNN | 8 → 9 → 10 | 0,2704 → **0,4488** → 0,2405 | 98 → **216** → 82 |
| CNN | 17 → 18 → 19 | 0,1910 → **0,5188** → 0,1722 | 67 → **281** → 57 |
| ResNet-18 | 18 → 19 → 20 | 0,0302 → **0,0455** → 0,0279 | 15 → **18** → 15 |
| MobileNetV2 | 13 → 14 → 15 | 0,0171 → **0,0696** → 0,0204 | 6 → **18** → 6 |

Cách giải thích: tập validation cố định 986 ảnh, không có augmentation. Sau mỗi epoch trọng số đổi, nên dự đoán trên cùng 986 ảnh cũng đổi. Dao động có thể liên quan đến thứ tự batch xáo trộn và bước cập nhật của Adam; nhóm chưa chạy thêm seed để kiểm tra (mục 3.3.2). Mỗi đỉnh hồi phục ngay ở epoch sau, nên đây là dao động tạm thời, không phải lỗi code. Lịch sử chỉ lưu số tổng hợp theo epoch, nên nhóm **chưa xác định được nguyên nhân riêng** của từng đỉnh.

**Vì sao loss nhảy mạnh mà accuracy chỉ giảm ít?** Accuracy chỉ đếm đúng/sai. Loss còn phạt độ tự tin. Một ảnh lỗi bị cho `p_ok` = 0,60 hay 0,99 đều là "sai một ảnh", nhưng 0,99 bị phạt nặng hơn nhiều. Ví dụ MobileNetV2 epoch 14: loss tăng khoảng 4 lần, accuracy chỉ giảm khoảng 1,22 điểm phần trăm.

**Vì sao validation có lúc tốt hơn train?** Chỉ số train là trung bình qua các batch trong lúc trọng số đang thay đổi và Dropout đang bật. Validation dùng trọng số cuối epoch và tắt Dropout.

**So sánh ba đường validation loss (mục 3.3.5, Hình 3.4).** Hình 3.4 vẽ validation loss ba mô hình trên cùng trục, thang logarit, dấu sao là checkpoint được chọn.

- Sau epoch 1: MobileNetV2 0,0923, ResNet-18 0,2464, CNN 0,6838. Số 0,6838 gần ln 2 ≈ 0,693, tức mức loss khi gán 0,5 cho mọi ảnh.
- Hai mạng học chuyển giao vượt 95% validation accuracy ngay epoch 1. CNN đến epoch 21 mới lần đầu đạt 95%. Chỉ MobileNetV2 vượt 99% (từ epoch 5).
- Thứ tự MobileNetV2 < ResNet-18 < CNN giữ ở **28/30 epoch**; hai ngoại lệ là epoch 14 và 17 (MobileNetV2 0,0696 và 0,0311 so với ResNet-18 0,0398 và 0,0289). Vì vậy việc chọn MobileNetV2 không phụ thuộc một epoch riêng lẻ.
- 10 epoch cuối, số ảnh validation sai: CNN 41–65, ResNet-18 11–15, MobileNetV2 4–6.
- Validation loss thấp hơn train loss ở 26/30 epoch với CNN, 11 với ResNet-18, 4 với MobileNetV2. Epoch 30: validation loss MobileNetV2 gấp khoảng 3,9 lần train loss (0,0179 so với 0,0046), ResNet-18 gấp khoảng 2,0 lần.
- CNN và ResNet-18 có checkpoint ở epoch 29, nên chưa biết học quá 30 epoch có giảm tiếp không.

### 4.8. Đánh giá: lớp lỗi là lớp quan tâm

<!-- Nguồn: mục 2.4.4, Bảng 2.11, công thức (2.5), (2.6) -->

Nhóm coi **lớp lỗi là lớp "dương tính"** (lớp cần phát hiện), dù mã nhãn của nó là 0. Ma trận nhầm lẫn đọc theo **hàng = nhãn thật, cột = nhãn dự đoán**, thứ tự lỗi rồi đạt:

| Nhãn thật \ Dự đoán | Lỗi | Đạt |
|---|---|---|
| **Lỗi** | **TP**: phát hiện đúng ảnh lỗi | **FN**: **bỏ sót lỗi** (sản phẩm lỗi bị cho qua) |
| **Đạt** | **FP**: báo nhầm (ảnh đạt bị báo lỗi) | **TN**: nhận đúng ảnh đạt |

- **FN là loại sai nguy hiểm hơn** trong kiểm tra chất lượng: sản phẩm lỗi lọt qua. FP chỉ làm tốn công kiểm tra lại.
- **Accuracy** = (TP + TN) / tổng: tỷ lệ đúng chung.
- **Precision lớp lỗi** = TP / (TP + FP): trong số ảnh bị báo lỗi, bao nhiêu ảnh lỗi thật.
- **Recall lớp lỗi** = TP / (TP + FN): trong số ảnh lỗi thật, tìm ra được bao nhiêu. Chỉ số này theo dõi trực tiếp việc bỏ sót.
- **F1** = 2 × P × R / (P + R): trung bình điều hòa của precision và recall.
- **Macro F1** = (F1 lớp lỗi + F1 lớp đạt) / 2: hai lớp được coi nặng như nhau, nên phản ánh cả lớp ít ảnh hơn.
- Vì test lệch lớp (63,4% là ảnh lỗi), một mô hình đoán toàn "lỗi" đã đạt khoảng 63,4% accuracy. Do đó cần thêm các chỉ số theo lớp.

### 4.9. Kết quả trên tập test và cách đọc ma trận

<!-- Nguồn: comparison.csv, metrics_*.json, test_predictions_*.csv; Bảng 3.3–3.7; Hình 3.5–3.7; mục 3.4.1–3.4.4 -->

| Mô hình | TP | FN (bỏ sót) | FP (báo nhầm) | TN | Đúng / 715 |
|---|---:|---:|---:|---:|---:|
| CNN | 428 | 25 | 12 | 250 | 678 |
| ResNet-18 | 446 | 7 | 2 | 260 | 706 |
| MobileNetV2 | **451** | **2** | **2** | **260** | **711** |

Đọc ma trận MobileNetV2 `[[451, 2], [2, 260]]`:

- Hàng lỗi: 453 ảnh lỗi thật, 451 phát hiện đúng, **2 bị bỏ sót**.
- Hàng đạt: 262 ảnh đạt thật, 260 nhận đúng, **2 bị báo nhầm**.
- Accuracy = (451 + 260) / 715 = **99,44%**. Precision lỗi = 451 / 453 = 99,56%. Recall lỗi = 451 / 453 = 99,56%. Hai mẫu số cùng là 453 chỉ vì FN = FP = 2; ý nghĩa hai chỉ số vẫn khác nhau.
- Một ảnh test bằng khoảng **0,14 điểm phần trăm** (1/715). Chênh 0,70 điểm giữa MobileNetV2 và ResNet-18 là **5 ảnh**.

So sánh: so với CNN, MobileNetV2 bỏ sót ít hơn 23 ảnh lỗi và báo nhầm ít hơn 10 ảnh. So với ResNet-18, bỏ sót ít hơn 5 ảnh, cùng số báo nhầm. CNN sai chủ yếu kiểu bỏ sót (25 FN so với 12 FP).

**Chỉ số theo từng lớp (Bảng 3.4, mục 3.4.1).** Bảng 3.3 chỉ có lớp lỗi; Bảng 3.4 thêm lớp đạt và support (453 lỗi, 262 đạt).

| Mô hình | Lớp | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| CNN | Lỗi | 97,27% | 94,48% | 95,86% |
| CNN | Đạt | 90,91% | 95,42% | 93,11% |
| ResNet-18 | Lỗi | 99,55% | 98,45% | 99,00% |
| ResNet-18 | Đạt | 97,38% | 99,24% | 98,30% |
| MobileNetV2 | Lỗi | 99,56% | 99,56% | 99,56% |
| MobileNetV2 | Đạt | 99,24% | 99,24% | 99,24% |

- CNN: precision lớp đạt chỉ 90,91% vì 275 ảnh được dự đoán đạt có 25 ảnh lỗi. Recall lỗi (94,48%) thấp hơn recall đạt (95,42%): bỏ sót trong lớp lỗi nhiều hơn báo nhầm trong lớp đạt. ResNet-18 cùng xu hướng (98,45% so với 99,24%).
- MobileNetV2 có FN = FP = 2 nhưng F1 đạt (99,24%) thấp hơn F1 lỗi (99,56%), vì lớp đạt chỉ 262 ảnh, mỗi ảnh sai chiếm tỷ lệ lớn hơn. Cả ba mô hình đều có F1 đạt < F1 lỗi, nên macro F1 < F1 lỗi.

**FN, FP nghĩa là gì ngoài nhà máy (mục 3.4.2).** FN là sản phẩm lỗi bị cho qua; FP là sản phẩm đạt bị giữ lại kiểm tra hoặc loại oan. CNN bỏ sót 25/453 ≈ 5,52%, báo nhầm 12/262 ≈ 4,58%; MobileNetV2 bỏ sót 0,44%, báo nhầm 0,76%. Phép tính minh họa của báo cáo: nếu dây chuyền chỉ có 5% sản phẩm lỗi, precision lớp lỗi còn khoảng 87,3% (MobileNetV2) và 52,1% (CNN). Đây là giả định, không phải kết quả đo.

**Hai mô hình học chuyển giao có thật sự khác nhau không (mục 3.4.3).**

- Test gần với validation: số ảnh sai tại checkpoint là 41 / 12 / 5 trên 986 ảnh validation, 37 / 9 / 4 trên 715 ảnh test. BCE trên test 0,1298 / 0,0354 / 0,0128, cùng thứ tự với validation loss.
- MobileNetV2 và ResNet-18: 11 ảnh chỉ một trong hai mô hình sai (ResNet-18 sai riêng 8, MobileNetV2 sai riêng 3). Nếu hai mô hình ngang nhau, mỗi ảnh như vậy rơi về một bên với xác suất 0,5; xác suất lệch từ 3 : 8 trở lên (tính cả hai phía) ≈ **0,23**. Vì vậy **chưa đủ khẳng định** MobileNetV2 tốt hơn ResNet-18.
- CNN với hai mô hình học chuyển giao: 35 so với 7 (ResNet-18), 36 so với 3 (MobileNetV2); xác suất tương ứng đều **< 10⁻⁴**. Chênh lệch này rõ.

**Chi phí (Bảng 3.6).**

| Mô hình | Tổng tham số | Tham số học | Epoch | Thời gian 30 epoch (s) | Tệp .keras (MB) | Ảnh test sai |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 100.993 | 100.993 | 29 | 300,36 | 1,26 | 37 |
| ResNet-18 | 11.219.009 | 32.897 | 29 | 990,97 | 45,35 | 9 |
| MobileNetV2 | 2.340.033 | 82.049 | 21 | 557,11 | 10,61 | 4 |

- Mỗi epoch: CNN 10,01 s, ResNet-18 33,03 s, MobileNetV2 18,57 s. ResNet-18 học ít tham số nhất mà chậm nhất, khoảng 1,78 lần MobileNetV2, vì 11.186.112 tham số cố định vẫn chạy xuôi mỗi batch (MobileNetV2: 2.257.984).
- Dung lượng tệp chủ yếu theo **tổng** tham số. Tệp ResNet-18 ≈ 4 byte/tham số (số thực 32 bit); tệp CNN ≈ 12 byte/tham số vì tệp `.keras` lưu kèm trạng thái Adam (hai moment cho mỗi tham số được học). Ba tệp demo tổng ≈ 57,23 MB. Chưa đo thời gian dự đoán một ảnh.

**Phạm vi kết luận:** ba cấu hình khác nhau đồng thời ở nhiều yếu tố (Bảng 2.10): kiến trúc, độ phân giải đầu vào, trọng số ImageNet. Kết quả chỉ xếp hạng **cả cấu hình** trong thực nghiệm này, chưa tách được đóng góp riêng của từng yếu tố (mục 3.4.3).

**Đổi ngưỡng (mục 3.4.4, Bảng 3.7).** Xem bảng đủ ba mô hình ở [mục 8.3](#83-ảnh-hưởng-của-ngưỡng-bảng-37). Ý chính: tăng ngưỡng thì FN giảm, FP tăng. Đây là phân tích bổ sung trên test, **không dùng để chọn ngưỡng**.

### 4.10. Ảnh dự đoán sai

<!-- Nguồn: errors_*.csv, test_predictions_*.csv; Bảng 3.8, 3.9, Hình 3.8–3.11, mục 3.5.1 -->

Số ảnh sai: CNN 37, ResNet-18 9, MobileNetV2 4. Hình 3.8 (CNN) và Hình 3.9 (ResNet-18) là ảnh lỗi nhận đúng và bị bỏ sót; Hình 3.10 là ví dụ đúng/sai của MobileNetV2. Bốn ảnh MobileNetV2 sai (Bảng 3.8):

| Ảnh | Nhãn thật | p_ok | Dự đoán |
|---|---|---:|---|
| `cast_def_0_150.jpeg` | Lỗi | 0,955552 | Đạt (bỏ sót, điểm đạt cao) |
| `cast_def_0_2102.jpeg` | Lỗi | 0,502379 | Đạt (bỏ sót, sát ngưỡng 0,5) |
| `cast_ok_0_16.jpeg` | Đạt | 0,228884 | Lỗi (báo nhầm) |
| `cast_ok_0_7839.jpeg` | Đạt | 0,244992 | Lỗi (báo nhầm) |

- `cast_def_0_150.jpeg` bị **cả ba mô hình** đoán là đạt (p_ok 0,708656 ở CNN, 0,979930 ở ResNet-18, 0,955552 ở MobileNetV2). Đây là ảnh test duy nhất cả ba cùng sai. Nhìn bằng mắt, nhóm không thấy vết lỗi rõ ở mép và bề mặt như các ảnh lỗi khác (mục 3.5.1).
- **Nhóm chưa biết nguyên nhân.** Dữ liệu không có chú giải vị trí lỗi; muốn tìm hiểu cần công cụ giải thích như Grad-CAM, nhóm **chưa làm**. Không tự bịa nguyên nhân.
- Điểm sigmoid **chưa hiệu chuẩn**: `p_ok = 0,96` không có nghĩa "đúng 96%". Một ảnh sai vẫn có thể được cho điểm cao.

**Ảnh sai chung giữa các mô hình (Bảng 3.9).** Đường chéo là tổng ảnh sai của từng mô hình; ô ngoài đường chéo là số ảnh hai mô hình cùng sai.

| Mô hình | FN | FP | Sai chung với CNN | Sai chung với ResNet-18 | Sai chung với MobileNetV2 |
|---|---:|---:|---:|---:|---:|
| CNN | 25 | 12 | **37** | 2 | 1 |
| ResNet-18 | 7 | 2 | 2 | **9** | 1 |
| MobileNetV2 | 2 | 2 | 1 | 1 | **4** |

- Sai ở cả ba: 1 ảnh (`cast_def_0_150.jpeg`). Sai ở ít nhất một mô hình: 47 ảnh, trong đó 45 ảnh chỉ sai ở một mô hình. Đúng ở cả ba: 668 ảnh.
- Ngoài `cast_def_0_150.jpeg`, chỉ `cast_def_0_1591.jpeg` sai ở hai mô hình (CNN và ResNet-18), cả hai là FN. 16 ảnh FP của ba mô hình là 16 ảnh khác nhau.
- Có ảnh một mô hình sai rất tự tin mà mô hình khác đúng rất chắc: `cast_def_0_2177.jpeg` có p_ok 0,9878 ở ResNet-18, trong khi CNN cho 0,0003, MobileNetV2 cho 0,0063.
- Kết luận: phần lớn ảnh sai **gắn với từng mô hình**, không phải ảnh mà mọi mô hình đều thấy khó.

**Nhóm 37 ảnh `new__0_*` (chỉ nói số liệu).** Tập test có 37 ảnh lỗi tên dạng `new__0_*`, mẫu tên này chỉ có trong test. CNN bỏ sót 7/37 (18,9%), so với 18/416 ≈ 4,3% ở các ảnh lỗi còn lại. ResNet-18 và MobileNetV2 không sai ảnh nào trong nhóm. Báo cáo không suy đoán nguyên nhân; khi bị hỏi, chỉ nêu số và nói nhóm chưa có thông tin về nguồn gốc nhóm ảnh này.

**Phân bố p_ok (Hình 3.11).** Hình vẽ phân bố p_ok của ảnh lỗi và ảnh đạt trên test, trục tung thang logarit, tam giác dưới trục hoành là ảnh sai.

| | CNN | ResNet-18 | MobileNetV2 |
|---|---:|---:|---:|
| Ảnh có p_ok trong 0,2–0,8 | 76 (28 sai) | 11 | 7 |
| Ảnh có độ tự tin max(p_ok, 1 − p_ok) ≥ 0,99, tất cả đúng | 323 | 649 | 686 |
| Tỷ lệ đúng khi độ tự tin < 0,8 | 48/76 | 8/11 | 4/7 |
| Ảnh sai dù độ tự tin ≥ 0,9 | 6 | 3 | 1 (`cast_def_0_150.jpeg`) |

- MobileNetV2: 448/453 ảnh lỗi có p_ok < 0,05 và 253/262 ảnh đạt có p_ok > 0,95. CNN chỉ có 62/262 ảnh đạt p_ok > 0,95, phân bố rộng hơn rõ.
- Ý nghĩa: hai lớp càng tách biệt thì đổi ngưỡng càng ít gây báo nhầm. Do vùng trung gian rất ít ảnh (7–76), nhóm chưa đủ dữ liệu đánh giá hiệu chuẩn; p_ok chỉ dùng để xem mô hình nghiêng về phía nào.

### 4.11. Lưu mô hình, kiểm chứng và demo

<!-- Nguồn: Bảng 3.10, casting_full_3models_01/demo/verification.json, demo/models/comparison_verification.json -->

- Checkpoint lưu dạng `.keras` (gồm kiến trúc và trọng số), kèm metadata: quy ước nhãn, kích thước đầu vào, chuỗi tiền xử lý, mã băm.
- **Kiểm chứng sau nạp trên CPU** (Bảng 3.10): nạp lại mô hình, so với dự đoán đã lưu từ Colab.

| Mô hình | Tập đối chiếu | Nhãn trùng | Sai lệch điểm lớn nhất |
|---|---|---:|---:|
| MobileNetV2 | 10 ảnh validation | 10/10 | 4,2142 × 10⁻⁸ |
| CNN | 10 ảnh test (5 lỗi, 5 đạt) | 10/10 | 8,2600 × 10⁻⁸ |
| ResNet-18 | 10 ảnh test | 10/10 | 3,1825 × 10⁻⁶ |
| MobileNetV2 | 10 ảnh test | 10/10 | 6,2771 × 10⁻⁷ |

  Dung sai: tương đối 10⁻³, tuyệt đối 10⁻⁴. Phép này chỉ chứng minh **tệp nạp lại cho cùng kết quả**, không phải đo độ chính xác. Độ chính xác là 715 ảnh test ở mục 3.4.
- Trên Windows, lớp `ResNetImageConverter` của ResNet-18 được dựng lại bằng các tầng Keras (resize, nhân scale, cộng offset) rồi nạp trọng số gốc (`demo/inference.py`).
- **Demo** (mục 3.5.3, Hình 3.12, 3.13): chọn một hoặc cả ba mô hình, nhận nhiều ảnh, hiện nhãn và điểm 6 chữ số thập phân, xuất CSV/JSON. Khi nạp, ứng dụng đối chiếu tệp mô tả với cấu hình thực nghiệm và chỉ nhận mô hình có mã băm SHA-256 trùng checkpoint lúc huấn luyện. Tệp ảnh hỏng được báo lỗi riêng, các ảnh hợp lệ vẫn chạy.
- Điểm của **một ảnh** (ví dụ 0,998192) khác với **accuracy toàn tập** (99,44%).

### 4.12. Hạn chế (mục 3.5.4)

1. **Một seed, một cách chia, test cùng nguồn.** Chưa đo độ dao động khi đổi seed.
2. **Rò rỉ gần giống chưa loại hết.** Nguồn có sẵn ảnh tăng cường và không có mã sản phẩm gốc. SHA-256 chỉ loại 64 bản y hệt; biến thể xoay hoặc đổi sáng của cùng ảnh có thể nằm ở cả train và test. Vì vậy **99,44% có thể lạc quan** so với ảnh chụp mới tại nhà máy. Ba mô hình chịu cùng rủi ro này, nhưng mức lạc quan có thể khác nhau giữa các mô hình, nên nhóm không khẳng định chênh lệch giữa chúng hoàn toàn không bị ảnh hưởng.
3. **Backbone đóng băng**, chưa fine-tuning; kiến trúc CNN, độ phân giải, siêu tham số chưa được tìm kiếm có hệ thống.
4. **Thời gian** chỉ là thời gian huấn luyện một lượt trên T4. Chưa đo độ trễ từng ảnh, bộ nhớ.
5. **Điểm sigmoid chưa hiệu chuẩn**; nhãn chỉ ở cấp toàn ảnh, không định vị lỗi.
6. Demo là **bản minh họa**, cần kiểm định thêm trước khi dùng trong sản xuất.
7. `history_*.csv` không ghi learning rate, nên không biết ReduceLROnPlateau giảm LR ở epoch nào.
8. Dự đoán trên validation chỉ lưu cho 10 ảnh, nên chưa chọn được ngưỡng theo chi phí FN/FP hay đánh giá hiệu chuẩn trên validation (mục 3.4.4).
9. Seed 42 cố định chia tập, khởi tạo, thứ tự xáo trộn, nhưng GPU không chạy chế độ tất định; mỗi cấu hình một lượt nên chưa có khoảng dao động của chỉ số.

**Hướng phát triển** (phần Kết luận): thêm ảnh độc lập, chia theo nhóm sản phẩm khi có thông tin, chạy nhiều seed, thử fine-tuning, chọn ngưỡng theo chi phí bỏ sót/báo nhầm trên validation, đo độ trễ và hiệu chuẩn điểm.

---

## 5. Bảng số liệu chính

<!-- Nguồn: comparison.csv, fit_*.json, metrics_*.json, kích thước tệp *.keras; khớp Bảng 2.8, 3.2, 3.3, 3.5, 3.6 của báo cáo v4 bản in -->

<!-- BCE test tính lại từ test_predictions_*.csv; số ảnh sai chung ở Bảng 3.9 -->


| | CNN | ResNet-18 | MobileNetV2 |
|---|---:|---:|---:|
| Khởi tạo | Ngẫu nhiên | ImageNet | ImageNet |
| Phần được học | Toàn mạng | Đầu Dense mới | Đầu Dense mới |
| Đầu vào | 256 × 256 × 1 | 224 × 224 × 3 | 224 × 224 × 3 |
| Tổng tham số | 100.993 | 11.219.009 | 2.340.033 |
| Tham số được học | 100.993 | 32.897 | 82.049 |
| Epoch được chọn | 29 | 29 | **21** |
| Validation loss tại epoch chọn | 0,1210 | 0,0269 | **0,0150** |
| Validation accuracy tại epoch chọn | 95,84% | 98,78% | 99,49% |
| Thời gian huấn luyện (giây, T4) | 300,36 | 990,97 | 557,11 |
| **Accuracy test** | 94,83% | 98,74% | **99,44%** |
| Precision lớp lỗi | 97,27% | 99,55% | 99,56% |
| Recall lớp lỗi | 94,48% | 98,45% | 99,56% |
| F1 lớp lỗi | 95,86% | 99,00% | 99,56% |
| Macro F1 | 94,48% | 98,65% | 99,40% |
| TP / FN / FP / TN | 428 / 25 / 12 / 250 | 446 / 7 / 2 / 260 | 451 / 2 / 2 / 260 |
| Ảnh đúng / 715 | 678 | 706 | 711 |
| Ảnh test sai | 37 | 9 | 4 |
| BCE trên test | 0,1298 | 0,0354 | 0,0128 |
| Tệp .keras (MB) | 1,26 | 45,35 | 10,61 |
| Thời gian mỗi epoch (giây) | 10,01 | 33,03 | 18,57 |

Giá trị validation loss đầy đủ (trong `fit_*.json`): 0,121010 / 0,026946 / 0,015027.

---

<a id="5-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi"></a>

## 6. Ôn bảo vệ: câu hỏi hội đồng hay hỏi

Trả lời **ngắn, có số, chỉ đúng mục/bảng**. Không biết thì nói thật.

### A. Bài toán và dữ liệu

**1. Bài toán của nhóm là gì? Đầu ra là gì?**
Phân loại cả ảnh sản phẩm đúc thành lỗi hoặc đạt. Mô hình cho `p_ok` (điểm lớp đạt); `p_ok ≥ 0,5` là đạt. Không định vị vết lỗi vì dữ liệu chỉ có nhãn cấp ảnh (mục 2.1).

**2. Dữ liệu bao nhiêu ảnh, chia thế nào?**
7.348 ảnh nguồn. Loại 64 ảnh train trùng test, còn 7.284. Chia 5.583 train, 986 validation, 715 test (Bảng 3.1). Validation là 15% của 6.569 ảnh train sạch, chia phân tầng, seed 42. Số ảnh qua từng bước kiểm tra ở Bảng 2.2.

**3. Vì sao phải kiểm ảnh trùng? Vì sao dùng SHA-256?**
Ảnh vừa ở train vừa ở test làm accuracy test cao giả tạo. SHA-256 trên pixel ảnh xám trước resize cho biết hai ảnh có giống hệt nhau không, chính xác hơn so tên tệp (mục 2.2.2).

**4. 64 ảnh trùng là ảnh gì? Vì sao loại bên train mà không loại bên test?**
Cả 64 là ảnh đạt ở `train/ok_front` trùng ảnh ở `test/ok_front`. Nhóm giữ nguyên tập test của nguồn để tập đánh giá không bị thay đổi, nên bỏ bản ở train (mục 2.2.2, 3.2).

**5. Vì sao dùng ảnh xám?**
Ảnh nguồn lưu ba kênh nhưng ba kênh bằng nhau, tức là nội dung đã là ảnh xám. Chuyển về một kênh không mất thông tin và nhẹ hơn. Hai backbone cần 3 kênh nên nhóm lặp kênh xám ba lần (mục 2.2.1, 2.2.4).

**6. Có dùng augmentation không?**
Không thêm. Bộ nguồn đã có sẵn ảnh tăng cường, nên ba mô hình học trên cùng tập train cố định (mục 2.2.4). Nhóm chưa đo tác dụng của việc thêm augmentation.

### B. Mô hình

**7. CNN "tự xây" là tự xây thế nào?**
Nhóm tự chọn 3 khối Conv (32/64/128 bộ lọc) + MaxPooling, rồi GAP, Dense 64, Dropout 0,3, Dense 1 sigmoid, và học toàn bộ 100.993 tham số từ đầu. Các tầng đều là tầng Keras phổ biến; nhóm không nhận là kiến trúc mới (mục 2.3.2, Bảng 2.5, Hình 2.3).

**8. Tính giúp số tham số tầng Conv đầu và Dense 64.**
Conv đầu: (3 × 3 × 1 + 1) × 32 = 320. Dense 64: (128 + 1) × 64 = 8.256. Số "+1" là bias.

**9. Vì sao dùng Global Average Pooling mà không Flatten?**
Flatten cho 131.072 đầu vào, Dense 64 cần 8.388.672 tham số. GAP chỉ còn 128 đầu vào, cần 8.256 tham số. Ít tham số hơn nhiều, giảm rủi ro quá khớp; đổi lại mất thông tin vị trí (mục 2.3.2).

**10. Vì sao đầu ra 1 nơ-ron sigmoid mà không softmax 2 nơ-ron?**
Bài toán hai lớp. Một sigmoid cho `p_ok`, điểm lỗi là `1 − p_ok`. Cách này đi cùng binary cross-entropy và ít tham số hơn.

**11. Học chuyển giao, đóng băng backbone là gì?**
Lấy phần trích đặc trưng đã học trên ImageNet, giữ nguyên trọng số (`trainable=False`, gọi với `training=False`), chỉ học đầu phân loại mới. ResNet-18 học 32.897 tham số, MobileNetV2 học 82.049 (Bảng 2.8). Nhóm **không fine-tune**.

**12. Vì sao MobileNetV2 nhỏ hơn ResNet-18 mà tham số học nhiều hơn?**
Đặc trưng trước GAP của MobileNetV2 có 1.280 kênh, ResNet-18 có 512. Dense 64: (1.280 + 1) × 64 + 65 = 82.049 so với (512 + 1) × 64 + 65 = 32.897.

**13. Inverted residual của MobileNetV2 khác basic block của ResNet-18 ra sao?**
Basic block của ResNet-18: hai Conv 3 × 3 giữ nguyên số kênh, cộng nối tắt rồi ReLU. Inverted residual: Conv 1 × 1 mở rộng (64 → 384), depthwise 3 × 3, Conv 1 × 1 nén tuyến tính về 64 rồi cộng nối tắt (mỏng → dày → mỏng). Kiểu "dày → mỏng → dày" là bottleneck của ResNet-50 trở lên, **không phải** ResNet-18 (Hình 2.4, 2.6).

### C. Huấn luyện

**14. Vì sao chọn checkpoint theo validation loss mà không theo accuracy?**
Accuracy trên 986 ảnh chỉ nhảy theo bước khoảng 0,1 điểm và hay bằng nhau giữa các epoch. Loss thay đổi liên tục theo xác suất nên phân biệt tốt hơn (mục 2.4.5). Ví dụ MobileNetV2: epoch 25 có val accuracy cao nhất (99,59%) nhưng val loss 0,0155, vẫn cao hơn 0,0150 của epoch 21.

**15. Vì sao không chọn mô hình theo kết quả test?**
Nếu dùng test để chọn thì test thành một phần của quá trình xây mô hình, kết quả test không còn khách quan. Nhóm ghi quyết định vào `selection.json` trước khi chạy test (mục 2.4.5).

**16. Vì sao không dùng early stopping?**
Nhóm muốn lưu đủ 30 epoch cho cả ba mô hình để so sánh đường học. ModelCheckpoint theo validation loss đã giữ bản tốt nhất, tương đương việc chọn điểm dừng (mục 2.4.3).

**17. ReduceLROnPlateau làm gì?**
Nếu validation loss không cải thiện trong 3 epoch thì nhân learning rate với 0,5, thấp nhất 10⁻⁶. Lưu ý: lịch sử không ghi learning rate, nên nhóm không nói được LR giảm ở epoch nào.

**18. Đỉnh nhọn trên biểu đồ loss có phải lỗi code không?**
Không. Mỗi đỉnh hồi phục ngay epoch sau. Ví dụ CNN epoch 17 → 18 → 19: 67 → 281 → 57 ảnh validation sai trên 986. Trọng số đổi sau mỗi epoch nên dự đoán trên cùng tập validation đổi theo; nguyên nhân riêng của từng đỉnh chưa xác định (mục 3.3).

**19. Có quá khớp không?**
MobileNetV2 có dấu hiệu nhẹ sau epoch 21: train loss giảm 0,0119 → 0,0046, validation loss kết thúc 0,0179 (cao hơn 0,0150). Nhóm dùng checkpoint epoch 21. CNN chưa thấy quá khớp. Accuracy 99% một mình chưa đủ để kết luận; phải xem train và validation qua nhiều epoch.

**20. ResNet-18 học ít tham số nhất sao lại chạy lâu nhất (990,97 s)?**
Backbone đóng băng vẫn phải tính xuôi cho mọi batch. ResNet-18 có backbone nặng hơn MobileNetV2 về phép tính. Thời gian phụ thuộc lượng tính toán, không phụ thuộc số tham số được học.

### D. Kết quả

**21. Kết quả chính là gì?**
Accuracy test: CNN 94,83%, ResNet-18 98,74%, MobileNetV2 99,44%. MobileNetV2 đúng 711/715, bỏ sót 2 ảnh lỗi, báo nhầm 2 ảnh đạt (Bảng 3.3, 3.5).

**22. Đọc giúp ma trận nhầm lẫn của MobileNetV2.**
Hàng là nhãn thật, cột là dự đoán, thứ tự lỗi rồi đạt: `[[451, 2], [2, 260]]`. Ô hàng lỗi, cột đạt là 2 ảnh lỗi bị bỏ sót (FN). Ô hàng đạt, cột lỗi là 2 ảnh đạt bị báo nhầm (FP) (Hình 3.7).

**23. FN và FP, cái nào nguy hiểm hơn? Muốn giảm bỏ sót thì chỉnh ngưỡng thế nào?**
FN nguy hiểm hơn vì sản phẩm lỗi bị cho qua. Quy tắc là `p_ok ≥ ngưỡng` thì đạt, nên muốn bớt bỏ sót phải **tăng** ngưỡng; đổi lại FP tăng. **Không nói "hạ ngưỡng"**. Ngưỡng mới phải chọn trên validation. Mức đánh đổi trên test ở Bảng 3.7, mục 3.4.4; xem câu 37–38.

**24. 99,44% có cao bất thường không? Có rò rỉ dữ liệu không?**
Chủ động thừa nhận giới hạn: nguồn có sẵn ảnh tăng cường, không có mã sản phẩm gốc; SHA-256 chỉ loại 64 bản y hệt. Biến thể gần giống có thể còn ở cả train và test, nên 99,44% có thể lạc quan với ảnh mới (mục 3.5.4). Ba mô hình chịu cùng rủi ro, nhưng mức ảnh hưởng có thể khác nhau; nhóm chưa đo được điều này.

**25. Có thể kết luận MobileNetV2 tốt nhất không? Học chuyển giao tốt hơn CNN không?**
Chỉ trong thực nghiệm này: ba cấu hình, một seed, test cùng nguồn. Hai mô hình học chuyển giao khác CNN cùng lúc về kiến trúc, độ phân giải và trọng số ImageNet, nên chưa tách được yếu tố nào tạo ra chênh lệch (mục 3.4.3, Bảng 2.10). Hơn CNN thì rõ; hơn ResNet-18 thì chưa đủ khẳng định (câu 39–40).

**26. Vì sao ảnh `cast_def_0_150.jpeg` bị đoán là đạt với p_ok 0,9556?**
Nhóm chưa xác định được. Cả ba mô hình đều đoán đạt (0,7087 / 0,9799 / 0,9556), đây là ảnh test duy nhất cả ba cùng sai; nhìn bằng mắt không thấy vết lỗi rõ. Dữ liệu không có chú giải vị trí lỗi, nhóm chưa dùng Grad-CAM (mục 3.5.1, Bảng 3.8).

### E. Demo

**27. Demo dùng mô hình nào? Hoạt động thế nào?**
Ba checkpoint đã chọn: CNN epoch 29, ResNet-18 epoch 29, MobileNetV2 epoch 21. Ảnh đi qua đúng chuỗi tiền xử lý lúc đánh giá (EXIF → xám → 256 → với mạng học trước thì 224 và 3 kênh), mô hình cho `p_ok`, áp ngưỡng 0,5 (mục 3.5.3).

**28. Ảnh demo hiện 0,998192, sao báo cáo ghi 99,44%?**
0,998192 là điểm đạt của **một ảnh** `cast_ok_0_45.jpeg`. 99,44% là tỷ lệ đúng trên **715 ảnh test**. Hai số đo hai thứ khác nhau.

**29. Đưa ảnh không phải sản phẩm đúc vào thì sao?**
Mô hình chỉ có hai lớp nên vẫn trả lỗi hoặc đạt. Demo chỉ kiểm tra tệp có đọc được không, không có bước nhận biết ảnh lạ. Kết quả trên ảnh như vậy không có ý nghĩa.

### F. "Nếu… thì sao" và phân công

**30. Nếu fine-tune, chạy nhiều seed hay dùng ảnh nhà máy thì sao?**
Nhóm chưa làm nên không có số. Đây là hướng phát triển ghi ở mục 3.5.4 và Kết luận. Với ảnh nhà máy khác camera, ánh sáng, phải đánh giá lại trên dữ liệu đó.

**31. Em làm phần nào? Phần nhóm tự làm là gì?**
Trả lời theo bảng phân công ở mục 7. Phần cả nhóm tự làm: kiểm trùng SHA-256 và chia tập, chọn cấu hình CNN, gắn đầu phân loại và huấn luyện ba mô hình trong cùng điều kiện, chọn checkpoint trước khi test, đánh giá theo lớp lỗi, phân tích ảnh sai, kiểm chứng sau nạp và demo so sánh ba mô hình. Dữ liệu (Kaggle) và kiến trúc ResNet-18, MobileNetV2 (Keras) có nguồn bên ngoài.

**Không nhận những việc chưa làm:** fine-tuning, Grad-CAM, nhiều seed, đo độ trễ từng ảnh, hiệu chuẩn xác suất, chọn ngưỡng trên validation, định vị lỗi.
**Tránh câu tuyệt đối:** "chắc chắn đúng", "không hề quá khớp", "dùng ngay trong nhà máy được".

### G. Nội dung mới của bản in

Các câu dưới bám phần bổ sung của bản in 25/11/2026. Số liệu là kết quả Chương 3, trừ câu 32–33 (Chương 2).

**32. Hình 2.1 và Bảng 2.2 nói gì?**
Hình 2.1 là sơ đồ xử lý dữ liệu: kiểm tra ảnh → băm SHA-256, loại bản train trùng test → chia train/validation, test giữ nguyên → resize 256 chung → bước riêng từng mô hình. Bảng 2.2 đếm ảnh qua các bước: 7.348 nguồn, 0 ảnh hỏng, 64 ảnh đạt bị loại, 6.569 train sạch (57,21% lỗi). Loại trùng làm trước khi chia validation, nên validation không chứa ảnh trùng test (mục 2.2).

**33. Bảng 2.10 dùng để làm gì?**
Tách yếu tố giữ chung (dữ liệu, thiết lập huấn luyện, cách chọn và đánh giá, cấu trúc đầu phân loại) với yếu tố khác nhau (phần trích đặc trưng, đầu vào 256 × 256 × 1 hay 224 × 224 × 3, khởi tạo, 128/512/1.280 đặc trưng, 0/11.186.112/2.257.984 tham số cố định). Vì nhiều yếu tố cùng khác, kết quả chỉ xếp hạng cấu hình, không tách được đóng góp của ImageNet, kiến trúc hay độ phân giải (mục 2.4.1).

**34. Hình 3.4 so sánh gì?**
Validation loss ba mô hình trên cùng trục, thang logarit, dấu sao là checkpoint. Sau epoch 1: MobileNetV2 0,0923, ResNet-18 0,2464, CNN 0,6838 (gần ln 2 ≈ 0,693, tức gán 0,5 cho mọi ảnh). Thứ tự MobileNetV2 < ResNet-18 < CNN giữ ở 28/30 epoch, trừ epoch 14 và 17. Nên việc chọn MobileNetV2 không phụ thuộc một epoch (mục 3.3.5).

**35. Bảng 3.4 thêm được gì so với Bảng 3.3?**
Chỉ số của lớp đạt và support (453 lỗi, 262 đạt). CNN có precision lớp đạt chỉ 90,91%: 275 ảnh được dự đoán đạt có 25 ảnh lỗi. MobileNetV2 có FN = FP = 2 nhưng F1 đạt 99,24% < F1 lỗi 99,56%, vì lớp đạt ít ảnh hơn nên mỗi ảnh sai nặng hơn. Cả ba mô hình đều có F1 đạt < F1 lỗi, nên macro F1 thấp hơn F1 lỗi (mục 3.4.1).

**36. Bảng 3.6 cho thấy chi phí thế nào?**
Thời gian 30 epoch: 300,36 / 990,97 / 557,11 s; tệp .keras 1,26 / 45,35 / 10,61 MB; ảnh test sai 37 / 9 / 4. ResNet-18 học ít tham số nhất (32.897) nhưng chậm nhất, khoảng 1,78 lần MobileNetV2, vì 11.186.112 tham số cố định vẫn chạy xuôi mỗi batch. Dung lượng tệp theo tổng tham số; tệp CNN khoảng 12 byte/tham số vì lưu kèm hai moment của Adam. Chưa đo thời gian dự đoán một ảnh (mục 3.4.3).

**37. Bảng 3.7 là gì? Sao không chọn ngưỡng theo bảng đó?**
Bảng 3.7 đếm lại FN và FP của ba mô hình ở các ngưỡng 0,1–0,99 từ p_ok đã lưu của 715 ảnh test. Tăng ngưỡng thì FN giảm, FP tăng. Đây là **phân tích bổ sung trên test để minh họa đánh đổi**, không dùng để chọn ngưỡng, vì chọn ngưỡng trên test làm kết quả test lạc quan. Ngưỡng đúng phải chọn trên validation; lượt chạy này chỉ lưu dự đoán 10 ảnh validation nên chưa làm được (mục 3.4.4).

**38. Muốn không bỏ sót ảnh lỗi nào thì ngưỡng phải bao nhiêu?**
Trên test: MobileNetV2 phải vượt 0,9556 (p_ok của `cast_def_0_150.jpeg`); tại t = 0,96 báo nhầm tăng từ 2 lên 9. CNN chỉ đạt FN = 0 từ t = 0,96 trong các ngưỡng của Bảng 3.7, khi đó báo nhầm **229/262 ≈ 87,4%** ảnh đạt. ResNet-18 cần ngưỡng **> 0,9878** (p_ok của `cast_def_0_2177.jpeg`); tại t = 0,99 báo nhầm 36 ảnh. Ý chính: ngưỡng để không bỏ sót do vài ảnh lỗi có p_ok rất cao quyết định; mô hình tách hai lớp càng tốt thì báo nhầm tăng càng ít.

**39. MobileNetV2 hơn ResNet-18 5 ảnh. Có đủ nói MobileNetV2 tốt hơn không?**
Chưa đủ. Chỉ xét 11 ảnh mà hai mô hình bất đồng (một đúng, một sai): ResNet-18 sai riêng 8, MobileNetV2 sai riêng 3. Nếu hai mô hình ngang nhau, mỗi ảnh như vậy rơi về một bên với xác suất 0,5. Xác suất một bên chỉ sai từ 3 ảnh trở xuống trong 11 (tính cả hai phía) ≈ **0,23**. Con số này lớn, nên chênh lệch có thể do ngẫu nhiên (mục 3.4.3, Kết luận).
*Nếu thầy hỏi "phép tính này là gì?":* "Dạ đây là kiểm định McNemar dạng chính xác (kiểm định nhị thức trên các ảnh hai mô hình bất đồng); báo cáo mô tả bằng lời, không nêu tên."

**40. Còn so với CNN thì sao?**
Chênh rõ. Số ảnh chỉ CNN sai là 35 so với 7 (với ResNet-18) và 36 so với 3 (với MobileNetV2). Xác suất tương ứng đều **< 10⁻⁴**. Nên nói được: hai cấu hình học chuyển giao tốt hơn CNN trên tập test này; vẫn chưa tách được yếu tố nào tạo ra chênh lệch (Bảng 2.10).

**41. Ba mô hình có sai cùng những ảnh không? (Bảng 3.9)**
Gần như không. CNN và ResNet-18 cùng sai 2 ảnh; CNN và MobileNetV2 cùng sai 1; ResNet-18 và MobileNetV2 cùng sai 1. Chỉ `cast_def_0_150.jpeg` sai ở cả ba. 47 ảnh sai ở ít nhất một mô hình, 45 trong số đó chỉ sai ở một mô hình; 668 ảnh đúng ở cả ba. 16 ảnh FP là 16 ảnh khác nhau. Vậy phần lớn ảnh sai gắn với từng mô hình (mục 3.5.1).

**42. Hình 3.11 (phân bố p_ok) đọc thế nào?**
Mỗi mô hình một biểu đồ: phân bố p_ok của ảnh lỗi và ảnh đạt, trục tung thang logarit, tam giác là ảnh sai. MobileNetV2 tách rõ: 448/453 ảnh lỗi có p_ok < 0,05, 253/262 ảnh đạt có p_ok > 0,95, chỉ 7 ảnh trong khoảng 0,2–0,8. CNN trải rộng: 76 ảnh trong khoảng đó, 28 ảnh sai. Ở cả ba mô hình, mọi ảnh có độ tự tin ≥ 0,99 đều đúng (323 / 649 / 686 ảnh), nhưng vẫn có ảnh sai dù độ tự tin ≥ 0,9 (6 / 3 / 1). Điểm chưa hiệu chuẩn nên không đọc p_ok là xác suất thật (mục 3.5.1).

**43. Nhóm ảnh `new__0_*` là gì?**
Chỉ nói số liệu. Test có 37 ảnh lỗi tên dạng `new__0_*`, mẫu tên chỉ có trong test. CNN bỏ sót 7/37 (18,9%), so với 18/416 ≈ 4,3% ở các ảnh lỗi còn lại. ResNet-18 và MobileNetV2 không sai ảnh nào trong nhóm. Nhóm không có thông tin về nguồn gốc nhóm ảnh này nên không suy đoán nguyên nhân.

Đọc thêm câu hỏi và cách giải thích chi tiết: [CAU_HOI_VA_GIAI_THICH_BAO_VE.md](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) · [HUONG_DAN_BAO_VE.md](HUONG_DAN_BAO_VE.md).

---

## 7. Ai trình bày phần nào

Theo bảng **Phân công nhiệm vụ** trong báo cáo, dự án có 8 nhiệm vụ:

| Stt | Nhiệm vụ | Người thực hiện |
|---|---|---|
| 1 | Tìm hiểu cơ sở lý thuyết | Minh, Thịnh, Quý |
| 2 | Thu thập và tiền xử lý dữ liệu | Thịnh |
| 3 | Phát biểu bài toán và xây dựng CNN baseline | Minh |
| 4 | Huấn luyện ResNet-18, MobileNetV2 bằng học chuyển giao | Minh |
| 5 | Đánh giá và so sánh ba mô hình | Minh, Quý |
| 6 | Phân tích ảnh dự đoán sai và hạn chế | Quý |
| 7 | Lưu mô hình và xây dựng demo | Quý |
| 8 | Viết báo cáo | Cả nhóm |

Gợi ý chia phần trình bày theo phân công (nhóm tự chốt lại thứ tự và thời lượng):

| Phần trình bày | Mục báo cáo | Người trình bày | Người hỗ trợ khi bị hỏi |
|---|---|---|---|
| Mở đầu, cơ sở lý thuyết | Mở đầu, Chương 1 | Cả nhóm (chia nhỏ theo mục 1.1–1.5) | Ai cũng phải trả lời được 1.5 (CNN, học chuyển giao) |
| Phát biểu bài toán | 2.1 | Minh | Thịnh |
| Dữ liệu, kiểm trùng, chia tập, tiền xử lý | 2.2, 3.2 | **Thịnh** | Minh |
| Kiến trúc CNN, ResNet-18, MobileNetV2 | 2.3 | **Minh** | Quý |
| Huấn luyện, tiêu chí đánh giá, chọn mô hình | 2.4, 3.1, 3.3 | **Minh** | Quý |
| Kết quả test, ma trận nhầm lẫn, so sánh | 3.4 | **Minh và Quý** | Thịnh |
| Ảnh dự đoán sai, hạn chế | 3.5.1, 3.5.4 | **Quý** | Minh |
| Lưu mô hình, kiểm chứng, chạy demo | 3.5.2, 3.5.3 | **Quý** | Minh |
| Kết luận, hướng phát triển | Kết luận | Cả nhóm | |

**Mẹo khi bị hỏi phần không phải mình làm:**

1. **Trả lời trước, đừng đẩy ngay cho bạn.** Hội đồng hỏi để xem cả nhóm có hiểu không. Nói ý chính bằng số liệu trong README này, rồi mới nói "bạn X làm trực tiếp phần này, xin phép bạn bổ sung".
2. **Chỉ vào mục/bảng.** Ví dụ: "Dạ, MobileNetV2 bỏ sót 2 ảnh lỗi, ở Bảng 3.5 ạ." Mở báo cáo đúng chỗ khi cần.
3. **Dùng khung ba ý:** làm gì → vì sao → kết quả là số nào. Ví dụ: "Nhóm loại 64 ảnh trùng → để test không bị rò rỉ → còn 5.583/986/715."
4. **Không chắc thì nói phạm vi:** "Phần này nhóm chưa kiểm chứng, đã ghi ở mục hạn chế 3.5.4." Không đoán, không bịa.
5. **Không nói lệch nhau:** cả nhóm dùng cùng câu tóm tắt 30 giây ở mục 1 và cùng bảng số ở mục 5.

---

## 8. Bản đồ báo cáo v4 và phép tính tay

### 8.1. Báo cáo gồm những gì

Bản in 25/11/2026: 69 tờ PDF = bìa + 10 trang La Mã (i–x: Lời cảm ơn, Mục lục, Phân công, Danh mục hình vii, Danh mục bảng viii, Viết tắt ix, Mở đầu x) + **58 trang Ả Rập** (Chương 1 từ tr. 1, Chương 2 tr. 20, Chương 3 tr. 38, Kết luận tr. 55, Tài liệu tham khảo tr. 56, Phụ lục tr. 58). Toàn báo cáo có 34 hình và 23 bảng.

| Phần | Nội dung | Hình, bảng, công thức |
|---|---|---|
| Phân công nhiệm vụ | 8 nhiệm vụ, phân theo tiến độ và theo thành viên | 2 bảng phân công |
| Mở đầu | Lý do, mục tiêu, phạm vi, bố cục 3 chương | |
| **Chương 1. Cơ sở lý thuyết** | 1.1 học máy, học sâu · 1.2 toán nền tảng (tensor, xác suất, entropy, gradient) · 1.3 học từ dữ liệu · 1.4 hàm mục tiêu, tối ưu, khái quát hóa · 1.5 mạng nơ-ron và CNN (1.5.3 tích chập và tầng gộp, 1.5.4 cấu trúc CNN và học chuyển giao) · 1.6 kết luận | Hình 1.1–1.13, Bảng 1.1–1.2, công thức (1.1)–(1.11); (1.11) là công thức kích thước đầu ra Conv |
| **Chương 2. Phương pháp và mô hình** | 2.1 phát biểu bài toán · 2.2 dữ liệu (2.2.1 nguồn và nhãn, 2.2.2 kiểm trùng, 2.2.3 chia tập, 2.2.4 chuẩn hóa) · 2.3 ba mô hình (2.3.2 CNN, 2.3.3 ResNet-18, 2.3.4 MobileNetV2) · 2.4 huấn luyện (2.4.1 ba cấu hình, 2.4.3 checkpoint, 2.4.4 tiêu chí đánh giá, 2.4.5 chọn mô hình trước test) · 2.5 kết luận | Bảng 2.1 nhãn · **2.2 số ảnh qua các bước** · 2.3 vai trò ba tập · 2.4 đầu vào/chuẩn hóa · 2.5 CNN · 2.6 ResNet-18 · 2.7 MobileNetV2 · 2.8 ba cấu hình · 2.9 thiết lập huấn luyện · **2.10 yếu tố chung/khác** · 2.11 ma trận nhầm lẫn. **Hình 2.1 quy trình xử lý dữ liệu** · 2.2 ảnh mẫu · 2.3 CNN · 2.4 basic block · 2.5 ResNet-18 · 2.6 inverted residual · 2.7 MobileNetV2 · **2.8 quy trình thực nghiệm**. Công thức (2.1) mô hình, (2.2)–(2.3) bài toán tối ưu và BCE, (2.4) sigmoid, (2.5) accuracy và precision, (2.6) recall và F1 |
| **Chương 3. Kết quả thực nghiệm bài toán kiểm tra lỗi sản phẩm đúc kim loại** | 3.1 môi trường · 3.2 kiểm tra và chia dữ liệu · 3.3 kết quả huấn luyện (3.3.1 checkpoint, 3.3.2–3.3.4 từng mô hình, **3.3.5 so sánh ba mô hình**) · 3.4 đánh giá trên test (3.4.1 chỉ số, 3.4.2 ma trận, 3.4.3 so sánh, **3.4.4 ảnh hưởng của ngưỡng**) · 3.5 phân tích và minh họa (3.5.1 ảnh đúng/sai, 3.5.2 lưu và kiểm tra mô hình, 3.5.3 demo, 3.5.4 hạn chế) · 3.6 kết luận | Bảng 3.1 phân bố dữ liệu · 3.2 checkpoint và thời gian · **3.3 accuracy/precision/recall/F1 lớp lỗi** · **3.4 chỉ số theo từng lớp** · **3.5 TP/FN/FP/TN** · **3.6 chi phí** · **3.7 FN/FP theo ngưỡng** · 3.8 bốn ảnh MobileNetV2 sai · **3.9 ảnh sai chung** · 3.10 kiểm chứng sau nạp. Hình 3.1–3.3 đường học · **3.4 validation loss chung trục** · 3.5–3.7 ma trận nhầm lẫn · 3.8–3.10 ảnh đúng/sai · **3.11 phân bố pₒₖ** · 3.12–3.13 demo |
| Kết luận, Tài liệu tham khảo (15 tài liệu), Phụ lục (link Google Drive mã nguồn) | | |

**Lật nhanh khi bảo vệ (số trang in ở chân trang):**

| Cần chỉ | Trang | Cần chỉ | Trang |
|---|---:|---|---:|
| Hình 2.1 quy trình xử lý dữ liệu | 22 | Bảng 3.4 chỉ số theo lớp | 43 |
| Bảng 2.2 số ảnh qua các bước | 25 | Bảng 3.5, Hình 3.5–3.7 ma trận nhầm lẫn | 44 |
| Bảng 2.5 CNN baseline | 27 | Mục 3.4.3 so sánh, xác suất ≈ 0,23 | 45 |
| Hình 2.8, Bảng 2.10 | 34 | Bảng 3.6 chi phí; mục 3.4.4 | 46 |
| Bảng 2.11 ma trận nhầm lẫn | 36 | Bảng 3.7 FN/FP theo ngưỡng | 47 |
| Bảng 3.1 phân bố dữ liệu | 38 | Bảng 3.8 bốn ảnh MobileNetV2 sai | 49 |
| Bảng 3.2 checkpoint | 39 | Bảng 3.9, Hình 3.11 | 50 |
| Mục 3.3.5 | 41 | Bảng 3.10 kiểm chứng sau nạp | 51 |
| Hình 3.4 validation loss | 42 | Hình 3.12, 3.13 demo | 52, 53 |
| Bảng 3.3 chỉ số lớp lỗi | 43 | Mục 3.5.4 hạn chế | 53 |

**Số đã đổi so với bản trước (để không chỉ nhầm):** Hình 2.1 → 2.2 (ảnh mẫu), 2.2 → 2.3, 2.3 → 2.4, 2.4 → 2.5, 2.5 → 2.6, 2.6 → 2.7; Bảng 2.2 → 2.3, 2.3 → 2.4, 2.4 → 2.5, 2.5 → 2.6, 2.6 → 2.7, 2.7 → 2.8, 2.8 → 2.9, 2.9 → 2.11; Hình 3.4–3.6 → 3.5–3.7, 3.7–3.9 → 3.8–3.10, 3.10–3.11 → 3.12–3.13; Bảng 3.4 → 3.5, 3.5 → 3.8, 3.6 → 3.10. Mới hoàn toàn: Hình 2.1, 2.8, 3.4, 3.11; Bảng 2.2, 2.10, 3.4, 3.6, 3.7, 3.9; mục 3.3.5, 3.4.4.

### 8.2. Phép tính tay có thể được yêu cầu viết lên bảng

| Nội dung | Công thức | Ví dụ trong đề tài |
|---|---|---|
| Kích thước đầu ra Conv | o = ⌊(n + 2p − k) / s⌋ + 1 (1.11) | Ảnh 5 × 5, bộ lọc 3 × 3, p = 0, s = 1 → 3 × 3 (Hình 1.11). Padding same, s = 1 → giữ 256 × 256 |
| MaxPooling 2 × 2, stride 2 | chia đôi mỗi chiều | 256 → 128 → 64 → 32 |
| Tham số Conv | (k × k × C_vào + 1) × C_ra | 320; 18.496; 73.856 |
| Tham số Dense | (số đầu vào + 1) × số nơ-ron | 8.256; 65. Tổng CNN = 100.993 |
| Đầu phân loại học chuyển giao | (số đặc trưng + 1) × 64 + 65 | ResNet-18: 32.897; MobileNetV2: 82.049 |
| Sigmoid | σ(z) = 1 / (1 + e^(−z)); σ′ = σ(1 − σ) | σ(0) = 0,5; đạo hàm lớn nhất 0,25 tại z = 0 |
| BCE một ảnh | L = −[y·ln p + (1 − y)·ln(1 − p)] | Ảnh đạt: p = 0,9 → 0,105; p = 0,1 → 2,303. Gán 0,5 cho mọi ảnh → ln 2 ≈ 0,693 (CNN epoch 1: 0,6838) |
| Accuracy, precision, recall | (TP+TN)/tổng; TP/(TP+FP); TP/(TP+FN) | MobileNetV2: 711/715 = 99,44%; 451/453 = 99,56% |
| Precision, recall lớp đạt | TN/(TN+FN); TN/(TN+FP) | CNN: 250/275 = 90,91%; 250/262 = 95,42% (Bảng 3.4) |
| Một ảnh test | 1 / 715 | ≈ 0,14 điểm phần trăm |
| Chia tập | 6.633 − 64 = 6.569; 15% | 986 validation, 5.583 train |
| Số bước mỗi epoch | ⌈5.583 / 32⌉ | 175 |
| Tỷ lệ lỗi trong validation | 564 / 986 | 57,20% (bằng val acc của CNN 3 epoch đầu) |
| Tỷ lệ bỏ sót, báo nhầm | FN/453; FP/262 | CNN 25/453 ≈ 5,52%, 12/262 ≈ 4,58%; MobileNetV2 0,44%, 0,76% |
| Precision lỗi khi dây chuyền 5% lỗi (giả định, mục 3.4.2) | R·0,05 / (R·0,05 + tỷ lệ báo nhầm·0,95) | MobileNetV2 ≈ 87,3%; CNN ≈ 52,1% |
| So hai mô hình trên 11 ảnh bất đồng (mục 3.4.3) | 2 × [C(11,0) + C(11,1) + C(11,2) + C(11,3)] / 2¹¹ | 2 × (1 + 11 + 55 + 165) / 2.048 = 464/2.048 ≈ 0,23 |
| Thời gian mỗi epoch | thời gian 30 epoch / 30 | 10,01 / 33,03 / 18,57 s |

### 8.3. Ảnh hưởng của ngưỡng (Bảng 3.7)

<!-- Bảng 3.7 của bản in; đã đếm lại từ test_predictions_*.csv ngày 25/11/2026, khớp từng ô. -->

Bảng 3.7 (mục 3.4.4) đếm FN, FP của ba mô hình trên 715 ảnh test theo ngưỡng t, quy tắc "đạt khi p_ok ≥ t". Đây là **phân tích bổ sung để minh họa đánh đổi**, không dùng để chọn ngưỡng: chọn ngưỡng trên test làm kết quả test lạc quan. Ngưỡng báo cáo dùng vẫn là 0,5.

| Ngưỡng t | CNN FN | CNN FP | ResNet-18 FN | ResNet-18 FP | MobileNetV2 FN | MobileNetV2 FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0,1 | 60 | 0 | 13 | 0 | 5 | 0 |
| 0,3 | 39 | 4 | 10 | 1 | 3 | 2 |
| **0,5 (đang dùng)** | 25 | 12 | 7 | 2 | 2 | 2 |
| 0,7 | 10 | 24 | 5 | 4 | 1 | 2 |
| 0,9 | 6 | 106 | 3 | 11 | 1 | 6 |
| 0,96 | 0 | 229 | 2 | 19 | 0 | 9 |
| 0,99 | 0 | 262 | 0 | 36 | 0 | 22 |

- MobileNetV2: FN giữ ở 1 trong khoảng 0,6–0,95 vì `cast_def_0_150.jpeg` có p_ok 0,9556. Tại t = 0,96, FN = 0 nhưng FP tăng từ 2 lên 9.
- CNN: trong các ngưỡng ở bảng, chỉ đạt FN = 0 **từ t = 0,96**, khi đó báo nhầm **229/262 ≈ 87,4%** ảnh đạt.
- ResNet-18: cần ngưỡng **> 0,9878** (p_ok của `cast_def_0_2177.jpeg`); tại t = 0,99 báo nhầm 36 ảnh.
- Ý chính: hai lớp càng tách biệt (Hình 3.11) thì tăng ngưỡng càng ít làm tăng báo nhầm. Ngưỡng để không bỏ sót do vài ảnh lỗi có p_ok rất cao quyết định.
- Làm đúng: lưu p_ok toàn bộ validation, chọn ngưỡng theo tỷ lệ chi phí FN/FP trên validation, rồi mới báo FN, FP trên test. Lượt này chỉ lưu dự đoán 10 ảnh validation nên chưa làm được.

### 8.4. Thuật ngữ, mỗi người nói được trong một câu

| Thuật ngữ | Một câu |
|---|---|
| SHA-256 | "Dấu vân tay" của ảnh: giống hệt từng pixel thì cùng mã |
| EXIF | Thông tin đi kèm ảnh, có hướng xoay; nhóm xoay đúng chiều trước khi xử lý |
| Stratify | Chia sao cho tỷ lệ lỗi/đạt hai phần giống nhau |
| Epoch / batch | Một lượt qua hết tập train / một nhóm 32 ảnh cho một lần cập nhật |
| Checkpoint | Bản lưu trọng số tại một epoch |
| Baseline | Mô hình mốc để so sánh; ở đây là CNN tự xây |
| Backbone | Phần trích đặc trưng của mạng học sẵn trên ImageNet |
| Đóng băng | Giữ nguyên trọng số backbone, chỉ học đầu Dense mới |
| Fine-tuning | Mở khóa một phần backbone để học tiếp; nhóm chưa làm |
| Batch Normalization | Chuẩn hóa đầu ra một tầng; khi đóng băng thì giữ thống kê cũ |
| Dropout | Khi train, tắt ngẫu nhiên một tỷ lệ giá trị để giảm quá khớp |
| GAP | Lấy trung bình mỗi bản đồ đặc trưng thành một số |
| Depthwise convolution | Mỗi kênh có bộ lọc riêng, ít phép tính hơn Conv thường |
| ReLU6 | ReLU chặn trên ở 6, dùng trong MobileNetV2 |
| ReduceLROnPlateau | Validation loss không cải thiện 3 epoch thì giảm LR một nửa |
| Quá khớp / thiếu khớp | Học thuộc train mà kém trên dữ liệu mới / học chưa đủ, kém cả trên train |
| Hiệu chuẩn (calibration) | Kiểm tra "90% tự tin" có thật sự đúng 90% không; nhóm chưa làm |
| Macro F1 | Trung bình F1 lớp lỗi và lớp đạt |

---

## 9. Liên kết và checklist trước buổi bảo vệ

### 9.1. Liên kết

| Tài liệu | Dùng khi |
|---|---|
| [Báo cáo v4 (.docx)](bao_cao/baocaocodong_v4.docx) · [PDF](bao_cao/baocaocodong_v4.pdf) | Bản in 25/11/2026 (58 trang Ả Rập), bản nộp và bản dùng khi bảo vệ |
| [HUONG_DAN_BAO_VE.md](HUONG_DAN_BAO_VE.md) | Đọc thêm: giải thích từng phần Chương 2–3 và trình tự nói |
| [CAU_HOI_VA_GIAI_THICH_BAO_VE.md](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) | Đọc thêm: câu hỏi theo chủ đề, cách đọc biểu đồ và ma trận |
| [Casting.ipynb](Casting.ipynb) | Xem mã và output lượt chính thức |
| [HUONG_DAN_CASTING.md](HUONG_DAN_CASTING.md) | Chạy lại notebook trên Colab |
| [demo/README.md](demo/README.md) · [demo/app.py](demo/app.py) | Chạy và giải thích demo |
| [comparison.csv](casting_full_3models_01/comparison.csv) · [selection.json](casting_full_3models_01/selection.json) · [split_counts.csv](casting_full_3models_01/split_counts.csv) | Đối chiếu số liệu khi bị hỏi |
| [test_predictions_cnn.csv](casting_full_3models_01/test_predictions_cnn.csv) · [resnet18](casting_full_3models_01/test_predictions_resnet18.csv) · [mobilenetv2](casting_full_3models_01/test_predictions_mobilenetv2.csv) | p_ok từng ảnh test: nguồn của Bảng 3.7, 3.9, Hình 3.11 và phép so 11 ảnh bất đồng |
| [Lịch sử CNN](casting_full_3models_01/history_cnn.csv) · [ResNet-18](casting_full_3models_01/history_resnet18.csv) · [MobileNetV2](casting_full_3models_01/history_mobilenetv2.csv) | Xem từng epoch (cột `epoch` đếm từ 0, báo cáo đếm từ 1) |
| [Kiểm chứng MobileNetV2](casting_full_3models_01/demo/verification.json) · [Kiểm chứng ba mô hình](demo/models/comparison_verification.json) | Bảng 3.10 (tệp thứ hai nằm trong `demo/models/`, chỉ có sau khi chạy `prepare_models.py`) |
| [Thư mục Google Drive của nhóm](https://drive.google.com/drive/folders/1otbkaT-wvJfNERf_CL0uapmz2M2ebSnR?usp=sharing) | Mã nguồn nộp kèm (Phụ lục) |
| [Nguồn dữ liệu Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Tài liệu [12] |
| [prompt/](prompt/) | Prompt và meta-prompt của dự án (quy ước trong AGENTS.md) |

### 9.2. Checklist trước buổi bảo vệ

- [ ] Mỗi người đọc mục 1, 4, 5 của README và tự trả lời to 43 câu ở mục 6 (câu 32–43 là nội dung mới của bản in).
- [ ] Thuộc câu tóm tắt 30 giây và năm con số: 5.583/986/715 · 94,83/98,74/99,44% · MobileNetV2 epoch 21 · 711/715, FN 2, FP 2 · 100.993 / 32.897 / 82.049 tham số được học.
- [ ] Chạy thử demo trên laptop sẽ mang đi (lần đầu nạp mô hình lâu); để sẵn `cast_ok_0_45.jpeg` và `cast_def_0_1059.jpeg`.
- [ ] Mở sẵn PDF báo cáo v4; nếu demo lỗi thì chỉ vào Hình 3.12 và 3.13 (tr. 52–53).
- [ ] Mở link Google Drive ở cửa sổ ẩn danh để chắc người khác xem được.
- [ ] Sạc đầy laptop, có mặt sớm.
