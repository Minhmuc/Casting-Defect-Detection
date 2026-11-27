# Tài liệu bảo vệ – Kiểm tra lỗi sản phẩm đúc kim loại

**Nhóm 10 · Học phần Học sâu · ICTU** · Nguyễn Quang Minh, Nông Hữu Thịnh, Vũ Ngọc Quý · GVHD: TS. Ngô Hữu Huy
Bám báo cáo `bao_cao/baocaocodong_v4.pdf`, bản in 27/11/2026 (60 trang Ả Rập, 37 hình, 23 bảng, 17 tài liệu tham khảo). Mọi con số lấy từ lượt chạy chính thức `casting_full_3models_01/`.

> **Cách dùng.** Đây là tài liệu duy nhất cần đọc. Đọc theo thứ tự: phần 1 (thuộc lòng) → phần 2 (hiểu cả dự án như một câu chuyện) → phần 3 (lý thuyết nền) → phần 4 (phần riêng của mình) → phần 5–7 (câu hỏi, bẫy, demo). Ai đọc hết đều phải trả lời được mọi câu ở phần 5, kể cả phần mình không làm.
> Khi nói, phân biệt rõ: **cách làm** thì dẫn "theo mục 2.x"; **con số kết quả** thì nói "ở Chương 3".

---

## 1. Phải thuộc lòng

**Câu tóm tắt 30 giây (cả nhóm nói giống nhau):**
> Nhóm phân loại ảnh sản phẩm đúc thành lỗi hoặc đạt. Sau khi loại 64 ảnh trùng giữa train và test, nhóm có 5.583 ảnh train, 986 ảnh validation và 715 ảnh test. Nhóm dùng một CNN học từ đầu làm mốc, so với ResNet-18 và MobileNetV2 dùng trọng số ImageNet, đóng băng backbone và chỉ học đầu phân loại. Mô hình được chọn theo validation loss là MobileNetV2 ở epoch 21. Trên 715 ảnh test, mô hình đúng 711 ảnh, bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.

**Năm nhóm số:**

| Nội dung | Số |
|---|---|
| Dữ liệu | 7.348 ảnh nguồn → loại 64 ảnh trùng → 7.284 ảnh; **5.583 train / 986 validation / 715 test** |
| Tham số được học | CNN **100.993** (toàn bộ) · ResNet-18 **32.897** / 11.219.009 · MobileNetV2 **82.049** / 2.340.033 |
| Checkpoint chọn | CNN epoch 29 (val loss 0,1210) · ResNet-18 epoch 29 (0,0269) · **MobileNetV2 epoch 21 (0,0150)** |
| Accuracy test | CNN **94,83%** · ResNet-18 **98,74%** · MobileNetV2 **99,44%** (711/715) |
| FN / FP trên test | CNN 25 / 12 · ResNet-18 7 / 2 · MobileNetV2 **2 / 2** |

**Quy ước không được nói sai:** nhãn **0 = lỗi** (`def_front`), **1 = đạt** (`ok_front`). Mô hình xuất **pₒₖ = điểm lớp đạt**; điểm lỗi = 1 − pₒₖ. Dự đoán đạt khi pₒₖ ≥ 0,5. Lớp **lỗi là lớp quan tâm** khi tính TP, precision, recall. **FN = ảnh lỗi bị cho là đạt (bỏ sót)**, **FP = ảnh đạt bị báo lỗi (báo nhầm)**.

---

## 2. Cả dự án như một câu chuyện (Chương 2–3)

### 2.1. Bài toán (mục 2.1)
Đầu vào là ảnh chụp mặt trước một sản phẩm đúc (cánh bơm chìm). Đầu ra là nhãn **lỗi/đạt cho cả ảnh** kèm điểm pₒₖ. Dữ liệu chỉ có nhãn cấp ảnh nên đề tài **không định vị** vết lỗi. Huấn luyện bằng cách cực tiểu **binary cross-entropy (BCE)** trên train; Adam chỉ tìm nghiệm xấp xỉ, nên nhóm giữ trạng thái tại epoch có **validation loss thấp nhất** (không phải train loss thấp nhất) để hạn chế quá khớp.

### 2.2. Dữ liệu (mục 2.2, Bảng 2.1–2.3, Hình 2.1–2.5)

- **Nguồn [12]:** bộ *Casting Product Image Data for Quality Inspection* (Kaggle, giấy phép CC BY-NC-ND 4.0, dùng cho học tập). Ảnh xám, chụp từ trên xuống, chiếu sáng ổn định. Nguồn liệt kê 7 dạng khuyết tật đúc (rỗ khí, lỗ kim, bavia, co ngót, vật liệu khuôn, kim loại rót, luyện kim) nhưng **nhãn chỉ có lỗi/đạt**.
- **Hai phiên bản:** 1.300 ảnh 512 × 512 chưa tăng cường, hoặc **7.348 ảnh 300 × 300 đã tăng cường sẵn** và chia sẵn train 6.633 / test 715. Nhóm dùng bản thứ hai: nhiều hơn ≈ 5,7 lần, test cố định dễ đối chiếu.
- Ảnh JPEG lưu 3 kênh nhưng R = G = B → chuyển về ảnh xám 1 kênh, không mất thông tin. Kiểm cả 7.348 ảnh: **0 ảnh hỏng**.

**Kiểm trùng bằng SHA-256 (mục 2.2.2).** Băm *kích thước + pixel ảnh xám trước khi resize*. Hai ảnh cùng mã = giống hệt từng pixel.

- Kết quả: **64 cặp trùng, đều là ảnh đạt**, mỗi cặp gồm một ảnh ở train và một ảnh **cùng tên** ở test. Nhóm **loại bản ở train**, giữ nguyên test. Không có trùng nội bộ train/test, không có trùng khác nhãn.
- Vì sao phải loại: nếu giữ, 64/262 ≈ **24,4% ảnh đạt của test** là ảnh mô hình đã học → điểm test cao giả tạo (rò rỉ dữ liệu). Loại chỉ mất 0,96% train.
- Vì sao không so theo tên tệp: còn **80 cặp cùng tên nhưng khác nội dung** (44 lỗi, 36 đạt) → so theo tên sẽ loại nhầm.
- Giới hạn: SHA-256 chỉ bắt bản **y hệt**. Ảnh xoay/lật/đổi sáng của cùng sản phẩm có mã khác → có thể còn rò rỉ "gần giống" (mục 3.5.4).

**Chia tập (mục 2.2.3, Bảng 2.2, 3.1).** Train sạch 6.633 − 64 = 6.569 → chia **85/15 phân tầng theo nhãn, seed 42** → train 5.583 (3.194 lỗi, 2.389 đạt), validation 986 (564, 422). Test giữ nguyên 715 (453 lỗi, 262 đạt). Tỷ lệ lỗi: train/validation 57,2%, test 63,4%. Loại trùng **trước** khi chia validation nên không ảnh trùng nào lọt vào validation.

- Hệ quả của lệch lớp: đoán "lỗi" cho mọi ảnh đã đúng 57,20% (validation) và 63,36% (test) → phải đọc thêm chỉ số theo lớp và **macro F1**. Lệch ở train nhẹ nên **không dùng class weight**.

**Phân tích khám phá dữ liệu – EDA (cuối 2.2.1, Hình 2.3–2.5),** tính trên 5.583 ảnh train:

- Hình 2.3: lớp lỗi chiếm đa số ở cả ba tập nhưng lớp đạt vẫn trên một phần ba → lệch **nhẹ**.
- Hình 2.4: ảnh trung bình hai lớp đều **đồng tâm** (sản phẩm luôn đặt giữa). Hiệu giữa hai ảnh trung bình là các vòng đồng tâm, **không chỉ ra vị trí lỗi** vì lỗi mỗi ảnh một chỗ, bị lấy trung bình làm mờ → mô hình phải học đặc trưng cục bộ.
- Hình 2.5: ảnh đạt **sáng hơn ảnh lỗi khoảng 10,5 mức xám** (149,82 so với 139,28), nhưng hai phân bố chồng lấn ở 140–150. → Mô hình **có thể dựa một phần vào độ sáng**; nhóm chưa kiểm chứng, kết quả Chương 3 đọc kèm lưu ý này.

**Chuẩn hóa đầu vào (mục 2.2.4, Bảng 2.4).** Mọi ảnh resize bilinear về 256 × 256 × 1 (0–255). Sau đó:

| Mô hình | Đầu vào mạng | Chuẩn hóa (nằm **trong** mô hình) |
|---|---|---|
| CNN | 256 × 256 × 1 | x / 255 |
| ResNet-18 | 224 × 224 × 3 (lặp kênh xám) | (x/255 − μ)/σ theo thống kê ImageNet |
| MobileNetV2 | 224 × 224 × 3 (lặp kênh xám) | x / 127,5 − 1 |

Chuẩn hóa đặt trong mô hình nên demo chỉ cần đưa ảnh 0–255 vào là đúng. **Không thêm augmentation** vì bộ nguồn đã có sẵn ảnh tăng cường.

### 2.3. Ba mô hình (mục 2.3, Bảng 2.5–2.8, Hình 2.6–2.10)
Cả ba có **cùng đầu phân loại**: GAP → Dense 64 ReLU → Dropout 0,3 → Dense 1 sigmoid. Khác nhau ở phần trích đặc trưng phía trước.

**CNN baseline (Hình 2.6, Bảng 2.5)** – nhóm tự thiết kế, học toàn bộ từ đầu:

| Tầng | Đầu ra | Tham số |
|---|---|---:|
| Rescaling /255 | 256 × 256 × 1 | 0 |
| Conv2D 32, 3 × 3, same, ReLU → MaxPool 2 × 2 | 256×256×32 → 128×128×32 | (3·3·1+1)·32 = **320** |
| Conv2D 64 → MaxPool | 128×128×64 → 64×64×64 | (3·3·32+1)·64 = **18.496** |
| Conv2D 128 → MaxPool | 64×64×128 → 32×32×128 | (3·3·64+1)·128 = **73.856** |
| Global Average Pooling | 128 | 0 |
| Dense 64 ReLU (+ Dropout 0,3) | 64 | (128+1)·64 = **8.256** |
| Dense 1 sigmoid | 1 | (64+1)·1 = **65** |
| **Tổng** | | **100.993** |

- Mỗi lần chia đôi kích thước thì gấp đôi số bộ lọc → biểu diễn nhiều mẫu hơn mà chi phí mỗi tầng không tăng.
- GAP thay Flatten: Flatten cần (131.072 + 1) × 64 = 8.388.672 tham số cho Dense 64, GAP chỉ cần 8.256 → ít quá khớp hơn, đổi lại mất thông tin vị trí.

**ResNet-18 (Hình 2.7–2.8, Bảng 2.6) [13, 15]** – trọng số ImageNet [16] từ KerasHub, **backbone đóng băng**. Basic block: hai Conv 3 × 3 + Batch Normalization tạo F(x), đầu ra **y = ReLU(F(x) + x)**. Đường nối tắt giúp khối chỉ cần học phần sai khác và gradient truyền ngược dễ hơn. Ảnh 224 → … → 7 × 7 × 512 → GAP 512. Học (512+1)·64 + 65 = **32.897** tham số.

**MobileNetV2 (Hình 2.9–2.10, Bảng 2.7) [9, 14]** – Keras Applications, ImageNet, α = 1,0, đóng băng. **Inverted residual:** Conv 1 × 1 mở rộng (64 → 384) → depthwise 3 × 3 lọc riêng từng kênh → Conv 1 × 1 chiếu tuyến tính về 64 kênh (**không ReLU** để không mất thông tin khi nén) → cộng với đầu vào. "Ngược" vì rộng ở giữa, hẹp ở hai đầu. Tích chập tách biệt rẻ hơn Conv 3 × 3 thường khoảng 8–9 lần. Đầu ra 7 × 7 × 1280 → GAP 1280. Học (1280+1)·64 + 65 = **82.049** tham số.

**Vì sao chọn hai mạng này:** ResNet-18 là mạng nhỏ nhất họ ResNet; MobileNetV2 là mạng nhẹ, hợp demo chạy CPU; cả hai có sẵn trọng số ImageNet trong Keras. **Vì sao đóng băng:** train chỉ ~5.600 ảnh, mục tiêu đo đóng góp của đặc trưng ImageNet với chi phí thấp. Fine-tuning **chưa làm**.

### 2.4. Huấn luyện và chọn mô hình (mục 2.4, Bảng 2.8–2.11, Hình 2.11)

- Thiết lập chung (Bảng 2.9): **30 epoch, batch 32, Adam lr 0,001** [7], BCE, seed 42, giảm tốc độ học ×0,5 khi validation loss không giảm 3 epoch (tối thiểu 10⁻⁶), **không dừng sớm**, lưu checkpoint validation loss thấp nhất, ngưỡng 0,5.
- Train xáo trộn mỗi epoch; validation/test giữ thứ tự cố định để gắn đúng pₒₖ với từng ảnh. Ba lượt dùng **chung một danh sách chia tập** (mã băm danh sách trùng nhau ở cả ba lượt).
- **Chọn mô hình trước khi xem test (mục 2.4.5):** so validation loss của ba checkpoint, chọn nhỏ nhất, ghi vào `selection.json` rồi mới chạy test. Dùng loss thay accuracy vì loss thay đổi liên tục theo xác suất, còn accuracy trên 986 ảnh nhảy theo bước ~0,1% và dễ hòa.
- **Giới hạn thiết kế (Bảng 2.10):** ba cấu hình chung dữ liệu, thiết lập, ngưỡng, thước đo, nhưng khác **cùng lúc** kiến trúc, độ phân giải, khởi tạo, phần được học → chỉ **xếp hạng ba cấu hình**, không tách được đóng góp riêng của ImageNet hay kiến trúc.
- Thước đo (mục 2.4.4): accuracy, precision/recall/F1 lớp lỗi, **macro F1** = (F1 lỗi + F1 đạt)/2.

### 2.5. Kết quả huấn luyện (mục 3.1–3.3, Bảng 3.2, Hình 3.1–3.4)
Môi trường: Google Colab, GPU Tesla T4, TensorFlow 2.20, Keras 3.12, KerasHub 0.23.

| Mô hình | Epoch chọn | Val loss | Val acc | Thời gian 30 epoch |
|---|---:|---:|---:|---:|
| CNN | 29 | 0,1210 | 95,84% | 300 s |
| ResNet-18 | 29 | 0,0269 | 98,78% | 991 s |
| **MobileNetV2** | **21** | **0,0150** | **99,49%** | 557 s |

- **CNN:** 3 epoch đầu đứng ở 57,20% (= tỷ lệ ảnh lỗi validation → chưa tách được lớp). Có hai đỉnh dao động ở epoch 9 và 18 rồi phục hồi. Validation loss thấp hơn train loss vì Dropout chỉ bật khi train và train loss là trung bình trong lúc trọng số đang đổi → không thấy quá khớp.
- **ResNet-18:** sau epoch 1 đã 95,44%. Huấn luyện **lâu nhất** dù học ít tham số nhất, vì backbone 11 triệu tham số vẫn phải chạy xuôi mỗi batch.
- **MobileNetV2:** sau epoch 1 đã 97,26%. Epoch 14 loss vọt lên 0,0696 dù chỉ thêm 12 ảnh sai → BCE phạt nặng dự đoán sai mà tự tin. Sau epoch 21 train loss vẫn giảm còn validation loss không giảm → **quá khớp nhẹ**, nên dùng checkpoint epoch 21.
- **So sánh (Hình 3.4):** sau epoch 1, val loss CNN 0,6838 ≈ ln 2 (đoán 0,5 cho mọi ảnh), ResNet-18 0,2464, MobileNetV2 0,0923. Thứ tự MobileNetV2 < ResNet-18 < CNN giữ ở **28/30 epoch** → lựa chọn không phụ thuộc một epoch may mắn.

### 2.6. Kết quả trên 715 ảnh test (mục 3.4, Bảng 3.3–3.7)

| Mô hình | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi | Macro F1 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 94,83 | 97,27 | 94,48 | 95,86 | 94,48 | 428 | 25 | 12 | 250 |
| ResNet-18 | 98,74 | 99,55 | 98,45 | 99,00 | 98,65 | 446 | 7 | 2 | 260 |
| MobileNetV2 | **99,44** | 99,56 | 99,56 | 99,56 | 99,40 | 451 | 2 | 2 | 260 |

- CNN chủ yếu **bỏ sót** (FN 25 gấp đôi FP 12). Chuyển sang học chuyển giao giảm bỏ sót 25 → 7 → 2.
- Mỗi ảnh test = 1/715 ≈ **0,14 điểm %**; MobileNetV2 hơn ResNet-18 0,70 điểm = 5 ảnh.
- Kết quả test **khớp với lựa chọn trên validation**; BCE trên test cùng thứ tự (0,1298 / 0,0354 / 0,0128).
- **MobileNetV2 có thật sự tốt hơn ResNet-18?** Chỉ 11 ảnh mà đúng một trong hai mô hình sai (ResNet-18 sai riêng 8, MobileNetV2 sai riêng 3). Nếu hai mô hình ngang nhau, xác suất lệch đến mức này ≈ **0,23** → **chưa đủ khẳng định**. Còn so với CNN thì chênh rõ (35 so với 7; 36 so với 3; xác suất < 10⁻⁴).
- **Chi phí (Bảng 3.6):** tệp .keras CNN 1,26 MB, ResNet-18 45,35 MB, MobileNetV2 10,61 MB. Chưa đo tốc độ suy luận.
- **Ý nghĩa thực tế:** FN = sản phẩm lỗi lọt ra ngoài (nguy hiểm hơn); FP = sản phẩm tốt bị giữ lại kiểm tay. Ví dụ minh họa (giả định, không phải số đo): nếu dây chuyền chỉ 5% lỗi thì precision lỗi còn ≈ 87,3% (MobileNetV2) và ≈ 52,1% (CNN).
- **Ngưỡng (mục 3.4.4, Bảng 3.7):** quy tắc "đạt khi pₒₖ ≥ t" → **tăng** t thì bỏ sót giảm, báo nhầm tăng. MobileNetV2 muốn không bỏ sót ảnh nào trên test phải đặt t > 0,9556, báo nhầm tăng từ 2 lên 9. Bảng này chỉ **minh họa đánh đổi**, không dùng để chọn ngưỡng (chọn trên test là gian lận); ngưỡng vẫn 0,5.

### 2.7. Phân tích ảnh sai, demo, hạn chế (mục 3.5)

- **Ảnh sai hầu như không trùng giữa các mô hình (Bảng 3.9):** 47 ảnh sai ở ít nhất một mô hình, 45 ảnh chỉ sai ở một mô hình, 668 ảnh đúng ở cả ba. Chỉ **`cast_def_0_150.jpeg`** sai ở cả ba (pₒₖ 0,71 / 0,98 / 0,96); nhìn bằng mắt không thấy lỗi rõ, nhóm chưa rõ nguyên nhân. Trong 25 ảnh CNN bỏ sót, ResNet-18 nhận đúng 23.
- **Bốn ảnh MobileNetV2 sai (Bảng 3.8):** 2 lỗi bị nhận là đạt (`cast_def_0_150` pₒₖ 0,9556; `cast_def_0_2102` pₒₖ 0,5024 – sát ngưỡng) và 2 đạt bị báo lỗi (`cast_ok_0_16`, `cast_ok_0_7839`).
- **Nhóm ảnh `new__0_*`:** 37 ảnh lỗi chỉ có trong test; CNN bỏ sót 18,9% nhóm này (so với ~4,3% ở ảnh lỗi khác); hai mạng học chuyển giao không sai ảnh nào. Nhóm không suy đoán nguồn gốc.
- **Phân bố pₒₖ (Hình 3.11):** MobileNetV2 dồn sát 0 và 1 (chỉ 7 ảnh trong khoảng 0,2–0,8), CNN trải rộng (76 ảnh). Ảnh có độ tự tin ≥ 0,99 đều đúng ở cả ba mô hình. Nhưng **pₒₖ chưa hiệu chuẩn** [5] → không nói "pₒₖ = 0,96 là đúng 96%".
- **Lưu và kiểm chứng (Bảng 3.10):** nạp lại checkpoint trên CPU, so với dự đoán đã lưu: 10/10 nhãn trùng, sai lệch điểm lớn nhất 3,2 × 10⁻⁶.
- **Demo Streamlit [17] (Hình 3.12–3.13):** chọn một hoặc cả ba mô hình, nhiều ảnh, xuất CSV/JSON; chỉ nạp mô hình có mã băm SHA-256 khớp checkpoint lúc huấn luyện; ảnh hỏng không làm dừng các ảnh khác.
- **Hạn chế (mục 3.5.4):** một seed, một lần chia; không bắt được ảnh gần giống; 99,44% chỉ đúng trong phạm vi bộ dữ liệu này; backbone đóng băng, chưa fine-tune, chưa tìm siêu tham số; chưa đo độ trễ, bộ nhớ; pₒₖ chưa hiệu chuẩn; nhãn cấp ảnh; chưa lưu pₒₖ toàn bộ validation nên chưa chọn ngưỡng theo chi phí; demo chưa kiểm định trong sản xuất.

---

## 3. Lý thuyết nền cần nói được (Chương 1)

- **AI ⊃ học máy ⊃ học sâu.** Học sâu = nhiều tầng biến đổi nối tiếp, **tự học biểu diễn** thay vì thiết kế đặc trưng thủ công (HOG, SIFT, LBP + SVM).
- **Nơ-ron:** z = w·x + b, a = f(z). **ReLU** f(z) = max(0, z). **Sigmoid** σ(z) = 1/(1 + e⁻ᶻ) ∈ (0, 1), σ(0) = 0,5, đạo hàm lớn nhất 0,25.
- **BCE một ảnh:** L = −[y ln p + (1 − y) ln(1 − p)]. Ảnh đạt (y = 1): p = 0,9 → 0,105; p = 0,1 → 2,303. Đoán 0,5 cho mọi ảnh → ln 2 ≈ 0,693. BCE = log-hợp lý âm của phân phối Bernoulli.
- **Gradient descent:** w ← w − η·∂L/∂w. **Lan truyền ngược** dùng quy tắc dây chuyền để tính gradient từ đầu ra về đầu vào. **Adam** giữ trung bình trượt của gradient và bình phương gradient → bước riêng cho từng tham số.
- **Tích chập:** bộ lọc k × k trượt trên ảnh, **kết nối cục bộ và dùng chung trọng số** → ít tham số, bắt được mẫu ở mọi vị trí. Kích thước ra o = ⌊(n + 2p − k)/s⌋ + 1 (công thức 1.11); padding same, stride 1 giữ nguyên kích thước. Tham số Conv = (k·k·C_vào + 1)·C_ra.
- **Pooling:** MaxPool 2 × 2 stride 2 chia đôi mỗi chiều, không có tham số. **GAP:** lấy trung bình mỗi kênh → 1 số/kênh.
- **Dropout:** tắt ngẫu nhiên nơ-ron khi huấn luyện (0,3 = 30%), tắt khi dự đoán → giảm quá khớp. **Batch Normalization:** chuẩn hóa theo batch, đóng băng thì dùng thống kê đã học.
- **Quá khớp:** train tốt, validation kém dần; **thiếu khớp:** cả hai đều kém. Validation dùng để chọn, test chỉ dùng **một lần** để đánh giá.
- **Học chuyển giao:** dùng lại phần trích đặc trưng học trên tập lớn (ImageNet [16], ~1,2 triệu ảnh, 1.000 lớp) cho bài toán mới; **trích đặc trưng cố định** (nhóm làm) hoặc **fine-tune** (chưa làm).
- **Precision** = TP/(TP+FP): trong các ảnh bị báo lỗi, bao nhiêu thật sự lỗi. **Recall** = TP/(TP+FN): trong các ảnh lỗi thật, bắt được bao nhiêu. **F1** = 2PR/(P+R).

---

## 4. Phân công và phần riêng của từng người

| Stt | Nhiệm vụ (theo báo cáo) | Người thực hiện |
|---|---|---|
| 1 | Tìm hiểu cơ sở lý thuyết | Minh, Thịnh, Quý |
| 2 | Thu thập và tiền xử lý dữ liệu | Thịnh |
| 3 | Phát biểu bài toán và xây dựng CNN baseline | Minh |
| 4 | Huấn luyện ResNet-18, MobileNetV2 bằng học chuyển giao | Minh |
| 5 | Đánh giá và so sánh ba mô hình | Minh, Quý |
| 6 | Phân tích ảnh dự đoán sai và hạn chế | Quý |
| 7 | Lưu mô hình và xây dựng demo | Quý |
| 8 | Viết báo cáo | Cả nhóm |

**Gợi ý trình bày (nhóm tự chốt):** Mở đầu + Chương 1: chia đều ba người · 2.1: Minh · 2.2 + 3.2 (dữ liệu, EDA): **Thịnh** · 2.3, 2.4, 3.1, 3.3: **Minh** · 3.4: **Minh và Quý** · 3.5 (ảnh sai, hạn chế, demo): **Quý** · Kết luận: cả nhóm.

**Thịnh – phải nói sâu được:** nguồn dữ liệu và lý do chọn bản 7.348 ảnh; quy trình Hình 2.1; SHA-256 trên pixel trước resize; 64 ảnh trùng (vì sao loại, vì sao không so tên, 80 cặp cùng tên khác nội dung); stratify 85/15 seed 42 và các số 6.569 → 5.583/986; ba hình EDA và lưu ý độ sáng; chuẩn hóa từng mô hình (Bảng 2.4); vì sao không augmentation, không class weight.

**Minh – phải nói sâu được:** công thức bài toán (2.1)–(2.4); tự tính 100.993, 32.897, 82.049 trên bảng; basic block và inverted residual; vì sao đóng băng; thiết lập huấn luyện; vì sao chọn theo validation loss và chọn trước test; đọc Hình 3.1–3.4; Bảng 2.10 và giới hạn thiết kế; Bảng 3.3–3.6; vì sao ResNet-18 lâu nhất.

**Quý – phải nói sâu được:** TP/FN/FP/TN và precision, recall, F1, macro F1 (tính tay từ Bảng 3.5); ý nghĩa FN/FP trong sản xuất; xác suất 0,23; Bảng 3.7 ngưỡng; Bảng 3.8–3.9, ảnh `cast_def_0_150`, nhóm `new__`; Hình 3.11 và vì sao pₒₖ chưa hiệu chuẩn; kiểm chứng sau nạp; **chạy demo**; toàn bộ hạn chế 3.5.4.

**Khi bị hỏi phần không phải mình:** trả lời trước bằng khung *làm gì → vì sao → số nào*, chỉ vào bảng/mục trong báo cáo, rồi mới mời bạn bổ sung. Không chắc thì nói "nhóm chưa kiểm chứng, đã ghi ở mục hạn chế 3.5.4". Không bịa số.

---

## 5. Câu hỏi hay gặp và cách trả lời

**Bài toán, dữ liệu**

1. *Vì sao phân loại cả ảnh mà không phát hiện vị trí lỗi?* – Dữ liệu chỉ có nhãn cấp ảnh, không có khung hay mặt nạ vị trí lỗi.
2. *Vì sao 7.348 ảnh mà chỉ dùng 7.284?* – Loại 64 ảnh train trùng pixel với test, không phải ảnh hỏng.
3. *Nếu không loại 64 ảnh thì sao?* – Gần một phần tư ảnh đạt của test là ảnh mô hình đã học → điểm lớp đạt cao giả tạo. Nhóm không đo mức chênh vì không chạy bản giữ ảnh trùng.
4. *Sao không chia lại test cho cân bằng?* – Giữ test của nguồn để kết quả đối chiếu được và nhóm không can thiệp vào "đề thi"; chỉ loại bản trùng ở phía train.
5. *85/15 là của cái gì?* – Của 6.569 ảnh train sạch, không phải của toàn bộ dữ liệu. Trên 7.284 ảnh, train ≈ 76,6%, validation ≈ 13,5%, test ≈ 9,8%.
6. *Stratify là gì?* – Chia sao cho tỷ lệ lỗi/đạt hai phần giống nhau (57,2%).
7. *Ảnh màu hay xám?* – Lưu 3 kênh nhưng R = G = B; chuyển 1 kênh, khi cần 3 kênh thì lặp lại.
8. *Mô hình có "ăn gian" bằng độ sáng không?* – Có thể. Ảnh đạt sáng hơn ~10,5 mức xám nhưng phân bố chồng lấn nên độ sáng một mình không tách được. Nhóm chưa tách riêng yếu tố này; hướng làm: chuẩn hóa độ sáng từng ảnh rồi huấn luyện lại, hoặc xem Grad-CAM.
9. *Có còn rò rỉ không?* – Có thể còn ảnh gần giống (biến thể tăng cường của cùng sản phẩm) vì SHA-256 chỉ bắt bản y hệt; nên 99,44% chỉ phản ánh bộ dữ liệu này. Hướng làm: băm cảm nhận (perceptual hash) hoặc so véc tơ đặc trưng.

**Mô hình**

10. *CNN có giống VGG không?* – Báo cáo không nói theo VGG. Chỉ nói: nhóm tự chọn 3 khối Conv 3 × 3 + MaxPool, số bộ lọc tăng gấp đôi khi kích thước giảm một nửa.
11. *Tính tham số Conv thứ hai?* – (3 × 3 × 32 + 1) × 64 = 18.496. Viết công thức trước, thay số sau.
12. *Vì sao GAP thay Flatten?* – 8.256 so với 8.388.672 tham số ở Dense 64 → ít quá khớp; đổi lại mất thông tin vị trí.
13. *Đóng băng thì backbone có chạy không?* – Có, vẫn chạy xuôi mỗi batch để tạo đặc trưng, chỉ không cập nhật trọng số. Vì vậy ResNet-18 vẫn huấn luyện lâu nhất.
14. *ResNet-18 có dùng bottleneck 1×1–3×3–1×1 không?* – Không, đó là ResNet-50 trở lên. ResNet-18 dùng **basic block** hai Conv 3 × 3.
15. *Vì sao phép chiếu cuối của MobileNetV2 không có ReLU?* – Biểu diễn bị nén về ít kênh, ReLU sẽ cắt mất thông tin âm.
16. *Ảnh xám vào mạng học trên ảnh màu ImageNet có ổn không?* – Lặp kênh để đúng đầu vào, không thêm thông tin; kết quả thực nghiệm vẫn tốt, nhưng thu nhỏ 224 có thể làm mờ lỗi rất nhỏ.

**Huấn luyện, chọn mô hình**

17. *Vì sao chọn theo validation loss mà không theo accuracy test?* – Chọn trên test là dùng "đề thi" để chọn, kết quả sẽ lạc quan. Loss mịn hơn accuracy, ít hòa.
18. *Vì sao validation loss của CNN thấp hơn train loss?* – Dropout chỉ bật khi train; train loss là trung bình khi trọng số đang thay đổi, validation đo bằng trọng số cuối epoch.
19. *Vì sao loss vọt lên mà accuracy giảm ít?* – BCE phạt rất nặng ảnh sai mà tự tin; vài ảnh như vậy đủ làm loss tăng mạnh.
20. *Vì sao 30 epoch, batch 32, lr 0,001, ảnh 256?* – Nhóm **không khảo sát**: 0,001 là mặc định đề xuất cho Adam; 256 nhỏ hơn 300 và chia đôi gọn 256 → 128 → 64 → 32. Tìm siêu tham số là hướng tiếp theo.
21. *Sao không dừng sớm?* – Để có đủ lịch sử 30 epoch; checkpoint tốt nhất thay vai trò điểm dừng.
22. *Chạy lại có ra y hệt không?* – Seed 42 cố định chia tập, khởi tạo, xáo trộn, nhưng GPU không bật tất định → có thể lệch nhỏ. Mỗi cấu hình chỉ chạy một lần.

**Kết quả**

23. *Đọc ma trận nhầm lẫn MobileNetV2?* – Hàng nhãn thật, cột dự đoán, thứ tự lỗi rồi đạt: 451 lỗi đúng, 2 lỗi bị cho đạt (FN), 2 đạt bị báo lỗi (FP), 260 đạt đúng.
24. *Tính precision và recall lớp lỗi của MobileNetV2?* – 451/(451+2) = 99,56%; 451/(451+2) = 99,56%.
25. *Vì sao macro F1 thấp hơn F1 lỗi?* – Lớp đạt ít ảnh hơn, mỗi ảnh sai chiếm tỷ lệ lớn hơn nên F1 đạt thấp hơn.
26. *MobileNetV2 chắc chắn tốt nhất?* – Hơn CNN thì rõ; hơn ResNet-18 thì chưa đủ khẳng định (xác suất ≈ 0,23, một seed).
27. *Muốn bỏ sót ít hơn thì làm gì?* – **Tăng** ngưỡng pₒₖ, chấp nhận báo nhầm nhiều hơn, và phải chọn ngưỡng trên validation.
28. *pₒₖ = 0,96 nghĩa là đúng 96%?* – Không, điểm chưa hiệu chuẩn; `cast_def_0_150` là ảnh lỗi bị cho 0,9556.
29. *Ảnh nào cả ba mô hình sai?* – Chỉ `cast_def_0_150.jpeg`.
30. *Kết quả có dùng được ở nhà máy không?* – Chưa. Cần đánh giá trên ảnh chụp tại nhà máy, điều kiện ánh sáng khác, đo độ trễ và kiểm định.

**Demo, mở rộng**

31. *Demo có xử lý ảnh giống lúc huấn luyện không?* – Có. Chuẩn hóa nằm trong mô hình, demo đối chiếu thông tin tiền xử lý và mã băm checkpoint trước khi nạp.
32. *Đưa ảnh không phải sản phẩm đúc thì sao?* – Mô hình vẫn trả lỗi hoặc đạt vì chỉ biết hai lớp; demo chưa có bước phát hiện ảnh lạ. Đây là hạn chế.
33. *Hướng phát triển?* – Fine-tune; nhiều seed; lọc ảnh gần giống; chọn ngưỡng theo chi phí trên validation; hiệu chuẩn; Grad-CAM; thu ảnh thật; đo độ trễ.
34. *Tài liệu tham khảo?* – 17 mục, bài báo dùng bản đã xuất bản (ResNet CVPR 2016, MobileNetV2 CVPR 2018, Adam ICLR 2015, Guo ICML 2017, ImageNet CVPR 2009) và tài liệu chính thức Keras, TensorFlow, scikit-learn, Streamlit.

---

## 6. Bẫy hay nói sai

| Dễ nói sai | Nói đúng |
|---|---|
| "Lớp Conv thứ hai" | **Tầng** là thành phần mạng; **lớp** là nhãn lỗi/đạt |
| "pₒₖ là xác suất lỗi" | pₒₖ là điểm lớp **đạt**; điểm lỗi = 1 − pₒₖ |
| "Lỗi là nhãn 1" | Lỗi là nhãn **0**, nhưng là lớp quan tâm |
| "FN là báo nhầm" | FN = **bỏ sót** lỗi; FP = **báo nhầm** ảnh đạt |
| "Nhóm có fine-tune / có augmentation" | **Không**: backbone đóng băng; không thêm augmentation |
| "Chọn MobileNetV2 vì test cao nhất" | Chọn theo **validation loss**, ghi lại **trước** khi test |
| "Loại 64 ảnh hỏng" | Loại 64 ảnh **trùng**; 0 ảnh hỏng |
| "SHA-256 loại hết rò rỉ" | Chỉ loại bản **y hệt** |
| "Muốn bớt bỏ sót thì hạ ngưỡng" | Phải **tăng** ngưỡng pₒₖ |
| "Nhóm chọn ngưỡng 0,96 theo Bảng 3.7" | Bảng 3.7 chỉ minh họa trên test; ngưỡng vẫn **0,5** |
| "Đóng băng nên huấn luyện nhanh" | Backbone vẫn chạy xuôi; ResNet-18 lâu nhất |

---

## 7. Demo và lật nhanh báo cáo

**Chạy demo (PowerShell tại thư mục gốc, lần đầu cài thư viện):**

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r .\demo\requirements.txt
.\.venv-demo\Scripts\python.exe .\demo\prepare_models.py
.\.venv-demo\Scripts\python.exe -m streamlit run .\demo\app.py
```

Mở `http://localhost:8501`, giữ cả ba mô hình, chọn hai ảnh rồi bấm **Phân loại tất cả**. Kết quả phải khớp Hình 3.12–3.13:

| Ảnh | CNN pₒₖ | ResNet-18 pₒₖ | MobileNetV2 pₒₖ | Nhãn |
|---|---:|---:|---:|---|
| `casting_full_3models_01/demo_test/ok_front/cast_ok_0_45.jpeg` | 0,848358 | 0,999593 | 0,998192 | Đạt |
| `casting_full_3models_01/demo_test/def_front/cast_def_0_1059.jpeg` | 0,000039 | 0,003175 | 0,000013 | Lỗi |

Nếu demo lỗi: mở PDF, chỉ vào Hình 3.12–3.13 (tr. 54–55).

**Số trang để lật (chân trang Ả Rập):**

| Cần chỉ | Trang | Cần chỉ | Trang |
|---|---:|---|---:|
| Hình 2.1 quy trình xử lý dữ liệu | 22 | Bảng 3.3, 3.4 chỉ số | 45 |
| Hình 2.3–2.5 EDA | 24–25 | Bảng 3.5, Hình 3.5–3.7 ma trận | 46 |
| Bảng 2.2 số ảnh qua các bước | 27 | Mục 3.4.3, xác suất ≈ 0,23 | 47 |
| Bảng 2.5 CNN baseline | 30 | Bảng 3.6 chi phí; mục 3.4.4 | 48 |
| Hình 2.11, Bảng 2.10 | 36–37 | Bảng 3.7 ngưỡng | 49 |
| Bảng 2.11 ma trận quy ước | 38 | Bảng 3.8 bốn ảnh sai | 51 |
| Bảng 3.1 phân bố dữ liệu | 40 | Bảng 3.9, Hình 3.11 | 52 |
| Bảng 3.2 checkpoint | 41 | Bảng 3.10 kiểm chứng | 53 |
| Mục 3.3.5, Hình 3.4 | 43–44 | Mục 3.5.4 hạn chế | 55 |

**Checklist trước buổi bảo vệ:** thuộc phần 1 · tự trả lời to 34 câu ở phần 5 · chạy thử demo trên đúng laptop mang đi · mở sẵn PDF · kiểm tra link Google Drive mã nguồn mở được ở cửa sổ ẩn danh · sạc đầy máy.
