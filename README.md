# Kiểm tra lỗi sản phẩm đúc bằng học sâu

**Nhóm 10 · Học phần Học sâu · Trường ĐH Công nghệ Thông tin và Truyền thông (ICTU)**
Thành viên: Nguyễn Quang Minh, Nông Hữu Thịnh, Vũ Ngọc Quý · GVHD: TS. Ngô Hữu Huy

> README này viết để **một thành viên bất kỳ đọc xong có thể bảo vệ cả dự án**, kể cả phần mình không trực tiếp làm.
> Báo cáo chính thức là **[bao_cao/baocaocodong_v4.docx](bao_cao/baocaocodong_v4.docx)** (bản PDF: [baocaocodong_v4.pdf](bao_cao/baocaocodong_v4.pdf)). Bản v3 và các bản cũ hơn chỉ để đối chiếu, **không dùng để ôn**.
> README dẫn theo **số mục, số hình, số bảng** của báo cáo (ví dụ "mục 2.4.5", "Bảng 3.3"), không dẫn số trang, vì câu chữ báo cáo còn có thể được sửa nhỏ.
> Mọi con số trong README lấy từ thư mục kết quả chính thức [`casting_full_3models_01/`](casting_full_3models_01/) và khớp với báo cáo v4.

## Mục lục

1. [Tóm tắt trong 5 dòng](#1-tóm-tắt-trong-5-dòng)
2. [Cấu trúc thư mục và cách chạy](#2-cấu-trúc-thư-mục-và-cách-chạy)
3. [Hiểu dự án trong 15 phút](#3-hiểu-dự-án-trong-15-phút)
4. [Bảng số liệu chính](#4-bảng-số-liệu-chính)
5. [Ôn bảo vệ: câu hỏi hội đồng hay hỏi](#5-ôn-bảo-vệ-câu-hỏi-hội-đồng-hay-hỏi)
6. [Ai trình bày phần nào](#6-ai-trình-bày-phần-nào)
7. [Bản đồ báo cáo v4 và phép tính tay](#7-bản-đồ-báo-cáo-v4-và-phép-tính-tay)
8. [Liên kết và checklist trước buổi bảo vệ](#8-liên-kết-và-checklist-trước-buổi-bảo-vệ)

---

## 1. Tóm tắt trong 5 dòng

<!-- Nguồn: data_check.json, split_counts.csv, comparison.csv, selection.json, metrics_mobilenetv2.json; báo cáo mục 3.2, Bảng 3.2–3.4 -->

1. **Bài toán:** nhận ảnh mặt trước một sản phẩm đúc (cánh bơm), phân loại cả ảnh thành **lỗi** (nhãn 0) hoặc **đạt** (nhãn 1).
2. **Dữ liệu:** bộ *Casting Product Image Data for Quality Inspection* (Kaggle), 7.348 ảnh. Loại 64 ảnh train trùng pixel với test, còn 7.284 ảnh, chia **5.583 train / 986 validation / 715 test**.
3. **Ba mô hình:** CNN baseline tự xây, học từ đầu; **ResNet-18** và **MobileNetV2** học chuyển giao từ ImageNet, đóng băng backbone, chỉ học đầu phân loại mới.
4. **Kết quả trên 715 ảnh test:** CNN **94,83%** · ResNet-18 **98,74%** · MobileNetV2 **99,44%** (đúng 711/715, bỏ sót 2 ảnh lỗi, báo nhầm 2 ảnh đạt).
5. **Mô hình được chọn:** **MobileNetV2 ở epoch 21**, vì có validation loss thấp nhất (0,0150). Quyết định này được ghi vào `selection.json` **trước** khi chạy test (mục 2.4.5). Demo Streamlit cho chạy riêng hoặc so sánh cả ba mô hình.

**Câu tóm tắt 30 giây (cả nhóm nói giống nhau):**

> Nhóm phân loại ảnh sản phẩm đúc thành lỗi hoặc đạt. Sau khi loại 64 ảnh trùng giữa train và test, nhóm có 5.583 ảnh train, 986 ảnh validation và 715 ảnh test. Nhóm dùng một CNN học từ đầu làm mốc, so với ResNet-18 và MobileNetV2 dùng trọng số ImageNet, đóng băng backbone và chỉ học đầu phân loại. Mô hình được chọn theo validation loss là MobileNetV2 ở epoch 21. Trên 715 ảnh test, mô hình đúng 711 ảnh, bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.

---

## 2. Cấu trúc thư mục và cách chạy

### 2.1. Những gì quan trọng trong thư mục

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

### 2.2. Môi trường

| Nơi chạy | Phiên bản đã dùng | Nguồn |
|---|---|---|
| Huấn luyện (Google Colab, GPU Tesla T4) | Python 3.13.16, TensorFlow 2.20.0, Keras 3.12.0, KerasHub 0.23.0, NumPy 2.2.6 | `environment.json`, mục 3.1 |
| Demo trên Windows (CPU) | Python 3.12.14, TensorFlow 2.20.0, Keras 3.12.0, KerasHub 0.23.0, NumPy 2.2.6, Pillow 11.3.0, Streamlit 1.67.0 | `.venv` của dự án, `demo/requirements.txt` |

<!-- Phiên bản demo kiểm bằng .venv/Scripts/python.exe ngày 24/11/2026 -->

### 2.3. Chạy lại notebook trên Google Colab

Muốn **xem** kết quả thì chỉ cần mở [Casting.ipynb](Casting.ipynb): notebook đã lưu output. Muốn **chạy lại**:

1. Nén thư mục `data/casting_data` thành ZIP. Trong ZIP phải có `train/def_front`, `train/ok_front`, `test/def_front`, `test/ok_front`.
2. Upload `Casting.ipynb` lên Colab, chọn GPU. Chạy mục 1 (cài thư viện), **khởi động lại session**, rồi chạy từ mục 2.
3. Ở mục 3, upload ZIP dữ liệu. Ở mục 4, kiểm tra output báo **64 ảnh trùng bị loại** và ba số **5.583 / 986 / 715**.
4. Chạy mục 5–8 để huấn luyện CNN, ResNet-18, MobileNetV2. Mỗi mô hình lưu checkpoint `.keras`, lịch sử và gói sao lưu. Nếu chạy thêm lượt mới, đặt `RUN_NAME` khác để không ghi đè.
5. Chạy mục 9 để chọn mô hình và đánh giá test. Mục 10 tải kết quả. Mục 11–13 chuẩn bị mô hình demo, kiểm chứng trên CPU, thử một ảnh và tải gói `demo_bundle.zip`, `complete_*.zip`.

Chi tiết từng bước: [HUONG_DAN_CASTING.md](HUONG_DAN_CASTING.md).

Lưu ý: notebook không bật chế độ tính toán tất định trên GPU, nên chạy lại có thể ra số lệch nhỏ dù cùng seed 42. Số trong báo cáo là của lượt `casting_full_3models_01`.

### 2.4. Chạy demo Streamlit trên máy Windows

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

Kết quả đúng phải như Hình 3.10 và 3.11 của báo cáo:

<!-- Nguồn: báo cáo mục 3.5.3; demo/reference/demo_cast_def_0_1059.txt -->

| Ảnh | Mô hình | Nhãn | Điểm đạt p_ok | Điểm lỗi = 1 − p_ok |
|---|---|---|---:|---:|
| `cast_ok_0_45.jpeg` (Hình 3.10) | CNN | Đạt | 0,848358 | 0,151642 |
| | ResNet-18 | Đạt | 0,999593 | 0,000407 |
| | MobileNetV2 | Đạt | 0,998192 | 0,001808 |
| `cast_def_0_1059.jpeg` (Hình 3.11) | CNN | Lỗi | 0,000039 | 0,999961 |
| | ResNet-18 | Lỗi | 0,003175 | 0,996825 |
| | MobileNetV2 | Lỗi | 0,000013 | 0,999987 |

Nếu máy demo hỏng khi bảo vệ: mở Hình 3.10 và 3.11 trong báo cáo, hoặc ảnh chụp giao diện `demo/reference/demo_cast_ok_0_45.png` và `demo/reference/demo_cast_def_0_1059.png`.

### 2.5. Lỗi thường gặp

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

## 3. Hiểu dự án trong 15 phút

### 3.1. Bài toán

- **Đầu vào:** một ảnh mặt trước sản phẩm đúc.
- **Đầu ra:** một số `p_ok` trong khoảng 0–1, là điểm của lớp **đạt**. Điểm lỗi = `1 − p_ok`.
- **Quy tắc ra nhãn:** `p_ok ≥ 0,5` → **đạt**; `p_ok < 0,5` → **lỗi**. Ngưỡng 0,5 nghĩa là chọn lớp có điểm lớn hơn. Bằng nhau thì gán đạt (mục 2.1).
- **Nhãn:** 0 = `def_front` = lỗi; 1 = `ok_front` = đạt (Bảng 2.1).
- **Phạm vi:** phân loại **cả ảnh**. Dữ liệu chỉ có nhãn cấp ảnh, nên mô hình **không chỉ ra vị trí lỗi**.

Công thức (2.1) trong báo cáo viết gọn: `p = f(P(x); θ)`. Trong đó `x` là ảnh đã tiền xử lý, `P` là bước riêng của từng mô hình, `f` là mạng, `θ` là phần tham số được học.

### 3.2. Dữ liệu: 7.348 ảnh → 7.284 ảnh dùng được

<!-- Nguồn: data_check.json (original_counts, dimensions, bad_images, excluded 64), split_counts.csv; báo cáo mục 2.2, 3.2, Bảng 3.1 -->

- Nguồn: thư mục `casting_data`, **7.348 ảnh JPEG 300 × 300**. Thư mục train có 6.633 ảnh (3.758 lỗi, 2.875 đạt), thư mục test có 715 ảnh (453 lỗi, 262 đạt).
- Ảnh lưu ba kênh nhưng ba kênh bằng nhau (nội dung xám). Khi đọc, nhóm chuyển về ảnh xám một kênh.
- Bộ nguồn **đã có sẵn ảnh tăng cường** (xoay, đổi sáng...). Nhóm **không thêm** augmentation (`pipeline_augmentation = false` trong `config.json`).
- Kiểm tra toàn bộ 7.348 ảnh: **không có ảnh hỏng**.

**Kiểm trùng bằng SHA-256.** SHA-256 là hàm băm: hai ảnh giống hệt từng pixel cho cùng một mã. Nhóm băm *kích thước + pixel của ảnh xám, trước khi resize* (mục 2.2.2).

- **Vì sao phải kiểm trùng?** Nếu một ảnh vừa nằm trong train vừa nằm trong test, mô hình "đã thấy đáp án". Accuracy test sẽ cao giả tạo. Đây là rò rỉ dữ liệu (data leakage).
- **Vì sao so pixel, không so tên tệp?** Ảnh cùng tên ở train và test có thể khác nội dung, và ảnh khác tên có thể giống hệt nhau.
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

### 3.3. Tiền xử lý cho từng mô hình

<!-- Nguồn: báo cáo mục 2.2.4, Bảng 2.3; doi_chieu_so_lieu.md mục 2 -->

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

### 3.4. CNN baseline: từng tầng và số tham số

<!-- Nguồn: Bảng 2.4, model_summary_cnn.txt, fit_cnn.json (100993/100993) -->

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

### 3.5. Học chuyển giao: ResNet-18 và MobileNetV2

<!-- Nguồn: mục 2.3.1, 2.3.3, 2.3.4, Bảng 2.5–2.7; fit_resnet18.json, fit_mobilenetv2.json -->

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
| Ý tưởng chính | Khối dư (residual) có đường nối tắt: đầu ra = ReLU(F(x) + x) (Hình 2.3) | Khối inverted residual: mở rộng kênh → depthwise 3 × 3 → nén tuyến tính (Hình 2.5) |
| Đặc trưng trước GAP | 7 × 7 × **512** | 7 × 7 × **1.280** |
| Tổng tham số | 11.219.009 | 2.340.033 |
| Tham số được học | **32.897** = (512 + 1) × 64 + 65 | **82.049** = (1.280 + 1) × 64 + 65 |

**ResNet-18 (basic block):** hai Conv 3 × 3 giữ nguyên số kênh (ví dụ 64 → 64), kèm Batch Normalization, tạo `F(x)`; cộng với đầu vào `x` qua đường nối tắt rồi ReLU. Đường nối tắt giúp gradient đi thẳng về các tầng trước, giảm hiện tượng gradient biến mất. Khi đổi số kênh hoặc độ phân giải, đường tắt dùng phép chiếu (Conv 1 × 1). Bốn nhóm, mỗi nhóm 2 basic block, số kênh 64/128/256/512 (Bảng 2.5).

**MobileNetV2 (inverted residual, ví dụ block_7):** Conv 1 × 1 mở rộng 64 → 384 kênh → depthwise 3 × 3 lọc riêng từng kênh → Conv 1 × 1 chiếu về 64 kênh, **không ReLU** (linear bottleneck) → cộng với đầu vào. Gọi là "ngược" vì mở rộng ở giữa, hẹp ở hai đầu. Tích chập tách biệt (depthwise + 1 × 1) cần ít phép tính hơn Conv 3 × 3 thường khoảng 8–9 lần. Hàm kích hoạt là ReLU6 (ReLU chặn trên ở 6).

**Vì sao MobileNetV2 ít tham số tổng hơn mà tham số học lại nhiều hơn ResNet-18?** Vì đặc trưng trước GAP của MobileNetV2 có 1.280 kênh, của ResNet-18 chỉ 512. Dense 64 của MobileNetV2 nhận nhiều đầu vào hơn.

**Vì sao chọn hai mạng này?** ResNet-18 là mạng nhỏ nhất họ ResNet, đại diện kiến trúc khối dư. MobileNetV2 là mạng nhẹ, hợp với demo chạy CPU. Cả hai có sẵn trọng số ImageNet trong hệ Keras (mục 2.3.1).

### 3.6. Huấn luyện

<!-- Nguồn: Bảng 2.8, mục 2.4.2–2.4.3, config.json -->

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

### 3.7. Diễn biến huấn luyện và checkpoint được chọn

<!-- Nguồn: history_*.csv (epoch trong CSV đếm từ 0, ở đây +1), fit_*.json; Bảng 3.2, mục 3.3.2–3.3.4 -->

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

### 3.8. Đánh giá: lớp lỗi là lớp quan tâm

<!-- Nguồn: mục 2.4.4, Bảng 2.9, công thức (2.5), (2.6) -->

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

### 3.9. Kết quả trên tập test và cách đọc ma trận

<!-- Nguồn: comparison.csv, metrics_*.json; Bảng 3.3, 3.4; Hình 3.4–3.6 -->

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

**Phạm vi kết luận:** hai mô hình học chuyển giao khác CNN đồng thời ở kiến trúc, độ phân giải đầu vào và trọng số ImageNet. Kết quả cho biết hiệu quả của **cả cấu hình**, chưa tách được đóng góp riêng của từng yếu tố (mục 3.4.3).

### 3.10. Ảnh dự đoán sai

<!-- Nguồn: errors_*.csv; Bảng 3.5, Hình 3.7–3.9, mục 3.5.1 -->

Số ảnh sai: CNN 37, ResNet-18 9, MobileNetV2 4. Bốn ảnh MobileNetV2 sai (Bảng 3.5):

| Ảnh | Nhãn thật | p_ok | Dự đoán |
|---|---|---:|---|
| `cast_def_0_150.jpeg` | Lỗi | 0,955552 | Đạt (bỏ sót, điểm đạt cao) |
| `cast_def_0_2102.jpeg` | Lỗi | 0,502379 | Đạt (bỏ sót, sát ngưỡng 0,5) |
| `cast_ok_0_16.jpeg` | Đạt | 0,228884 | Lỗi (báo nhầm) |
| `cast_ok_0_7839.jpeg` | Đạt | 0,244992 | Lỗi (báo nhầm) |

- `cast_def_0_150.jpeg` bị **cả ba mô hình** đoán là đạt (p_ok 0,708656 ở CNN, 0,979930 ở ResNet-18, 0,955552 ở MobileNetV2). Đây là ảnh test duy nhất cả ba cùng sai. Nhìn bằng mắt, nhóm không thấy vết lỗi rõ ở mép và bề mặt như các ảnh lỗi khác (mục 3.5.1).
- **Nhóm chưa biết nguyên nhân.** Dữ liệu không có chú giải vị trí lỗi; muốn tìm hiểu cần công cụ giải thích như Grad-CAM, nhóm **chưa làm**. Không tự bịa nguyên nhân.
- Điểm sigmoid **chưa hiệu chuẩn**: `p_ok = 0,96` không có nghĩa "đúng 96%". Một ảnh sai vẫn có thể được cho điểm cao.

### 3.11. Lưu mô hình, kiểm chứng và demo

<!-- Nguồn: Bảng 3.6, casting_full_3models_01/demo/verification.json, demo/models/comparison_verification.json -->

- Checkpoint lưu dạng `.keras` (gồm kiến trúc và trọng số), kèm metadata: quy ước nhãn, kích thước đầu vào, chuỗi tiền xử lý, mã băm.
- **Kiểm chứng sau nạp trên CPU** (Bảng 3.6): nạp lại mô hình, so với dự đoán đã lưu từ Colab.

| Mô hình | Tập đối chiếu | Nhãn trùng | Sai lệch điểm lớn nhất |
|---|---|---:|---:|
| MobileNetV2 | 10 ảnh validation | 10/10 | 4,2142 × 10⁻⁸ |
| CNN | 10 ảnh test (5 lỗi, 5 đạt) | 10/10 | 8,2600 × 10⁻⁸ |
| ResNet-18 | 10 ảnh test | 10/10 | 3,1825 × 10⁻⁶ |
| MobileNetV2 | 10 ảnh test | 10/10 | 6,2771 × 10⁻⁷ |

  Dung sai: tương đối 10⁻³, tuyệt đối 10⁻⁴. Phép này chỉ chứng minh **tệp nạp lại cho cùng kết quả**, không phải đo độ chính xác. Độ chính xác là 715 ảnh test ở mục 3.4.
- Trên Windows, lớp `ResNetImageConverter` của ResNet-18 được dựng lại bằng các tầng Keras (resize, nhân scale, cộng offset) rồi nạp trọng số gốc (`demo/inference.py`).
- **Demo** (mục 3.5.3): chọn một hoặc cả ba mô hình, nhận nhiều ảnh, hiện nhãn và điểm 6 chữ số thập phân, xuất CSV/JSON. Tệp ảnh hỏng được báo lỗi riêng, các ảnh hợp lệ vẫn chạy.
- Điểm của **một ảnh** (ví dụ 0,998192) khác với **accuracy toàn tập** (99,44%).

### 3.12. Hạn chế (mục 3.5.4)

1. **Một seed, một cách chia, test cùng nguồn.** Chưa đo độ dao động khi đổi seed.
2. **Rò rỉ gần giống chưa loại hết.** Nguồn có sẵn ảnh tăng cường và không có mã sản phẩm gốc. SHA-256 chỉ loại 64 bản y hệt; biến thể xoay hoặc đổi sáng của cùng ảnh có thể nằm ở cả train và test. Vì vậy **99,44% có thể lạc quan** so với ảnh chụp mới tại nhà máy. Ba mô hình chịu cùng điều kiện, nên **phép so sánh giữa chúng vẫn công bằng**.
3. **Backbone đóng băng**, chưa fine-tuning; kiến trúc CNN, độ phân giải, siêu tham số chưa được tìm kiếm có hệ thống.
4. **Thời gian** chỉ là thời gian huấn luyện một lượt trên T4. Chưa đo độ trễ từng ảnh, bộ nhớ.
5. **Điểm sigmoid chưa hiệu chuẩn**; nhãn chỉ ở cấp toàn ảnh, không định vị lỗi.
6. Demo là **bản minh họa**, cần kiểm định thêm trước khi dùng trong sản xuất.
7. `history_*.csv` không ghi learning rate, nên không biết ReduceLROnPlateau giảm LR ở epoch nào.

**Hướng phát triển** (phần Kết luận): thêm ảnh độc lập, chia theo nhóm sản phẩm khi có thông tin, chạy nhiều seed, thử fine-tuning, chọn ngưỡng theo chi phí bỏ sót/báo nhầm trên validation, đo độ trễ và hiệu chuẩn điểm.

---

## 4. Bảng số liệu chính

<!-- Nguồn: comparison.csv, fit_*.json, metrics_*.json; khớp Bảng 2.7, 3.2, 3.3, 3.4 của báo cáo v4 -->

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

Giá trị validation loss đầy đủ (trong `fit_*.json`): 0,121010 / 0,026946 / 0,015027.

---

## 5. Ôn bảo vệ: câu hỏi hội đồng hay hỏi

Trả lời **ngắn, có số, chỉ đúng mục/bảng**. Không biết thì nói thật.

### A. Bài toán và dữ liệu

**1. Bài toán của nhóm là gì? Đầu ra là gì?**
Phân loại cả ảnh sản phẩm đúc thành lỗi hoặc đạt. Mô hình cho `p_ok` (điểm lớp đạt); `p_ok ≥ 0,5` là đạt. Không định vị vết lỗi vì dữ liệu chỉ có nhãn cấp ảnh (mục 2.1).

**2. Dữ liệu bao nhiêu ảnh, chia thế nào?**
7.348 ảnh nguồn. Loại 64 ảnh train trùng test, còn 7.284. Chia 5.583 train, 986 validation, 715 test (Bảng 3.1). Validation là 15% của 6.569 ảnh train sạch, chia phân tầng, seed 42.

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
Nhóm tự chọn 3 khối Conv (32/64/128 bộ lọc) + MaxPooling, rồi GAP, Dense 64, Dropout 0,3, Dense 1 sigmoid, và học toàn bộ 100.993 tham số từ đầu. Các tầng đều là tầng Keras phổ biến; nhóm không nhận là kiến trúc mới (mục 2.3.2, Bảng 2.4).

**8. Tính giúp số tham số tầng Conv đầu và Dense 64.**
Conv đầu: (3 × 3 × 1 + 1) × 32 = 320. Dense 64: (128 + 1) × 64 = 8.256. Số "+1" là bias.

**9. Vì sao dùng Global Average Pooling mà không Flatten?**
Flatten cho 131.072 đầu vào, Dense 64 cần 8.388.672 tham số. GAP chỉ còn 128 đầu vào, cần 8.256 tham số. Ít tham số hơn nhiều, giảm rủi ro quá khớp; đổi lại mất thông tin vị trí (mục 2.3.2).

**10. Vì sao đầu ra 1 nơ-ron sigmoid mà không softmax 2 nơ-ron?**
Bài toán hai lớp. Một sigmoid cho `p_ok`, điểm lỗi là `1 − p_ok`. Cách này đi cùng binary cross-entropy và ít tham số hơn.

**11. Học chuyển giao, đóng băng backbone là gì?**
Lấy phần trích đặc trưng đã học trên ImageNet, giữ nguyên trọng số (`trainable=False`, gọi với `training=False`), chỉ học đầu phân loại mới. ResNet-18 học 32.897 tham số, MobileNetV2 học 82.049 (Bảng 2.7). Nhóm **không fine-tune**.

**12. Vì sao MobileNetV2 nhỏ hơn ResNet-18 mà tham số học nhiều hơn?**
Đặc trưng trước GAP của MobileNetV2 có 1.280 kênh, ResNet-18 có 512. Dense 64: (1.280 + 1) × 64 + 65 = 82.049 so với (512 + 1) × 64 + 65 = 32.897.

**13. Inverted residual của MobileNetV2 khác basic block của ResNet-18 ra sao?**
Basic block của ResNet-18: hai Conv 3 × 3 giữ nguyên số kênh, cộng nối tắt rồi ReLU. Inverted residual: Conv 1 × 1 mở rộng (64 → 384), depthwise 3 × 3, Conv 1 × 1 nén tuyến tính về 64 rồi cộng nối tắt (mỏng → dày → mỏng). Kiểu "dày → mỏng → dày" là bottleneck của ResNet-50 trở lên, **không phải** ResNet-18 (Hình 2.3, 2.5).

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
Accuracy test: CNN 94,83%, ResNet-18 98,74%, MobileNetV2 99,44%. MobileNetV2 đúng 711/715, bỏ sót 2 ảnh lỗi, báo nhầm 2 ảnh đạt (Bảng 3.3, 3.4).

**22. Đọc giúp ma trận nhầm lẫn của MobileNetV2.**
Hàng là nhãn thật, cột là dự đoán, thứ tự lỗi rồi đạt: `[[451, 2], [2, 260]]`. Ô hàng lỗi, cột đạt là 2 ảnh lỗi bị bỏ sót (FN). Ô hàng đạt, cột lỗi là 2 ảnh đạt bị báo nhầm (FP) (Hình 3.6).

**23. FN và FP, cái nào nguy hiểm hơn? Muốn giảm bỏ sót thì chỉnh ngưỡng thế nào?**
FN nguy hiểm hơn vì sản phẩm lỗi bị cho qua. Quy tắc là `p_ok ≥ ngưỡng` thì đạt, nên muốn bớt bỏ sót phải **tăng** ngưỡng; đổi lại FP tăng. **Không nói "hạ ngưỡng"**. Ngưỡng mới phải chọn trên validation (xem bảng minh họa ở mục 7.3).

**24. 99,44% có cao bất thường không? Có rò rỉ dữ liệu không?**
Chủ động thừa nhận giới hạn: nguồn có sẵn ảnh tăng cường, không có mã sản phẩm gốc; SHA-256 chỉ loại 64 bản y hệt. Biến thể gần giống có thể còn ở cả train và test, nên 99,44% có thể lạc quan với ảnh mới (mục 3.5.4). Ba mô hình chịu cùng điều kiện nên so sánh giữa chúng vẫn công bằng.

**25. Có thể kết luận MobileNetV2 tốt nhất không? Học chuyển giao tốt hơn CNN không?**
Chỉ trong thực nghiệm này: ba cấu hình, một seed, test cùng nguồn. Hai mô hình học chuyển giao khác CNN cùng lúc về kiến trúc, độ phân giải và trọng số ImageNet, nên chưa tách được yếu tố nào tạo ra chênh lệch (mục 3.4.3).

**26. Vì sao ảnh `cast_def_0_150.jpeg` bị đoán là đạt với p_ok 0,9556?**
Nhóm chưa xác định được. Cả ba mô hình đều đoán đạt (0,7087 / 0,9799 / 0,9556), đây là ảnh test duy nhất cả ba cùng sai; nhìn bằng mắt không thấy vết lỗi rõ. Dữ liệu không có chú giải vị trí lỗi, nhóm chưa dùng Grad-CAM (mục 3.5.1).

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
Trả lời theo bảng phân công ở mục 6. Phần cả nhóm tự làm: kiểm trùng SHA-256 và chia tập, chọn cấu hình CNN, gắn đầu phân loại và huấn luyện ba mô hình trong cùng điều kiện, chọn checkpoint trước khi test, đánh giá theo lớp lỗi, phân tích ảnh sai, kiểm chứng sau nạp và demo so sánh ba mô hình. Dữ liệu (Kaggle) và kiến trúc ResNet-18, MobileNetV2 (Keras) có nguồn bên ngoài.

**Không nhận những việc chưa làm:** fine-tuning, Grad-CAM, nhiều seed, đo độ trễ từng ảnh, hiệu chuẩn xác suất, chọn ngưỡng trên validation, định vị lỗi.
**Tránh câu tuyệt đối:** "chắc chắn đúng", "không hề quá khớp", "dùng ngay trong nhà máy được".

Đọc thêm câu hỏi và cách giải thích chi tiết: [CAU_HOI_VA_GIAI_THICH_BAO_VE.md](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) · [HUONG_DAN_BAO_VE.md](HUONG_DAN_BAO_VE.md).

---

## 6. Ai trình bày phần nào

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
2. **Chỉ vào mục/bảng.** Ví dụ: "Dạ, MobileNetV2 bỏ sót 2 ảnh lỗi, ở Bảng 3.4 ạ." Mở báo cáo đúng chỗ khi cần.
3. **Dùng khung ba ý:** làm gì → vì sao → kết quả là số nào. Ví dụ: "Nhóm loại 64 ảnh trùng → để test không bị rò rỉ → còn 5.583/986/715."
4. **Không chắc thì nói phạm vi:** "Phần này nhóm chưa kiểm chứng, đã ghi ở mục hạn chế 3.5.4." Không đoán, không bịa.
5. **Không nói lệch nhau:** cả nhóm dùng cùng câu tóm tắt 30 giây ở mục 1 và cùng bảng số ở mục 4.

---

## 7. Bản đồ báo cáo v4 và phép tính tay

### 7.1. Báo cáo gồm những gì

| Phần | Nội dung | Hình, bảng, công thức |
|---|---|---|
| Phân công nhiệm vụ | 8 nhiệm vụ, phân theo tiến độ và theo thành viên | 2 bảng phân công |
| Mở đầu | Lý do, mục tiêu, phạm vi, bố cục 3 chương | |
| **Chương 1. Cơ sở lý thuyết** | 1.1 học máy, học sâu · 1.2 toán nền tảng (tensor, xác suất, entropy, gradient) · 1.3 học từ dữ liệu · 1.4 hàm mục tiêu, tối ưu, khái quát hóa · 1.5 mạng nơ-ron và CNN (1.5.3 tích chập và tầng gộp, 1.5.4 cấu trúc CNN và học chuyển giao) · 1.6 kết luận | Hình 1.1–1.13, Bảng 1.1–1.2, công thức (1.1)–(1.11); (1.11) là công thức kích thước đầu ra Conv |
| **Chương 2. Phương pháp và mô hình** | 2.1 phát biểu bài toán · 2.2 dữ liệu (2.2.2 kiểm trùng, 2.2.3 chia tập, 2.2.4 chuẩn hóa) · 2.3 ba mô hình (2.3.2 CNN, 2.3.3 ResNet-18, 2.3.4 MobileNetV2) · 2.4 huấn luyện (2.4.3 checkpoint, 2.4.4 tiêu chí đánh giá, 2.4.5 chọn mô hình trước test) · 2.5 kết luận | Bảng 2.1 nhãn · 2.2 vai trò ba tập · 2.3 đầu vào/chuẩn hóa · 2.4 CNN · 2.5 ResNet-18 · 2.6 MobileNetV2 · 2.7 ba cấu hình · 2.8 thiết lập huấn luyện · 2.9 ma trận nhầm lẫn. Hình 2.1 ảnh mẫu, 2.2 CNN, 2.3 basic block, 2.4 ResNet-18, 2.5 inverted residual, 2.6 MobileNetV2. Công thức (2.1) mô hình, (2.2)–(2.3) bài toán tối ưu và BCE, (2.4) sigmoid, (2.5) accuracy và precision, (2.6) recall và F1 |
| **Chương 3. Thực nghiệm và đánh giá kết quả** | 3.1 môi trường · 3.2 kiểm tra và chia dữ liệu · 3.3 kết quả huấn luyện (3.3.1 checkpoint, 3.3.2–3.3.4 từng mô hình) · 3.4 đánh giá trên test (3.4.1 chỉ số, 3.4.2 ma trận, 3.4.3 so sánh) · 3.5 phân tích và minh họa (3.5.1 ảnh đúng/sai, 3.5.2 lưu và kiểm tra mô hình, 3.5.3 demo, 3.5.4 hạn chế) · 3.6 kết luận | Bảng 3.1 phân bố dữ liệu · 3.2 checkpoint và thời gian · **3.3 accuracy/precision/recall/F1** · **3.4 TP/FN/FP/TN** · 3.5 bốn ảnh MobileNetV2 sai · 3.6 kiểm chứng sau nạp. Hình 3.1–3.3 đường học · 3.4–3.6 ma trận nhầm lẫn · 3.7–3.9 ảnh đúng/sai · 3.10–3.11 demo |
| Kết luận, Tài liệu tham khảo (15 tài liệu), Phụ lục (link Google Drive mã nguồn) | | |

### 7.2. Phép tính tay có thể được yêu cầu viết lên bảng

| Nội dung | Công thức | Ví dụ trong đề tài |
|---|---|---|
| Kích thước đầu ra Conv | o = ⌊(n + 2p − k) / s⌋ + 1 (1.11) | Ảnh 5 × 5, bộ lọc 3 × 3, p = 0, s = 1 → 3 × 3 (Hình 1.11). Padding same, s = 1 → giữ 256 × 256 |
| MaxPooling 2 × 2, stride 2 | chia đôi mỗi chiều | 256 → 128 → 64 → 32 |
| Tham số Conv | (k × k × C_vào + 1) × C_ra | 320; 18.496; 73.856 |
| Tham số Dense | (số đầu vào + 1) × số nơ-ron | 8.256; 65. Tổng CNN = 100.993 |
| Đầu phân loại học chuyển giao | (số đặc trưng + 1) × 64 + 65 | ResNet-18: 32.897; MobileNetV2: 82.049 |
| Sigmoid | σ(z) = 1 / (1 + e^(−z)); σ′ = σ(1 − σ) | σ(0) = 0,5; đạo hàm lớn nhất 0,25 tại z = 0 |
| BCE một ảnh | L = −[y·ln p + (1 − y)·ln(1 − p)] | Ảnh đạt: p = 0,9 → 0,105; p = 0,1 → 2,303 |
| Accuracy, precision, recall | (TP+TN)/tổng; TP/(TP+FP); TP/(TP+FN) | MobileNetV2: 711/715 = 99,44%; 451/453 = 99,56% |
| Một ảnh test | 1 / 715 | ≈ 0,14 điểm phần trăm |
| Chia tập | 6.633 − 64 = 6.569; 15% | 986 validation, 5.583 train |
| Số bước mỗi epoch | ⌈5.583 / 32⌉ | 175 |
| Tỷ lệ lỗi trong validation | 564 / 986 | 57,20% (bằng val acc của CNN 3 epoch đầu) |

### 7.3. Minh họa đổi ngưỡng (chỉ để thấy xu hướng)

<!-- Tính lại từ test_predictions_mobilenetv2.csv ngày 24/11/2026. Không có trong báo cáo; báo cáo chỉ nói ngưỡng có thể chọn lại trên validation (mục 3.4.3). -->

Số liệu dưới đây tính từ dự đoán test của MobileNetV2, **không nằm trong báo cáo**. Chỉ dùng khi được hỏi về đánh đổi FN/FP. Khi làm thật, ngưỡng phải chọn trên validation.

| Ngưỡng p_ok | FN (bỏ sót) | FP (báo nhầm) |
|---:|---:|---:|
| 0,50 (đang dùng) | 2 | 2 |
| 0,60 | 1 | 2 |
| 0,80 | 1 | 3 |
| 0,90 | 1 | 6 |
| 0,96 | 0 | 9 |

Ảnh `cast_def_0_150.jpeg` có p_ok 0,9556, nên phải đẩy ngưỡng lên khoảng 0,96 mới bắt được, trong khi báo nhầm tăng từ 2 lên 9.

### 7.4. Thuật ngữ, mỗi người nói được trong một câu

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

## 8. Liên kết và checklist trước buổi bảo vệ

### 8.1. Liên kết

| Tài liệu | Dùng khi |
|---|---|
| [Báo cáo v4 (.docx)](bao_cao/baocaocodong_v4.docx) · [PDF](bao_cao/baocaocodong_v4.pdf) | Bản nộp và bản dùng khi bảo vệ |
| [HUONG_DAN_BAO_VE.md](HUONG_DAN_BAO_VE.md) | Đọc thêm: giải thích từng phần Chương 2–3 và trình tự nói |
| [CAU_HOI_VA_GIAI_THICH_BAO_VE.md](CAU_HOI_VA_GIAI_THICH_BAO_VE.md) | Đọc thêm: câu hỏi theo chủ đề, cách đọc biểu đồ và ma trận |
| [Casting.ipynb](Casting.ipynb) | Xem mã và output lượt chính thức |
| [HUONG_DAN_CASTING.md](HUONG_DAN_CASTING.md) | Chạy lại notebook trên Colab |
| [demo/README.md](demo/README.md) · [demo/app.py](demo/app.py) | Chạy và giải thích demo |
| [comparison.csv](casting_full_3models_01/comparison.csv) · [selection.json](casting_full_3models_01/selection.json) · [split_counts.csv](casting_full_3models_01/split_counts.csv) | Đối chiếu số liệu khi bị hỏi |
| [Lịch sử CNN](casting_full_3models_01/history_cnn.csv) · [ResNet-18](casting_full_3models_01/history_resnet18.csv) · [MobileNetV2](casting_full_3models_01/history_mobilenetv2.csv) | Xem từng epoch (cột `epoch` đếm từ 0, báo cáo đếm từ 1) |
| [Kiểm chứng MobileNetV2](casting_full_3models_01/demo/verification.json) · [Kiểm chứng ba mô hình](demo/models/comparison_verification.json) | Bảng 3.6 (tệp thứ hai nằm trong `demo/models/`, chỉ có sau khi chạy `prepare_models.py`) |
| [Thư mục Google Drive của nhóm](https://drive.google.com/drive/folders/1otbkaT-wvJfNERf_CL0uapmz2M2ebSnR?usp=sharing) | Mã nguồn nộp kèm (Phụ lục) |
| [Nguồn dữ liệu Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Tài liệu [12] |
| [prompt/](prompt/) | Prompt và meta-prompt của dự án (quy ước trong AGENTS.md) |

### 8.2. Checklist trước buổi bảo vệ

- [ ] Mỗi người đọc mục 1, 3, 4 của README và tự trả lời to 31 câu ở mục 5.
- [ ] Thuộc câu tóm tắt 30 giây và năm con số: 5.583/986/715 · 94,83/98,74/99,44% · MobileNetV2 epoch 21 · 711/715, FN 2, FP 2 · 100.993 / 32.897 / 82.049 tham số được học.
- [ ] Chạy thử demo trên laptop sẽ mang đi (lần đầu nạp mô hình lâu); để sẵn `cast_ok_0_45.jpeg` và `cast_def_0_1059.jpeg`.
- [ ] Mở sẵn PDF báo cáo v4; nếu demo lỗi thì chỉ vào Hình 3.10 và 3.11.
- [ ] Mở link Google Drive ở cửa sổ ẩn danh để chắc người khác xem được.
- [ ] Sạc đầy laptop, có mặt sớm.
