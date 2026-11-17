# Bàn giao dự án Học sâu Nhóm 10 ICTU

Bạn tiếp nhận dự án tại `D:\Casting-Defect-Detection-CNN`. Hãy dùng thông tin dưới đây để tiếp tục đúng công việc người dùng yêu cầu. Đọc `AGENTS.md`, `README.md` và các tệp liên quan trực tiếp trước khi chỉnh sửa. Tệp thực tế và output chính thức là căn cứ xác minh.

## 1. Mục tiêu và cách làm việc

Người dùng là sinh viên ICTU, học ML và DL song song, cần hiểu và bảo vệ được bài làm. Đề tài là **Kiểm tra lỗi sản phẩm đúc bằng học sâu**, phân loại toàn ảnh thành lỗi hoặc đạt yêu cầu. Mục tiêu là hoàn thiện báo cáo học phần và demo upload ảnh đơn giản.

Viết tiếng Việt rõ ràng, tự nhiên, giải thích từ mức nhập môn. Đánh giá hoàn thành theo `yeucau.docx` và phạm vi người dùng đã chốt. Mỗi đề xuất phải gắn với một lỗi hoặc khoảng trống cụ thể có bằng chứng. Người dùng rất khó chịu với vòng lặp sửa xong, báo hoàn thành, rồi tiếp tục sinh thêm yêu cầu cải thiện. Tập hợp các vấn đề có căn cứ trong một lượt và kiểm tra đầy đủ phần đã sửa trước khi kết luận.

Người dùng tự thực hiện huấn luyện trên Google Colab. Việc chỉnh notebook đi kèm hướng dẫn để người dùng tự chạy và hiểu. Kết quả mới chỉ được đưa vào báo cáo khi có output thực tế tương ứng. Các sơ đồ kiến trúc sử dụng cấu hình thật và công cụ vẽ kỹ thuật. Người dùng yêu cầu hình minh họa từ bài giảng, tài liệu có nguồn hoặc sơ đồ kỹ thuật, tránh ảnh tạo sinh.

## 2. Tệp và trạng thái bàn giao

- `bao_cao/baocaocodong.docx` là bản biên tập mới nhất đang được xem xét. Người dùng đã chọn tạo bản riêng với tên này. Các chỉnh sửa báo cáo tiếp theo cần xác định đúng tệp theo yêu cầu mới.
- `bao_cao/baocao_DL_theo_dangbaocao.docx` là báo cáo nguồn để đối chiếu.
- `README.md` là tài liệu nhập môn tổng hợp cho thành viên, gồm phương pháp, kết quả, cách chạy và giải thích bảo vệ.
- `Casting.ipynb` là notebook chính có output đã chạy.
- `casting_full_3models_01/` là bộ kết quả chính thức dùng cho báo cáo và demo.
- `casting_full_3models_01_decay/` lưu một lượt thử lịch tốc độ học khác. Số liệu đang sử dụng trong báo cáo lấy từ bộ chính thức ở trên.
- `demo/README.md`, `demo/app.py`, `demo/inference.py`, `demo/prepare_models.py`, `demo/style.css` và `.streamlit/config.toml` phục vụ demo local.
- `CAU_HOI_VA_GIAI_THICH_BAO_VE.md`, `HUONG_DAN_BAO_VE.md`, `HUONG_DAN_CASTING.md` là tài liệu hướng dẫn hiện có. Đọc phiên bản thực tế khi làm việc liên quan.
- Prompt mới được lưu trong `prompt/` theo AGENTS.md.

Lần bàn giao ngày 17/11/2026 xác nhận báo cáo `baocaocodong.docx` có SHA-256 `0ee20eec73bed5ebd32a82da67916f55273fa82bbe630111dfa3dfbe6b582e9e`, khớp bản đã render và kiểm tra. Báo cáo có **66 trang**, Chương 1 có **22 trang**, Kết luận 1 trang và phụ lục 2 trang.

Các tệp kiểm tra gần nhất nằm trong `report_work/bao_cao_co_dong_20261115/`, gồm `after.pdf`, `pages/`, `validation.json`, `delivery_validation.json`, bản nguồn và bản README dự phòng. Đây là tư liệu đối chiếu nội bộ.

Thư mục làm việc hiện có thay đổi chưa commit ở README và hai tài liệu bảo vệ, cùng tệp `prompt/META_PROMPT_BO_SUNG_NOI_DUNG_BAO_VE.md`. Kiểm tra trạng thái thực tế để bảo toàn các thay đổi đang có.

## 3. Phương pháp đã chốt

Ba mô hình dùng TensorFlow/Keras, đầu ra một nút sigmoid biểu diễn `p_ok`.

**CNN baseline** học từ đầu. Kiến trúc là Rescaling chia 255 → ba khối Conv2D 32/64/128 bộ lọc, kernel 3×3, padding same, ReLU và MaxPooling2D 2×2 → Global Average Pooling → Dense 64 ReLU → Dropout 0,3 → Dense 1 sigmoid. Tổng tham số là **100.993**, toàn bộ được cập nhật.

**ResNet-18** dùng KerasHub preset `resnet_18_imagenet` và converter của cùng preset. Backbone giữ `trainable=False` và gọi với `training=False`. Đầu mới gồm GAP → Dense 64 ReLU → Dropout 0,3 → Dense 1 sigmoid. Tổng **11.219.009** tham số, phần được cập nhật **32.897**.

**MobileNetV2** dùng ImageNet, `include_top=False`, `alpha=1.0`, backbone cố định và đầu mới cùng cấu trúc trên. Tổng **2.340.033** tham số, phần được cập nhật **82.049**. Cả ba kiến trúc đều thuộc họ CNN.

Ảnh được hiệu chỉnh EXIF, chuyển xám, resize Pillow bilinear về 256×256×1. Hai mạng học trước tiếp tục resize TensorFlow bilinear có antialias về 224×224, rồi lặp kênh xám thành ba kênh. CNN chuẩn hóa chia 255, MobileNetV2 dùng x/127,5−1, ResNet-18 dùng converter đã lưu. Metadata và `demo/inference.py` mô tả chuỗi xử lý khi suy luận.

Mỗi mô hình chạy 30 epoch, batch 32, seed 42, Adam khởi đầu 0,001. ReduceLROnPlateau giảm một nửa, patience 3, min_lr 10⁻⁶. ModelCheckpoint chọn loss validation nhỏ nhất. Ba lượt hoàn tất đủ 30 epoch.

## 4. Dữ liệu và kết quả chính thức

Bộ Kaggle `casting_data` có 7.348 ảnh nguồn 300×300, đã có biến thể tăng cường. Nhóm loại 64 ảnh đạt phía train trùng pixel với test, giữ 715 ảnh test và tách validation phân tầng từ train đã làm sạch.

| Tập | Lỗi | Đạt | Tổng |
|---|---:|---:|---:|
| Train | 3.194 | 2.389 | 5.583 |
| Validation | 564 | 422 | 986 |
| Test | 453 | 262 | 715 |

Nhãn 0 là `def_front`, nhãn 1 là `ok_front`. Dự đoán đạt khi `p_ok ≥ 0,5`, điểm lỗi bằng `1−p_ok`. Khi đánh giá, lớp lỗi là lớp quan tâm. Các hình ma trận dùng hàng nhãn thật, cột nhãn dự đoán, thứ tự lỗi rồi đạt.

| Mô hình | Epoch chọn | Loss validation | Accuracy test | Bỏ sót lỗi | Báo nhầm lỗi |
|---|---:|---:|---:|---:|---:|
| CNN | 29 | 0,121010 | 94,83% | 25 | 12 |
| ResNet-18 | 29 | 0,026946 | 98,74% | 7 | 2 |
| MobileNetV2 | 21 | 0,015027 | 99,44% | 2 | 2 |

Ma trận tương ứng là `[[428,25],[12,250]]`, `[[446,7],[2,260]]`, `[[451,2],[2,260]]`. MobileNetV2 được đề xuất theo validation loss thấp nhất trước bước đánh giá test. Căn cứ gồm `comparison.csv`, `history_*.csv`, `fit_*.json`, `metrics_*.json`, `test_predictions_*.csv`, `selection.json` và `split.json`.

Giới hạn đã được báo cáo gồm một seed, các biến thể tăng cường có thể liên quan giữa các tập dù đã loại trùng pixel chính xác, và tập test cùng nguồn. Phép so sánh đánh giá ba cấu hình có khởi tạo, độ phân giải và phần tham số học khác nhau. Nguyên nhân riêng của từng spike còn cần thêm bằng chứng. Các nhận xét cần giữ đúng mức kết luận này.

## 5. Demo và báo cáo

Demo Streamlit local hỗ trợ cả ba mô hình, nhiều ảnh, bảng so sánh và tải CSV/JSON. Điểm hiển thị sáu chữ số thập phân. Ba checkpoint trong `demo/models/` là `cnn.keras`, `resnet18.keras`, `casting_classifier.keras`, kèm metadata tương ứng. Các tệp đã được đối chiếu mã băm với kết quả chính thức và kiểm chứng CPU trên 10 ảnh test mỗi mô hình. Hướng dẫn chạy có trong README gốc và README demo.

Ảnh minh họa đã chọn là `cast_ok_0_45.jpeg` thuộc test. Báo cáo mới dùng một hình giao diện so sánh cả ba mô hình, Hình 3.10. Hai hình giải thích khối mạng được bổ sung dựa trên cấu hình thật. Phần tiền xử lý 2.3.5 đã gộp vào 2.2.4 và có danh mục viết tắt, công thức ReLU/sigmoid.

Báo cáo theo mẫu ICTU tại `D:\Crop-Recommendation-ML\nhom08\Template_DA_KLTN_ICTU\` và hướng dẫn sử dụng mẫu cùng cấp. Định dạng hiện tại là A4, Times New Roman 13 cho nội dung, lề trái 3 cm, phải/trên/dưới 2 cm, giãn dòng 1,5. Người dùng chọn số trang ở chân trang. Chương 1 được ưu tiên bảo toàn nội dung và cấu trúc.

## 6. Những điểm còn chờ xử lý

Lần rà soát cuối đã ghi nhận bốn chi tiết biên tập sau trong `baocaocodong.docx`.

1. Mục 3.3.2 gọi epoch 9 là “đỉnh đầu tiên”, trong khi epoch 7 có một cực đại nhỏ. Cách viết chính xác hơn là “Một đỉnh đáng chú ý xuất hiện tại epoch 9”.
2. Mục 3.1.2 sử dụng Macro F1 và Bảng 3.4 có số liệu này. Cần thêm định nghĩa trung bình F1 của hai lớp với trọng số bằng nhau.
3. Còn cụm “Validation loss” tại 3.3.4, tiêu đề cột “Fit (giây)” tại Bảng 3.3 và cách gọi “mã băm trọng số” tại 3.5.2. Tên chính xác cho phép băm đã thực hiện là “mã băm tệp checkpoint”.
4. Mục 3.3.1 nhắc lại việc chạy đủ 30 epoch và checkpoint MobileNetV2 epoch 21 trong đoạn cuối. Có thể rút đoạn này.

Thông tin chi tiết của nguồn bài giảng [5] và quyền xem, nội dung gói Google Drive còn cần xác nhận. Liên kết mã nguồn đã có trong phụ lục là `https://drive.google.com/drive/folders/1otbkaT-wvJfNERf_CL0uapmz2M2ebSnR?usp=sharing`.

Đây là trạng thái bàn giao. Hãy tiếp nhận yêu cầu tiếp theo của người dùng, thực hiện trọn phần được yêu cầu và báo kết quả ngắn gọn. Khi sửa DOCX, lưu dự phòng, kiểm tra phần chỉnh sửa cùng số liệu liên quan, cập nhật tham chiếu và render để kiểm tra bố cục trước khi giao tệp.
