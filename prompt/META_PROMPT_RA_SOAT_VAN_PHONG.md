# Rà soát văn phong báo cáo Học sâu Nhóm 10

Bạn là người biên tập báo cáo học phần Học sâu của nhóm sinh viên ICTU. Hãy đọc toàn bộ báo cáo hiện tại, rà soát văn phong và chỉnh sửa trực tiếp theo các giới hạn dưới đây.

Mục tiêu là một báo cáo nộp cuối kỳ, viết theo cách sinh viên trình bày học thuật, rõ ràng, tự nhiên, cụ thể và dễ giải thích khi bảo vệ.

## 1. Tệp và căn cứ sử dụng

- Báo cáo chính `D:/Casting-Defect-Detection-CNN/bao_cao/baocao_DL_theo_dangbaocao.docx`.
- Yêu cầu giảng viên `D:/Casting-Defect-Detection-CNN/yeucau.docx`.
- Notebook có output `D:/Casting-Defect-Detection-CNN/Casting.ipynb`.
- Kết quả chính thức `D:/Casting-Defect-Detection-CNN/casting_full_3models_01/`.
- Mẫu chuẩn ICTU `D:/Crop-Recommendation-ML/nhom08/Template_DA_KLTN_ICTU/`, gồm Main_DA_KLTN_ICTU.tex, Main_DA_KLTN_ICTU.pdf và reportICTU.cls.
- Hướng dẫn mẫu `D:/Crop-Recommendation-ML/nhom08/Huong_dan_su_dung_mau_DA_KLTN_ICTU.pdf`.

Đọc nội dung thực tế và AGENTS.md áp dụng trong workspace trước khi biên tập. Báo cáo đã được hoàn thiện khoảng 65 trang ở lần kiểm tra trước; cần xác định lại bố cục từ tệp hiện tại.

## 2. Giới hạn bắt buộc với Chương 1

Chương 1 chỉ được sửa chính tả, dấu câu, ngữ pháp và lặp từ trong câu.

Giữ nội dung kiến thức, ý nghĩa, thứ tự các đoạn, hệ thống đề mục, ví dụ, công thức, hình, bảng và trích dẫn. Giữ nguyên việc phân bổ nội dung trong chương. Các đoạn tương tự nhau về chủ đề chỉ được ghi nhận trong phần bàn giao nếu việc xử lý đòi hỏi gộp, cắt hoặc viết lại nội dung.

Nếu một câu có thể sửa bằng thay từ, dấu câu hoặc điều chỉnh ngữ pháp nhỏ, ưu tiên cách sửa đó. Mỗi thay đổi tại Chương 1 phải xác định được lỗi câu chữ cụ thể. Giới hạn này cũng áp dụng khi Chương 1 có văn phong khác các chương sau.

## 3. Phạm vi biên tập Chương 2–3 và các phần còn lại

Rà soát kỹ mở đầu, Chương 2, Chương 3 và kết luận. Được rút gọn câu dài, chia câu khó đọc, gộp hoặc lược bỏ các đoạn thực sự lặp ý trong cùng mục. Giữ đủ luận điểm, căn cứ và thông tin cần để hiểu phương pháp, thực nghiệm.

Tập trung xử lý các vấn đề sau.

- Lặp từ, lặp câu hoặc diễn đạt cùng một ý nhiều lần mà chưa bổ sung thông tin.
- Câu nhiều mệnh đề, chủ ngữ thiếu rõ ràng, liên kết đoạn gượng hoặc dùng thuật ngữ chưa giải thích.
- Những câu chung chung như “có ý nghĩa quan trọng”, “cho thấy tiềm năng lớn”, hoặc kết luận vượt quá bằng chứng.
- Các đoạn mang giọng hướng dẫn người đọc chạy phần mềm, lời trao đổi với trợ lý hoặc ghi chú biên tập.
- Việc nhắc notebook, Google Colab, tên tệp và thao tác kỹ thuật quá nhiều trong phần diễn giải học thuật. Giữ tên công cụ, môi trường hoặc tệp khi cần để xác định phương pháp và khả năng đối chiếu.
- Các câu hạn chế hoặc cảnh báo giống nhau xuất hiện dày đặc. Tập hợp tại mục giới hạn phù hợp, giữ điều kiện cần thiết ngay cạnh những nhận xét dễ bị hiểu quá mức.

Phân biệt sự lặp thừa với việc nhắc lại có chức năng. Chương 2 cần trình bày cách làm, Chương 3 cần trình bày điều đã quan sát, kết luận cần tổng hợp kết quả. Một số liệu xuất hiện ở bảng, phần phân tích và kết luận có thể cần thiết. Chỉ rút gọn sau khi xác định vai trò của từng chỗ.

Ưu tiên giữ cấu trúc đề mục hiện tại. Những thay đổi cần cho tính nhất quán như tên mô hình, nhãn, thuật ngữ và tham chiếu được chỉnh đồng bộ.

## 4. Kiểm tra nhận xét bằng output

Khi gặp câu diễn giải kết quả thiếu chính xác hoặc khẳng định quá mức trong Chương 2–3, đối chiếu output rồi sửa cả cách diễn đạt và nhận xét. Giữ nguyên các số liệu đã được xác minh.

Các căn cứ cần đọc gồm comparison.csv, fit_*.json, history_*.csv, metrics_*.json, test_predictions_*.csv, split_counts.csv, data_check.json, selection.json và demo/verification.json. Với ảnh minh họa, đối chiếu hình và output thật trong notebook.

Những thông tin sau định hướng việc kiểm tra; output thực tế là căn cứ xác nhận.

- CNN Conv2D/MaxPooling học từ đầu làm baseline, tổng 100.993 tham số. ResNet-18 và MobileNetV2 dùng backbone ImageNet cố định, huấn luyện đầu phân loại mới.
- Bộ nguồn casting_data gồm 7.348 ảnh 300 × 300, đã có augmentation. Pipeline của nhóm không bổ sung augmentation.
- Loại 64 bản sao khỏi train, giữ 715 ảnh test. Train thực dùng 5.583 ảnh, validation 986 ảnh, test 715 ảnh.
- Cả ba mô hình hoàn tất 30 epoch. Checkpoint CNN và ResNet-18 được chọn ở epoch 29, MobileNetV2 ở epoch 21.
- Accuracy test lần lượt 94,83%, 98,74% và 99,44%. MobileNetV2 còn bỏ sót 2 ảnh lỗi và báo nhầm 2 ảnh đạt.
- Mô hình demo được chọn theo validation loss thấp nhất trước khi đánh giá test.
- Nhãn 0 là lỗi, nhãn 1 là đạt. Đầu ra sigmoid là p_ok, ngưỡng 0,5.
- Mô hình lưu bằng .keras. Kiểm chứng nạp lại trên CPU dùng 10 ảnh validation, nhãn trùng hoàn toàn, sai lệch xác suất lớn nhất khoảng 4,21 × 10⁻⁸.
- Minh họa demo dùng ảnh cast_ok_0_45.jpeg từ output notebook, hiển thị nhãn đạt và xác suất đạt 99,82%.

Phân biệt epoch tốt nhất với epoch cuối; accuracy với loss; kết quả validation với test; bỏ sót lỗi với báo nhầm. Khi nhận xét quá khớp hoặc thiếu khớp, đọc cả lịch sử loss và accuracy, xét cách đo train/validation cùng trạng thái Dropout.

Giữ phạm vi kết luận tương ứng một seed, dữ liệu cùng nguồn và các cấu hình đã thực nghiệm. Chỉ khẳng định sự độc lập giữa các tập ở mức đã được kiểm tra. Đánh giá trùng pixel chưa xác định mọi biến thể augmentation hoặc ảnh của cùng sản phẩm.

Nếu phát hiện số liệu trong báo cáo không khớp output, ghi rõ vị trí và hai giá trị trong phần bàn giao. Giữ vấn đề đó tách khỏi các sửa đổi văn phong để người học xác nhận. Không tự chạy chương trình để tạo bằng chứng mới.

## 5. Văn phong và định dạng

Viết tiếng Việt học thuật ở mức sinh viên, dùng câu rõ chủ thể và động từ cụ thể. Ưu tiên “nhóm” khi mô tả công việc nhóm đã thực hiện. Giải thích ngắn các thuật ngữ như baseline, backbone và checkpoint tại vị trí cần thiết, sau đó dùng nhất quán.

Giữ độ chính xác kỹ thuật khi rút gọn. Các giới hạn có căn cứ cần được trình bày ngắn, gắn đúng phạm vi. Bỏ các câu thuyết minh về quá trình AI biên tập khỏi bản nộp.

Bám mẫu ICTU và định dạng báo cáo hiện tại. Giữ bìa, hai bảng phân công, font/cỡ chữ, lề, giãn dòng, tiêu đề và số trang ở cuối trang. Tên bảng phía trên, tên hình phía dưới, đánh số theo chương. Giữ công thức Word và các hình minh họa thật. Giữ khoảng 15 tài liệu tham khảo và quan hệ giữa trích dẫn với tài liệu tương ứng.

Độ dài có thể giảm hợp lý khi bỏ lặp. Chất lượng biên tập được ưu tiên hơn việc giữ đúng 65 trang. Không kéo giãn định dạng, thêm đoạn lặp hoặc trang trắng để bù độ dài.

## 6. Trình tự thực hiện và bàn giao

1. Lưu một bản sao báo cáo hiện tại trước khi sửa. Ghi nhận nội dung, cấu trúc, hình và bảng của Chương 1 để đối chiếu sau biên tập.
2. Đọc toàn văn, xác định các điểm trùng lặp và đối chiếu những nhận xét cần kiểm tra với output.
3. Chỉnh sửa các phần trong phạm vi được phép, cập nhật báo cáo chính sau khi kiểm tra. Nếu Word đang giữ tệp, hoàn thiện bản làm việc rồi báo người học lưu và đóng trước khi ghi đè.
4. Kiểm tra lại Chương 1 để bảo đảm mọi thay đổi chỉ thuộc lỗi câu chữ đã cho phép. Kiểm tra các số liệu, công thức, nhãn, hình và bảng được bảo toàn.
5. Cập nhật mục lục, danh mục và tham chiếu khi phân trang thay đổi. Render để kiểm tra từng trang, đặc biệt các bảng, caption và vị trí ngắt trang sau rút gọn.
6. Trả lại báo cáo đã sửa, đường dẫn bản sao và tóm tắt ngắn các thay đổi. Nêu riêng những thay đổi nhỏ ở Chương 1, số trang mới và các vấn đề cần người học xác nhận.

Phạm vi công việc là biên tập báo cáo. Không sửa notebook, mã demo, dữ liệu hoặc requirements; không cài thư viện, huấn luyện hay chạy suy luận; không tạo ảnh AI hoặc kết quả thực nghiệm mới.
