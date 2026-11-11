# Chỉnh báo cáo và hướng dẫn bảo vệ đề tài phân loại lỗi sản phẩm đúc

Bạn là người biên tập báo cáo Học sâu và hướng dẫn nhóm sinh viên ICTU hiểu kết quả thực nghiệm để bảo vệ. Thực hiện trực tiếp các công việc dưới đây bằng tiếng Việt rõ ràng, tự nhiên, phù hợp sinh viên mới học ML và DL.

## Yêu cầu đã chốt

1. Tạo tệp hướng dẫn riêng `D:/Casting-Defect-Detection-CNN/HUONG_DAN_BAO_VE.md`, bám từng mục và số liệu thực của báo cáo. Giải thích cách hiểu, cách trình bày miệng và cách trả lời câu hỏi.
2. Đổi tên các mục 3.3.2–3.3.4 thành “Kết quả huấn luyện CNN”, “Kết quả huấn luyện ResNet-18”, “Kết quả huấn luyện MobileNetV2”. Đọc lại phần nhận xét để câu chữ cụ thể, dễ hiểu, có căn cứ từ biểu đồ và lịch sử huấn luyện.
3. Bỏ đề mục 2.4.5 “Lưu mô hình và chuẩn bị suy luận”. Tích hợp thông tin cần thiết vào 3.5.2 và 3.5.3, gộp những nội dung đã có để tránh lặp.
4. Phân tích các đỉnh validation bằng số liệu của epoch trước, tại đỉnh và sau đỉnh. Phân biệt hiện tượng quan sát được, cơ chế có thể giải thích và nguyên nhân chưa được kiểm chứng. Cần giải thích vì sao một đỉnh riêng lẻ chưa đủ kết luận quá khớp và vì sao validation đôi lúc tốt hơn train.
5. Mọi đề xuất mới làm thay đổi phạm vi, cấu hình hoặc thực nghiệm phải được hỏi người dùng trước khi thực hiện.

## Tài liệu phải đọc

- Báo cáo chính `D:/Casting-Defect-Detection-CNN/bao_cao/baocao_DL_theo_dangbaocao.docx`.
- Notebook có output `D:/Casting-Defect-Detection-CNN/Casting.ipynb`.
- Thư mục kết quả chính thức `D:/Casting-Defect-Detection-CNN/casting_full_3models_01/`, đặc biệt history_*.csv, learning_curves_*.png, comparison.csv, fit_*.json, selection.json, split_counts.csv, data_check.json, metrics_*.json và demo/verification.json.
- Mẫu ICTU và hướng dẫn tại `D:/Crop-Recommendation-ML/nhom08/Template_DA_KLTN_ICTU/` và `D:/Crop-Recommendation-ML/nhom08/Huong_dan_su_dung_mau_DA_KLTN_ICTU.pdf`.

Đối chiếu tên lớp và tiền xử lý bằng code. Kiểm tra tài liệu chính thức khi giải thích hành vi Keras. Không suy nguyên nhân chỉ từ hình dạng đường cong.

## Các điểm kỹ thuật phải giữ đúng

- CNN học từ đầu gồm Conv2D 32/64/128 và MaxPooling, GAP, Dense64, Dropout0,3, Dense1 sigmoid. Tổng 100.993 tham số. CNN này không có Batch Normalization.
- ResNet-18 và MobileNetV2 giữ cố định backbone ImageNet, chỉ huấn luyện đầu phân loại mới. Đây là so sánh ba cấu hình có cách khởi tạo và tiền xử lý riêng.
- Train 5.583, validation 986, test 715. Loại 64 ảnh trùng pixel từ train trước khi chia validation, giữ nguyên test. Nguồn đã có augmentation, pipeline nhóm không bổ sung.
- Mỗi cấu hình 30 epoch. CNN và ResNet-18 chọn epoch29, MobileNetV2 chọn epoch21 theo validation loss nhỏ nhất. Mô hình demo được chọn trước khi đánh giá test.
- Trong CSV, cột epoch bắt đầu từ0. Số epoch trình bày trong báo cáo và hướng dẫn bắt đầu từ1.
- Accuracy test lần lượt 94,83%, 98,74%, 99,44%. MobileNetV2 còn bỏ sót2 ảnh lỗi, báo nhầm2 ảnh đạt.
- Đầu ra p_ok, nhãn0 lỗi, nhãn1 đạt, ngưỡng0,5. Tệp demo dùng định dạng .keras.
- Ảnh minh họa là cast_ok_0_45.jpeg với xác suất đạt99,82% từ output có sẵn. Kiểm chứng CPU10 ảnh validation là công việc đã có output, tách riêng với chất lượng trên715 ảnh test.

## Hướng dẫn bảo vệ cần có

- Một lời giới thiệu ngắn về bài toán và phần nhóm đã làm.
- Cách giải thích các mục quan trọng của Chương2, gồm chia tập, baseline, backbone, đóng băng, tiền xử lý, các lớp CNN, loss, accuracy và checkpoint.
- Cách đọc ba biểu đồ huấn luyện, giải thích đúng các đỉnh validation, mức quá khớp hoặc giới hạn kết luận.
- Cách đọc bảng so sánh, ma trận nhầm lẫn, precision, recall, F1 và giải thích ý nghĩa số ảnh lỗi bị bỏ sót.
- Cách trình bày demo và trả lời những câu hỏi thường gặp bằng số liệu của nhóm.
- Trình tự trình bày gợi ý, có câu nói mẫu ngắn để người học diễn đạt lại bằng lời của mình.
- Đường dẫn đến các output giúp người học tự kiểm tra và nguồn chính thức cho những cơ chế kỹ thuật đã sử dụng.

## Giới hạn và kiểm tra bàn giao

Giữ nguyên Chương1, kiến trúc, số liệu, hình, bảng và định dạng ICTU hiện tại ngoài các nội dung được phép chỉnh. Hướng dẫn học và lời luyện nói nằm trong tệp riêng, báo cáo giữ văn phong bản nộp.

Không chạy notebook, huấn luyện, suy luận hoặc cài thư viện dự án. Không sửa notebook, mã demo, dữ liệu hoặc requirements. Không tạo hình AI, biểu đồ hay kết quả mới. Không tự bổ sung thí nghiệm hoặc tuyên bố đã xác định chính xác nguyên nhân đỉnh validation.

Lưu bản sao dự phòng trước khi sửa báo cáo. Làm việc trên bản sao, cập nhật mục lục và render bằng công cụ khả dụng, kiểm tra bố cục, bảng, hình, số trang và bảo toàn Chương1. Cập nhật báo cáo chính sau khi kiểm tra và xác nhận tệp không bị Word giữ. Nếu bị giữ, tiếp tục hoàn thiện bản làm việc và yêu cầu người dùng lưu, đóng tệp.

Trả lại báo cáo đã sửa, meta-prompt và hướng dẫn bảo vệ. Tóm tắt các thay đổi, số trang thực tế, vị trí bản sao và các điểm cần người học xác nhận.
