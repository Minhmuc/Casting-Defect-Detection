# Prompt khảo sát đề tài PRODUCT METAL DEFECT INSPECTION

Bạn là người hỗ trợ nghiên cứu cho một nhóm sinh viên thực hiện báo cáo học phần Học sâu tại ICTU. Hãy thực sự tìm kiếm trên Internet, đọc các nguồn có thể truy cập và khảo sát đề tài **PRODUCT METAL DEFECT INSPECTION** theo đúng thứ tự: **bài báo khoa học trước, mã nguồn GitHub sau**. Từ bằng chứng thu được, đề xuất phương án thực nghiệm có lý do rõ ràng.

Mục tiêu là chọn được hướng làm phù hợp với bài toán và nguồn lực, không chọn mô hình theo mức độ quen tên, thứ tự xuất hiện trong bài giảng hoặc năm ra đời.

## 1. Bối cảnh và giới hạn

- Đây là dự án môn Học sâu, cần triển khai được, giải thích được và có thực nghiệm so sánh. Không xây dựng hệ thống phần mềm phức tạp.
- Phạm vi hiện tại: **phân loại toàn ảnh sản phẩm đúc thành sản phẩm lỗi và sản phẩm đạt yêu cầu**. Đây chưa phải bài toán khoanh vùng hoặc phân đoạn khuyết tật.
- Dự án hiện có tại `D:\Casting-Defect-Detection-CNN`. Đọc README và mã nguồn notebook để xác nhận dữ liệu, kiến trúc và thiết kế đang có trước khi đánh giá. Không dùng output chạy thử cũ làm kết quả chính thức.
- Dữ liệu đang hướng đến bộ ảnh sản phẩm đúc công khai trên Kaggle, gồm hai thư mục `def_front` và `ok_front`. Xác minh đúng nguồn, phiên bản và số ảnh từ tài liệu hoặc tệp dữ liệu; không mặc định số ảnh của một phiên bản khác là số ảnh nhóm sử dụng.
- Mô hình chính hiện là CNN tự xây dựng bằng TensorFlow/Keras, huấn luyện từ đầu. Tăng cường dữ liệu là một yếu tố dự kiến khảo sát.
- Ưu tiên TensorFlow/Keras. Thiết bị có thể sử dụng: Ryzen 7 5800H, RTX 3060 Laptop 6 GB hoặc Google Colab. Dự kiến khoảng 30 epoch cho mỗi cấu hình; đây là ngân sách ban đầu, không phải bằng chứng mô hình đã hội tụ.
- Hướng triển khai sau thực nghiệm: chuyển mô hình sang định dạng phục vụ suy luận, như ONNX, và demo tải lên một ảnh. Đây là phần phụ, không chi phối việc chọn hướng nghiên cứu.
- **Chỉ khảo sát và lập phương án trong lần này. Không sửa notebook, báo cáo hoặc môi trường; không cài thư viện, tải bộ dữ liệu/trọng số, chạy chương trình hay huấn luyện mô hình.**

## 2. Bước 1 — Tìm và đọc bài báo khoa học

### 2.1. Chiến lược tìm kiếm

Tìm bằng nhiều nhóm từ khóa, chẳng hạn:

- `casting product defect classification deep learning`
- `casting defect inspection convolutional neural network`
- `metal surface defect classification transfer learning`
- `industrial visual inspection metal defects deep learning`
- `casting product image data quality inspection`

Ưu tiên bài báo gắn trực tiếp với phân loại ảnh sản phẩm đúc. Mở rộng sang bề mặt kim loại hoặc kiểm tra công nghiệp khi cần bổ sung phương pháp, nhưng phải ghi rõ khác biệt với bài toán hiện tại.

Phân biệt ba hướng để tránh so sánh sai:

1. Phân loại có giám sát: nhãn lỗi/đạt hoặc loại lỗi cho toàn ảnh.
2. Phát hiện/phân đoạn: có hộp giới hạn hoặc mặt nạ vùng lỗi.
3. Phát hiện bất thường: thường học từ ảnh bình thường, có thể trả điểm bất thường hoặc bản đồ vị trí.

Không coi một phương pháp phát hiện hoặc phân đoạn là đối chứng trực tiếp cho phân loại nhị phân nếu chưa phân tích cách chuyển bài toán và yêu cầu nhãn.

### 2.2. Tiêu chí nguồn

- Chọn khoảng **6–8 bài có liên quan thực chất**, ưu tiên chất lượng và độ phù hợp hơn số lượng. Nếu không tìm đủ, báo đúng số đã xác minh.
- Ưu tiên trang nhà xuất bản, hội nghị, tạp chí, kho lưu trữ của tác giả hoặc arXiv. Phân biệt bài đã xuất bản với bản tiền công bố.
- Kết hợp công trình nền tảng có ích và công trình gần đây; không mặc định mô hình mới hơn sẽ phù hợp hơn.
- Mở trang bài báo và đọc phần phương pháp, dữ liệu, thực nghiệm khi có toàn văn. Nếu chỉ đọc được tóm tắt, ghi rõ giới hạn đó.
- Không dùng bài quảng cáo, trang tổng hợp hoặc câu trả lời AI làm bằng chứng thay cho bài gốc.
- Không bịa tên tác giả, DOI, đường dẫn, chỉ số hoặc nội dung bài báo. Thông tin chưa xác minh thì ghi “chưa xác minh”.

### 2.3. Nội dung cần trích xuất

Với từng bài, xác định:

- Tên, tác giả, năm, nơi công bố và liên kết gốc/DOI.
- Loại sản phẩm, dạng khuyết tật và nhiệm vụ thực sự giải quyết.
- Bộ dữ liệu và phiên bản; cách chia train/validation/test nếu được mô tả.
- Mô hình, học từ đầu hay học chuyển giao, tiền xử lý và tăng cường dữ liệu.
- Mô hình đối chứng, chỉ số đánh giá và kết quả liên quan nhất nếu có căn cứ đọc được.
- Hạn chế và mức phù hợp với đề tài hiện tại: có thể tham khảo trực tiếp, chỉ tham khảo phương pháp hoặc khác phạm vi.

Khi ghi kết quả của bài báo, gắn chỉ số với đúng dữ liệu và giao thức đánh giá. **Không xếp hạng các mô hình bằng cách đặt cạnh accuracy của những bài dùng dữ liệu hoặc cách chia khác nhau.** Chú ý nguy cơ rò rỉ nếu ảnh tăng cường hoặc ảnh gần trùng xuất hiện ở nhiều tập.

## 3. Bước 2 — Tìm và kiểm tra kho GitHub

Sau khi khảo sát bài báo, tìm mã nguồn chính thức được tác giả liên kết. Sau đó bổ sung các kho triển khai liên quan nếu mã chính thức không có hoặc chưa phù hợp.

Chọn khoảng **4–6 kho đáng xem**, có thể ít hơn nếu nguồn phù hợp hạn chế. Với mỗi kho:

- Mở README, tệp phụ thuộc và các tệp thực hiện đọc dữ liệu, xây dựng mô hình, huấn luyện, đánh giá. Không chỉ đọc mô tả trên trang tìm kiếm.
- Ghi rõ là mã chính thức của bài báo hay dự án độc lập. Chỉ gọi là “chính thức” khi có liên kết hoặc bằng chứng từ tác giả/bài báo.
- Xác nhận nhiệm vụ, dữ liệu, kiến trúc, framework và cách sử dụng trọng số học trước.
- Kiểm tra cách chia dữ liệu, tăng cường, chọn checkpoint và đánh giá test. Nêu những điểm có thể làm sai lệch kết quả nếu nhận thấy từ mã nguồn.
- Kiểm tra tệp hướng dẫn, phiên bản phụ thuộc, checkpoint công bố và khả năng tái lập. Phân biệt “đã đọc mã” với “đã chạy xác nhận”; lần này không thực thi.
- Kiểm tra giấy phép thực tế. Nếu không có LICENSE, ghi “không thấy giấy phép”; không kết luận rằng kho công khai thì được tự do sao chép.
- Ghi thời điểm cập nhật hoặc commit được khảo sát nếu truy cập được. Không dùng số sao làm tiêu chí chất lượng chính.
- Đánh giá có thể tham khảo phần nào và phần nào nhóm cần tự triển khai. Không đề xuất sao chép toàn bộ rồi nhận là công trình tự xây dựng.

Nếu kho dùng PyTorch nhưng phương pháp phù hợp, vẫn có thể đưa vào khảo sát. Nêu rõ công sức chuyển sang TensorFlow và những phần cần xác nhận; không loại phương pháp chỉ vì khác framework, cũng không tự chuyển framework của dự án.

## 4. Tổng hợp bằng chứng trước khi chọn mô hình

Sau hai bước trên, đánh giá phương án theo các tiêu chí:

1. Phù hợp với nhãn lỗi/đạt và loại ảnh sản phẩm đúc.
2. Ý nghĩa đối chứng so với CNN hiện có: mỗi mô hình bổ sung trả lời câu hỏi gì?
3. Nhu cầu dữ liệu, độ phân giải và chế độ học từ đầu/học chuyển giao.
4. Mức độ sẵn có và tương thích của triển khai TensorFlow/Keras.
5. Khả năng thực hiện trong phạm vi môn học và bộ nhớ GPU 6 GB.
6. Khả năng giải thích, tái lập và phân tích lỗi.

Tách rõ điều rút ra từ nguồn với nhận định hoặc đề xuất của bạn. Không khẳng định thời gian train, mức dùng VRAM hoặc khả năng xuất mô hình nếu chưa có bằng chứng phù hợp; nếu chỉ ước lượng thì nói rõ giả định.

Đề xuất **một phương án chính**, gồm CNN hiện có và tối đa hai mô hình đối chứng nếu có lý do. Có thể giữ so sánh CNN có/không tăng cường dữ liệu. Không chốt trước LeNet, VGG, ResNet, MobileNet hoặc EfficientNet chỉ để có đủ tên mô hình; giữ hoặc thay mỗi lựa chọn phải có căn cứ từ khảo sát.

Với mỗi cấu hình đề xuất, nêu ngắn:

- Vai trò: mô hình chính, đối chứng kiến trúc hay đối chứng học chuyển giao.
- Giả thuyết cần kiểm tra.
- Đầu vào, tiền xử lý và phần tham số được huấn luyện.
- Những điều kiện giữ chung và những khác biệt phải công bố.
- Đầu ra cần lưu để hoàn thiện Chương 2–3.

Không gọi so sánh là hoàn toàn công bằng nếu một mô hình dùng ImageNet, mô hình khác học từ đầu hoặc có độ phân giải khác nhau. Phân biệt đối chiếu các cấu hình ứng dụng với thí nghiệm tách riêng ảnh hưởng của kiến trúc.

## 5. Cấu trúc kết quả cần trả về

Trình bày bằng tiếng Việt, rõ ràng và có liên kết kiểm tra được:

1. **Kết luận khảo sát:** hướng phù hợp nhất và căn cứ chính, chưa khẳng định mô hình thắng trước thực nghiệm.
2. **Bài báo:** bảng ngắn gồm bài/năm, nhiệm vụ–dữ liệu, phương pháp, điểm liên quan và nguồn. Chi tiết giao thức hoặc hạn chế đặt trong đoạn giải thích bên dưới khi cần.
3. **GitHub:** bảng gồm kho, chính thức/độc lập, framework, giấy phép, phần đáng tham khảo và hạn chế.
4. **Phương án thực nghiệm:** mô hình chính, các đối chứng được chọn, giả thuyết, điều kiện so sánh và thứ tự thực hiện.
5. **Những điểm còn thiếu căn cứ:** dữ liệu hoặc yêu cầu giảng viên cần xác nhận, giới hạn truy cập nguồn và các kiểm tra chỉ có thể hoàn thành sau khi nhóm tự chạy.

Ưu tiên kết luận có lý do và nguồn cụ thể. Không trả lại một danh sách tên mô hình dài nhưng không giúp nhóm quyết định. Không viết sẵn kết quả thực nghiệm hoặc thay đổi dự án trong bước khảo sát này.
