# BÁO CÁO KIỂM TRA GIẤY PHÉP VÀ NGUỒN GỐC DỮ LIỆU TOÀN DIỆN (STAGE 2 – LICENSE AND DATA PROVENANCE REVIEW)

- **Dự án:** Hệ thống Nhận diện Ngôn ngữ Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)  
- **Workspace:** `D:\NhanDienCuChi`  
- **Repository:** [phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)  
- **Branch:** `main` | **Git HEAD tham chiếu:** `dce96c2`  
- **Ngày thực hiện:** 17/11/2026  
- **Chế độ kiểm tra:** STRICT READ-ONLY FORENSIC REVIEW (Chỉ rà soát tài liệu và dữ liệu, tuyệt đối không sửa mã nguồn, không sửa model/manifest, không tạo file ở thư mục gốc, không commit/push/tag/release).

---

## 1. MA TRẬN NGUỒN GỐC DỮ LIỆU VÀ GIẤY PHÉP (DATA PROVENANCE & LICENSE MATRIX)

Bảng tổng hợp toàn diện đối soát từng nguồn dữ liệu, đường dẫn lưu trữ, quy mô mẫu, giấy phép công bố, chứng cứ lưu trữ và các nghĩa vụ pháp lý bắt buộc:

| Nguồn dữ liệu / Tác giả | Phân hệ sử dụng | Quy mô & Dạng thức | Giấy phép công bố | Bằng chứng lưu trữ | Quyền sử dụng dữ liệu | Quyền phân phối Model | Nghĩa vụ pháp lý & Attribution bắt buộc |
|---|---|---|---|---|:---:|:---:|---|
| **Google MediaPipe**<br>(Google LLC) | Cả 4 phân hệ (Trích xuất 21 landmarks) | Gói mô hình nhị phân `.task` (7.819.105 B) | **Apache License 2.0** | Tài liệu MediaPipe Solutions chính thức; GCS URL chính thức | `CÓ` | `CÓ` | Giữ nguyên thông báo bản quyền Apache 2.0 của Google LLC trong tài liệu phân phối. |
| **Ardamavi Digits**<br>(Arda Mavi & Zeynep Dikle) | Nhánh 123 (Digits) | 170 mẫu vector (10 số) trích từ ảnh 100x100 | **Apache License 2.0** | Tệp `LICENSE` trên GitHub repository `ardamavi/Sign-Language-Digits-Dataset` | `CÓ` | `CÓ` | 1. Đính kèm bản sao giấy phép Apache 2.0.<br>2. Giữ nguyên thông báo bản quyền.<br>3. Trích dẫn công trình: *arXiv:2011.08927 [cs.CV]*.<br>4. Ghi rõ các sửa đổi (trích xuất landmarks 63 chiều). |
| **ASL-HG Mendeley**<br>(Pranto et al., 2023) | Nhánh 123 (Digits) | 30 mẫu vector (chữ số 4, 6, 8) trích từ ảnh | **Creative Commons Attribution 4.0 (CC BY 4.0)** | Trang Mendeley Data chính thức DOI: [10.17632/j4y5w2c8w9.1](https://doi.org/10.17632/j4y5w2c8w9.1) | `CÓ` | `CÓ` | 1. Ghi công nhóm tác giả Pranto et al.<br>2. Dẫn liên kết đến giấy phép CC BY 4.0.<br>3. Dẫn liên kết đến DOI gốc trên Mendeley.<br>4. Mô tả rõ việc trích xuất tọa độ landmark cho 3 lớp. |
| **Kaggle ASL Alphabet**<br>(Akash Nagaraj, 2018) | Nhánh ABC (Static Letters) | 45.236 mẫu ảnh ($50,59\%$ tập ABC) | **GNU General Public License v2.0 (GPL-2.0)** | Siêu dữ liệu trang Kaggle `grassknoted/asl-alphabet` | `CÓ` (Nghiên cứu) | `CHƯA RÕ` (Tranh luận phái sinh) | Ràng buộc điều khoản Copyleft của GPL 2.0 nếu trọng số mạng nơ-ron bị xác định là tác phẩm phái sinh; cần tham vấn pháp lý. |
| **Color-coded Images**<br>(Không rõ tác giả) | Nhánh ABC (Static Letters) | 41.676 mẫu ảnh ($46,61\%$ tập ABC) | **LICENSE_UNKNOWN** (Chưa xác định) | Không có metadata, không có URL nguồn gốc trong kho dữ liệu cũ | `KHÔNG RÕ` | `KHÔNG ĐỦ CĂN CỨ` | Rủi ro bản quyền cao nhất. Nguy cơ vi phạm DMCA hoặc khiếu nại bản quyền nếu phát hành công khai trọng số học trực tiếp từ nhóm ảnh này. |
| **Hand Segmentation**<br>(Không rõ tác giả) | Nhánh ABC (Static Letters) | 2.503 mẫu ảnh ($2,80\%$ tập ABC) | **LICENSE_UNKNOWN** (Chưa xác định) | Tệp ảnh `hand1_c_bot_seg`, `A0 (4).jpg`... | `KHÔNG RÕ` | `KHÔNG ĐỦ CĂN CỨ` | Tương tự nhóm ảnh color-coded. |
| **SigNN ASL Video [J, Z]**<br>(SigNN Team) | Nhánh J/Z (Dynamic) | 640 video chuỗi động (.avi) | **CC0 1.0 Universal (Public Domain)** | Ảnh chụp màn hình Kaggle `signnteam_kaggle_cc0_license.png` (SHA256: `16e7e727...`) | `CÓ` (Bản quyền) | `CHƯA ĐỦ ĐIỀU KIỆN RỘNG RÃI` | CC0 chỉ từ bỏ bản quyền tác giả. Chưa có thỏa thuận giải phóng quyền hình ảnh cá nhân (Model Release) của người xuất hiện trong video theo Điều 2(d) CC0. |
| **SUBTLEX-US**<br>(Brysbaert & New, 2009) | Bộ gợi ý từ vựng (Dictionary) | 25.000 từ tiếng Anh trích từ kho 74k từ | **CC BY-SA 4.0** | Tệp chứng từ chi tiết tại [data/dictionary/NOTICE.md](file:///D:/NhanDienCuChi/data/dictionary/NOTICE.md) | `CÓ` | `CÓ` | Bắt buộc chia sẻ tệp từ vựng phái sinh dưới giấy phép CC BY-SA 4.0; ghi công Brysbaert & New (2009). |
| **Dữ liệu số tự thu**<br>(Tác giả đề tài) | Nhánh 123 (base_train_v1) | 349 mẫu vector landmarks cơ sở | **Bản quyền tác giả** (Chưa công bố giấy phép mở) | [train_v1.csv](file:///D:/NhanDienCuChi/ai/dataset/kytuso/datatrain/train_v1.csv) | `CÓ` (Sở hữu) | `CẦN CÔNG BỐ` | Tác giả cần ký phát hành tuyên bố cấp phép mở (ví dụ MIT License) để người khác có quyền sử dụng hợp pháp. |
| **Dữ liệu SPACE tự thu**<br>(Tác giả đề tài) | Nhánh SPACE (Mẫu dương) | 1.499 mẫu vector (1.049 train, 225 val, 225 test) | **Bản quyền tác giả** (Chưa công bố giấy phép mở) | `space_landmarks.csv` | `CÓ` (Sở hữu) | `CẦN CÔNG BỐ` | Tương tự như trên; cần cấp phép mở cho mẫu dương và giải quyết sự phụ thuộc vào tập âm. |

---

## 2. KIỂM TOÁN NGUỒN GỐC VÀ RỦI RO BẢN QUYỀN NHÁNH ABC (STATIC LETTERS)

### 2.1. Phân loại cấu trúc tập ảnh chữ cái tĩnh
Căn cứ tệp đóng băng dữ liệu chuẩn `ai/dataset/preprocessing_frozen_v3/final_freeze_manifest.json` và phân loại từ `sample_group_map_frozen_v3.csv` trên tổng số **89.415 mẫu ảnh** sạch:

1. **Nhóm Kaggle ASL Alphabet (45.236 mẫu — chiếm 50,59%):**
   - *Tác giả:* Akash Nagaraj (2018).
   - *URL nguồn:* `https://www.kaggle.com/datasets/grassknoted/asl-alphabet`.
   - *Quy cách tệp:* `A1.jpg`..`A3000.jpg`, `B1.jpg`..`B3000.jpg`...
   - *Giấy phép công bố trên Kaggle:* **GNU General Public License v2.0 (GPL-2.0)**.
   - *Đặc điểm:* Ảnh chụp tay trước phông nền đơn sắc xám/đen của tác giả.
2. **Nhóm ảnh Color-coded Segmentation (41.676 mẫu — chiếm 46,61%):**
   - *Quy cách tệp:* Tiền tố `color_0_*`, `color_10_0460 (3).png`...
   - *Tình trạng:* **`LICENSE_UNKNOWN`**. Toàn bộ kho lưu trữ lịch sử `dulieu/chucai` không lưu lại metadata, nguồn tải ban đầu, tên tác giả, ngày thu thập hoặc điều khoản cấp phép.
   - *Đặc điểm:* Ảnh được tạo ra từ thuật toán phân đoạn màu nền nhân tạo hoặc các cuộc thi trích xuất trên mạng internet.
3. **Nhóm phân đoạn bàn tay phụ trợ và ảnh web (2.503 mẫu — chiếm 2,80%):**
   - *Quy cách tệp:* `hand1_c_bot_seg`, `A0 (4).jpg`...
   - *Tình trạng:* **`LICENSE_UNKNOWN`**.

### 2.2. Phân tích pháp lý: License Dataset vs. License Model Weights
- **Vấn đề GPL-2.0:** Giấy phép GPL-2.0 quy định cơ chế "lây lan" (Copyleft) đối với các "tác phẩm phái sinh" (derivative works). Tuy nhiên, trong giới luật học công nghệ thế giới hiện nay:
  - *Quan điểm thứ nhất (Fair Use / Biểu diễn thống kê độc lập):* Mô hình học máy chỉ học các quan hệ thống kê toán học (weights) từ tọa độ landmark (x, y, z), không chứa bất kỳ pixel ảnh hay bản sao mã nguồn nào của dữ liệu gốc. Do đó, model weights không tự động biến thành tác phẩm phái sinh theo định nghĩa của luật bản quyền.
  - *Quan điểm thứ hai (Ràng buộc dây chuyền):* Nếu dữ liệu GPL được coi là "mã nguồn" để biên dịch ra mô hình toán học (binary weights), việc phân phối mô hình có thể bị yêu cầu phải chịu sự điều chỉnh của GPL.
  > **Khuyến nghị:** Cần sự tư vấn từ luật sư chuyên môn về sở hữu trí tuệ để có kết luận pháp lý chính thức.
- **Vấn đề cốt lõi của 41.676 ảnh color-coded:** Rủi ro nghiêm trọng nhất của nhánh ABC **không nằm ở GPL-2.0 mà nằm ở 41.676 ảnh không rõ nguồn gốc**. Việc huấn luyện mô hình thương mại hoặc công khai từ dữ liệu không rõ quyền sử dụng vi phạm nguyên tắc thẩm định nguồn (Due Diligence), có thể dẫn đến rủi ro gỡ bỏ (DMCA takedown notice).
- **Kết luận:** Mô hình `uvw_medium_model.joblib` được xếp loại **`LICENSE_UNKNOWN`** và **`BLOCKED`** cho việc phân phối công khai cho đến khi dữ liệu được làm sạch.

---

## 3. KIỂM TOÁN NGUỒN GỐC VÀ QUYỀN HÌNH ẢNH NHÁNH J/Z (DYNAMIC GESTURES)

### 3.1. Nguồn dữ liệu và hiện trạng xử lý
- *Bộ dữ liệu:* SigNN ASL Sign Language Alphabet Videos [J, Z].
- *URL:* `https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z`.
- *Quy mô:* Ban đầu gồm 643 video định dạng `.avi` (316 video J + 327 video Z). Trong đó 3 video bị loại bỏ do MediaPipe không phát hiện bàn tay (`J/99.avi`, `J/103.avi`, `J/250.avi`), còn lại đúng **640 chuỗi huấn luyện** (265 chuỗi J + 375 chuỗi Z) được phân bổ 512 Train / 64 Val / 64 Test.
- *Giấy phép công bố:* **CC0 1.0 Universal (Public Domain)** (Được xác minh qua ảnh chụp màn hình `signnteam_kaggle_cc0_license.png`, SHA-256: `16e7e727...`).

### 3.2. Phân biệt Bản quyền và Quyền riêng tư / Quyền hình ảnh cá nhân
Đây là điểm pháp lý trọng yếu cần phân định rõ:
1. **Về Bản quyền (Copyright):** Người đăng tải (SigNN Team) đã chọn nhãn CC0 1.0 Universal trên Kaggle, đồng nghĩa với việc từ bỏ quyền tác giả đối với tác phẩm video ở mức tối đa theo luật định.
2. **Về Quyền riêng tư và Quyền hình ảnh cá nhân (Privacy & Publicity Rights):**
   - Căn cứ **Điều 2(d) của văn bản pháp lý Creative Commons CC0 1.0 Universal**:
     > *"No trademark or patent rights held by Affirmer are waived, abandoned, surrendered, licensed or otherwise affected by this document. Affirmers also do not waive or affect any rights of publicity or privacy held by Affirmer or any other person."*
   - Video SigNN ghi hình trực tiếp khuôn mặt, cơ thể và nhận dạng của các cá nhân tham gia thực hiện ký hiệu. Tuyên bố CC0 của chủ sở hữu tài khoản Kaggle **không thay thế và không làm suy giảm quyền hình ảnh cá nhân của các tình nguyện viên xuất hiện trong video**.
   - Căn cứ `data/experiments/dynamic_jz_v1/metadata/signnteam_source_rights_evidence_v1.json`, hiện dự án **hoàn toàn thiếu văn bản giải phóng quyền hình ảnh (Model Release / Participant Consent Forms)**.
3. **Phạm vi áp dụng:**
   - Việc chỉ sử dụng mô hình trong phạm vi nghiên cứu học thuật nội bộ là giải pháp thận trọng của dự án. Để phân phối thương mại hoặc công khai rộng rãi, cần có chứng từ xác nhận các tình nguyện viên đã đồng thuận cho phép sử dụng hình ảnh.
- **Kết luận:** Mô hình J/Z được xếp loại **`LICENSE_REVIEW`** do thiếu văn bản bảo đảm về quyền hình ảnh cá nhân.

---

## 4. KIỂM TOÁN MỐI LIÊN HỆ PHỤ THUỘC DỮ LIỆU CỦA NHÁNH SPACE

### 4.1. Nguồn gốc dữ liệu huấn luyện SPACE
Căn cứ tệp kiểm toán `staging/space_training_readiness_audit_phase3_9_4_20261106/dataset_integrity_audit.csv` và `space_model_metadata.json`:
Mô hình `final_space_model.joblib` là một bộ phân loại Logistic Regression ($C=10.0$) với không gian 63 đặc trưng. Toàn bộ tập huấn luyện gồm **64.189 mẫu**:
- **Mẫu dương (SPACE = 1):** Đúng **1.049 mẫu** ($1,63\%$), do tác giả đề tài tự quay và thu thập đặc trưng bàn tay.
- **Mẫu âm (NOT_SPACE = 0):** Đúng **63.140 mẫu** ($98,37\%$), gồm:
  - **62.591 mẫu chữ cái (ABC):** Lấy trực tiếp từ tập Train cơ sở `ai/dataset/train_validation_test_split_v1/train.csv`.
  - **549 mẫu chữ số (123):** Lấy trực tiếp từ tập Train cơ sở `data/processed/digits/digits_final_train_frozen.csv`.

### 4.2. Phân tích ảnh hưởng pháp lý dây chuyền từ ABC sang SPACE
1. **Cơ chế hoạt động toán học của Logistic Regression:** Mô hình tối ưu hóa hàm log-loss để tìm một siêu phẳng nhị phân phân tách giữa hai lớp. Tọa độ ma trận trọng số $w$ và hệ số chặn $b$ được xác định đồng thời bởi cả mẫu dương và mẫu âm:
   $$P(	ext{SPACE}=1 \mid x) = \sigma(w^T x + b)$$
   Do đó, mô hình `final_space_model.joblib` phụ thuộc trực tiếp vào phân bố đặc trưng của 62.591 mẫu chữ cái ABC.
2. **Hệ quả pháp lý dây chuyền:**
   - Vì mẫu âm chiếm tới **98,37%** tổng dữ liệu huấn luyện, và $62.591 / 63.140 = \mathbf{99,13\%}$ mẫu âm lấy từ nhánh ABC (trong đó có 41.676 ảnh color-coded chưa rõ bản quyền), mô hình SPACE thừa hưởng toàn bộ rủi ro pháp lý của nhánh ABC.
   - Tác giả không thể tuyên bố toàn quyền phát hành mở cho mô hình SPACE chỉ dựa vào việc tác giả tự thu thập 1.049 mẫu dương.

### 4.3. Đề xuất phương án xử lý
- **Phương án 1 (Kỹ thuật — RẤT KHUYẾN NGHỊ):**
  - Thu thập lại một tập âm NOT_SPACE độc lập, sạch và hoàn toàn mới: Quay các tư thế tay bình thường khi không bấm dấu cách (tay buông lơi, tay nghỉ ngơi, các cử chỉ bàn tay thông thường, cử chỉ số tự thu).
  - Quy mô tập âm chỉ cần khoảng 2.000 – 5.000 mẫu (thay vì 63.000 mẫu như cũ).
  - Huấn luyện lại (retrain) mô hình SPACE. Khi đó mô hình SPACE sẽ hoàn toàn tách rời và độc lập với nhánh ABC, đủ điều kiện phát hành công khai ngay lập tức.
- **Phương án 2 (Pháp lý):** Làm rõ nguồn gốc và giấy phép của tập dữ liệu ABC trước khi phát hành mô hình SPACE.

---

## 5. ĐÁNH GIÁ DỮ LIỆU TỰ THU THẬP NỘI BỘ (SELF-COLLECTED DATA)

Dự án có hai nhóm dữ liệu do tác giả đề tài tự thu thập trực tiếp qua webcam:
1. **349 mẫu chữ số cơ sở (Nhánh 123):** Thu thập tại `ai/dataset/kytuso/datatrain/train_v1.csv` (gồm 34 mẫu số 0 và 35 mẫu cho mỗi số từ 1 đến 9).
2. **1.499 mẫu cử chỉ SPACE dương:** Thu thập tại `space_landmarks.csv` (1.049 train, 225 validation, 225 test).

### Phân biệt quyền sở hữu và việc công bố giấy phép mở:
- **Quyền sở hữu:** Tác giả là người sáng tạo và có đầy đủ quyền sở hữu độc quyền đối với các mẫu dữ liệu và vector đặc trưng do chính mình thu thập. Tác giả hoàn toàn có quyền sử dụng nội bộ để huấn luyện mô hình.
- **Điều kiện phát hành mở:** Quyền sở hữu nội bộ **không đồng nghĩa với việc bên thứ ba được tự do sử dụng** nếu tác giả chưa đưa ra một tuyên bố cấp phép mở (Explicit License Grant).
- **Hành động bắt buộc:** Tác giả cần đưa ra văn bản công bố chính thức:
  - Cấp phép mã nguồn và trọng số mô hình phái sinh theo giấy phép **MIT License**.
  - Cấp phép bộ dữ liệu vector đặc trưng trích xuất theo giấy phép mở **Creative Commons Attribution 4.0 International (CC BY 4.0)** hoặc **CC0 1.0 Universal**.

---

## 6. NỘI DUNG ĐÃ XÁC MINH VÀ CHƯA XÁC MINH (VERIFIED VS. UNVERIFIED)

| Hạng mục kiểm toán | Nội dung đã xác minh (Verified) | Nội dung chưa xác minh (Unverified / Gaps) |
|---|---|---|
| **MediaPipe** | - Giấy phép Apache 2.0 từ Google LLC.<br>- SHA-256 tệp `.task`: `fbc2a30...` khớp Google CDN. | Không có. Đã xác minh 100%. |
| **Ardamavi Digits** | - Giấy phép Apache 2.0 trên GitHub.<br>- Bài báo trích dẫn: *arXiv:2011.08927*.<br>- Số mẫu lấy vào train: Đúng 170 mẫu. | Chưa có file `THIRD_PARTY_NOTICES.md` trong repo thực hiện nghĩa vụ ghi công. |
| **ASL-HG Mendeley** | - Giấy phép CC BY 4.0 trên Mendeley Data.<br>- DOI: *10.17632/j4y5w2c8w9.1*.<br>- Số mẫu lấy vào train: Đúng 30 mẫu cho 4, 6, 8. | Tương tự Ardamavi, thiếu file văn bản ghi nhận công khai nghĩa vụ Attribution. |
| **Kaggle ASL Alphabet** | - Tác giả Akash Nagaraj.<br>- Giấy phép hiển thị trên Kaggle: GPL-2.0.<br>- Số mẫu sạch: 45.236 mẫu. | Chưa có án lệ chính thức kết luận trọng số MLP có chịu hiệu ứng copyleft của GPL 2.0 hay không. |
| **Color-coded ABC** | - Số lượng: 41.676 mẫu.<br>- Dạng thức: Ảnh nhãn phân đoạn màu nhân tạo. | **HOÀN TOÀN KHÔNG RÕ NGUỒN GỐC, TÁC GIẢ, URL VÀ GIẤY PHÉP**. |
| **SigNN J/Z Video** | - Nhãn giấy phép Kaggle: CC0 1.0 Public Domain.<br>- 640 chuỗi huấn luyện sau khi trừ 3 video lỗi. | **THIẾU THỎA THUẬN QUYỀN HÌNH ẢNH (MODEL RELEASE)** của các tình nguyện viên xuất hiện trong video. |
| **SPACE Gesture** | - 1.049 mẫu dương tự thu.<br>- 63.140 mẫu âm (62.591 từ ABC, 549 từ 123). | Chưa có tuyên bố cấp phép mở cho mẫu dương; rủi ro bản quyền từ 62.591 mẫu âm ABC. |
| **SUBTLEX-US** | - Đã lưu trữ đầy đủ tại `data/dictionary/NOTICE.md`.<br>- Giấy phép CC BY-SA 4.0, trích dẫn Brysbaert & New (2009). | Đã hoàn tất 100% trong thư mục dictionary. |

---

## 7. ĐÁNH GIÁ RỦI RO PHÁP LÝ VÀ QUYỀN RIÊNG TƯ (LEGAL & PRIVACY RISK ASSESSMENT)

1. **Rủi ro vi phạm nghĩa vụ Attribution (Rủi ro Thấp - Dễ khắc phục):**
   - *Nguyên nhân:* Thiếu văn bản ghi nhận quyền tác giả cho Arda Mavi và ASL-HG Mendeley.
   - *Giải pháp:* Ban hành tệp `THIRD_PARTY_NOTICES.md` theo bản nháp tại Mục 8 dưới đây.
2. **Rủi ro bản quyền tập dữ liệu Color-coded của ABC (Rủi ro Cao):**
   - *Nguyên nhân:* Sử dụng 41.676 hình ảnh không rõ nguồn gốc để huấn luyện mô hình sản phẩm.
   - *Hậu quả tiềm tàng:* Bị khiếu nại bản quyền hoặc nhận yêu cầu gỡ bỏ theo DMCA nếu phát hành mô hình công khai.
   - *Giải pháp:* Tách riêng mô hình ABC khỏi gói phát hành công khai ban đầu; tiến hành thay thế và huấn luyện lại trên tập dữ liệu mở minh bạch.
3. **Rủi ro quyền riêng tư / hình ảnh cá nhân đối với video J/Z (Rủi ro Trung bình):**
   - *Nguyên nhân:* CC0 trên Kaggle không bao gồm quyền hình ảnh cá nhân (Điều 2(d) CC0).
   - *Giải pháp:* Duy trì mô hình J/Z dưới trạng thái tính năng thử nghiệm nghiên cứu (`EXPERIMENTAL_RESEARCH_ONLY`) và ghi rõ giới hạn trách nhiệm.
4. **Vấn đề cần tư vấn chuyên gia pháp lý (Legal Advice Needed):**
   - Xác định phạm vi ảnh hưởng của giấy phép GPL-2.0 đối với mô hình mạng nơ-ron được huấn luyện từ dữ liệu ảnh mang giấy phép GPL-2.0 theo luật sở hữu trí tuệ hiện hành.

---

## 8. BẢN NHÁP VĂN BẢN `THIRD_PARTY_NOTICES.md` (DRAFT VERSION)

> [!IMPORTANT]
> **Lưu ý:** Dưới đây là bản nháp văn bản tuân thủ bản quyền bên thứ ba. Bản nháp này chỉ được lưu trữ trong báo cáo kiểm toán này và **chưa được ghi đè hay tạo ra tại thư mục gốc repository**. Tác giả đề tài cần phê duyệt trước khi tạo file chính thức.

```markdown
# THIRD-PARTY SOFTWARE AND DATA NOTICES AND INFORMATION

This project, **Hệ thống Nhận diện Ngôn ngữ Ký hiệu (Sign Language Recognition System)**, incorporates, references, or was trained on third-party software, models, and datasets under various open-source and open-data licenses. This document acknowledges the respective authors and provides the applicable license terms and attribution notices.

---

## 1. Google MediaPipe Hand Landmarker

- **Project**: MediaPipe Hand Landmarker Model Bundle (`hand_landmarker.task`)
- **Copyright**: Copyright (c) Google LLC. All rights reserved.
- **License**: Apache License, Version 2.0
- **Official Source**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **Official Download URL**: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- **Notice**:
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:
      http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## 2. Arda Mavi Sign Language Digits Dataset

- **Project**: Sign Language Digits Dataset
- **Authors**: Arda Mavi and Zeynep Dikle (2020)
- **Repository**: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- **License**: Apache License, Version 2.0
- **Academic Citation**:
  > Mavi, A., (2020), "A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits", arXiv:2011.08927 [cs.CV]. Available at: https://arxiv.org/abs/2011.08927
- **Modifications**:
  The project extracted 63-dimensional MediaPipe landmark feature vectors from 170 right-hand images (10 samples for digits 4, 6, 8 and 20 samples for digits 0, 1, 2, 3, 5, 7, 9) to train the static digits MLP classification model (`digits_final.joblib`).

---

## 3. ASL-HG: American Sign Language Hand Gesture Image Dataset

- **Project**: ASL-HG Dataset (Version 1)
- **Authors**: Md. Famidul Islam Pranto, Md. Rifatul Islam, Md. Ali Akbor, Nabonita Ghosh, Md. Rahatun Alam, Sudipto Chaki, Md. Masudul Islam (2023)
- **Publisher**: Mendeley Data / Data in Brief, Elsevier
- **DOI**: 10.17632/j4y5w2c8w9.1 (https://doi.org/10.17632/j4y5w2c8w9.1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  https://creativecommons.org/licenses/by/4.0/
- **Modifications**:
  The project extracted 63-dimensional MediaPipe landmark feature vectors from 30 image samples (10 samples each for digits 4, 6, and 8 across distinct volunteer signers) to augment the static digits classification model (`digits_final.joblib`).

---

## 4. SUBTLEX-US Word Frequency Dataset

- **Project**: SUBTLEX-US (American English Word Frequency Norms)
- **Authors**: Marc Brysbaert (Ghent University) & Boris New (Université Savoie Mont Blanc / CNRS)
- **Publisher**: Department of Experimental Psychology, Ghent University (UGent), Belgium & OpenLexicon Project
- **Source URL**: https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
  https://creativecommons.org/licenses/by-sa/4.0/
- **Academic Citation**:
  > Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. *Behavior Research Methods*, 41(4), 977–990. DOI: 10.3758/BRM.41.4.977
- **Derivative Notice**:
  The filtered English dictionary `data/dictionary/english_words_25k.tsv` is distributed under CC BY-SA 4.0 in accordance with the ShareAlike clause. Full ingestion details are provided in `data/dictionary/NOTICE.md`.

---

## 5. SigNN ASL Sign Language Alphabet Videos [J, Z]

- **Project**: ASL Sign Language Alphabet Videos [J, Z]
- **Publisher / Team**: SigNN Team
- **Source URL**: https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z
- **License as Displayed**: Creative Commons CC0 1.0 Universal (Public Domain Dedication)
  https://creativecommons.org/publicdomain/zero/1.0/
- **Notice on Personality & Privacy Rights**:
  In accordance with Section 2(d) of CC0 1.0 Universal, publicity and privacy rights of individuals appearing in video recordings are not waived. Models derived from this dataset (`model_pipeline.joblib`) are distributed strictly for academic research and experimental benchmarking purposes.

---

## 6. ASL Alphabet Dataset (Akash Nagaraj)

- **Project**: ASL Alphabet Dataset
- **Author**: Akash Nagaraj (2018)
- **Source URL**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **License**: GNU General Public License, Version 2.0 (GPL-2.0)
  https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
- **Notice**:
  A subset of 45,236 static letter images from this dataset was utilized for feature extraction and research benchmarking.
```

---

## 9. DANH SÁCH QUYẾT ĐỊNH CẦN CHỦ DỰ ÁN PHÊ DUYỆT (PROJECT OWNER APPROVAL)

Trước khi tiến hành phân phối mã nguồn hoặc mô hình ra cộng đồng, chủ dự án cần phê duyệt 5 quyết định chiến lược:

1. **Phê duyệt Quyết định 1 (Văn bản ghi công):** Phê duyệt bản nháp tại Mục 8 và cho phép khởi tạo tệp chính thức `THIRD_PARTY_NOTICES.md` tại thư mục gốc repository.
2. **Phê duyệt Quyết định 2 (Cấp phép dữ liệu tự thu):** Tuyên bố cấp phép mở chính thức (ví dụ MIT License cho trọng số và CC BY 4.0 cho dataset vector trích xuất) đối với 349 mẫu số cơ sở và 1.499 mẫu cử chỉ SPACE.
3. **Phê duyệt Quyết định 3 (Xử lý mô hình SPACE):** Lựa chọn phương án:
   - *Phương án A (Khuyến nghị):* Phê duyệt kế hoạch huấn luyện lại mô hình SPACE với tập âm NOT_SPACE sạch để tách rời hoàn toàn khỏi ABC.
   - *Phương án B:* Giữ nguyên mô hình hiện tại nhưng xếp vào diện nghiên cứu nội bộ, không phân phối công khai trọng số.
4. **Phê duyệt Quyết định 4 (Xử lý mô hình J/Z):** Xác nhận phân phối mô hình J/Z với nhãn ghi rõ giới hạn: *"Dành cho nghiên cứu học thuật và kiểm nghiệm kỹ thuật (Academic & Experimental Research Only)"* để phòng ngừa rủi ro về quyền hình ảnh cá nhân.
5. **Phê duyệt Quyết định 5 (Xử lý mô hình ABC):** Giữ mô hình ABC tĩnh theo chính sách **Source-Only Distribution** (không phân phối file nhị phân `.joblib` lên GitHub Releases) cho đến khi hoàn thành việc huấn luyện lại mô hình trên tập dữ liệu hoàn toàn minh bạch.

---

## 10. BẢNG TRẠNG THÁI PHÂN PHỐI TỪNG MODEL (FINAL MODEL DISTRIBUTION STATUS)

| Mô hình | Trạng thái kỹ thuật | Trạng thái bản quyền | Phân loại phân phối | Đánh giá & Điều kiện phát hành |
|---|:---:|:---:|:---:|---|
| **MediaPipe Hand Landmarker** (`hand_landmarker.task`) | `READY` | `READY` | **`READY`** | **ĐỦ ĐIỀU KIỆN PHÁT HÀNH CÔNG KHAI**. Bản quyền Apache 2.0 từ Google LLC rõ ràng. |
| **123 Digits Final** (`digits_final.joblib`) | `READY` | `REVIEW` | **`REVIEW` (Chờ tài liệu)** | **CÓ THỂ SẴN SÀNG NGAY**. Quyền sử dụng hợp pháp đã có đủ; chỉ cần phê duyệt và tạo file `THIRD_PARTY_NOTICES.md` và tuyên bố cấp phép cho 349 mẫu cơ sở để chuyển sang `READY`. |
| **J/Z Dynamic Pipeline** (`model_pipeline.joblib`) | `READY` | `REVIEW` | **`REVIEW` (Hạn chế)** | **CHỈ PHÂN PHỐI NỘI BỘ / NGHIÊN CỨU**. Bản quyền CC0 nhưng thiếu chứng từ Model Release của tình nguyện viên xuất hiện trong video. |
| **SPACE Gesture** (`final_space_model.joblib`) | `READY` | `REVIEW` | **`REVIEW` (Phụ thuộc)** | **TẠM DỪNG PHÂN PHỐI CÔNG KHAI**. Phụ thuộc $97,51\%$ mẫu Train vào tập âm ABC. Cần huấn luyện lại với tập âm sạch. |
| **ABC Letters MLP** (`uvw_medium_model.joblib`) | `READY` | `UNKNOWN` | **`BLOCKED`** | **KHÔNG ĐƯỢC PHÁT HÀNH CÔNG KHAI**. $46,61\%$ dữ liệu (41.676 ảnh color-coded) hoàn toàn không rõ nguồn gốc và $50,59\%$ dữ liệu mang tính chất copyleft của GPL 2.0. Áp dụng chính sách Source-Only. |

---
*Báo cáo Stage 2 được lập hoàn toàn khách quan, dựa trên các bằng chứng tài liệu và dữ liệu nhị phân trên đĩa của workspace D:\NhanDienCuChi.*
