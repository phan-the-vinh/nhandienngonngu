# BÁO CÁO RÀ SOÁT VÀ ĐÁNH GIÁ PHÁP LÝ TOÀN DIỆN (FINAL REVIEW LICENSE AUDIT): NGUỒN GỐC DỮ LIỆU VÀ KHẢ NĂNG PHÂN PHỐI MODEL

**Dự án:** Hệ thống Nhận diện Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)  
**Workspace:** `D:\NhanDienCuChi`  
**Repository mục tiêu:** [phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)  
**Branch:** `main` | **HEAD tham chiếu:** `dce96c2`  
**Ngày thực hiện kiểm toán:** 17/11/2026 (Phiên rà soát pháp lý cuối cùng)  
**Chế độ:** STRICT FORENSIC DOCUMENT AUDIT (Chỉ rà soát tài liệu, không sửa mã nguồn, không sửa model, không sửa dataset, không commit/push, không tạo GitHub Release).

---

## 1. TÓM TẮT ĐIỀU CHỈNH VÀ CHUẨN HÓA THUẬT NGỮ PHÁP LÝ

Phiên rà soát cuối cùng (Final Review) tập trung chuẩn hóa các nhận định pháp lý, loại bỏ các từ ngữ tuyệt đối hóa (như "tuyệt đối cấm", "hoàn toàn hợp pháp", "tự động chịu GPL"), đồng thời phân định rạch ròi giữa các phạm trù quyền:

1. **Chuẩn hóa nhận định về GPL 2.0 của Nhánh ABC:**
   - Xác nhận bằng chứng trực tiếp: Bộ dữ liệu `grassknoted/asl-alphabet` trên Kaggle mang giấy phép **GPL 2.0 (GNU General Public License v2.0)**.
   - Hiệu chỉnh kết luận: **Không kết luận mô hình mạng nơ-ron MLP tự động chịu ràng buộc copyleft của GPL 2.0 hoặc tự động bị cấm phân phối**. Mối quan hệ giữa dữ liệu huấn luyện GPL và trọng số mạng nơ-ron là một câu hỏi pháp lý mở chưa có án lệ thống nhất trong luật bản quyền quốc tế. Rủi ro của nhánh ABC chủ yếu đến từ **41.676 mẫu ảnh color-coded hoàn toàn không rõ nguồn gốc** (`LICENSE_UNKNOWN`).
2. **Làm rõ bản chất pháp lý của Nhánh 123:**
   - Phân biệt rõ: Dự án **CÓ QUYỀN HỢP PHÁP** sử dụng dữ liệu Ardamavi (Apache 2.0) và ASL-HG (CC BY 4.0) để huấn luyện mô hình.
   - Việc repository hiện tại chưa có file `THIRD_PARTY_NOTICES.md` là **thiếu sót về mặt tài liệu tuân thủ (compliance documentation gap)**, chứ không phải là không có quyền sử dụng dữ liệu.
3. **Phân định ba phạm trù quyền đối với Nhánh J/Z (SigNN Team):**
   - Phân biệt rạch ròi giữa **Bản quyền (Copyright)**, **Quyền riêng tư (Privacy Rights)** và **Quyền hình ảnh cá nhân (Publicity Rights)**.
   - Khẳng định: Nhãn CC0 trên Kaggle chỉ từ bỏ bản quyền, không bao gồm quyền riêng tư/hình ảnh cá nhân theo Điều 2(d) của CC0.
   - Đính chính: Bỏ nhận định rằng dataset "chỉ được dùng cho học thuật nội bộ". Giấy phép CC0 không có điều khoản giới hạn này; "học thuật nội bộ" là phạm vi tự giới hạn thận trọng của dự án do thiếu chứng từ đồng thuận của người quay video.
4. **Phân tích rủi ro phụ thuộc dữ liệu của Nhánh SPACE:**
   - Giữ nguyên kết luận: Mô hình SPACE sử dụng lớp âm NOT_SPACE gồm 62.591 mẫu từ ABC và 549 mẫu từ 123.
   - Đưa ra 2 hướng xử lý thực tế: (1) Giải quyết pháp lý nguồn dữ liệu ABC; hoặc (2) Huấn luyện lại mô hình SPACE với tập âm NOT_SPACE được xây dựng từ dữ liệu có nguồn gốc xác định.

---

## 2. KIỂM TOÁN CHI TIẾT NHÁNH ABC (STATIC LETTERS)

### 2.1. Bằng chứng trực tiếp về dữ liệu và giấy phép
Căn cứ tệp đóng băng dữ liệu `ai/dataset/preprocessing_frozen_v3/final_freeze_manifest.json` và phân loại từ `sample_group_map_frozen_v3.csv` (tổng 89.415 mẫu):

1. **Nhóm Kaggle ASL Alphabet (45.236 mẫu — $50.59\%$):**
   - *URL chính thức:* [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Tác giả: Akash Nagaraj).
   - *Quy ước tệp:* `A1.jpg`..`A3000.jpg`, `B1.jpg`..`B3000.jpg`...
   - *Giấy phép công bố chính thức trên Kaggle:* **GNU General Public License v2.0 (GPL-2.0)** (`VERIFIED` trực tiếp từ metadata trang dataset Kaggle).
   - *Điều khoản cốt lõi của GPL 2.0:* Quy định về quyền phân phối tác phẩm và các tác phẩm phái sinh (works based on the Program), yêu cầu các tác phẩm phái sinh khi phân phối phải áp dụng cùng điều khoản GPL mà không thu thêm phí bản quyền.
2. **Nhóm ảnh phân đoạn màu (Color-coded Segmentation) (41.676 mẫu — $46.61\%$):**
   - *Quy ước tệp:* `color_0_*`, `color_10_0460 (3).png`...
   - *Tình trạng:* **`LICENSE_UNKNOWN`**. Toàn bộ kho lưu trữ lịch sử `dulieu/chucai` không lưu lại siêu dữ liệu, URL tải ban đầu, tác giả hoặc điều khoản cấp phép của nhóm ảnh này.
3. **Nhóm phân đoạn bàn tay phụ trợ và ảnh web khác (2.503 mẫu — $2.80\%$):**
   - *Quy ước tệp:* `hand1_c_bot_seg`, `A0 (4).jpg`... Tình trạng: **`LICENSE_UNKNOWN`**.

### 2.2. Phân tích pháp lý: License Dataset vs. License Model Weights
- **Học thuyết tác phẩm phái sinh đối với Machine Learning:**
  - *Quan điểm 1 (Biểu diễn thống kê độc lập / Fair Use):* Quá trình huấn luyện trích xuất các mẫu hình thống kê (statistical patterns), chuyển hóa 63 tọa độ landmark thành ma trận trọng số toán học float32 ($1 	imes 63 	o 64 	o 32 	o 24$). Trọng số không chứa bản sao nhị phân hay biểu đạt nghệ thuật của hình ảnh gốc, do đó theo quan điểm của nhiều chuyên gia luật công nghệ, weights không tự động bị coi là tác phẩm phái sinh chịu sự chi phối của GPL 2.0.
  - *Quan điểm 2 (Hiệu ứng Copyleft dây chuyền):* Một số lập luận cho rằng nếu dữ liệu đầu vào chịu GPL 2.0 được sử dụng như một thành phần thiết yếu để tạo ra artifact phần mềm (tương tự mã nguồn biên dịch ra binary), thì sản phẩm phái sinh phải tuân thủ GPL 2.0.
- **Rủi ro cốt lõi của Nhánh ABC:**
  - Rủi ro lớn nhất **không chỉ nằm ở GPL 2.0**, mà nằm ở **41.676 mẫu ảnh color-coded hoàn toàn không rõ nguồn gốc** ($46.61\%$).
  - Nếu chủ sở hữu bản quyền của 41.676 mẫu ảnh này đưa ra khiếu nại, mô hình được huấn luyện trực tiếp trên dữ liệu của họ có nguy cơ bị yêu cầu thu hồi hoặc đối mặt với thông báo vi phạm DMCA.
- **Kết luận:** Chưa đủ căn cứ để khẳng định model ABC tự động chịu GPL 2.0, nhưng do thiếu nguồn gốc của $46.61\%$ dữ liệu, mô hình `uvw_medium_model.joblib` được xếp loại **`LICENSE_UNKNOWN`** và **chưa đủ căn cứ xác minh để phân phối công khai**.

---

## 3. KIỂM TOÁN CHI TIẾT NHÁNH 123 (STATIC DIGITS)

### 3.1. Đối soát số lượng và nguồn dữ liệu thực tế
Căn cứ tệp manifest `data/metadata/digits/digits_final_repro/training_manifest_digits_final.json`:
Tập Train gồm 549 mẫu sử dụng trực tiếp để tối ưu hóa trọng số mô hình `digits_final.joblib`, kết hợp từ đúng 3 nguồn:
1. **Arda Mavi Digits:** **170 mẫu** (Lớp 0,1,2,3,5,7,9: 20 mẫu/lớp; Lớp 4,6,8: 10 mẫu/lớp).
2. **ASL-HG:** **30 mẫu** (Lớp 4,6,8: 10 mẫu/lớp).
3. **Dữ liệu tự thu thập nội bộ (base_train_v1):** **349 mẫu** (Lớp 0: 34 mẫu; Lớp 1–9: 35 mẫu/lớp).

### 3.2. Xác minh giấy phép và nghĩa vụ Attribution
1. **Ardamavi Sign Language Digits Dataset:**
   - *URL:* [https://github.com/ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset) (Kaggle: `ardamavi/sign-language-digits-dataset`).
   - *Tác giả:* Arda Mavi và Zeynep Dikle (2020).
   - *Giấy phép:* **Apache License 2.0** (`VERIFIED` qua tệp `LICENSE` trên GitHub).
   - *Quyền sử dụng:* Cho phép sử dụng tự do cho nghiên cứu, thương mại, chỉnh sửa và tạo sản phẩm phái sinh.
   - *Nghĩa vụ Attribution (Section 4):*
     * Phải gửi kèm bản sao của giấy phép Apache 2.0 khi phân phối.
     * Phải giữ lại toàn bộ thông báo bản quyền, nhãn hiệu và thông báo ghi công.
     * Trích dẫn tài liệu công bố: `Mavi, A., (2020), "A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits", arXiv:2011.08927 [cs.CV]`.
2. **ASL-HG (American Sign Language Hand Gesture Image Dataset):**
   - *DOI:* [10.17632/j4y5w2c8w9.1](https://doi.org/10.17632/j4y5w2c8w9.1) (Mendeley Data / Data in Brief, Elsevier).
   - *Tác giả:* Md. Famidul Islam Pranto, Md. Rifatul Islam, Md. Ali Akbor, Nabonita Ghosh, Md. Rahatun Alam, Sudipto Chaki, Md. Masudul Islam (2023).
   - *Phiên bản:* **Version 1** (`VERIFIED` qua tệp trích xuất `tools/extract_asl_hg_digits_v2.py` với regex `P\d+_[468]_\d+` tương ứng 10 tình nguyện viên).
   - *Giấy phép:* **Creative Commons Attribution 4.0 International (CC BY 4.0)** (`VERIFIED` trên Mendeley Data).
   - *Quyền sử dụng:* Cho phép chia sẻ và tạo sản phẩm chuyển thể (Adapted Material, bao gồm cả mô hình học máy) cho mọi mục đích, kể cả thương mại.
   - *Nghĩa vụ Attribution (Section 3):*
     * Phải ghi nhận danh tính các tác giả được chỉ định.
     * Phải cung cấp liên kết đến giấy phép CC BY 4.0.
     * Phải cung cấp đường dẫn URI đến nguồn dữ liệu gốc trên Mendeley Data.
     * Phải ghi chú rõ ràng các sửa đổi đã thực hiện (dự án chỉ trích xuất tọa độ landmark của 30 mẫu cho 3 chữ số 4, 6, 8).
3. **Dữ liệu tự thu thập nội bộ (349 mẫu cơ sở):**
   - Do tác giả đề tài tự thu thập trực tiếp (`ai/dataset/kytuso/datatrain/train_v1.csv`). Chưa có tuyên bố phát hành giấy phép mở.

### 3.3. Phân biệt quyền sử dụng và nghĩa vụ ghi công
- Dự án **CÓ ĐẦY ĐỦ QUYỀN HỢP PHÁP** sử dụng cả hai bộ dữ liệu Ardamavi và ASL-HG để huấn luyện mô hình học máy theo điều khoản của Apache 2.0 và CC BY 4.0.
- Việc repository hiện tại chưa có file `THIRD_PARTY_NOTICES.md` là **thiếu sót về mặt tài liệu tuân thủ (compliance documentation gap)**.
- **Công việc tác giả cần hoàn thành trước khi phân phối:**
  1. Tạo file `THIRD_PARTY_NOTICES.md` thực hiện đầy đủ 4 nghĩa vụ ghi công cho Arda Mavi và 4 nghĩa vụ cho ASL-HG.
  2. Bổ sung tuyên bố cấp phép mở của tác giả đối với 349 mẫu số cơ sở.
  3. Sau khi hoàn thành 2 bước trên, mô hình `digits_final.joblib` đủ điều kiện chuyển từ `LICENSE_REVIEW` sang `READY`.

---

## 4. KIỂM TOÁN CHI TIẾT NHÁNH J/Z (DYNAMIC GESTURES)

### 4.1. Phân định ba phạm trù quyền: Bản quyền, Quyền riêng tư và Quyền hình ảnh
1. **Bản quyền (Copyright):**
   - Quyền của người sáng tạo đối với tác phẩm video như một sản phẩm trí tuệ.
   - Nhãn hiển thị trên trang Kaggle [ASL Sign Language Alphabet Videos [J, Z]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z) là **`CC0 1.0 Universal: Public Domain`** (`PARTIALLY_VERIFIED` qua ảnh chụp màn hình `signnteam_kaggle_cc0_license.png`, SHA-256: `16e7e727...`).
   - Tuyên bố CC0 từ bỏ tối đa các quyền tác giả (copyright) theo luật định.
2. **Quyền riêng tư (Privacy Rights) & Quyền hình ảnh cá nhân (Publicity Rights):**
   - Theo Điều 2(d) của văn bản pháp lý Creative Commons CC0 1.0 Universal:
     > *"No trademark or patent rights held by Affirmer are waived, abandoned, surrendered, licensed or otherwise affected by this document. Affirmers also do not waive or affect any rights of publicity or privacy held by Affirmer or any other person."*
   - Video SigNN ghi hình nhận diện khuôn mặt, biểu cảm và hình thể của các cá nhân tình nguyện viên. Tuyên bố CC0 không đồng nghĩa với việc các tình nguyện viên đã từ bỏ quyền riêng tư hoặc cho phép sử dụng hình ảnh của họ cho mục đích thương mại hoặc phát hành rộng rãi.
   - Siêu dữ liệu nội bộ `signnteam_source_rights_evidence_v1.json` ghi nhận rõ ràng: `"signer_session_metadata: ABSENT; source-video split is not signer-independent"`.
3. **Đính chính phạm vi sử dụng:**
   - Giấy phép CC0 trên Kaggle **không chứa điều khoản giới hạn "chỉ dùng cho học thuật"**.
   - Cụm từ "học thuật nội bộ" là đánh giá rủi ro thận trọng của nhóm nghiên cứu nhằm phòng ngừa rủi ro về quyền hình ảnh cá nhân, không phải là điều khoản pháp lý ràng buộc từ nguồn dữ liệu.
4. **Chứng từ còn thiếu:**
   - Cần văn bản thỏa thuận tham gia / giải phóng quyền hình ảnh (Model Release / Participant Consent Form) từ SigNN Team xác nhận các tình nguyện viên trong video đã đồng thuận cho phép phân phối dữ liệu và các sản phẩm phái sinh.
- **Kết luận:** Mô hình J/Z được xếp loại **`LICENSE_REVIEW`** do còn thiếu chứng từ xác minh quyền hình ảnh cá nhân của người xuất hiện trong video gốc.

---

## 5. KIỂM TOÁN CHI TIẾT NHÁNH SPACE (GESTURE SPACE)

### 5.1. Mối quan hệ phụ thuộc dữ liệu của mô hình nhị phân
Căn cứ tệp kiểm toán `staging/space_training_readiness_audit_phase3_9_4_20261106/dataset_integrity_audit.csv` và metadata `staging/space_training_phase3_9_15/space_model_metadata.json`:

Mô hình SPACE (`final_space_model.joblib`) là một bộ phân loại nhị phân Logistic Regression ($C=10.0$) hoạt động trên không gian 63 đặc trưng landmark. Tập huấn luyện chính thức gồm **64.189 mẫu**:

```
Tập Train SPACE (64.189 mẫu) = 1.049 mẫu dương (SPACE) + 63.140 mẫu âm (NOT_SPACE)
- Mẫu dương: 1.049 mẫu (1.63%) do tác giả tự thu thập (Phase 3.2).
- Mẫu âm:   63.140 mẫu (98.37%), gồm:
             ├── 62.591 mẫu chữ cái (ABC) từ ai/dataset/train_validation_test_split_v1/train.csv
             └── 549 mẫu chữ số (123) từ data/processed/digits/digits_final_train_frozen.csv
```

### 5.2. Phân tích rủi ro phụ thuộc dữ liệu
1. **Bản chất toán học của thuật toán phân loại:**
   - Bộ phân loại Logistic Regression tìm kiếm siêu phẳng (hyperplane) tối ưu hóa hàm mất mát nhị phân. Các hệ số trọng số ($1 	imes 63$) và hệ số chặn (intercept) được xác định trực tiếp bởi tọa độ của cả mẫu dương lẫn mẫu âm.
   - Do đó, trọng số của mô hình SPACE không chỉ đại diện cho cử chỉ SPACE, mà đại diện cho **ranh giới phân biệt giữa cử chỉ SPACE và các cử chỉ chữ cái/chữ số**.
2. **Rủi ro pháp lý dây chuyền:**
   - Vì lớp âm (NOT_SPACE) phụ thuộc tới $97.5\%$ vào tập dữ liệu ABC (trong đó có 41.676 mẫu chưa rõ nguồn gốc và 45.236 mẫu GPL 2.0), mô hình SPACE thừa hưởng toàn bộ những điểm chưa chắc chắn về pháp lý của nhánh ABC.
   - Tác giả không thể khẳng định có toàn quyền cấp phép độc lập cho model SPACE chỉ dựa trên việc tự thu thập 1.499 mẫu dương.

### 5.3. Hai hướng xử lý đề xuất cho tác giả
- **Hướng 1 (Hướng pháp lý):** Tiến hành truy vết, xác minh quyền sử dụng và làm rõ giấy phép của tập dữ liệu ABC để giải quyết tính hợp lệ của dữ liệu lớp âm.
- **Hướng 2 (Hướng kỹ thuật — Khuyến nghị):** Huấn luyện lại (retrain) mô hình SPACE bằng cách xây dựng lại tập âm NOT_SPACE từ các nguồn dữ liệu có nguồn gốc rõ ràng và giấy phép mở xác định (hoặc tác giả tự quay thêm các cử chỉ tay thông thường không phải SPACE để làm tập âm). Phương án này giúp tách rời hoàn toàn mô hình SPACE khỏi sự phụ thuộc vào dữ liệu chưa rõ nguồn gốc của ABC.
- **Kết luận:** Mô hình SPACE được xếp loại **`LICENSE_REVIEW`** do rủi ro phụ thuộc dữ liệu từ nhánh ABC.

---

## 6. KIỂM TOÁN MEDIAPIPE HAND LANDMARKER

### 6.1. Bằng chứng và điều khoản
- **Tệp phân phối:** [models/mediapipe/hand_landmarker.task](file:///D:/NhanDienCuChi/models/mediapipe/hand_landmarker.task) (7.819.105 bytes, SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`).
- **Nguồn phát hành:** Google LLC.
- **URL tải chính thức:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.
- **Giấy phép:** **Apache License 2.0** (`VERIFIED` qua tài liệu Google MediaPipe Solutions).
- **Quyền sử dụng:** Phép cấp quyền Apache 2.0 bao quát cả mã nguồn SDK và gói mô hình nhị phân `.task`, cho phép tích hợp, chỉnh sửa, sử dụng thương mại và tái phân phối.
- **Nghĩa vụ tuân thủ:** Phải giữ nguyên thông báo bản quyền Apache 2.0 của Google LLC.
- **Kết luận:** Thành phần duy nhất đạt trạng thái **`READY`** cho việc phân phối.

---

## 7. BẢNG KẾT LUẬN TỔNG HỢP KHẢ NĂNG PHÂN PHỐI (DISTRIBUTION MATRIX)

Áp dụng phân định đa chiều và các trạng thái chuẩn hóa: `READY`, `LICENSE_REVIEW`, `LICENSE_UNKNOWN`, `TECHNICALLY_READY`, `NOT_VERIFIED`.

| Thành phần | Nguồn dữ liệu | Quyền sử dụng dữ liệu | Quyền phân phối dữ liệu gốc | Quyền phân phối Model phái sinh | Nghĩa vụ Attribution & Rủi ro quyền hình ảnh | Trạng thái kỹ thuật | Trạng thái phân phối | Hành động tiếp theo |
|---|---|:---:|:---:|:---:|---|:---:|:---:|---|
| **MediaPipe Landmarker** | Google AI Edge internal dataset | Có | Có | Có | Giữ nguyên thông báo bản quyền Google Apache 2.0 | `TECHNICALLY_READY` | **`READY`** | Duy trì trong repository hoặc phân phối qua script tải tự động `download_models.py` từ Google CDN. |
| **123 Digits Final** (`digits_final.joblib`) | 349 mẫu tác giả + 170 Arda Mavi + 30 ASL-HG | Có | Có (cho Ardamavi & ASL-HG); Cần công bố (cho 349 base) | Có quyền tạo và phân phối model | Bắt buộc đính kèm văn bản Attribution cho Arda Mavi và ASL-HG; chưa phát sinh rủi ro quyền hình ảnh | `TECHNICALLY_READY` | **`LICENSE_REVIEW`** | Soạn thảo `THIRD_PARTY_NOTICES.md` hoàn tất nghĩa vụ ghi công; tác giả xác nhận điều khoản cấp phép cho 349 mẫu cơ sở. |
| **J/Z Dynamic Pipeline** (`model_pipeline.joblib`) | 640 video SigNN ASL Video Dataset | Có | Có (theo CC0 copyright) | Cần đánh giá rủi ro hình ảnh | Thiếu chứng từ đồng thuận quyền hình ảnh cá nhân (publicity/privacy) của người xuất hiện trong video | `TECHNICALLY_READY` | **`LICENSE_REVIEW`** | Xác minh với SigNN Team về thỏa thuận tình nguyện viên; giữ trong phạm vi nghiên cứu khi chưa có kết luận pháp lý. |
| **SPACE Gesture** (`final_space_model.joblib`) | 1.499 mẫu dương (tác giả) + 90.314 mẫu âm (từ ABC & 123) | Có cho mẫu dương; Phụ thuộc tập ABC cho mẫu âm | Không phân phối dữ liệu âm | Phụ thuộc rủi ro dữ liệu âm | Thừa hưởng rủi ro pháp lý từ tập âm ABC (41k ảnh chưa rõ nguồn); không thể đơn phương cấp phép MIT | `TECHNICALLY_READY` | **`LICENSE_REVIEW`** | Huấn luyện lại model SPACE với tập âm NOT_SPACE được xây dựng từ nguồn dữ liệu sạch/xác định rõ ràng. |
| **ABC Letters MLP** (`uvw_medium_model.joblib`) | 45.236 Kaggle ASL + 41.676 color-coded + 2.503 mẫu khác | Có quyền nghiên cứu | Không được phân phối tập 41k ảnh | Chưa đủ căn cứ xác minh | 41.676 ảnh color-coded chưa xác định nguồn gốc; tranh luận pháp lý mở về phạm vi copyleft của GPL 2.0 | `TECHNICALLY_READY` | **`LICENSE_UNKNOWN`** | Chưa đủ căn cứ xác minh để phân phối công khai. Cần loại bỏ dữ liệu không rõ nguồn và huấn luyện lại trên tập dữ liệu mở xác định. |

---

## 8. DANH SÁCH VẤN ĐỀ CÒN TỒN TẠI VÀ CHƯA THỂ XÁC MINH

1. **Nguồn gốc của 41.676 mẫu ảnh color-coded (ABC):** Chưa tìm thấy thông tin tác giả, trang web ban đầu hoặc giấy phép sử dụng. Đây là điểm nghẽn pháp lý lớn nhất của hệ thống.
2. **Chứng từ đồng thuận quyền hình ảnh của video SigNN (J/Z):** Thiếu thỏa thuận giải phóng quyền hình ảnh (Model Release) của các cá nhân xuất hiện trong video.
3. **Phạm vi tác động của GPL 2.0 đối với trọng số mạng nơ-ron:** Hiện chưa có án lệ tư pháp chính thức xác định liệu ma trận trọng số trích xuất từ dữ liệu GPL 2.0 có bắt buộc phải chịu sự chi phối của GPL 2.0 hay không.
4. **Văn bản cấp phép của tác giả đối với dữ liệu nội bộ:** Cả 349 mẫu số cơ sở (123) và 1.499 mẫu cử chỉ SPACE (SPACE) đều do tác giả tự thu thập nhưng chưa có văn bản công bố giấy phép chính thức trong repository.

---

## 9. DANH SÁCH CÔNG VIỆC TÁC GIẢ CẦN HOÀN THÀNH TRƯỚC KHI PHÁT HÀNH

- [ ] **Việc 1 (Nhánh 123):** Soạn thảo và thêm file `THIRD_PARTY_NOTICES.md` vào thư mục gốc repository để hoàn thành nghĩa vụ Attribution cho Arda Mavi (Apache 2.0) và Mendeley ASL-HG (CC BY 4.0), đồng thời công bố giấy phép cho 349 mẫu số cơ sở.
- [ ] **Việc 2 (Nhánh SPACE):** Lựa chọn một trong hai hướng xử lý: (a) Xác minh bản quyền dữ liệu ABC, hoặc (b) Huấn luyện lại mô hình SPACE với tập âm NOT_SPACE sạch để tách rời hoàn toàn sự phụ thuộc.
- [ ] **Việc 3 (Nhánh J/Z):** Liên hệ hoặc kiểm tra tài liệu bổ sung từ SigNN Team về chứng từ quyền hình ảnh của tình nguyện viên trước khi cân nhắc phát hành rộng rãi ngoài mục đích nghiên cứu.
- [ ] **Việc 4 (Nhánh ABC):** Đánh giá phương án huấn luyện lại mô hình ABC trên tập dữ liệu hoàn toàn mở có giấy phép xác định (như Kaggle ASL với license rõ ràng hoặc tự thu thập mới).
- [ ] **Việc 5 (Kỹ thuật):** Chạy thử nghiệm clean-machine integration test trên một máy tính độc lập để xác minh khả năng cài đặt và chạy tự động thông qua `tools/download_models.py`.

---

## 10. KẾT LUẬN RÕ RÀNG VỀ TRẠNG THÁI PHÁT HÀNH TỪNG MODEL

1. **Model duy nhất ĐỦ ĐIỀU KIỆN phát hành công khai ngay:**
   - **`MediaPipe Hand Landmarker` (`hand_landmarker.task`):** Đầy đủ căn cứ bản quyền Apache 2.0 từ Google LLC.
2. **Model CÓ THỂ ĐỦ ĐIỀU KIỆN phát hành sau khi hoàn tất thủ tục tài liệu:**
   - **`123 Digits Final` (`digits_final.joblib`):** Quyền sử dụng dữ liệu đã rõ ràng. Chỉ cần bổ sung file `THIRD_PARTY_NOTICES.md` thực hiện nghĩa vụ Attribution và công bố giấy phép cho dữ liệu cơ sở.
3. **Các model CHƯA ĐỦ CĂN CỨ XÁC MINH để phát hành công khai theo bằng chứng hiện có:**
   - **`ABC Static Letters` (`uvw_medium_model.joblib`):** Chưa đủ căn cứ xác minh do $46.61\%$ dữ liệu huấn luyện không rõ nguồn gốc (`LICENSE_UNKNOWN`) và $50.59\%$ dữ liệu mang tính chất copyleft của GPL 2.0.
   - **`SPACE Gesture` (`final_space_model.joblib`):** Chưa đủ căn cứ xác minh do lớp âm phụ thuộc $98.37\%$ vào tập dữ liệu ABC chưa rõ bản quyền.
   - **`J/Z Dynamic Pipeline` (`model_pipeline.joblib`):** Chưa đủ căn cứ xác minh để phát hành công khai rộng rãi do thiếu chứng từ quyền riêng tư/hình ảnh cá nhân của các tình nguyện viên xuất hiện trong video SigNN.

---
*Báo cáo kiểm toán cuối cùng được hoàn thành trên nguyên tắc trung thực khoa học, thận trọng pháp lý và bảo vệ an toàn cho kho mã nguồn mở của tác giả.*
