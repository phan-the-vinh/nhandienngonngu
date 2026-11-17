# BÁO CÁO KIỂM TRA ĐỘ SẴN SÀNG PHÂN PHỐI DỰ ÁN VÀ TÍNH NHẤT QUÁN DỮ LIỆU (FINAL DISTRIBUTION READINESS & DATA CONSISTENCY AUDIT)

**Dự án:** Hệ thống Nhận diện Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)  
**Workspace:** `D:\NhanDienCuChi`  
**Repository:** [phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)  
**Branch:** `main` | **HEAD tham chiếu:** `dce96c2`  
**Ngày kiểm toán:** 17/11/2026  
**Chế độ:** READ-ONLY FORENSIC CONSISTENCY AUDIT (Chỉ kiểm tra, không sửa source code, không sửa model, không commit/push).

---

## 1. KẾT QUẢ ĐỐI SOÁT SỐ LIỆU CHI TIẾT (DATA AUDIT MATRIX)

Quá trình kiểm toán đã đối soát trực tiếp từng tệp manifest, bảng phân chia dữ liệu (split files) và dữ liệu huấn luyện nhị phân thực tế trên đĩa:

| Phân hệ / Nhánh | Chỉ số đối soát | Số liệu ghi nhận | Tệp nguồn / Manifest đối chứng | Trạng thái xác minh |
|---|---|---:|---|:---:|
| **ABC (Frozen v3)** | Tổng mẫu vũ trụ sạch | **89.415** | `ai/dataset/preprocessing_frozen_v3/final_freeze_manifest.json` | `VERIFIED` |
| **ABC (Split v1)** | Tập Train cơ sở (70%) | **62.591** | `ai/dataset/train_validation_test_split_v1/train.csv` (split_summary.json) | `VERIFIED` |
| **ABC (Split v1)** | Tập Validation (15%) | **13.412** | `ai/dataset/train_validation_test_split_v1/validation.csv` | `VERIFIED` |
| **ABC (Split v1)** | Tập Test đóng băng (15%) | **13.412** | `ai/dataset/train_validation_test_split_v1/test.csv` | `VERIFIED` |
| **ABC (Model Train)**| Tập Train thực tế fit mô hình `uvw_medium` | **62.648** | `ai/dataset/mlp_uvw_training_data_v1/train_uvw_v1.csv` (manifest.json) | `VERIFIED` |
| **123 (Digits Final)** | Tổng mẫu toàn nhánh | **899** | `data/metadata/digits/digits_final_repro/training_manifest_digits_final.json` | `VERIFIED` |
| **123 (Digits Final)** | Tập Train chính thức (fit trọng số) | **549** | `data/processed/digits/digits_final_train_frozen.csv` (349 base + 170 Arda Mavi + 30 ASL-HG) | `VERIFIED` |
| **123 (Digits Final)** | Validation Reference | **100** | `data/splits/digits/validation_reference.csv` (10 mẫu/lớp) | `VERIFIED` |
| **123 (Digits Final)** | Regression Reference | **250** | `data/splits/digits/regression_reference.csv` (Gốc từ `test_v1.csv`) | `VERIFIED` |
| **J/Z (Dynamic)** | Tổng video nguồn ban đầu | **643** | `data/experiments/dynamic_jz_binary_baseline_v1/training_manifest.json` | `VERIFIED` |
| **J/Z (Dynamic)** | Video bị loại do lỗi MediaPipe | **3** | `J/99.avi`, `J/103.avi`, `J/250.avi` | `VERIFIED` |
| **J/Z (Dynamic)** | Chuỗi hợp lệ đưa vào huấn luyện | **640** | 265 chuỗi J + 375 chuỗi Z | `VERIFIED` |
| **J/Z (Dynamic)** | Train (80%) / Val (10%) / Test (10%) | **512 / 64 / 64** | `data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json` | `VERIFIED` |
| **SPACE (Dataset)** | Tổng mẫu vũ trụ toàn bộ các tập | **91.813** | `staging/space_training_readiness_audit_phase3_9_4_20261106/dataset_integrity_audit.csv` | `VERIFIED` |
| **SPACE (Train)** | Mẫu dương (SPACE = 1) | **1.049** | `space_landmarks.csv` (assigned_split == 'train') | `VERIFIED` |
| **SPACE (Train)** | Mẫu âm (NOT_SPACE = 0) | **63.140** | 62.591 (từ ABC `train.csv`) + 549 (từ 123 `digits_final_train_frozen.csv`) | `VERIFIED` |
| **SPACE (Train)** | Tổng số mẫu Train chính thức | **64.189** | `space_model_metadata.json` (Phase 3.9.15) | `VERIFIED` |
| **SPACE (Train)** | Tỷ lệ mẫu âm trong tập Train | **98.37%** | $63.140 / 64.189 = 98.36576...\%$ | `VERIFIED` |
| **SPACE (Train)** | Tỷ lệ mẫu dương trong tập Train | **1.63%** | $1.049 / 64.189 = 1.63423...\%$ | `VERIFIED` |
| **SPACE (Val / Test)**| Mẫu dương Val / Test | **225 / 225** | Mỗi tập gồm 225 mẫu dương tự thu | `VERIFIED` |
| **SPACE (Val / Test)**| Mẫu âm Val / Test | **13.512 / 13.662** | Val âm: 13.412 ABC + 100 Digits; Test âm: 13.412 ABC + 250 Digits | `VERIFIED` |

---

## 2. NGUYÊN NHÂN CÁC SAI LỆCH VÀ MÂU THUẪN ĐÃ ĐƯỢC LÀM RÕ

### 2.1. Làm rõ sai lệch 57 mẫu giữa 62.648 và 62.591 trong Nhánh ABC
- **Vấn đề đặt ra:** Vì sao trong một số báo cáo cũ ghi nhận số mẫu huấn luyện ABC là **62.648**, trong khi báo cáo kiểm toán Vòng 2 lại xuất hiện con số **62.591**? Có sự mất mát dữ liệu, lọc trùng hay thay đổi phân chia tập (split leakage) không?
- **Nguyên nhân chính xác được xác minh từ tệp gốc:**
  1. Theo tệp phân chia tập ban đầu `ai/dataset/train_validation_test_split_v1/split_summary.json`, toàn bộ 89.415 mẫu sạch của ABC được chia theo tỷ lệ 70/15/15:
     - **Train cơ sở (train_v1):** Đúng **62.591 mẫu**.
     - **Validation cơ sở:** Đúng **13.412 mẫu**.
     - **Test cơ sở:** Đúng **13.412 mẫu**.
  2. Trong quá trình thử nghiệm camera ở giai đoạn sau, mô hình nhận diện tĩnh gặp hiện tượng nhầm lẫn trầm trọng giữa 3 chữ cái U, V và W. Để giải quyết, nhóm nghiên cứu đã lập tệp bổ sung `ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv` gồm đúng **57 mẫu tăng cường** (15 mẫu OLD_U, 15 mẫu OLD_V, 15 mẫu V_TO_W, 12 mẫu V_TO_U).
  3. Căn cứ tệp `ai/dataset/mlp_uvw_training_data_v1/manifest.json`:
     $$	ext{train\_new} = 	ext{train\_v1 (62.591)} + 	ext{supplement (57)} = \mathbf{62.648\ 	ext{mẫu}}$$
     Tập này được lưu thành `train_uvw_v1.csv` và dùng để huấn luyện mô hình production `uvw_medium_model.joblib`.
  4. Khi xây dựng bộ dữ liệu nhị phân cho nhánh SPACE ở Phase 3.9.4, kịch bản tạo dữ liệu đã liên kết trực tiếp vào tệp phân vùng cơ sở `ai/dataset/train_validation_test_split_v1/train.csv` (chứa **62.591 mẫu**), chứ không lấy tệp tăng cường `train_uvw_v1.csv` (62.648 mẫu).
- **Kết luận:**
  - Không có bất kỳ sự mất mát dữ liệu, sai sót thuật toán hay rò rỉ phân vùng nào.
  - Con số **62.648** là kích thước tập huấn luyện thực tế của mô hình chữ cái ABC `uvw_medium`.
  - Con số **62.591** là kích thước tập Train cơ sở của ABC đồng thời là số mẫu âm lấy từ ABC để nạp vào tập Train của SPACE.
  - Cả hai con số đều chính xác $100\%$ theo đúng ngữ cảnh của từng artifact.

### 2.2. Xác minh tính chính xác của tập Train SPACE (64.189 mẫu) và tỷ lệ 98.37%
- **Công thức đối soát từ dữ liệu gốc:**
  $$	ext{Train SPACE} = 1.049\ (	ext{Dương SPACE}) + 62.591\ (	ext{Âm ABC}) + 549\ (	ext{Âm Digits}) = \mathbf{64.189\ 	ext{mẫu}}$$
- Tỷ lệ mẫu âm trong tập Train:
  $$rac{63.140}{64.189} = 0.9836576... pprox \mathbf{98.37\%}$$
- Tỷ lệ mẫu dương trong tập Train:
  $$rac{1.049}{64.189} = 0.016342... pprox \mathbf{1.63\%}$$
- Tỷ lệ mẫu âm ABC chiếm trong tổng số mẫu âm:
  $$rac{62.591}{63.140} = 0.991305... pprox \mathbf{99.13\%}$$
- **Kết luận:** Tỷ lệ $98.37\%$ và tổng số 64.189 mẫu Train đã được xác minh chính xác toán học từ dữ liệu nhị phân trên đĩa.

---

## 3. KIỂM TRA HỒ SƠ PHÁT HÀNH (RELEASE ASSETS & DOCUMENTATION AUDIT)

| Hồ sơ / Tài liệu bắt buộc | Đường dẫn kiểm tra | Trạng thái hiện tại | Đánh giá & Rủi ro |
|---|---|:---:|---|
| **README.md** | [README.md](file:///D:/NhanDienCuChi/README.md) | **CÓ SẴN (8.616 B)** | Đầy đủ tổng quan, sơ đồ kiến trúc, bảng cử chỉ, phím tắt. Tuy nhiên **chưa có link tải model releases** và chưa có link Google Drive tải dataset. |
| **LICENSE** | [LICENSE](file:///D:/NhanDienCuChi/LICENSE) | **CÓ SẴN (1.104 B)** | MIT License chuẩn. Đã hoàn tất cho mã nguồn dự án. |
| **docs/dataset.md** | [docs/dataset.md](file:///D:/NhanDienCuChi/docs/dataset.md) | **CÓ SẴN (1.321 B)** | Mô tả quy ước đặc trưng 63 landmarks và bố cục thư mục dữ liệu trên đĩa. |
| **docs/MODEL_DISTRIBUTION.md** | [docs/MODEL_DISTRIBUTION.md](file:///D:/NhanDienCuChi/docs/MODEL_DISTRIBUTION.md) | **CÓ SẴN (5.826 B)** | Tài liệu chính sách Source-Only giải thích lý do không commit file nhị phân `.joblib` vào Git tree. |
| **THIRD_PARTY_NOTICES.md** | `THIRD_PARTY_NOTICES.md` | **CHƯA CÓ (MISSING)** | **RỦI RO CAO**. Chưa có file tập hợp nghĩa vụ Attribution cho Arda Mavi (Apache 2.0), ASL-HG Mendeley (CC BY 4.0) và Google MediaPipe (Apache 2.0). |
| **Manifest Dữ liệu** | `ai/dataset/` & `data/metadata/` | **CÓ SẴN** | Đầy đủ manifest đóng băng và phân chia tập cho ABC, 123, J/Z. |
| **Manifest Model** | `models/letters/`, `data/experiments/` | **CÓ SẴN** | Đầy đủ metadata huấn luyện, loss curve và cấu hình hyperparameters. |
| **Hướng dẫn tải dữ liệu từ Google Drive** | `docs/` hoặc `README.md` | **CHƯA CÓ (MISSING)** | Dự án chưa tạo folder chia sẻ trên Google Drive và chưa có tài liệu/script tải dataset vector trích xuất. |
| **Kịch bản tải tự động model** | [tools/download_models.py](file:///D:/NhanDienCuChi/tools/download_models.py) | **CHƯA HOÀN THIỆN** | Script hiện chỉ hỗ trợ tải `hand_landmarker.task` từ Google Storage. Nếu thiếu các model custom (`uvw_medium`, `digits_final`, `final_space_model`), script chỉ in thông báo `[MISSING]` và trả về Exit Code 1 mà **chưa có chức năng tự tải từ GitHub Releases**. |

---

## 4. ĐÁNH GIÁ KHẢ NĂNG TÁI LẬP (REPRODUCIBILITY AUDIT)

Một người dùng độc lập thực hiện clone repository mới: `git clone https://github.com/phan-the-vinh/nhandienngonngu.git`

| Tác vụ của người dùng mới | Khả năng thực hiện | Minh chứng & Cơ chế kiểm chứng |
|---|:---:|---|
| **1. Cài đặt môi trường & thư viện** | **KHẢ THI (Đã kiểm chứng)** | File `requirements.txt` và `environment.yml` đã được tối ưu hóa phiên bản chuẩn (`mediapipe`, `scikit-learn>=1.4.0`, `opencv-python`). |
| **2. Tải mô hình nền MediaPipe** | **KHẢ THI (Đã kiểm chứng)** | Lệnh `python tools/download_models.py` tải trực tiếp `hand_landmarker.task` từ Google Cloud Storage chính thức và kiểm tra mã băm SHA-256 thành công. |
| **3. Tải các custom model (ABC, 123, J/Z, SPACE)** | **CHƯA THỂ THỰC HIỆN (BỊ NGHẼN)** | Các file `.joblib` bị loại trừ trong `.gitignore`. Kho GitHub chưa có bản Release đính kèm file zip, và `tools/download_models.py` chưa có URL tải custom model. Người dùng clone về sẽ thiếu file model. |
| **4. Khởi chạy ứng dụng (`run.py` hoặc `src/main.py`)** | **BỊ CHẶN AN TOÀN (Fail-Closed)** | Khi thiếu file model, `app/camera/realtime_inference.py` phát hiện thiếu `uvw_medium_model.joblib` hoặc `digits_final.joblib`, dừng ngay lập tức với Exit Code 2 và in hướng dẫn đọc tài liệu `docs/MODEL_DISTRIBUTION.md`. |
| **5. Clean-Machine Integration Test** | **CHƯA THỰC HIỆN** | Toàn bộ kiểm thử hiện tại được thực thi trên máy phát triển. Chưa có kết quả chạy thực tế trên một hệ điều hành sạch độc lập. |

---

## 5. BẢNG TRẠNG THÁI TỔNG HỢP TỪNG MODEL

| Nhánh Model | Tệp Staging đối soát | Trạng thái kỹ thuật | Trạng thái giấy phép | Quyền công khai hiện tại |
|---|---|:---:|:---:|---|
| **MediaPipe** | `models/mediapipe/hand_landmarker.task` (7.819.105 B) | `TECHNICALLY_READY` | `READY` | **ĐỦ ĐIỀU KIỆN**. Bản quyền Apache 2.0 từ Google LLC. |
| **123 Digits** | `models/digits/production/digits_final.joblib` (464.265 B) | `TECHNICALLY_READY` | `LICENSE_REVIEW` | **TẠM THỜI CHƯA PHÁT HÀNH**. Đã có quyền sử dụng dữ liệu ngoại sinh (Arda Mavi Apache 2.0 + Mendeley CC BY 4.0). Bị chặn duy nhất bởi việc thiếu file `THIRD_PARTY_NOTICES.md` và tuyên bố license cho 349 mẫu cơ sở. |
| **SPACE** | `models/space/production/final_space_model.joblib` (1.375 B) | `TECHNICALLY_READY` | `LICENSE_REVIEW` | **CHƯA ĐỦ ĐIỀU KIỆN**. Phụ thuộc $98.37\%$ dữ liệu huấn luyện vào tập âm ABC không rõ bản quyền. Cần huấn luyện lại với tập âm sạch. |
| **J/Z Dynamic** | `data/experiments/.../model_pipeline.joblib` (36.657 B) | `TECHNICALLY_READY` | `LICENSE_REVIEW` | **CHƯA ĐỦ ĐIỀU KIỆN PHÁT HÀNH RỘNG RÃI**. Thiếu chứng từ đồng thuận quyền hình ảnh của người xuất hiện trong video SigNN. |
| **ABC Letters** | `models/letters/production/uvw_medium_model.joblib` (218.456 B) | `TECHNICALLY_READY` | `LICENSE_UNKNOWN` | **CHƯA ĐỦ ĐIỀU KIỆN**. $46.61\%$ dữ liệu (41.676 ảnh color-coded) hoàn toàn không rõ nguồn gốc. |

---

## 6. DANH SÁCH VIỆC BẮT BUỘC VÀ VIỆC CÓ THỂ ĐỂ SAU

### A. Những việc BẮT BUỘC hoàn thành trước khi phát hành (Hard Blockers)
1. **Khởi tạo tệp `THIRD_PARTY_NOTICES.md`:** Soạn thảo văn bản ghi nhận quyền tác giả và trích dẫn bản quyền cho:
   - Arda Mavi (Apache 2.0, arXiv:2011.08927).
   - Mendeley ASL-HG (CC BY 4.0, DOI: 10.17632/j4y5w2c8w9.1).
   - Google MediaPipe (Apache 2.0).
2. **Cập nhật công cụ tải model `tools/download_models.py`:**
   - Bổ sung logic tải tệp gói nén `nhandienngonngu-models-v1.0.0.zip` từ GitHub Releases tag `v1.0.0-models`.
   - Tự động giải nén và kiểm tra mã băm SHA-256 đối soát.
3. **Quyết định phương án phát hành mô hình SPACE và ABC:**
   - Xác nhận phạm vi phát hành (chỉ phát hành nội bộ/nghiên cứu hay phát hành công khai).
   - Nếu phát hành công khai, cần lên kế hoạch huấn luyện lại mô hình SPACE với tập âm sạch tách rời ABC.
4. **Tạo GitHub Release Asset:** Đóng gói thư mục `staging_distribution/` thành file ZIP và đính kèm vào Release trên GitHub.

### B. Những việc CÓ THỂ ĐỂ SAU (Post-Release / Non-Blockers)
1. **Đóng gói và tải dataset lên Google Drive:** Việc chia sẻ dataset gốc phục vụ tái huấn luyện có thể tiến hành sau khi đã phát hành bản chạy thử nghiệm (inference).
2. **Clean-Machine Dockerization:** Đóng gói môi trường Docker để tự động hóa toàn bộ quá trình cài đặt trên mọi hệ điều hành.
3. **Mở rộng nhận diện 2 tay:** Hiện hệ thống mới chỉ hỗ trợ nhận diện tay đơn (Single hand).

---

## 7. KẾT LUẬN CUỐI CÙNG: READY / NOT READY

$$\mathbf{KẾT\ LUẬN:\ NOT\ READY\ (CHƯA\ SẴN\ SÀNG\ PHÁT\ HÀNH)}$$

### Lý do:
1. **Về mặt pháp lý:** Thiếu tệp tin bắt buộc `THIRD_PARTY_NOTICES.md` để đáp ứng điều kiện ghi công của Apache 2.0 và CC BY 4.0; hai nhánh ABC và SPACE chưa giải quyết xong nguồn gốc của dữ liệu huấn luyện.
2. **Về mặt kỹ thuật triển khai:** Kịch bản cài đặt tự động `tools/download_models.py` chưa được tích hợp liên kết tải các mô hình custom từ GitHub Releases, dẫn đến người dùng clone repo mới về sẽ bị thiếu model và không thể chạy được hệ thống.

---
*Báo cáo được hoàn thành trên nguyên tắc đối soát nhị phân độc lập, phản ánh trung thực trạng thái thực tế của workspace D:\NhanDienCuChi.*
