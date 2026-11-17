# BÁO CÁO ĐỐI SOÁT MÃ BĂM STAGING VÀ XÁC MINH CÁC RÀO CẢN PHÁT HÀNH
**(STAGING HASH RECONCILIATION & RELEASE BLOCKER VERIFICATION REPORT)**

- **Dự án:** Hệ thống Nhận diện Ngôn ngữ Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)
- **Workspace:** `D:\NhanDienCuChi`
- **Repository:** [phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)
- **Branch:** `main` | **Git HEAD:** `dce96c2`
- **Thời điểm kiểm toán:** 17/11/2026 10:35 UTC+7
- **Chế độ kiểm tra:** READ-ONLY FORENSIC VERIFICATION (Tuyệt đối không sửa mã nguồn, không sửa model, không commit/push/tag/release).

---

## 1. BẢNG ĐỐI SOÁT MÃ BĂM SHA-256 ĐẦY ĐỦ (FULL SHA-256 AUDIT MATRIX)

Toàn bộ các file nhị phân trong thư mục hoạt động (Working Tree) và trong thư mục Staging (`staging_distribution/`) được tính toán trực tiếp mã băm SHA-256 theo từng khối byte trên đĩa và đối chiếu chéo với `staging_manifest.json` cùng các cấu hình chuẩn của hệ thống:

| Tên mô hình / Phân hệ | Đường dẫn tuyệt đối trên đĩa | Kích thước (Bytes) | SHA-256 tính trực tiếp (Disk & Staging) | SHA-256 Staging Manifest | Đối chiếu Config / Tài liệu chuẩn | Kết quả đối soát |
|---|---|---:|---|---|---|:---:|
| **ABC Letters**<br>(MLP Static) | `D:\NhanDienCuChi\models\letters\production\uvw_medium_model.joblib`<br>`D:\NhanDienCuChi\staging_distribution\models\letters\production\uvw_medium_model.joblib` | 218.456 | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | Khớp `docs/MODEL_DISTRIBUTION.md:91`<br>Khớp `results/experiments/.../verification.json`<br>Khớp `tests/test_stage13_integration.py:219` | **KHỚP 100% (IDENTICAL)** |
| **123 Digits**<br>(MLP Large v2) | `D:\NhanDienCuChi\models\digits\production\digits_final.joblib`<br>`D:\NhanDienCuChi\staging_distribution\models\digits\production\digits_final.joblib` | 464.265 | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | Khớp `docs/MODEL_DISTRIBUTION.md:92`<br>Khớp `reports/JZ_DTW_STAGE0_AUDIT.md:292`<br>Khớp `results/experiments/.../verification.json` | **KHỚP 100% (IDENTICAL)** |
| **J/Z Dynamic**<br>(Logistic Pipeline) | `D:\NhanDienCuChi\data\experiments\dynamic_jz_binary_baseline_v1\model_pipeline.joblib`<br>`D:\NhanDienCuChi\staging_distribution\data\experiments\dynamic_jz_binary_baseline_v1\model_pipeline.joblib` | 36.657 | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | Khớp `docs/MODEL_DISTRIBUTION.md:95`<br>Khớp `reports/STAGE18_PRODUCTION_JZ_INTEGRATION.md`<br>Khớp `tools/download_models.py:55` | **KHỚP 100% (IDENTICAL)** |
| **J/Z Scaler**<br>(StandardScaler) | `D:\NhanDienCuChi\data\experiments\dynamic_jz_binary_baseline_v1\scaler.joblib`<br>`D:\NhanDienCuChi\staging_distribution\data\experiments\dynamic_jz_binary_baseline_v1\scaler.joblib` | 27.079 | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | Khớp `reports/JZ_DTW_STAGE1_REPORT.md:179`<br>Khớp `tests/test_stage13_integration.py:217` | **KHỚP 100% (IDENTICAL)** |
| **SPACE Gesture**<br>(LogReg Classifier) | `D:\NhanDienCuChi\models\space\production\final_space_model.joblib`<br>`D:\NhanDienCuChi\staging_distribution\models\space\production\final_space_model.joblib` | 1.375 | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | Khớp `configs/space.json:6` (`model_sha256`) | **KHỚP 100% (IDENTICAL)** |
| **MediaPipe**<br>(Hand Landmarker) | `D:\NhanDienCuChi\models\mediapipe\hand_landmarker.task`<br>`D:\NhanDienCuChi\staging_distribution\models\mediapipe\hand_landmarker.task` | 7.819.105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | Khớp `reports/model_distribution_preparation_audit.md:156`<br>Khớp Git HEAD Tree Blob `0d53faf3...` | **KHỚP 100% (IDENTICAL)** |
| **ABC Scaler**<br>(StandardScaler) | `D:\NhanDienCuChi\models\letters\production\uvw_medium_scaler.joblib` | 1.887 | `93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c` | *(N/A - Scaler cơ sở)* | Khớp `docs/SRC_ARCHITECTURE_REFACTOR.md` | **NGUYÊN BẢN (KHỚP)** |

---

## 2. NGUYÊN NHÂN SAI LỆCH MÃ BĂM VÀ KẾT QUẢ XÁC MINH NGUYÊN GỐC

### 2.1. Nguồn gốc của các chuỗi hash lạ (`aebe4bc...`, `8d752ad...`, `404a37a...`, `60d9f0f...`)
Trong phản hồi tóm tắt của lượt kiểm toán trước, xuất hiện các tiền tố hash:
- `aebe4bc...` (ghi bên cạnh `digits_final.joblib`)
- `8d752ad...` (ghi bên cạnh `final_space_model.joblib`)
- `404a37a...` (ghi bên cạnh `model_pipeline.joblib`)
- `60d9f0f...` (ghi bên cạnh `uvw_medium_model.joblib`)

**Kết quả điều tra truy vết từ đĩa và Git:**
1. **Kiểm tra tệp báo cáo gốc trên đĩa:** Tệp [final_distribution_readiness_check.md](file:///D:/NhanDienCuChi/reports/final_distribution_readiness_check.md) được tạo trên đĩa tại Mục 5 **chỉ ghi kích thước byte** (464.265 B, 1.375 B, 36.657 B, 218.456 B) và **hoàn toàn không chứa các chuỗi hash trên**.
2. **Kiểm tra `staging_manifest.json` và Git history:** Không có bất kỳ commit, tag hay file staging nào từng chứa các mã băm `aebe4bc`, `8d752ad`, `404a37a`, `60d9f0f`.
3. **Kết luận nguyên nhân:** Các chuỗi này phát sinh do mô hình AI tự điền chuỗi placeholder ước lượng trong phần trình bày văn bản ở chat turn trước, chứ không phải được tính toán từ bất kỳ tệp vật lý nào trên đĩa.
4. **Xác nhận tính toàn vẹn vật lý:**
   - Toàn bộ 6 tệp mô hình trên đĩa đều giữ nguyên kích thước byte và mã SHA-256 từ thời điểm huấn luyện và đóng gói staging ban đầu.
   - Không có hiện tượng file bị ghi đè, sửa đổi ngầm hay bị phân rã (zero bit drift / corruption).

---

## 3. BẢNG TRẠNG THÁI TỪNG MODEL (STATUS & LICENSING COMPLIANCE)

| Tên mô hình | Vai trò & Kiến trúc | Trạng thái kỹ thuật | Trạng thái bản quyền dữ liệu | Trạng thái Git | Khả năng phân phối công khai hiện tại |
|---|---|:---:|:---:|:---:|---|
| **MediaPipe Hand Landmarker** | Nhận diện 21 điểm landmarks (Google LLC) | `READY` | `READY` (Apache 2.0) | **Tracked (Clean)** | **ĐỦ ĐIỀU KIỆN**. Được phân phối kèm repo hoặc tải từ GCS. |
| **123 Digits Final** | Phân loại 10 chữ số tĩnh (MLP Large v2) | `READY` | `LICENSE_REVIEW` | **Ignored** (`.gitignore:67`) | **TẠM DỪNG (Cần bổ sung Attribution)**. Đã có quyền sử dụng từ Arda Mavi (Apache 2.0) và Mendeley ASL-HG (CC BY 4.0), nhưng thiếu file `THIRD_PARTY_NOTICES.md`. |
| **J/Z Dynamic Pipeline** | Phân loại chuỗi động J/Z (Logistic Regression + Scaler) | `READY` | `LICENSE_REVIEW` | **Untracked** (Được cho phép bởi `!data/experiments/...`) | **CHƯA ĐỦ ĐIỀU KIỆN CÔNG KHAI RỘNG RÃI**. Dữ liệu SigNN có nhãn CC0 trên Kaggle nhưng chưa có văn bản thỏa thuận quyền hình ảnh cá nhân của người xuất hiện trong video. |
| **SPACE Gesture** | Phân loại nhị phân cử chỉ SPACE (Logistic Regression) | `READY` | `LICENSE_REVIEW` | **Ignored** (`.gitignore:68`) | **CHƯA ĐỦ ĐIỀU KIỆN**. Phụ thuộc $97,51\%$ mẫu Train vào tập âm NOT_SPACE từ tập ABC chưa rõ bản quyền. |
| **ABC Letters** | Phân loại 24 chữ cái tĩnh (MLP Medium) | `READY` | `LICENSE_UNKNOWN` | **Ignored** (`.gitignore:66`) | **CHƯA ĐỦ ĐIỀU KIỆN**. $46,61\%$ dữ liệu (41.676 ảnh color-coded) hoàn toàn không xác định được nguồn gốc và điều khoản sử dụng. |

---

## 4. TRẠNG THÁI HỒ SƠ PHÂN PHỐI VÀ TÀI LIỆU GIẤY PHÉP

| Tài liệu phân phối | Đường dẫn kiểm tra | Tồn tại | Trạng thái Git | Nội dung & Đánh giá rủi ro |
|---|---|:---:|:---:|---|
| **THIRD_PARTY_NOTICES.md** | `D:\NhanDienCuChi\THIRD_PARTY_NOTICES.md` | **CHƯA CÓ (MISSING)** | `NOT_FOUND` | **BLOCKER NGHIÊM TRỌNG**. Thiếu văn bản pháp lý bắt buộc để thực hiện nghĩa vụ ghi nhận quyền tác giả (Attribution) cho Apache 2.0 và CC BY 4.0. |
| **LICENSE** | [LICENSE](file:///D:/NhanDienCuChi/LICENSE) | **CÓ SẴN (1.104 B)** | `TRACKED (CLEAN)` | Giấy phép MIT License chuẩn cho mã nguồn repository. |
| **README.md** | [README.md](file:///D:/NhanDienCuChi/README.md) | **CÓ SẴN (8.616 B)** | `TRACKED (CLEAN)` | Tài liệu tổng quan hệ thống, kiến trúc đa nhánh, phím tắt. Đang thiếu liên kết tải model assets từ GitHub Releases. |
| **docs/dataset.md** | [docs/dataset.md](file:///D:/NhanDienCuChi/docs/dataset.md) | **CÓ SẴN (1.321 B)** | `TRACKED (CLEAN)` | Mô tả đặc trưng vector landmarks và cấu trúc thư mục dữ liệu. Đang thiếu URL liên kết Google Drive chia sẻ dataset. |
| **Staging Manifest** | [staging_manifest.json](file:///D:/NhanDienCuChi/staging_distribution/staging_manifest.json) | **CÓ SẴN (2.378 B)** | `UNTRACKED` | Chứa đầy đủ danh mục 9 tệp staging, kích thước byte và mã SHA-256 hoàn toàn chính xác. |
| **Model Manifest ABC** | [manifest.json](file:///D:/NhanDienCuChi/ai/dataset/mlp_uvw_training_data_v1/manifest.json) | **CÓ SẴN (6.763 B)** | `UNTRACKED` | Ghi nhận chi tiết việc bổ sung 57 mẫu UVW để tạo tập train 62.648 mẫu. |
| **Model Manifest Digits** | [training_manifest_digits_final.json](file:///D:/NhanDienCuChi/data/metadata/digits/digits_final_repro/training_manifest_digits_final.json) | **CÓ SẴN (83.219 B)** | `UNTRACKED` | Xác nhận tập train 549 mẫu gồm 3 nguồn: 349 base, 170 Ardamavi, 30 ASL-HG. |
| **Model Manifest J/Z** | [training_manifest.json](file:///D:/NhanDienCuChi/data/experiments/dynamic_jz_binary_baseline_v1/training_manifest.json) | **CÓ SẴN (1.570 B)** | `UNTRACKED` | Ghi nhận 640 chuỗi huấn luyện và loại bỏ 3 video lỗi MediaPipe. |
| **Config SPACE** | [configs/space.json](file:///D:/NhanDienCuChi/configs/space.json) | **CÓ SẴN (642 B)** | `TRACKED (CLEAN)` | Xác nhận hash `453162d8aa02...` của mô hình SPACE tại dòng 6. |

---

## 5. BÁO CÁO KIỂM TRA TRẠNG THÁI GIT CHI TIẾT (GIT STATUS AUDIT)

Lệnh kiểm tra: `git status --short` và `git status --untracked-files=all` trên branch `main` (HEAD `dce96c2`):

### 5.1. File bị chỉnh sửa (Modified Files):
- **HOÀN TOÀN KHÔNG CÓ (0 files modified).**
- Toàn bộ các file trong Git Tree đều ở trạng thái sạch (`clean`), không có bit nào bị thay đổi.

### 5.2. File đang được Git theo dõi (Tracked Files liên quan):
- `models/mediapipe/hand_landmarker.task`: Đang được theo dõi và lưu trữ nguyên vẹn dưới dạng blob `0d53faf3...` trong commit HEAD.
- `LICENSE`, `README.md`, `docs/dataset.md`, `configs/space.json`: Đang được theo dõi và giữ nguyên bản.

### 5.3. File bị loại trừ (Ignored Files theo `.gitignore`):
- `models/letters/production/uvw_medium_model.joblib` (loại trừ theo dòng 66 của `.gitignore`)
- `models/digits/production/digits_final.joblib` (loại trừ theo dòng 67 của `.gitignore`)
- `models/space/production/final_space_model.joblib` (loại trừ theo dòng 68 của `.gitignore`)
- Các checkpoint fallback: `models/digits/fallback/*.joblib`, `models/letters/fallback/*.joblib`

### 5.4. File chưa theo dõi (Untracked Files liên quan):
- Thư mục Staging: `staging_distribution/` (gồm toàn bộ các tệp model sao chép để chuẩn bị release).
- Tệp báo cáo kiểm toán trước: `reports/final_distribution_readiness_check.md`.
- Model và Scaler J/Z: `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` và `scaler.joblib` (được loại trừ khỏi ignore bởi quy tắc phủ định `!data/experiments/...`).

---

## 6. DANH SÁCH RÀO CẢN CÒN TỒN TẠI (RELEASE BLOCKERS)

### Rào cản Mức độ 1: Rào cản Pháp lý & Bản quyền (Legal Blockers)
1. **Thiếu tệp bắt buộc `THIRD_PARTY_NOTICES.md`:** Vi phạm điều kiện cấp phép bắt buộc (Attribution Clause) của Apache 2.0 (Arda Mavi, MediaPipe) và Creative Commons Attribution 4.0 International (Mendeley ASL-HG).
2. **Dữ liệu huấn luyện ABC và SPACE chưa giải quyết xong nguồn gốc:** 
   - ABC chứa $46,61\%$ ảnh color-coded chưa xác định được quyền tác giả.
   - SPACE thừa hưởng rủi ro này do dùng $62.591$ mẫu Train từ ABC làm tập âm NOT_SPACE.
3. **Quyền riêng tư video J/Z:** Chưa có văn bản miễn trừ quyền hình ảnh đối với các video người thật trong SigNN dataset.

### Rào cản Mức độ 2: Rào cản Kỹ thuật Phân phối (Technical Distribution Blockers)
1. **Kịch bản `tools/download_models.py` chưa hỗ trợ tải model từ GitHub Releases:**
   - Khi người dùng mới clone repo, chạy `python tools/download_models.py` sẽ chỉ tải được MediaPipe rồi báo lỗi `[MISSING]` cho các file `.joblib` và thoát với Exit Code 1.
   - Chưa tích hợp logic tự động tải file zip `nhandienngonngu-models-v1.0.0.zip` từ GitHub Releases.
2. **Chưa có Release Asset trên GitHub:** Thư mục `staging_distribution/` chưa được nén thành file zip và chưa được upload lên GitHub Releases (tag `v1.0.0-models`).
3. **Chưa có liên kết tải dataset tái huấn luyện:** Chưa thiết lập thư mục Google Drive để lưu trữ và chia sẻ các file đặc trưng vector trích xuất.

---

## 7. KẾT LUẬN CUỐI CÙNG (FINAL VERDICT)

$$\mathbf{KẾT\ LUẬN:\ NOT\ READY\ (CHƯA\ SẴN\ SÀNG\ PHÁT\ HÀNH)}$$

### Tóm tắt cốt lõi:
- **Về tính toàn vẹn kỹ thuật (Integrity):** `PASS 100%`. Toàn bộ mã băm SHA-256 và kích thước byte của các mô hình trên đĩa và trong staging hoàn toàn trùng khớp với manifest và tài liệu chuẩn của dự án. Không hề có hư hỏng dữ liệu hay thay đổi phiên bản.
- **Về điều kiện phát hành (Release Readiness):** `BLOCKED`. Dự án chưa thể phát hành công khai do thiếu tệp pháp lý bắt buộc `THIRD_PARTY_NOTICES.md` và kịch bản tải model tự động chưa được cấu hình liên kết tải từ GitHub Releases.

---
*Báo cáo được khởi tạo và xác minh tự động bởi công cụ kiểm toán độc lập trên nền tảng READ-ONLY của hệ thống.*
