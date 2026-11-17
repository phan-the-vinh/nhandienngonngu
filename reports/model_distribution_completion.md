# Báo Cáo Hoàn Thành Phân Phối Mô Hình Nhận Diện Ngôn Ngữ Ký Hiệu (NhanDienCuChi)

**Thời gian hoàn thành:** 17/11/2026  
**Workspace:** `D:\NhanDienCuChi`  
**GitHub Repository:** [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)  
**Branch:** `main`  
**Google Drive Dataset Archive:** [https://drive.google.com/drive/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)  

---

## 1. Tổng Quan Triển Khai Phân Phối (Distribution Architecture)

Dự án triển khai chiến lược phân phối mô hình theo mô hình **"Batteries-Included Runtime"**:
1. **Toàn bộ trọng số mô hình production, scaler chuẩn hóa, landmark task bundle và từ điển gợi ý từ** được lưu trữ trực tiếp trong Git tree trên branch `main`. Do tổng dung lượng các file checkpoint chỉ ~8.8 MB (trong đó MediaPipe task chiếm 7.8 MB, toàn bộ model scikit-learn nhẹ dưới 1 MB), người dùng chỉ cần `git clone` là có thể chạy inference ngay lập tức mà không gặp bất kỳ rào cản phụ thuộc nào về checkpoint.
2. **Dataset huấn luyện dung lượng lớn (video gốc, ảnh thô, feature CSV nhiều gigabyte, zip archives)** được lưu trữ an toàn trên Google Drive chính thức của dự án, tuân thủ đúng yêu cầu không làm phình kho lưu trữ GitHub và tôn trọng quyền riêng tư/giấy phép gốc.

---

## 2. Bảng Danh Mục Mô Hình, Checksum và Phân Loại Quyền Phân Phối

Tất cả 7 thành phần runtime cốt lõi đã được kiểm tra, tính toán mã băm SHA-256 và xác nhận tính toàn vẹn tuyệt đối:

| Thành phần | Đường dẫn canonical | Dung lượng (bytes) | Mã băm SHA-256 | Phân loại pháp lý & Nguồn gốc | Quyền phân phối |
|---|---|---:|---|---|:---:|
| **MediaPipe Hand Landmarker** | `models/mediapipe/hand_landmarker.task` | 7.819.105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | Google LLC (Apache License 2.0) | **Được phân phối công khai** (Public Verified) |
| **ASL Letters Model (`uvw_medium`)** | `models/letters/production/uvw_medium_model.joblib` | 218.456 | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | Kaggle ASL Alphabet (GPL-2.0, 45.236 mẫu) + Color-coded hand segmentation (41.676 mẫu) + Web supplementary (2.503 mẫu) + Tự thu bổ sung (57 mẫu) | **Phân phối phục vụ nghiên cứu & học tập** (Research Checkpoint with Attribution) |
| **ASL Letters Scaler** | `models/letters/production/uvw_medium_scaler.joblib` | 1.887 | `93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c` | Standard Normalizer (fit trên 62.648 mẫu train ABC v1) | **Được phân phối công khai** (Public Verified) |
| **Digits Model (`digits_final`)** | `models/digits/production/digits_final.joblib` | 464.265 | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | Tự thu nội bộ tác giả (349 mẫu) + Arda Mavi Digits (Apache-2.0, 170 mẫu) + ASL-HG Mendeley (CC BY 4.0, 30 mẫu). Pipeline: 2D-aligner + Scaler + MLP | **Được phân phối công khai** (Public Verified) |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | 1.375 | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | Logistic Regression (ngưỡng 0.889); Dương: 1.049 mẫu tác giả tự thu; Âm: 63.140 mẫu (62.591 từ ABC train.csv + 549 từ Digits train) | **Được phân phối công khai** (Public Verified) |
| **Dynamic J/Z Baseline Model** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 36.657 | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | SigNN ASL Video Dataset (CC0 1.0 Universal, Mục 2(d) bảo lưu quyền riêng tư/hình ảnh đối tượng trong video thô). Mô hình trích xuất landmark không chứa nhận diện khuôn mặt | **Bản thử nghiệm nghiên cứu nội bộ** (Experimental Research Baseline) |
| **Dynamic J/Z Scaler** | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 27.079 | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | Scaler chuẩn hóa quỹ đạo 16 khung hình (69 đặc trưng) | **Được phân phối công khai** (Public Verified) |
| **English Suggestion Dictionary** | `data/dictionary/english_words_25k.tsv` | 291.528 | `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` | SUBTLEX-US Corpus (CC BY-SA 4.0 với trích dẫn) | **Được phân phối công khai** (Public Verified) |

---

## 3. Nhật Ký Kiểm Thử Xác Minh Tính Khả Thi (Verification Results)

Tất cả các lệnh kiểm thử hệ thống trong chế độ không mở camera (`--check-only`) và bộ test tự động đều đạt kết quả **100% PASS**:

### 3.1. Kiểm tra tài nguyên mô hình (`tools/download_models.py --check-only`)
```text
=== NhanDienCuChi Model Asset Manager ===
[EXISTS] hand_landmarker.task: Verified matching SHA-256 at models\mediapipe\hand_landmarker.task

--- Verifying Project Production Runtime Assets ---
[OK] ASL Static Letters Model (uvw_medium) (models\letters\production\uvw_medium_model.joblib): Verified SHA-256
[OK] ASL Static Letters Scaler (uvw_medium_scaler) (models\letters\production\uvw_medium_scaler.joblib): Verified SHA-256
[OK] Static Digits Model (digits_final) (models\digits\production\digits_final.joblib): Verified SHA-256
[OK] SPACE Gesture Model (final_space_model) (models\space\production\final_space_model.joblib): Verified SHA-256
[OK] Dynamic J/Z Baseline Model (model_pipeline) (data\experiments\dynamic_jz_binary_baseline_v1\model_pipeline.joblib): Verified SHA-256
[OK] Dynamic J/Z Scaler (scaler.joblib) (data\experiments\dynamic_jz_binary_baseline_v1\scaler.joblib): Verified SHA-256
[OK] Offline English Word Suggestion Dictionary (25k words) (data\dictionary\english_words_25k.tsv): Verified SHA-256

[SUCCESS] All runtime models and assets are present and verified!
```

### 3.2. Kiểm tra khởi động hệ thống đa phân hệ (`run.py --check-only`)
- **Trạng thái:** `PASS`
- **Các phân hệ hoạt động:** Chữ cái tĩnh (ABC), Chữ số tĩnh (123), Cử chỉ động J/Z, Cử chỉ SPACE, Từ điển gợi ý 25.000 từ.
- **Ngưỡng SPACE thực tế:** `0.889`
- **Mã SHA-256 J/Z xác thực:** `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497`

### 3.3. Kiểm tra phân hệ Letters (`run.py --mode letters --check-only`)
- **Trạng thái:** `CAMERA_INFERENCE_CHECK_PASS`
- **Model:** `uvw_medium` (Epoch 81, 63 features, 24 classes: A–I, K–Y).
- **Mã băm model & scaler:** Khớp tuyệt đối với manifest.

### 3.4. Kiểm tra phân hệ Digits (`run.py --mode digits --check-only`)
- **Trạng thái:** `DIGIT_INFERENCE_CHECK_PASS`
- **Model:** `digits_final.joblib` (63 features, 10 classes: 0–9, module căn chỉnh `hand_centric_2d`).

### 3.5. Kiểm tra Unit Test tích hợp (`test_stage13_integration.py`)
```text
test_checkpoint_sha256_hashes ... ok
test_fail_safe_fallback_on_exception ... ok
test_feature_flag_disabled_matches_raw_model_100_percent ... ok
test_feature_flag_enabled_accepts_confident_gesture ... ok
test_feature_flag_enabled_rejects_ambiguous_margin ... ok
test_hold_confirmation_settings_invariance ... ok
test_switch_mode_invariance ... ok
test_threshold_boundary_cases ... ok
test_unknown_never_commits_to_text_composer ... ok
----------------------------------------------------------------------
Ran 9 tests in 0.010s
OK
```

---

## 4. Hướng Dẫn Tải và Chạy Hệ Thống Cho Người Dùng Mới

### Bước 1: Clone kho mã nguồn
```bash
git clone https://github.com/phan-the-vinh/nhandienngonngu.git
cd nhandienngonngu
```

### Bước 2: Thiết lập môi trường Python (Khuyến nghị dùng Conda)
```bash
conda env create -f environment.yml
conda activate nhandiencuchi
```
*Hoặc cài đặt qua venv thông thường:*
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 3: Kiểm tra tính sẵn sàng của mô hình (Không cần camera)
```bash
python tools/download_models.py --check-only
python run.py --check-only
```

### Bước 4: Chạy ứng dụng thời gian thực
- **Chế độ tự động đầy đủ (ABC, Digits, J/Z, SPACE, Từ điển gợi ý):**
  ```bash
  python run.py
  ```
- **Chế độ chỉ nhận diện chữ cái:**
  ```bash
  python run.py --mode letters
  ```
- **Chế độ chỉ nhận diện chữ số:**
  ```bash
  python run.py --mode digits
  ```

---

## 5. Kết Luận

1. Toàn bộ trọng số mô hình đã được tích hợp đầy đủ, nguyên vẹn trên kho GitHub [nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu).
2. Tài liệu hướng dẫn ([README.md](file:///D:/NhanDienCuChi/README.md), [docs/MODEL_DISTRIBUTION.md](file:///D:/NhanDienCuChi/docs/MODEL_DISTRIBUTION.md), [docs/dataset.md](file:///D:/NhanDienCuChi/docs/dataset.md), [THIRD_PARTY_NOTICES.md](file:///D:/NhanDienCuChi/THIRD_PARTY_NOTICES.md)) đã được cập nhật đồng bộ, chính xác về mã băm SHA-256 và chính sách cấp phép.
3. Không vi phạm chính sách Git về dữ liệu lớn: các tập dữ liệu gốc hàng chục nghìn ảnh/video được tổ chức độc lập trên Google Drive.
4. Dự án sẵn sàng bàn giao, nghiệm thu và triển khai trên môi trường máy sạch.
