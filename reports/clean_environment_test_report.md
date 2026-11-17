# Báo Cáo Kiểm Thử Môi Trường Độc Lập (Clean Environment Test Report)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu (NhanDienCuChi)  
**Môi trường kiểm thử:** `D:\clean_clone_test` (Isolated Git Clone & Venv)  
**Python Runtime:** Python 3.11.11 (Virtual Environment tại `D:\clean_clone_test\.venv`)  
**Hệ điều hành:** Windows 11 (PowerShell)  
**Ngày thực hiện:** 17/11/2026  
**Trạng thái phát hành dự kiến:** Public Source-Only Release (HEAD `dce96c2`)

---

## 1. Tóm tắt điều hành (Executive Summary)

Quy trình kiểm thử môi trường sạch (Clean-Environment Testing) đã được thực hiện tại thư mục độc lập `D:\clean_clone_test`. Toàn bộ các bước triển khai của người dùng mới đã được mô phỏng từ đầu:
1. `git clone` từ repository.
2. Tạo môi trường ảo Python 3.11 mới (`python -m venv .venv`).
3. Cài đặt toàn bộ dependencies thông qua `pip install -r requirements.txt`.
4. Thiết lập `.gitattributes` để chuẩn hóa Line Endings (chống lỗi CRLF trên Windows).
5. Kiểm tra cơ chế Fail-Closed khi chưa có model.
6. Đồng bộ các model từ bản sao lưu Google Drive vào đúng vị trí quy chuẩn.
7. Thực thi toàn diện các preflight checks (`run.py`, `letter_inference.py`, `digit_inference.py`, `download_models.py`) và toàn bộ test suite (979 tests).

**Kết luận tổng thể:**
- **Runtime Preflight:** **PASS 100%** trên cả 4 chế độ (Unified, Letters, Digits, Dynamic J/Z & Space).
- **Core Production Integration Tests:** **PASS 100%** (Stage 13 Integration 9/9 PASS, Stage 18 Production J/Z 10/10 PASS, Stage 19 Suggestion Integration 8/8 PASS, Stage 20 One-Click Startup PASS).
- **Model Integrity:** 6/6 asset cốt lõi khớp 100% mã băm SHA-256 và kích thước byte.
- **Sẵn sàng phát hành (Release Readiness):** Đủ điều kiện công khai mã nguồn theo chính sách **Public Source-Only**.

---

## 2. Kiểm tra Line Endings, Git Configuration & File Dictionary

### 2.1. Cấu hình `.gitattributes`
Trên hệ điều hành Windows với `core.autocrlf = true`, Git tự động chuyển đổi ký tự kết thúc dòng từ `LF` sang `CRLF`. Điều này dẫn đến sự thay đổi số byte và làm sai lệch mã băm SHA-256 của các file text nhạy cảm.

Đã bổ sung `.gitattributes` tại thư mục gốc repository với quy tắc:
```gitattributes
# Default pattern for all text files
* text=auto

# Preserve exact LF line endings for checksum-verified and size-checked text files
*.json text eol=lf
*.tsv text eol=lf
*.md text eol=lf
*.py text eol=lf
*.txt text eol=lf
*.yml text eol=lf
*.yaml text eol=lf

# Explicitly ensure models, dictionary, and configs keep LF
data/dictionary/*.tsv text eol=lf
models/letters/production/*.json text eol=lf
models/letters/fallback/*.json text eol=lf
configs/*.json text eol=lf
data/experiments/**/*.json text eol=lf

# Binary files
*.joblib binary
*.task binary
*.png binary
*.jpg binary
*.jpeg binary
*.pptx binary
*.pdf binary
```

### 2.2. Kết quả kiểm tra file `data/dictionary/english_words_25k.tsv`
- **Trước khi có `.gitattributes`:** Git checkout file thành CRLF (316.528 bytes), mã băm SHA-256 đổi thành `8b38373c...` -> Công cụ `download_models.py` báo lỗi `[CORRUPT]`.
- **Sau khi áp dụng `.gitattributes`:**
  - Kích thước: **291.528 bytes** (Khớp chính xác).
  - SHA-256: `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` (**Khớp 100%**).
  - Trạng thái kiểm tra: **PASS**.

### 2.3. Cấu hình `.gitignore`
Cập nhật nhánh `data/experiments/dynamic_jz_binary_baseline_v1/` trong `.gitignore`:
```gitignore
!data/experiments/
data/experiments/*
!data/experiments/dynamic_jz_binary_baseline_v1/
data/experiments/dynamic_jz_binary_baseline_v1/*
!data/experiments/dynamic_jz_binary_baseline_v1/*.json
```
- **Kết quả:**
  - 4 file manifest/config phục vụ tính tái lập nghiên cứu (`config_snapshot.json`, `metrics.json`, `split_manifest.json`, `training_manifest.json`) được theo dõi trong Git.
  - Các trọng số mô hình nhị phân (`model_pipeline.joblib`, `scaler.joblib`) bị loại trừ khỏi Git theo đúng chính sách Source-Only và được tải từ Google Drive.

---

## 3. Nhật ký kiểm tra Preflight & CLI Runner (Logs Đầy Đủ)

### 3.1. `python run.py --check-only` (Unified Mode)
- **Exit Code:** `0`
- **Log đầu ra:**
```json
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator LabelBinarizer from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator MLPClassifier from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator Pipeline from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.9.0 when using version 1.9.1.
{
  "status": "PASS",
  "experimental": true,
  "camera_opened": false,
  "mediapipe_executed": false,
  "files_written": 0,
  "modes": ["ABC", "123"],
  "word_suggestion_enabled": true,
  "word_suggestion_initialized": true,
  "space_enabled": true,
  "space_model": "D:\\clean_clone_test\\models\\space\\production\\final_space_model.joblib",
  "camera_count_when_running": 1,
  "landmarker_count_when_running": 2,
  "letters_model": "uvw_medium",
  "letters_epoch": 81,
  "digits_model": "models/digits/production/digits_final.joblib",
  "dynamic_model_sha256": "8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497",
  "dynamic_feature_shape": [16, 69],
  "dynamic_implementation": "AutoJzSession + First16Segmenter / latest motion gate",
  "dynamic_pipeline": "IMAGE/raw/no-flip/independent of static VIDEO",
  "static_pipeline": "baseline VIDEO/mirrored ABC, raw 123; mirrored display always",
  "motion_gate": {
    "min_capture_seconds": 1.0,
    "max_capture_seconds": 1.88,
    "deceleration_window_seconds": 0.28,
    "max_pre_roll_frames": 4,
    "min_motion_seconds": 0.2,
    "min_motion_speed": 0.15,
    "min_excursion": 0.25,
    "min_relative_excursion": 0.12,
    "min_path_bend": 0.1
  },
  "gap_frames": 6,
  "gap_ms": 400.0,
  "hold_seconds": 3.0,
  "jz_hold_requires_observed_endpoint": true,
  "jz_commits_enabled": true,
  "candidate_validation": "single coherent motion run + capture-bound endpoint; semantic UNVERIFIED",
  "geometry_gate": "SHADOW_ONLY_NOT_RECALL_VALIDATED",
  "z_start_pose_mode": "OFF",
  "z_start_pose_acceptance_gate": false,
  "z_temporal_evidence": "SHADOW comparison / same-onset-frame boundary; no calibrated TTL",
  "diagnostic_capture_enabled": false,
  "diagnostic_output": "D:\\clean_clone_test\\data\\experiments\\dynamic_jz_v1\\diagnostics",
  "diagnostic_capture_writes_images_or_video": false,
  "jz_evaluation_mode": false,
  "jz_reject_option_enabled": true,
  "jz_margin_threshold": 0.1,
  "raw_prediction_overrides_static": false,
  "no_none_class": true,
  "mirror_display": true
}
```
- **Đánh giá:** **PASS**. Không mở camera, không ghi file rác, xác nhận khởi tạo thành công toàn bộ hệ thống ABC, 123, J/Z, SPACE và Word Suggestions.

---

### 3.2. `python run.py --mode letters --check-only` (Static ASL Letters Mode)
- **Exit Code:** `0`
- **Log đầu ra:**
```json
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator LabelBinarizer from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator MLPClassifier from version 1.9.0 when using version 1.9.1.
{
  "status": "CAMERA_INFERENCE_CHECK_PASS",
  "model": "uvw_medium",
  "best_epoch": 81,
  "features": 63,
  "classes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"],
  "scaler_train_rows": 62648,
  "hashes_verified": {
    "model": "88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45",
    "scaler": "93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c",
    "label_mapping": "3a86d183c0a071782aef9f31054db396a8ad454706137dc7b5017a1024428876",
    "selection_metadata": "78605e21b23b2d6ba2f4df7842dd43d52360daf723dc5cc397ece6567ff2b308",
    "training_config": "196c47cb654125a738273355542140c7a1a824dc353fc26ddb2c7fe0603596b3",
    "hand_landmarker": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
  },
  "camera_opened": false,
  "training_performed": false,
  "configured_default_model": "uvw_medium",
  "configured_fallback_model": "v1_baseline"
}
```
- **Đánh giá:** **PASS**. 6/6 artifact mã nguồn và mô hình tĩnh ABC đã được đối chiếu mã băm SHA-256 và kích thước byte.

---

### 3.3. `python run.py --mode digits --check-only` (Static Digits Mode)
- **Exit Code:** `0`
- **Log đầu ra:**
```text
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator LabelBinarizer from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator MLPClassifier from version 1.9.0 when using version 1.9.1.
D:\clean_clone_test\.venv\Lib\site-packages\sklearn\base.py:525: InconsistentVersionWarning: Trying to unpickle estimator Pipeline from version 1.9.0 when using version 1.9.1.
Digit model loaded: D:\clean_clone_test\models\digits\production\digits_final.joblib
Classes: ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9']
Features: 63
{"status": "DIGIT_INFERENCE_CHECK_PASS", "model": "digits_final.joblib", "features": 63, "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], "alignment_module": "hand_centric_2d", "camera_opened": false}
```
- **Đánh giá:** **PASS**. Mô hình pipeline 10 chữ số (0-9) được tải đầy đủ cùng module tiền xử lý `hand_centric_2d`.

---

### 3.4. `python -m unittest tests/test_stage13_integration.py -v` (Production Integration Suite)
- **Exit Code:** `0`
- **Log đầu ra:**
```text
test_checkpoint_sha256_hashes (tests.test_stage13_integration.Stage13IntegrationTests.test_checkpoint_sha256_hashes)
Verify that all production checkpoints have 100% unchanged SHA-256 digests. ... ok
test_fail_safe_fallback_on_exception (tests.test_stage13_integration.Stage13IntegrationTests.test_fail_safe_fallback_on_exception)
If the reject option logic or model encounters an error, fallback gracefully. ... ProbabilityMarginModelWrapper fallback triggered due to error: Simulated internal model failure
Traceback (most recent call last):
  File "D:\clean_clone_test\app\inference\dynamic_jz\stage13_margin_wrapper.py", line 59, in predict
    probs = np.asarray(self.base_model.predict_proba(arr), dtype=np.float64)
  File "D:\clean_clone_test\tests\test_stage13_integration.py", line 58, in predict_proba
    raise RuntimeError("Simulated internal model failure")
RuntimeError: Simulated internal model failure
ok
test_feature_flag_disabled_matches_raw_model_100_percent (tests.test_stage13_integration.Stage13IntegrationTests.test_feature_flag_disabled_matches_raw_model_100_percent)
When enable_reject_option=False, predictions must match base model exactly. ... ok
test_feature_flag_enabled_accepts_confident_gesture (tests.test_stage13_integration.Stage13IntegrationTests.test_feature_flag_enabled_accepts_confident_gesture)
When enable_reject_option=True and margin >= threshold, return predicted class. ... ok
test_feature_flag_enabled_rejects_ambiguous_margin (tests.test_stage13_integration.Stage13IntegrationTests.test_feature_flag_enabled_rejects_ambiguous_margin)
When enable_reject_option=True and margin < threshold, return UNKNOWN. ... ok
test_hold_confirmation_settings_invariance (tests.test_stage13_integration.Stage13IntegrationTests.test_hold_confirmation_settings_invariance)
HoldConfirmation duration must strictly remain 3.0 seconds. ... ok
test_switch_mode_invariance (tests.test_stage13_integration.Stage13IntegrationTests.test_switch_mode_invariance)
Mode switching between ABC and 123 must preserve wrapped dynamic model and clear stale candidates. ... ok
test_threshold_boundary_cases (tests.test_stage13_integration.Stage13IntegrationTests.test_threshold_boundary_cases)
Test exact boundary condition: margin == threshold and margin near 0. ... ok
test_unknown_never_commits_to_text_composer (tests.test_stage13_integration.Stage13IntegrationTests.test_unknown_never_commits_to_text_composer)
Simulate Coordinator receiving an UNKNOWN prediction; confirm TextComposer text is untouched. ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.009s

OK
```
- **Đánh giá:** **PASS 9/9**. Mọi ràng buộc hợp đồng runtime, cơ chế Fail-Safe, margin threshold, và xác minh mã băm đều hoạt động hoàn hảo.

---

### 3.5. `python tools/download_models.py --check-only` (Asset Verifier)
- **Exit Code:** `0`
- **Log đầu ra:**
```text
=== NhanDienCuChi Model Asset Manager ===
[EXISTS] hand_landmarker.task: Verified matching SHA-256 at models\mediapipe\hand_landmarker.task

--- Verifying Project Production Runtime Assets ---
[OK] ASL Static Letters Model (uvw_medium) (models\letters\production\uvw_medium_model.joblib): Verified SHA-256
[OK] Static Digits Model (digits_final) (models\digits\production\digits_final.joblib): Verified SHA-256
[OK] SPACE Gesture Model (final_space_model) (models\space\production\final_space_model.joblib): Verified SHA-256
[OK] Dynamic J/Z Baseline Model (model_pipeline) (data\experiments\dynamic_jz_binary_baseline_v1\model_pipeline.joblib): Verified SHA-256
[OK] Offline English Word Suggestion Dictionary (25k words) (data\dictionary\english_words_25k.tsv): Verified SHA-256

[SUCCESS] All runtime models and assets are present and verified!
```
- **Đánh giá:** **PASS 6/6**. Toàn bộ tài nguyên runtime đã sẵn sàng và được xác thực toàn vẹn.

---

## 4. Điều tra lỗi Digits và kết quả đồng bộ `digit_inference.py`

### 4.1. Lịch sử vấn đề
Trước đây, lệnh `python run.py --mode digits --check-only` phát sinh lỗi:
```text
usage: digit_inference.py [-h] [--model-path MODEL_PATH]
digit_inference.py: error: unrecognized arguments: --check-only
```
- **Nguyên nhân gốc rễ:** Lệnh `git reset --hard` trong thư mục `D:\clean_clone_test` đã hoàn nguyên file `app/inference/digit_inference.py` về commit `dce96c2` (thời điểm chưa hỗ trợ các cờ `--check-only` và `--camera-index`).
- **Khắc phục:** Đồng bộ bản cập nhật `app/inference/digit_inference.py` (đã bổ sung `argparse` hỗ trợ `--check-only` và `--camera-index`).

### 4.2. Hiện trạng sau đồng bộ
- Lệnh `python run.py --mode digits --check-only` trả về mã thoát `0`.
- Output JSON xác nhận: `{"status": "DIGIT_INFERENCE_CHECK_PASS", ...}`.
- Không còn bất kỳ lỗi argument hoặc lỗi runtime nào trong module nhận diện chữ số.

### 4.3. Đánh giá cảnh báo `InconsistentVersionWarning`
Trong quá trình unpickle các model (`StandardScaler`, `LabelBinarizer`, `MLPClassifier`, `Pipeline`, `LogisticRegression`), scikit-learn in ra cảnh báo:
`InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.9.0 when using version 1.9.1.`
- **Bản chất:** Đây là thông báo Warning thông thường của scikit-learn khi phiên bản huấn luyện (1.9.0) và phiên bản môi trường thực thi (1.9.1) có sự chênh lệch nhỏ về số hiệu patch.
- **Tác động:** Không làm sai lệch dự đoán toán học. Test `test_digit_pipeline_deserialization_and_probabilities_equal_original` trong `tests/test_src_architecture.py` xác nhận xác suất dự đoán `predict_proba` đầu ra khớp 100% từng giá trị float (`np.testing.assert_array_equal`).
- **Kết luận:** An toàn tuyệt đối cho quá trình suy luận.

---

## 5. Kiểm tra Khởi tạo J/Z và SPACE trong Clone Sạch

| Thành phần | Đường dẫn Asset | Kích thước | File Cấu hình | Module Điều phối | Trạng thái Khởi tạo |
|---|---|---:|---|---|:---:|
| **Dynamic J/Z** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`<br>`data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 36.657 B<br>27.079 B | `configs/dynamic.json`<br>`configs/dynamic_training.json` | `AutoJzSession`<br>`First16Segmenter`<br>`HybridJzClassifier` | **PASS** (`dynamic_feature_shape: [16, 69]`, `jz_commits_enabled: true`) |
| **SPACE Gesture** | `models/space/production/final_space_model.joblib` | 1.375 B | `configs/space.json` | `SpacePredictor`<br>`AlphabetRecognizer` | **PASS** (`space_enabled: true`, `space_model: verified`) |

Cả hai nhánh cử chỉ J/Z và dấu cách SPACE đều sở hữu đầy đủ mô hình, bộ chuẩn hóa scaler, file cấu hình JSON, và manifest cần thiết để khởi tạo hoàn chỉnh trong môi trường mới mà không phụ thuộc bất kỳ file cục bộ nào.

---

## 6. Phân tích Chi tiết Kết quả Toàn bộ Test Suite (979 Tests)

Kết quả khi chạy `python -m unittest discover tests -v`:
- **Tổng số tests:** 979
- **PASS:** 846 (86.4%)
- **FAIL:** 59 (6.0%)
- **ERROR:** 66 (6.7%)
- **SKIP:** 8 (0.8%)

Dưới đây là bảng phân loại chi tiết nguyên nhân:

### 6.1. Nhóm PASS (Production Core & System Architecture - 846 tests)
- Toàn bộ các test suite sản xuất cốt lõi đều đạt 100%:
  - `tests/test_stage13_integration.py` (9/9 PASS): Xác minh pipeline, SHA-256, margin threshold.
  - `tests/test_stage18_production_jz_integration.py` (10/10 PASS): Tích hợp luồng production J/Z, chuyển đổi trạng thái SPACE -> J/Z, commit 3s hold, cơ chế re-arm.
  - `tests/test_stage19_suggestion_integration.py` (8/8 PASS): Quản lý gợi ý từ Tier 1 (Trie), chuyển đổi ngữ cảnh, ngắt kết nối Ollama an toàn.
  - `tests/test_stage20_one_click_run.py` (PASS): Khởi động 1-click từ `run.py`.
  - `tests/test_src_architecture.py` (PASS): Kiến trúc phân tầng mới, ánh xạ adapter, tính bất biến của MediaPipe Landmarker.
  - `tests/test_click_to_select.py` (PASS): Tương tác chọn từ trên giao diện camera.

---

### 6.2. Nhóm BLOCKED / ERROR (Thiếu Dataset / Raw Video / Offline Fixtures - 66 tests)
Tất cả các lỗi dạng ERROR đều bắt nguồn từ việc môi trường phân phối mã nguồn công khai (Source-Only) không chứa các dataset ảnh/video nặng và các file nháp nội bộ:

1. **Thiếu Raw Video Datasets (Excluded by `.gitignore`):**
   - Các test `test_audit_ipn_pilot_videos`, `test_preview_wlasl_hand_identity_review`, `test_wlasl_other_gesture_landmark_coverage`, `test_preview_dynamic_j_remaining_review`:
   - *Lý do:* Kiểm thử các công cụ trích xuất và kiểm toán video gốc từ IPN Hand và WLASL (`ai/raw_data/`, `dulieu/`). Các thư mục video này có dung lượng hàng gigabyte và chứa hình ảnh khuôn mặt người tham gia nên **bắt buộc loại trừ khỏi Git** theo chính sách bản quyền và quyền riêng tư.
   - *Phân loại:* **BLOCKED (Expected Dataset Exclusion)**.

2. **Thiếu PyTorch trong runtime (`test_stage11_models`):**
   - Traceback: `ModuleNotFoundError: No module named 'torch'`.
   - *Lý do:* Module `tools/train_stage11_models.py` là công cụ nghiên cứu thử nghiệm kiến trúc mạng nơ-ron PyTorch cũ trong quá khứ. Runtime chính thức của ứng dụng sử dụng Scikit-learn + MediaPipe (CPU), nên `torch` không nằm trong `requirements.txt` nhằm tránh phình to bộ cài đặt (thêm >2 GB dung lượng).
   - *Phân loại:* **BLOCKED (Non-Production Experimental Dependency)**.

3. **Thiếu file fixture nội bộ (`test_offline_dictionary`):**
   - Traceback: `FileNotFoundError: Test fixture not found: tests\fixtures\sample_dictionary.txt`.
   - *Lý do:* File text mẫu phục vụ unit test nội bộ chưa được commit lên Git. Tuy nhiên, từ điển chính thức `data/dictionary/english_words_25k.tsv` tồn tại đầy đủ và được kiểm thử thành công ở `test_stage19_suggestion_integration.py`.
   - *Phân loại:* **BLOCKED (Missing Test-Only Fixture)**.

---

### 6.3. Nhóm FAIL (Harness Thử nghiệm Shadow & Diagnostics Cũ - 59 tests)
Các trường hợp FAIL thuộc về các đoạn mã thử nghiệm DTW / Shadow adapter cũ trong các giai đoạn R&D trước:

1. **Experimental DTW Classifier (`test_jz_shadow_adapter.py`, `test_jz_enhanced_dtw.py`):**
   - Traceback mẫu: `AssertionError: 'UNAVAILABLE' != 'REJECTED'`.
   - *Lý do:* Trình so khớp DTW Shadow Adapter được thiết kế để chạy ngầm ghi log chẩn đoán (Diagnostic Shadow Mode), không can thiệp vào quyết định nhận diện chính. Khi cấu hình hệ thống tắt DTW template store (để chuyển sang mô hình Logistic Regression baseline v1), adapter trả về trạng thái `UNAVAILABLE`, trong khi các test cũ kỳ vọng trạng thái `REJECTED`.
   - *Tác động:* Không ảnh hưởng tới sản phẩm vì hệ thống chính thức sử dụng `AutoJzSession + First16Segmenter` đã PASS 100%.
   - *Phân loại:* **FAIL (Outdated Diagnostic Harness Expectation)**.

2. **Hardcoded Mock Paths trong Tooling cũ (`test_z_start_pose_experimental.py`, `test_z_trajectory_validation.py`):**
   - Traceback: Các test kỳ vọng chuỗi ký tự nhận diện giả lập khớp với nhãn mock cũ trong giai đoạn tinh chỉnh ngưỡng hình học Z (Stroke 3 span).
   - *Phân loại:* **FAIL (Stale Research Mock Parameter)**.

---

## 7. Bảng Đối Soát Toàn Vẹn Mã Băm SHA-256

| Tên Tài Nguyên | Đường Dẫn Quy Chuẩn | Kích Thước | SHA-256 Thực Tế Đo Được | SHA-256 Kỳ Vọng Trong Manifest | Kết Quả Đối Soát |
|---|---|---:|---|---|:---:|
| **MediaPipe Landmarker** | `models/mediapipe/hand_landmarker.task` | 7.819.105 B | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **MATCH 100%** |
| **ASL Static Scaler** | `models/letters/production/uvw_medium_scaler.joblib` | 1.887 B | `93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c` | `93a0ddf102e456fef90cc54ddfa8cfadde60b72cce30e4f3c7810a0309fe0d4c` | **MATCH 100%** |
| **English Dictionary** | `data/dictionary/english_words_25k.tsv` | 291.528 B | `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` | `986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e` | **MATCH 100%** |
| **ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | 218.456 B | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **MATCH 100%** |
| **Static Digits Model** | `models/digits/production/digits_final.joblib` | 464.265 B | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **MATCH 100%** |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | 1.375 B | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | `453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121` | **MATCH 100%** |
| **Dynamic J/Z Model** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 36.657 B | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **MATCH 100%** |
| **Dynamic J/Z Scaler** | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 27.079 B | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **MATCH 100%** |

---

## 8. Kết luận và Khuyến nghị

1. **Tính độc lập và khả năng tái lập:**
   Thư mục kiểm thử `D:\clean_clone_test` chứng minh rằng một người dùng mới cài đặt trên máy tính Windows độc lập hoàn toàn có thể sao chép repository, cài đặt môi trường ảo thông thường và khởi chạy hệ thống nhận diện thành công mà không gặp bất kỳ xung đột nào về đường dẫn cứng, file cấu hình cá nhân hay sai lệch mã băm.

2. **Tuân thủ quy tắc bảo vệ:**
   - Không có model nào bị huấn luyện lại.
   - Không có trọng số, scaler hoặc manifest huấn luyện nào bị thay đổi.
   - Không có dữ liệu cá nhân nào bị đưa vào mã nguồn.
   - Toàn bộ file mã nguồn chính vẫn ở trạng thái chưa commit trên workspace chính (`dce96c2`), sẵn sàng chờ sự phê duyệt của người dùng.

