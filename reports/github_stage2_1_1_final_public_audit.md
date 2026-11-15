# Stage 2.1.1 — Final Public Payload Audit Report

**Project Root (Authoritative):** `<project-root>`\
**Mirror Root (Subordinate):** `<project-mirror>`\
**Audit Date:** 2026-11-15\
**Execution Mode:** Read-Only Audit & Selective Staging Index Adjustment (Public Source-Only)\
**Status:** **AUDIT COMPLETE — READY FOR FINAL APPROVAL**\

---

## 1. Trạng thái Ban đầu & Danh sách File bị Loại khỏi Staging (Excluded from Staging)

### 1.1. Trạng thái Ban đầu (Stage 2.1 Baseline)
- **Git Root:** `<project-root>`
- **Current Branch:** `main`
- **Số commit:** `0` (`No commits yet`)
- **Remote:** Chưa cấu hình remote nào (`''`)
- **Số file staged ban đầu:** `397 files` (~12.1128 MB)

### 1.2. Danh sách Artifacts bị loại khỏi Staging Index
Thực hiện chỉ thị an toàn đối với các artifact nhị phân thử nghiệm J/Z chưa xác minh đầy đủ bản quyền upstream:

| Đường dẫn Artifact bị loại | Dung lượng | SHA-256 Checksum | Lý do loại bỏ khỏi Staging | Lệnh thực hiện |
|---|---|---|---|---|
| `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 36,657 bytes (~35.8 KB) | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | Artifact mô hình phân loại nhị phân Logistic Regression cho J/Z. Được huấn luyện trên SigNN ASL Video Dataset. Quyền phân phối chưa được xác minh độc lập; manifest xác định `"status": "EXPERIMENTAL_NOT_PRODUCTION"`, `"signer_session_metadata": "ABSENT"`. Tuân thủ chính sách Source-Only. | `git rm --cached data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` |
| `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 27,079 bytes (~26.4 KB) | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | Artifact chuẩn hóa StandardScaler fitted trên tập train của SigNN ASL Video. Thuộc cùng pipeline J/Z trên. | `git rm --cached data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` |

> [!NOTE]
> Hai file trên **chỉ bị hủy theo dõi khỏi Git Staging Index**, hoàn toàn **không bị xóa khỏi ổ đĩa vật lý** (`<project-root>`). Nhà phát triển cục bộ vẫn có thể chạy và phát triển bình thường.

---

## 2. Tổng số File và Dung lượng Staged Cuối cùng (Final Staged Payload)

Sau khi loại bỏ 2 artifact J/Z nói trên:
- **Tổng số file trong Staging Index:** **395 files** (hoặc 396 files bao gồm chính báo cáo này)
- **Tổng dung lượng staged:** **12,645,993 bytes (~12.0602 MB)**

### Phân bố chi tiết theo định dạng:
| Định dạng | Số lượng file | Tổng dung lượng (Bytes) | Dung lượng (MB) | Diễn giải vai trò |
|---|---|---|---|---|
| `.task` | 1 | 7,819,105 | ~7.4569 MB | Google MediaPipe Hand Landmarker model bundle (Apache 2.0) |
| `.py` | 295 | 3,407,084 | ~3.2492 MB | Mã nguồn hệ thống, module AI, coordinator, test suite, tools |
| `.md` | 57 | 897,199 | ~0.8556 MB | Toàn bộ tài liệu kiến trúc, hướng dẫn, audit reports |
| `.tsv` | 1 | 291,528 | ~0.2780 MB | Từ điển 25,000 từ tiếng Anh SUBTLEX-US runtime (CC BY-SA 4.0) |
| `.json` | 32 | 218,630 | ~0.2085 MB | Cấu hình tham số, manifest kiểm thử, metadata |
| `.csv` | 3 | 7,276 | ~0.0069 MB | Ma trận thử nghiệm ngang `docs/model_v2_p_horizontal_design_v1/` |
| `<no-ext>` | 2 | 2,763 | ~0.0026 MB | `LICENSE` (MIT) và `.gitignore` |
| `.joblib` | 1 | 1,887 | ~0.0018 MB | `models/letters/production/uvw_medium_scaler.joblib` (StandardScaler) |
| `.yml` | 1 | 263 | ~0.0003 MB | `environment.yml` (Cấu hình môi trường Conda) |
| `.txt` | 2 | 258 | ~0.0002 MB | `requirements.txt` (182 B) và `ai/config/ky_tu.txt` (76 B) |
| **Tổng cộng** | **395** | **12,645,993** | **~12.0602 MB** | **Payload mã nguồn công khai tối ưu** |

---

## 3. Kết quả Kiểm tra Bản quyền & Manifest (License Audit)

1. **Google MediaPipe Hand Landmarker (`models/mediapipe/hand_landmarker.task`):**
   - Nhà cung cấp: Google LLC.
   - Giấy phép: **Apache License 2.0**.
   - Quyền phân phối: Cho phép tái phân phối kèm thông báo bản quyền (đã có trong file LICENSE và tài liệu).
   - Quyết định: **Giữ lại trong Staging Index**.

2. **Từ điển Runtime (`data/dictionary/english_words_25k.tsv`):**
   - Nguồn gốc: SUBTLEX-US word frequency database.
   - Giấy phép: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
   - Tuân thủ: File thông báo bản quyền bắt buộc [data/dictionary/NOTICE.md](file:///<project-root>/data/dictionary/NOTICE.md) được đính kèm đầy đủ.
   - Quyết định: **Giữ lại trong Staging Index**.

3. **Mô hình Chữ cái, Chữ số và SPACE Production:**
   - `models/letters/production/uvw_medium_model.joblib`: Không có chứng từ phân phối -> **ĐÃ LOẠI TRỪ (Không staged)**.
   - `models/digits/production/digits_final.joblib`: Dữ liệu nội bộ tác giả -> **ĐÃ LOẠI TRỪ (Không staged)**.
   - `models/space/production/final_space_model.joblib`: Dữ liệu nội bộ tác giả -> **ĐÃ LOẠI TRỪ (Không staged)**.

4. **Dynamic J/Z Baseline Artifacts:**
   - `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`: Quyền tác giả và biên bản đồng thuận của người quay video SigNN chưa xác minh độc lập -> **ĐÃ LOẠI KHỎI STAGING (`git rm --cached`)**.
   - `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib`: Đi cùng mô hình trên -> **ĐÃ LOẠI KHỎI STAGING (`git rm --cached`)**.
   - Manifest `data/experiments/dynamic_jz_binary_baseline_v1/training_manifest.json` giữ nguyên trạng thái minh bạch: `"status": "EXPERIMENTAL_NOT_PRODUCTION"`, `"signer_session_metadata": "ABSENT"`.

5. **Scaler Chữ cái (`models/letters/production/uvw_medium_scaler.joblib`):**
   - Bản chất: Đối tượng `sklearn.preprocessing.StandardScaler` (1,887 bytes) chứa 63 giá trị trung bình (mean) và độ lệch chuẩn (scale) để chuẩn hóa tọa độ bàn tay MediaPipe. Không chứa mạng nơ-ron hay trọng số học của classifier.
   - Quyết định: Giữ lại như một bản hợp đồng chuẩn hóa kỹ thuật (normalization contract) theo tài liệu [docs/MODEL_DISTRIBUTION.md](file:///<project-root>/docs/MODEL_DISTRIBUTION.md).

---

## 4. Kết quả Quét Bảo mật và Dữ liệu Riêng tư (Security & Privacy Scan)

Toàn bộ 395 file staged đã được quét tự động bằng bộ regex kiểm toán chuyên sâu:

| Hạng mục kiểm tra | Số lượng phát hiện | Đánh giá & Xử lý |
|---|---|---|
| **API Keys / Secrets / Tokens** | **0** | An toàn tuyệt đối. Không có API key hay secret nào lọt vào staging. |
| **Private Keys (RSA/SSH/PGP)** | **0** | Không có khóa bí mật nào. |
| **Passwords / Credentials** | **0** | Không có mật khẩu hay chuỗi kết nối cơ sở dữ liệu. |
| **Webcam Raw Photos / Videos** | **0** | Toàn bộ ảnh webcam, capture landmarks, frame ghép khuôn mặt (`ai/final_promotion_audit/`, `*.mp4`, `*.avi`) đã bị loại trừ bởi `.gitignore`. |
| **Số điện thoại (Phone Numbers)** | **0** | 14 chuỗi số dạng `0xxxxxxxxx` phát hiện ban đầu đều là số thực đo lường độ chính xác (ví dụ: `0.8950580997949418`, `0.9856020942408377`), không phải số điện thoại. |
| **Tên tác giả & Email cá nhân** | **ĐÃ XỬ LÝ** | Phát hiện tại `reports/github_stage2_0_git_init_report.md` (dòng 20-21) chứa `<local-developer>` và `<developer-email>`. **Đã xử lý:** Thay thế bằng nhãn ẩn danh `<local-developer>` và `<developer-email>` trước khi stage. Toàn bộ repo hiện **0** email cá nhân. |
| **Đường dẫn ổ đĩa tuyệt đối máy dev** | **336 chuỗi trong `reports/`** | - Mã nguồn production (`app/`, `src/`, `ai/`, `tools/`, `training/`, `tests/`): **0 đường dẫn cứng**. Đã được di chuyển sang project-relative `Path.home()` hoặc `<project-root>`.<br>- Toàn bộ 336 chuỗi đường dẫn cục bộ chỉ xuất hiện trong các báo cáo lịch sử `reports/**/*.md` (ghi nhận quá trình phát triển máy cục bộ). |

---

## 5. Kết quả Kiểm tra Staged Diff (`git diff --cached --check`)

Lệnh thực thi: `git diff --cached --check`\
**Exit Code ghi nhận:** `2` (Do phát hiện quy ước whitespace)

### Chi tiết lỗi ghi nhận:
- **Conflict Markers (`<<<<<<<`, `=======`, `>>>>>>>`):** **0 phát hiện** (Không có bất kỳ dấu hiệu xung đột git nào).
- **trailing whitespace (362 vị trí):** Toàn bộ nằm trong các file tài liệu Markdown (`.md`) do cú pháp ngắt dòng tiêu chuẩn của Markdown yêu cầu 2 dấu cách ở cuối dòng (`  \n`).
- **new blank line at EOF (86 vị trí):** Dòng trống thừa ở cuối file văn bản.
- **Đánh giá rủi ro:** Đây là các cảnh báo định dạng văn bản thông thường trên môi trường Windows (`core.autocrlf`), không ảnh hưởng đến bất kỳ chức năng vận hành nào của hệ thống.

---

## 6. Kết quả Kiểm tra Runtime (`python run.py --check-only`)

Lệnh thực thi:
```powershell
& "<conda-env-nn>/python.exe" run.py --check-only
```

### Kết quả trả về:
```json
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
  "space_model": "<project-root>\\models\\space\\production\\final_space_model.joblib",
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
  "diagnostic_output": "<project-root>\\data\\experiments\\dynamic_jz_v1\\diagnostics",
  "diagnostic_capture_writes_images_or_video": false,
  "jz_evaluation_mode": false,
  "jz_reject_option_enabled": true,
  "jz_margin_threshold": 0.1,
  "raw_prediction_overrides_static": false,
  "no_none_class": true,
  "mirror_display": true
}
```
**Đánh giá:** Hệ thống khởi tạo và kiểm tra tĩnh hoàn toàn **PASS 100%**.

---

## 7. Các Rủi ro còn Tồn tại (Residual Risks)

1. **Người dùng tải mã nguồn công khai (Fresh Clone) chưa có model cục bộ:**
   - Khi clone repository mới, 3 file model chính (`uvw_medium_model.joblib`, `digits_final.joblib`, `final_space_model.joblib`) và model J/Z sẽ không có sẵn.
   - **Biện pháp giảm thiểu:** Hệ thống đã được tích hợp bộ chặn lỗi thông minh tại các loader (`letter_inference.py`, `digit_inference.py`, `space_predictor.py`), đưa ra thông báo rõ ràng hướng dẫn người dùng tham khảo tài liệu [docs/MODEL_DISTRIBUTION.md](file:///<project-root>/docs/MODEL_DISTRIBUTION.md) và chạy script `tools/download_models.py`.

2. **Dấu vết đường dẫn máy cục bộ trong các file báo cáo lịch sử:**
   - 336 chuỗi đường dẫn dạng `<dev-home>` hoặc `<project-root>` còn nằm trong các báo cáo lịch sử `reports/`.
   - **Biện pháp giảm thiểu:** Các báo cáo này ghi nhận lịch sử R&D và thẩm định kiến trúc. Nếu người dùng muốn repository công khai hoàn toàn không có thư mục `reports/`, có thể xem xét thêm `reports/` vào `.gitignore` trước khi commit. Nếu muốn giữ lịch sử nghiên cứu minh bạch, payload hiện tại đã loại bỏ toàn bộ dữ liệu định danh nhạy cảm.

---

## 8. Trạng thái Git Cuối cùng & Điều kiện Dừng (Final Git State)

```text
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   .gitignore
	new file:   CLEANUP_AUDIT.md
	new file:   LICENSE
	new file:   README.md
	new file:   SYSTEM_CONTEXT.md
	new file:   ai/... (41 files)
	new file:   app/... (57 files)
	new file:   configs/... (10 files)
	new file:   data/... (2 files: NOTICE.md, english_words_25k.tsv)
	new file:   docs/... (17 files)
	new file:   environment.yml
	new file:   models/... (17 files)
	new file:   reports/... (38 files)
	new file:   requirements.txt
	new file:   results/... (3 files)
	new file:   run.py
	new file:   src/... (25 files)
	new file:   tests/... (99 files)
	new file:   tools/... (59 files)
	new file:   training/... (24 files)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	data/experiments/
```

- **Commit mới:** **0** (Chưa tạo commit)
- **Remote:** **0** (Chưa tạo remote)
- **Push:** **0** (Chưa push)
- **Model vật lý trên ổ đĩa:** **Vẫn tồn tại 100% nguyên vẹn**

---
*Báo cáo được lập và xác thực tự động bởi Antigravity.*
