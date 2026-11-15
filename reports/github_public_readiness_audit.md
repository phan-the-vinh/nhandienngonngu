# BÁO CÁO KIỂM TOÁN VÀ CHUẨN BỊ PHÁT HÀNH GITHUB CÔNG KHAI
## (GITHUB PUBLIC RELEASE READINESS AUDIT REPORT)

**Dự án:** NhanDienCuChi (Hệ thống nhận diện cử chỉ bàn tay hỗ trợ gõ ký tự ASL, chữ số, dấu cách và gợi ý từ ngữ)\
**Đường dẫn gốc:** `<project-root>`\
**Ngày kiểm toán:** 14/11/2026\
**Mục tiêu:** Đánh giá tính sẵn sàng và lập kế hoạch chuẩn bị phát hành mã nguồn mở công khai (Public GitHub Repository).\
**Trạng thái kiểm toán an toàn:** CHỈ ĐỌC VÀ LẬP BÁO CÁO — Không thực hiện `git init`, `git add`, `git commit`, `git push`, không sửa đổi/xóa file dữ liệu hoặc mã nguồn production.

---

## 1. TỔNG QUAN (EXECUTIVE SUMMARY)

Dự án `NhanDienCuChi` đã hoàn thiện chuỗi chức năng nhận diện thời gian thực qua webcam:
- Nhận diện 24 chữ cái tĩnh ASL (A–Y trừ J, Z) bằng mô hình MLP `uvw_medium`.
- Nhận diện 10 chữ số tĩnh (0–9) bằng mô hình Pipeline `digits_final`.
- Nhận diện 2 chữ cái động J và Z bằng mô hình Logistic Regression + bộ lọc chuyển động First16 + cơ chế Hold Confirmation.
- Nhận diện cử chỉ ngửa lòng bàn tay tạo dấu cách (SPACE) qua mô hình Logistic Regression.
- Gợi ý từ ngữ (Offline Trie Dictionary 25,000 từ) và hỗ trợ xếp hạng ngữ cảnh/sinh câu hoàn chỉnh (QwenRanker / SentenceGenerator qua Ollama).
- Toàn bộ 49 test cases tích hợp và kiểm thử hồi quy đều đạt 100% OK.

### Thống kê tổng quan không gian lưu trữ hiện tại:
- **Tổng số file trong thư mục dự án:** 7,535 files.
- **Tổng dung lượng thư mục dự án:** 3,673.45 MB (~3.67 GB).
- **Số lượng file lớn (>= 10 MB):** 46 files.
- **Số lượng file vượt giới hạn GitHub (>= 100 MB):** 10 files (nếu push trực tiếp sẽ bị GitHub chặn 100%).

### Kết luận trạng thái sẵn sàng (Readiness Status):
> **TRẠNG THÁI: READY AFTER FIXES (SẴN SÀNG SAU KHI SỬA CHỮA CẤU HÌNH)**\
> Dự án **CHƯA ĐƯỢC PHÉP** chạy `git add .` hoặc `git push` ngay ở thời điểm hiện tại. Dự án cần thực hiện các bước chuẩn bị kỹ thuật (sửa `.gitignore`, chuẩn hóa đường dẫn tương đối cho mô hình SPACE và từ điển, làm sạch tài liệu README) để đảm bảo không rò rỉ dữ liệu video/hình ảnh cá nhân, không bị từ chối do file vượt 100MB, và người dùng mới clone về có thể chạy được ứng dụng.

---

## 2. PHÂN LOẠI CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY CLASSIFICATION)

Dự án được phân loại chi tiết theo 5 nhóm:
- **A. Required source code:** Mã nguồn bắt buộc — Cần đưa lên GitHub.
- **B. Required configuration / documentation:** Cấu hình và tài liệu bắt buộc — Cần đưa lên GitHub sau khi rà soát loại bỏ đường dẫn tuyệt đối.
- **C. Dataset / training artifacts:** Dữ liệu huấn luyện, trích xuất trung gian — Cần loại trừ (Exclude).
- **D. Generated results, logs, and debug data:** Kết quả sinh tự động, log thử nghiệm, ảnh chụp webcam — Cần loại trừ (Exclude).
- **E. Unclear / Empty:** Thư mục rỗng hoặc cần quyết định xử lý.

| Thư mục / Thành phần | Số file | Dung lượng | Phân loại | Đánh giá & Khuyến nghị |
|---|---|---|---|---|
| `src/` | 56 | 0.07 MB | **A** | Mã nguồn chuẩn hóa (public API, data loader, alphabet/digit/dynamic recognizers, OpenCV camera interface). **Publish 100%**. |
| `app/` | 136 | 2.21 MB | **A** | Mã nguồn ứng dụng thời gian thực (`app/camera/`, `app/inference/`, `app/interaction/`, `app/suggestion/`, `app/ui/`). **Publish 100%**. |
| `ai/preprocessing/` | 4 | ~0.05 MB | **A** | Mã nguồn chuẩn hóa landmark (`landmark_utils.py`, `hand_centric_2d.py`). **Publish**. |
| `ai/feature_extraction/` | 2 | ~0.02 MB | **A** | Mã nguồn trích xuất chuỗi đặc trưng động (`dynamic_sequence.py`). **Publish**. |
| `ai/common/` & `ai/config/` | 4 | ~0.02 MB | **A** | Hằng số và cấu hình dùng chung. **Publish**. |
| `ai/dataset/` | 382 | 1,480 MB | **C** | Hàng chục file CSV kích thước lớn (chứa landmark huấn luyện thô, audit nhãn sai, snapshots). **Loại trừ hoàn toàn (.gitignore)**. |
| `ai/raw_data/` | 2 | 123 MB | **C** | Chứa `raw_landmarks_chucai_all.csv` (122.9 MB). Vượt giới hạn GitHub. **Loại trừ (.gitignore)**. |
| `ai/quality_checkpoints/` | 3 | 59 MB | **C** | Báo cáo kiểm định hình học landmark và chất lượng ảnh thô. **Loại trừ (.gitignore)**. |
| `ai/phase2_outputs/` | 5 | 59.5 MB | **C/D** | Chứa model thử nghiệm trung gian `review_promotion_model.joblib` (43.7 MB) và out-of-fold CSVs. **Loại trừ (.gitignore)**. |
| `ai/audit_landmark_previews/` | ~250 | ~15 MB | **D** | Ảnh preview landmark chẩn đoán. **Loại trừ (.gitignore)**. |
| `ai/camera_debug_*/` | ~200 | ~35 MB | **D** | Ảnh chụp webcam chẩn đoán các cặp chữ cái nhầm lẫn. **Loại trừ (.gitignore)** (Bảo vệ quyền riêng tư). |
| `ai/models/` | 1 | 7.46 MB | **E** | Bản sao `hand_landmarker.task`. Nên dùng bản ở `models/mediapipe/`. **Loại trừ bản sao**. |
| `backend/` | 0 | 0 MB | **E** | Thư mục rỗng (không có file). Giữ lại kèm `.gitkeep` hoặc bỏ qua. |
| `database/` | 0 | 0 MB | **E** | Thư mục rỗng (không có file). Giữ lại kèm `.gitkeep` hoặc bỏ qua. |
| `debug/` | 0 | 0 MB | **E** | Thư mục rỗng (chỉ có subfolder con rỗng `digits/`, `letters/`). Giữ lại kèm `.gitkeep` hoặc bỏ qua. |
| `configs/` | 10 | 0.01 MB | **B** | Toàn bộ file JSON cấu hình hệ thống (`application.json`, `letters.json`, `digits.json`, `space.json`,...). **Publish sau khi sửa đường dẫn tuyệt đối trong `space.json`**. |
| `docs/` | 16 | 0.12 MB | **B** | Tài liệu kiến trúc, mô tả mô hình và refactor. **Publish**. |
| `models/letters/production/` | 8 | 0.25 MB | **B** | Mô hình ký tự tĩnh production (`uvw_medium_model.joblib`, scaler, label_mapping, manifests). **Publish**. |
| `models/digits/production/` | 4 | 0.95 MB | **B** | Mô hình chữ số production (`digits_final.joblib`, manifests). **Publish**. |
| `models/mediapipe/` | 1 | 7.46 MB | **B** | Model Google MediaPipe `hand_landmarker.task`. Cần quản lý phát hành riêng hoặc tải qua script. |
| `models/letters/fallback/` | 8 | 0.25 MB | **B** | Mô hình dự phòng đã qua kiểm định (`selected_model.joblib`). **Publish**. |
| `models/digits/candidates/` | 13 | 4.8 MB | **C/D** | Các checkpoint ứng viên cũ của chữ số. **Loại trừ khỏi git tracking**. |
| `models/dynamic_jz/dtw_templates/` | 32 | 0.9 MB | **D** | File JSON template DTW thử nghiệm trước đây. **Loại trừ hoặc lưu trữ**. |
| `reports/` | 51 | 1.63 MB | **B/D** | Các báo cáo kỹ thuật dạng Markdown (`STAGE*.md`) rất hữu ích để làm bằng chứng kỹ thuật. Tuy nhiên thư mục con `reports/data/` (chứa log webcam người dùng) cần được rà soát hoặc loại trừ. |
| `results/` | 3 | 0.06 MB | **B** | Chứa baseline hash phục vụ kiến trúc `test_src_architecture.py`. **Publish**. |
| `tests/` | 99 | 3.93 MB | **A** | Bộ kiểm thử tự động toàn diện (unit tests, integration tests). **Publish 100%**. |
| `tools/` | 58 | 1.92 MB | **A/B** | Các script công cụ trích xuất và kiểm tra webcam. **Publish sau khi chuẩn hóa đường dẫn**. |
| `training/` | 17 | 0.65 MB | **A** | Các script huấn luyện mô hình (không tự động chạy). **Publish**. |
| `data/` | 5,708 | 1,790 MB | **C/D** | Chứa video webcam, hàng ngàn ảnh JPG thực tế, kho lưu trữ `videos05.tgz` (890 MB). **Loại trừ 99.9%**, ngoại trừ: `data/dictionary/english_words_25k.tsv` và model J/Z `model_pipeline.joblib`. |
| Gốc: `run.py`, `README.md`, `requirements.txt`, `SYSTEM_CONTEXT.md` | 4 | ~0.05 MB | **A/B** | Entrypoint chính và tài liệu cốt lõi. **Publish**. |
| Gốc: `wlasl_sampled_16_review.zip` | 1 | 2.08 MB | **C** | File nén dữ liệu review WLASL. **Loại trừ (.gitignore)**. |
| Gốc: `jz_motion_log.txt` | 1 | 0.04 MB | **D** | File log runtime chuyển động bàn tay. **Loại trừ (.gitignore)**. |

---

## 3. LỖ HỔNG VÀ XUNG ĐỘT TRONG FILE `.GITIGNORE` HIỆN TẠI

File [.gitignore](file:///<project-root>/.gitignore) hiện tại có nội dung:
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# Notebooks / caches
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Large local data and generated artifacts (kept on disk, not source-controlled)
dulieu/
data/
models/**/*.joblib
models/**/*.task
reports/figures/
```

### Các xung đột nghiêm trọng đã phát hiện:
1. **Lỗi chặn toàn bộ mô hình Production (Fatal Runtime Blockers):**
   - Quy tắc `models/**/*.joblib` chặn toàn bộ:
     * `models/letters/production/uvw_medium_model.joblib` (218 KB)
     * `models/letters/production/uvw_medium_scaler.joblib` (1.8 KB)
     * `models/digits/production/digits_final.joblib` (464 KB)
     * `models/letters/fallback/selected_model.joblib` (218 KB)
   - Quy tắc `models/**/*.task` chặn:
     * `models/mediapipe/hand_landmarker.task` (7.46 MB)
   - Quy tắc `data/` chặn:
     * `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` (34 KB) — Mô hình nhận diện J/Z thực tế!
     * `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` (34 KB)
     * `data/dictionary/english_words_25k.tsv` (325 KB) — Từ điển gợi ý từ ngữ 25,000 từ!
   - **Hậu quả:** Nếu người dùng clone repository về theo `.gitignore` này, **100% mô hình và từ điển sẽ bị thiếu**, lệnh `python run.py` lập tức bị crash!

2. **Lỗi bỏ sót dữ liệu dung lượng lớn và video/ảnh nhạy cảm (Missing Exclusions):**
   - Không có quy tắc loại trừ `ai/dataset/`, `ai/raw_data/` khiến hàng chục file CSV (tổng cộng > 1.6 GB, trong đó có 7 file vượt 100 MB) bị lộ ra để git theo dõi.
   - Không có quy tắc loại trừ các định dạng nén (`*.zip`, `*.tgz`, `*.tar*`), dẫn đến `wlasl_sampled_16_review.zip` (2.1 MB) ở thư mục gốc không được bảo vệ.
   - Không có quy tắc loại trừ định dạng video (`*.avi`, `*.mp4`, `*.mov`), dẫn đến các video quay webcam cá nhân có thể bị theo dõi nếu đặt ngoài `data/`.
   - Không có quy tắc chặn log thử nghiệm (`*.log`, `jz_motion_log.txt`).
   - Không có quy tắc bảo vệ file môi trường `.env`.

### Đề xuất cấu hình `.gitignore` chuẩn hóa (Recommended `.gitignore`):
```gitignore
# ==============================================================================
# Python & Caches
# ==============================================================================
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.venv/
venv/
env/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# ==============================================================================
# OS & IDE
# ==============================================================================
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.swp
*.swo

# ==============================================================================
# Environment & Secrets
# ==============================================================================
.env
.env.*
*.pem
*.key
*.token

# ==============================================================================
# Media, Videos & Archives (Privacy & Bloat Protection)
# ==============================================================================
*.mp4
*.avi
*.mov
*.mkv
*.webm
*.zip
*.tar
*.tar.gz
*.tgz
*.rar
*.7z
*.log
jz_motion_log.txt

# ==============================================================================
# Raw Datasets & Intermediate Training Data
# ==============================================================================
dulieu/
ai/raw_data/
ai/dataset/
ai/quality_checkpoints/
ai/phase2_outputs/
ai/audit_landmark_previews/
ai/camera_debug_*/
ai/review_*/
ai/_cleanup_quarantine_v1/
ai/rescue_*.py
ai/visualize_*.py
ai/analyze_*.py
ai/audit_*.py

# ==============================================================================
# Data directory: Exclude datasets, keep runtime assets
# ==============================================================================
data/*
!data/dictionary/
!data/experiments/

# Inside data/experiments: exclude everything except dynamic J/Z runtime baseline
data/experiments/*
!data/experiments/dynamic_jz_binary_baseline_v1/
data/experiments/dynamic_jz_binary_baseline_v1/*
!data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib
!data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib
!data/experiments/dynamic_jz_binary_baseline_v1/metadata.json

# ==============================================================================
# Models: Ignore candidate/experimental checkpoints, preserve production models
# ==============================================================================
models/digits/candidates/
models/dynamic_jz/dtw_templates/
models/**/*.tmp

# MediaPipe bundle: can be tracked if using Git LFS or excluded if downloaded via script
# models/mediapipe/hand_landmarker.task

# Reports runtime artifacts
reports/figures/
reports/data/*.json
!reports/data/README.md
```

---

## 4. RÀ SOÁT BẢO MẬT VÀ QUYỀN RIÊNG TƯ (SECURITY & PRIVACY AUDIT)

### 4.1. Khóa bí mật, API Keys, Tokens và Mật khẩu
- **Kết quả quét regex:** Không tìm thấy API key, Token bí mật, Private Key, JWT hay Mật khẩu thực tế nào trong toàn bộ mã nguồn và cấu hình.
- `QwenRanker` kết nối tới Ollama qua endpoint HTTP nội bộ máy (`http://127.0.0.1:11434`), không sử dụng khóa xác thực.

### 4.2. Đường dẫn máy cục bộ và Tên người dùng cá nhân (Local Paths & Usernames)
Đã phát hiện **12 vị trí quan trọng** chứa đường dẫn tuyệt đối với ổ đĩa `C:\`, `D:\` và tên người dùng `<username>`:

1. **[configs/space.json](file:///<project-root>/configs/space.json#L5):**
   ```json
   "model_path": "<app-data-local>\\staging\\space_final_candidate_phase3_9_8_20261106\\final_space_model.joblib"
   ```
   *Rủi ro:* Cực kỳ nghiêm trọng. Khi chạy trên máy khác, file này không tồn tại, tính năng SPACE sẽ bị vô hiệu hóa.\
   *Khắc phục:* Di chuyển mô hình về `models/space/production/final_space_model.joblib` và đổi đường dẫn thành tương đối.

2. **[app/inference/space_predictor.py](file:///<project-root>/app/inference/space_predictor.py#L28):**
   ```python
   DEFAULT_MODEL_PATH = Path(r"<dev-home>\AppData\Local\NhanDienCuChi\staging\space_final_candidate_phase3_9_8_20261106\final_space_model.joblib")
   ```
   *Khắc phục:* Đổi thành đường dẫn tương đối trỏ vào `ROOT / "models" / "space" / "production" / "final_space_model.joblib"`.

3. **[app/inference/unified_abc123_jz_auto_experiment.py](file:///<project-root>/app/inference/unified_abc123_jz_auto_experiment.py#L495):**
   ```python
   Path(r"<project-mirror>\data\dictionary\english_words_25k.tsv")
   ```
   *Khắc phục:* Đã có fallback `ROOT / "data/dictionary/english_words_25k.tsv"`, nhưng cần xóa bỏ dòng chứa hardcoded `<dev-home>\...` để tránh lộ thông tin người dùng.

4. **[.vscode/settings.json](file:///<project-root>/.vscode/settings.json#L2):**
   ```json
   "python.defaultInterpreterPath": "<conda-env-nn>/python.exe"
   ```
   *Rủi ro:* Thư mục `.vscode/` đã được `.gitignore` bảo vệ, tuy nhiên nếu muốn chia sẻ thì nên cung cấp tài liệu hướng dẫn thay vì commit đường dẫn máy cá nhân.

5. **Các file công cụ phân tích trong `tools/`:**
   - [tools/run_stage2_experiments.py](file:///<project-root>/tools/run_stage2_experiments.py#L21) (`PROJECT_ROOT = Path("<project-mirror>")`)
   - [tools/audit_stage2_benchmark.py](file:///<project-root>/tools/audit_stage2_benchmark.py#L8) (`base = Path("<project-mirror>")`)
   - [tools/audit_stage2_config_comparison.py](file:///<project-root>/tools/audit_stage2_config_comparison.py#L6)
   - [tools/audit_stage2_templates.py](file:///<project-root>/tools/audit_stage2_templates.py#L7)
   - [tools/audit_stage2_validation_repro.py](file:///<project-root>/tools/audit_stage2_validation_repro.py#L9)
   - [tools/evaluate_stage10_wrist_relative.py](file:///<project-root>/tools/evaluate_stage10_wrist_relative.py#L467)
   - [tools/evaluate_stage9_trajectory.py](file:///<project-root>/tools/evaluate_stage9_trajectory.py#L324)
   - [tools/summarize_stage2_results.py](file:///<project-root>/tools/summarize_stage2_results.py#L3)\
   *Khắc phục:* Thay toàn bộ `Path("<project-mirror>")` bằng `Path(__file__).resolve().parents[1]`.

### 4.3. Dữ liệu hình ảnh và video người dùng (Webcam & Biometric Privacy)
- Thư mục `data/experiments/` chứa **4,658 ảnh JPG** và **209 file video** (`.avi`, `.mp4`, `.mov`).
- Các video này ghi lại hình ảnh khuôn mặt, bàn tay và phòng làm việc của người dùng trong quá trình kiểm thử webcam.
- **Bắt buộc:** Phải được loại trừ 100% bằng `.gitignore`, tuyệt đối không đưa lên kho lưu trữ công khai.

---

## 5. KIỂM TOÁN TẬP TIN DUNG LƯỢNG LỚN (LARGE FILE INVENTORY)

GitHub áp dụng các giới hạn:
- **Cảnh báo (Warning):** File lớn hơn 50 MB.
- **Chặn đẩy mã (Hard Block / Rejection):** File lớn hơn 100 MB.

### Bảng 1: Các file >= 100 MB (BẮT BUỘC LOẠI TRỪ)
| STT | Đường dẫn tương đối | Dung lượng (MB) | Định dạng | Mục đích | Khuyến nghị hành động |
|---|---|---|---|---|---|
| 1 | `data\experiments\dynamic_none_v1\raw\ipn_hand\archives\videos05.tgz` | 889.89 | `.tgz` | Kho nén video bộ dữ liệu IPN Hand bên ngoài | **Loại trừ (.gitignore)** |
| 2 | `ai\dataset\preprocessing_final_review_v1\wrong_label_audit_v2.csv` | 138.37 | `.csv` | Bảng landmark phân tích nhãn sai | **Loại trừ (.gitignore)** |
| 3 | `ai\dataset\wrong_label_audit.csv` | 136.45 | `.csv` | Bảng landmark phân tích nhãn sai | **Loại trừ (.gitignore)** |
| 4 | `ai\raw_data\raw_landmarks_chucai_all.csv` | 122.94 | `.csv` | Toàn bộ landmark thô của chữ cái | **Loại trừ (.gitignore)** |
| 5 | `ai\dataset\dataset_chucai_all.csv` | 121.67 | `.csv` | Toàn bộ tập dữ liệu chữ cái | **Loại trừ (.gitignore)** |
| 6 | `ai\dataset\phase2_quality_gate_v2\dataset_chucai_clean_reviewed_v2.csv` | 106.08 | `.csv` | Tập dữ liệu chữ cái đã làm sạch phase 2 | **Loại trừ (.gitignore)** |
| 7 | `ai\dataset\deep_cleanup_snapshot_v1\...\dataset_chucai_clean_reviewed_v2.csv` | 106.07 | `.csv` | Bản sao snapshot lưu trữ lịch sử | **Loại trừ (.gitignore)** |
| 8 | `ai\dataset\deep_cleanup_snapshot_v1\...\dataset_chucai_clean_reviewed_v2.csv` | 106.07 | `.csv` | Bản sao snapshot lưu trữ lịch sử | **Loại trừ (.gitignore)** |
| 9 | `ai\dataset\preprocessing_frozen_v3\dataset_chucai_clean_frozen_v3.csv` | 106.07 | `.csv` | Tập dữ liệu landmark chữ cái đóng băng v3 | **Loại trừ (.gitignore)** |
| 10 | `ai\dataset\dataset_chucai_clean.csv` | 105.80 | `.csv` | Tập dữ liệu chữ cái đã làm sạch | **Loại trừ (.gitignore)** |

### Bảng 2: Các file từ 50 MB đến 100 MB (CẦN LOẠI TRỪ TRÁNH WARNING)
| STT | Đường dẫn tương đối | Dung lượng (MB) | Định dạng | Mục đích | Khuyến nghị hành động |
|---|---|---|---|---|---|
| 11 | `data\experiments\dynamic_none_v1\raw\ipn_hand\pilot_videos\videos\4CM11_5_L_#10.avi` | 90.21 | `.avi` | Video cử chỉ IPN Hand | **Loại trừ (.gitignore)** |
| 12 | `ai\dataset\deep_cleanup_snapshot_v1\...\train_v2.csv` | 77.76 | `.csv` | Tập train MLP v2 snapshot | **Loại trừ (.gitignore)** |
| 13 | `ai\dataset\mlp_uvw_training_data_v1\train_uvw_v1.csv` | 77.73 | `.csv` | Tập train MLP uvw v1 | **Loại trừ (.gitignore)** |
| 14 | `ai\dataset\train_validation_test_split_v1\train.csv` | 74.40 | `.csv` | Tập train phân chia chính thức | **Loại trừ (.gitignore)** |
| 15 | `ai\dataset\landmark_quality_report.csv` | 70.11 | `.csv` | Báo cáo chất lượng landmark | **Loại trừ (.gitignore)** |

### Bảng 3: Các file đáng chú ý từ 10 MB đến 50 MB (Tổng cộng 31 files)
- `ai\phase2_outputs\run_20260917_111456_626441\review_promotion_model.joblib`: 43.73 MB (Model thử nghiệm trung gian, không dùng cho production).
- `ai\quality_checkpoints\geometry_quality.csv`: 40.35 MB.
- 7 video `.avi` trong `data\experiments\dynamic_none_v1\raw\ipn_hand\pilot_videos\`: ~23 MB mỗi file.
- 4 video preview `.avi` trong `data\experiments\dynamic_jz_v1\review\video_label_check_v1\`: 14 - 18 MB mỗi file.
- File nén `data\experiments\dynamic_jz_v1\review\quality_review_9_v1.zip`: 20.38 MB.
- 2 file `.jsonl` landmark trong `data\experiments\dynamic_jz_v1\processed\`: ~14.5 MB mỗi file.
- 1 file video MP4 WLASL `data\experiments\dynamic_none_v1\raw\wlasl_other_gesture\WLASL_70366.mp4`: 15.30 MB.

---

## 6. KIỂM TOÁN CÁC PHỤ THUỘC MÔ HÌNH THỜI GIAN THỰC (MODEL DEPENDENCY INVENTORY)

Để ứng dụng [run.py](file:///<project-root>/run.py) có thể chạy được trơn tru, hệ thống cần chính xác **6 thành phần mô hình và dữ liệu runtime**:

| Tên thành phần | Đường dẫn hiện tại | Dung lượng | Hash SHA-256 | Vai trò runtime | Chiến lược phát hành đề xuất |
|---|---|---|---|---|---|
| **MediaPipe Landmarker** | `models/mediapipe/hand_landmarker.task` | 7.46 MB | `c1d53284...` | Trích xuất 21 điểm tọa độ bàn tay 3D | Cung cấp script tự động tải `tools/download_models.py` từ Google Storage chính thức (Apache 2.0). |
| **ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | 218 KB | `829a4a75...` | Nhận diện 24 chữ cái tĩnh ASL (MLP) | Track trực tiếp trong Git (dung lượng nhỏ < 250 KB). |
| **ASL Letters Scaler** | `models/letters/production/uvw_medium_scaler.joblib` | 1.8 KB | `4043bbf6...` | Chuẩn hóa StandardScaler cho chữ cái | Track trực tiếp trong Git. |
| **Digits Model** | `models/digits/production/digits_final.joblib` | 464 KB | `6f38efb5...` | Pipeline nhận diện 10 chữ số (0–9) | Track trực tiếp trong Git (< 500 KB). |
| **Dynamic J/Z Model** | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | 34 KB | `8878b0c0...` | Logistic Regression phân loại J/Z | Di chuyển/Mirror vào `models/dynamic/production/` và track trong Git. |
| **Dynamic J/Z Scaler** | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 34 KB | `520025fa...` | Scaler cho đặc trưng chuỗi 16 frame | Di chuyển/Mirror vào `models/dynamic/production/` và track trong Git. |
| **SPACE Model** | `<dev-home>\AppData\Local\...\final_space_model.joblib` | 1.3 KB | `453162d8...` | Logistic Regression nhận diện SPACE | **BẮT BUỘC:** Di chuyển vào `models/space/production/final_space_model.joblib` và track trong Git. |
| **English Dictionary** | `data/dictionary/english_words_25k.tsv` | 325 KB | `a937a09d...` | Từ điển Trie 25,000 từ gợi ý | Whitelist trong `.gitignore` (`!data/dictionary/`) và track trong Git. |

*Nhận xét:* Tổng dung lượng của **toàn bộ mô hình học máy cần thiết** (bao gồm cả MediaPipe) chỉ khoảng **8.5 MB**. Toàn bộ các mô hình phân loại (chữ cái, chữ số, J/Z, SPACE) chỉ chiếm chưa đến **1 MB**, hoàn toàn phù hợp để quản lý phiên bản trực tiếp trong Git mà không cần dùng dịch vụ lưu trữ ngoài nặng nề.

---

## 7. BẢN QUYỀN VÀ NGUỒN GỐC TẬP DỮ LIỆU (DATASET & LICENSING CONCERNS)

1. **Google MediaPipe Hand Landmarker:**
   - Thuộc bản quyền của Google LLC.
   - Giấy phép: **Apache License 2.0**.
   - Hợp lệ để sử dụng và phân phối lại hoặc tải thông qua script chính thức.

2. **Dữ liệu bên thứ ba (Third-party Datasets) trong các thử nghiệm:**
   - **IPN Hand:** Dùng cho bài toán chẩn đoán cử chỉ chuyển tiếp và không hợp lệ. Phục vụ nghiên cứu khoa học.
   - **WLASL (World-Level ASL):** Dùng để đánh giá độ bao phủ landmark ngón tay.
   - **SigNN Video Data:** Dùng để trích xuất tập mẫu J/Z ban đầu.
   - *Khuyến cáo bản quyền:* Không được đưa các file nén (`videos05.tgz`, `wlasl_sampled_16_review.zip`) và các video gốc của các bộ dữ liệu này lên kho mã nguồn công khai. Chỉ nên trích dẫn link tài liệu học thuật trong phần References của README.

3. **Dữ liệu quay webcam cá nhân (`dulieu/chucai`, `dulieu/chuso`, `data/experiments/...`):**
   - Chứa hình ảnh khuôn mặt, bàn tay và phòng làm việc của thành viên nhóm.
   - *Bảo mật & Pháp lý:* Không phân phối công khai dữ liệu hình ảnh/video thô để bảo vệ quyền riêng tư cá nhân của sinh viên/nghiên cứu viên.

---

## 8. LỖ HỔNG TRONG README VÀ HỆ THỐNG PHỤ THUỘC (README & INSTALLATION GAPS)

Rà soát [README.md](file:///<project-root>/README.md) hiện tại và môi trường chạy:

1. **Thiếu hướng dẫn cài đặt mô hình:**
   - `README.md` không đề cập cách lấy file `hand_landmarker.task`. Nếu người dùng clone repo về khi file này bị `.gitignore` chặn, họ sẽ không biết tải từ đâu.
2. **Thiếu tài liệu về các tính năng mới:**
   - Chưa có mô tả về cử chỉ **SPACE** (ngửa lòng bàn tay để cách chữ).
   - Chưa có mô tả về thanh công cụ **Gợi ý từ ngữ (Word Suggestion)** và **Gợi ý câu (Sentence Generation)**.
   - Chưa hướng dẫn cách cài đặt và kết nối **Ollama** (với model `qwen3:4b`) nếu người dùng muốn trải nghiệm tính năng sinh câu bằng AI.
3. **Phụ thuộc gói thư viện trong `requirements.txt`:**
   - [requirements.txt](file:///<project-root>/requirements.txt) hiện đang ghim cứng các phiên bản nội bộ:
     * `opencv-python==5.0.0.93` (Phiên bản đặc biệt từ môi trường conda `nn`, trên PyPI chuẩn thông thường là `4.10.x` hoặc `4.11.x`).
     * `pandas==3.0.5`, `numpy==2.4.6`.
   - *Khuyến nghị:* Cung cấp `requirements.txt` chuẩn hóa với dải phiên bản tương thích linh hoạt (ví dụ: `opencv-python>=4.8.0`, `mediapipe>=0.10.0`, `scikit-learn>=1.3.0`, `numpy>=1.24.0`, `joblib>=1.3.0`, `pandas>=2.0.0`).
4. **Hướng dẫn chọn camera:**
   - Chưa có hướng dẫn chi tiết cách đổi `--camera-index` nếu người dùng sử dụng webcam gắn ngoài qua cổng USB thay vì webcam tích hợp.

---

## 9. DANH SÁCH FILE ĐỀ XUẤT ĐƯA LÊN GITHUB (RECOMMENDED FOR PUBLICATION)

Toàn bộ các file mã nguồn cốt lõi, cấu hình chuẩn hóa, tài liệu kỹ thuật và mô hình production:
1. **Entrypoint & Khởi động:**
   - `run.py`
   - `requirements.txt` (sau khi nới lỏng version)
   - `README.md` (sau khi cập nhật)
   - `.gitignore` (bản sửa đổi an toàn)
   - `LICENSE` (đề xuất: MIT License hoặc Apache 2.0)
2. **Mã nguồn ứng dụng:**
   - `app/**` (toàn bộ mã nguồn camera, inference, interaction, suggestion, ui)
   - `src/**` (toàn bộ API công khai, facades, contracts)
   - `ai/preprocessing/**` (`landmark_utils.py`, `hand_centric_2d.py`)
   - `ai/feature_extraction/**` (`dynamic_sequence.py`)
   - `ai/common/**`
   - `ai/config/**`
3. **Mã nguồn công cụ và kiểm thử:**
   - `tests/**` (toàn bộ 99 file test unit và integration)
   - `training/**` (các script pipeline huấn luyện chữ cái, chữ số, dynamic)
   - `tools/**` (các công cụ kiểm tra webcam, trích xuất)
4. **Cấu hình & Dữ liệu runtime bắt buộc:**
   - `configs/*.json` (bao gồm `application.json`, `letters.json`, `digits.json`, `space.json` sau khi sửa path)
   - `models/letters/production/*`
   - `models/letters/fallback/*`
   - `models/digits/production/*`
   - `models/digits/fallback/*`
   - `models/space/production/final_space_model.joblib` (sau khi sao chép từ AppData về repo)
   - `models/dynamic/production/` (hoặc giữ tại `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`)
   - `data/dictionary/english_words_25k.tsv`
   - `models/mediapipe/hand_landmarker.task` (hoặc tải qua setup script)
5. **Tài liệu kỹ thuật:**
   - `docs/**`
   - `SYSTEM_CONTEXT.md`
   - Các file báo cáo milestone quan trọng trong `reports/*.md`.

---

## 10. DANH SÁCH FILE VÀ THƯ MỤC CẦN LOẠI TRỪ (RECOMMENDED FOR EXCLUSION)

Tuyệt đối loại trừ khỏi kho lưu trữ Git công khai:
1. **Dữ liệu huấn luyện dung lượng lớn:**
   - `ai/dataset/` (~1.48 GB, chứa hàng trăm file CSV và bản sao snapshot).
   - `ai/raw_data/` (~123 MB, chứa `raw_landmarks_chucai_all.csv`).
   - `ai/quality_checkpoints/` (~59 MB).
   - `ai/phase2_outputs/` (~59.5 MB).
   - `dulieu/` (nếu có trên bất kỳ phân vùng nào).
2. **Video và hình ảnh cá nhân:**
   - Toàn bộ thư mục `data/experiments/` (ngoại trừ thư mục con chứa model J/Z `dynamic_jz_binary_baseline_v1`).
   - `ai/camera_debug_*/`, `ai/audit_landmark_previews/`, `ai/review_*/`.
   - Các file mở rộng: `*.avi`, `*.mp4`, `*.mov`, `*.jpg`, `*.png` trong các thư mục dữ liệu thử nghiệm.
3. **Các file nén và kho lưu trữ lớn:**
   - `data/experiments/dynamic_none_v1/raw/ipn_hand/archives/videos05.tgz` (889.9 MB).
   - `data/experiments/dynamic_jz_v1/review/quality_review_9_v1.zip` (20.4 MB).
   - `wlasl_sampled_16_review.zip` (2.08 MB).
4. **Log và chẩn đoán:**
   - `jz_motion_log.txt`
   - `reports/data/*.json` (các log ghi nhận phiên webcam của người dùng).
5. **Cấu hình máy cục bộ:**
   - `.vscode/` (đặc biệt là `settings.json` chứa đường dẫn `<conda-env>`).
   - `__pycache__/`, `*.pyc`, `.pytest_cache/`.

---

## 11. CÁC RỦI RO CÒN LẠI (REMAINING RISKS)

1. **Rủi ro người dùng mới clone về không chạy được mô hình SPACE:**
   - Nếu không sao chép `final_space_model.joblib` từ `<dev-home>\AppData\Local\...` vào trong thư mục dự án và cập nhật lại [configs/space.json](file:///<project-root>/configs/space.json), người dùng tải repo về sẽ không có tính năng SPACE.
2. **Rủi ro push nhầm file quá 100 MB làm hỏng lịch sử Git:**
   - Nếu thực hiện `git add .` trước khi cập nhật `.gitignore`, các file CSV như `wrong_label_audit_v2.csv` (138 MB) hay `videos05.tgz` (889 MB) sẽ bị thêm vào commit index, khiến lệnh `git push` bị từ chối và lịch sử `.git` bị phình to hàng gigabyte.
3. **Rủi ro bản quyền khi đưa model `hand_landmarker.task` lên GitHub:**
   - File `hand_landmarker.task` có dung lượng ~7.5 MB. Mặc dù thuộc Apache 2.0 của Google, cách chuyên nghiệp nhất cho kho nguồn mở là cung cấp script `tools/download_models.py` để tải trực tiếp từ Google Storage khi cài đặt, hoặc đính kèm vào GitHub Releases.
4. **Rủi ro tương thích hệ điều hành (Windows vs Linux/macOS):**
   - Hiện tại hệ thống đang được chạy trên Windows với các đường dẫn phân cách bằng dấu gạch chéo ngược `\` trong một số config. Khi lên GitHub, nên đảm bảo mã nguồn sử dụng `pathlib.Path` để tương thích đa nền tảng.

---

## 12. KẾ HOẠCH HÀNH ĐỘNG KHUYẾN NGHỊ (STEP-BY-STEP REMEDIATION PLAN)

Sau khi được người dùng phê duyệt, các bước thực hiện tiếp theo sẽ là:

- [ ] **Bước 1: Sao chép mô hình SPACE vào dự án**\
  Sao chép `final_space_model.joblib` từ `<dev-home>\AppData\Local\NhanDienCuChi\staging\space_final_candidate_phase3_9_8_20261106\` vào `<project-root>\models\space\production\final_space_model.joblib`.
- [ ] **Bước 2: Chuẩn hóa đường dẫn tương đối**\
  Cập nhật [configs/space.json](file:///<project-root>/configs/space.json) và [app/inference/space_predictor.py](file:///<project-root>/app/inference/space_predictor.py) để dùng đường dẫn tương đối trong dự án. Làm sạch đường dẫn hardcoded trong [app/inference/unified_abc123_jz_auto_experiment.py](file:///<project-root>/app/inference/unified_abc123_jz_auto_experiment.py) và các tool trong `tools/`.
- [ ] **Bước 3: Cập nhật file `.gitignore` an toàn**\
  Áp dụng bộ quy tắc `.gitignore` đã đề xuất tại Mục 3, mở khóa các mô hình production và từ điển, đồng thời chặn 100% video, dataset CSV nặng và dữ liệu webcam.
- [ ] **Bước 4: Tạo script tải mô hình tự động (`tools/download_models.py`)**\
  Hỗ trợ người dùng tự động tải `hand_landmarker.task` từ Google Storage chỉ với một lệnh.
- [ ] **Bước 5: Hoàn thiện `requirements.txt` và `README.md`**\
  Cập nhật tài liệu hướng dẫn cài đặt chuẩn, mô tả đầy đủ các tính năng (ABC, 123, J/Z, SPACE, Word Suggestions, Ollama).
- [ ] **Bước 6: Chạy kiểm thử xác minh độc lập trên thư mục sạch**\
  Chạy `python run.py --check-only` và bộ test 49 test cases để đảm bảo mọi chức năng hoạt động hoàn hảo trước khi khởi tạo Git.
