# Stage 2.1 — Selective Staging & Review Report

**Project Root (Authoritative):** `<project-root>`\
**Mirror Root (Subordinate):** `<project-mirror>`\
**Date:** 2026-11-15\
**Execution Mode:** Selective Git Staging & Staged Content Security Audit\
**Status:** **STAGING COMPLETE — AWAITING USER REVIEW & COMMIT APPROVAL**\

---

## 1. Kết quả kiểm tra trước Staging (Pre-Staging Verification)

Trước khi thực hiện lệnh stage, trạng thái repository đã được kiểm tra nghiêm ngặt:
- **Repository Root:** `<project-root>` (Chính xác).
- **Current Branch:** `main` (Chính xác).
- **Remote Configured:** Không có (Trống).
- **Staged Files trước khi chạy:** 0 files (`git diff --cached` không có dữ liệu).
- **Số commit hiện tại:** 0 commits (`No commits yet`).

---

## 2. Kết quả Staging và Tổng hợp Trọng tải (Staged Payload Summary)

Lệnh `git add -A` đã được thực thi sau khi thẩm định toàn diện `.gitignore`:

- **Tổng số file đã được Stage:** **396 files** (hoặc 397 files bao gồm chính báo cáo này)
- **Tổng dung lượng Staged:** **12,701,147 bytes (~12.1128 MB)**
- **Số dòng mã nguồn được bổ sung:** **120,052 insertions**
- **File có dung lượng lớn nhất:** `models/mediapipe/hand_landmarker.task` (**7,819,105 bytes / ~7.46 MB** — Apache 2.0 Google MediaPipe bundle)
- **File lớn thứ hai:** `data/dictionary/english_words_25k.tsv` (**291,528 bytes / ~0.28 MB** — CC BY-SA 4.0 SUBTLEX-US 25k word dictionary)

### Cơ cấu định dạng trong Staging Index
| Phần mở rộng | Số lượng file | Diễn giải |
|---|---|---|
| `.py` | 295 | Toàn bộ mã nguồn hệ thống, module AI, coordinator, test suite, tools |
| `.md` | 56 | Toàn bộ tài liệu kiến trúc, hướng dẫn sử dụng, báo cáo kiểm định |
| `.json` | 32 | Cấu hình tham số, manifest kiểm thử, siêu dữ liệu thí nghiệm |
| `.joblib` | 3 | Dynamic J/Z Baseline Pipeline (36 KB), Scaler J/Z (27 KB), Scaler chữ cái (1.8 KB) |
| `.csv` | 3 | Ma trận thử nghiệm và bảng metadata trong `docs/model_v2_p_horizontal_design_v1/` |
| `<no-ext>` | 2 | File `LICENSE` (MIT) và `.gitignore` |
| `.txt` | 2 | `requirements.txt` (182 B) và bảng ký tự tham chiếu `ai/config/ky_tu.txt` (76 B) |
| `.tsv` | 1 | Từ điển 25,000 từ tiếng Anh runtime `data/dictionary/english_words_25k.tsv` |
| `.yml` | 1 | Cấu hình môi trường Conda `environment.yml` |
| `.task` | 1 | Google MediaPipe Hand Landmarker model bundle `hand_landmarker.task` |
| **Tổng cộng** | **396** | **~12.11 MB** |

---

## 3. Các File bị Loại trừ và Lý do (Excluded Assets & Rationales)

| Hạng mục loại trừ | Đường dẫn / Mẫu | Dung lượng xấp xỉ | Lý do loại trừ |
|---|---|---|---|
| **Static ASL Letters Model** | `models/letters/production/uvw_medium_model.joblib` | 218 KB | Dữ liệu huấn luyện từ internet chưa có biên bản bản quyền rõ ràng; tuân thủ chính sách Source-Only. |
| **Static Digits Model** | `models/digits/production/digits_final.joblib` | 464 KB | Dữ liệu nội bộ cá nhân của tác giả; giữ trên máy dev theo chính sách Source-Only. |
| **SPACE Gesture Model** | `models/space/production/final_space_model.joblib` | 1.4 KB | Dữ liệu nội bộ tác giả; giữ trên máy dev theo chính sách Source-Only. |
| **Model ứng viên / Fallback** | `models/letters/fallback/*.joblib`, `models/digits/fallback/*.joblib`, `models/digits/production/mlp_large_v2a.joblib` | ~1.4 MB | Checkpoint thử nghiệm cũ / dư thừa, tránh làm phình repository. |
| **Dữ liệu hình ảnh thô** | `dulieu/chucai/`, `dulieu/chuso/` | ~1.8 GB | Dataset ảnh gốc nặng, không cần thiết cho mã nguồn phát hành. |
| **Dữ liệu trung gian & Video** | `ai/dataset/`, `ai/raw_data/`, `*.mp4`, `*.avi`, `*.zip` | ~2.0 GB | Landmark captures, video debug, file nén dung lượng lớn. |
| **Contact Sheet Webcam** | `ai/final_promotion_audit/` | ~3.5 MB | Chứa ảnh ghép frame khuôn mặt và bàn tay người thử nghiệm. |
| **File từ điển thô** | `data/dictionary/source/` | ~3.3 MB | Bản text thô SUBTLEX 74,000 từ chưa nén; đã có bản lọc runtime `25k.tsv`. |
| **Bộ nhớ đệm runtime** | `__pycache__/`, `.pytest_cache/`, `reports/figures/`, `reports/data/*.json` | ~15 MB | Artifacts sinh ra trong lúc chạy runtime. |

---

## 4. Kết quả Kiểm tra Bảo mật Nội dung Staged (Staged Security Audit)

Toàn bộ 396 files đã được quét nội dung tự động trước khi xác nhận staging:
- **API Keys / Secrets / Tokens:** **0 phát hiện** (Không có khóa bí mật nào lọt vào staging).
- **Private Keys:** **0 phát hiện**.
- **Connection Strings:** **0 phát hiện**.
- **Webcam Photos / Base64 Images:** **0 phát hiện**.
- **Personal Emails:** **0 phát hiện**.
- **Đường dẫn tuyệt đối trong mã nguồn:** **0 phát hiện** (Các script công cụ `tools/evaluate_stage9_trajectory.py`, `tools/evaluate_stage10_wrist_relative.py` và `docs/model_v2_p_horizontal_design_v1/evidence_manifest.json` đã được thay thế bằng đường dẫn động portable `Path.home()` và `<project-root>`).

---

## 5. Kết quả Kiểm tra Model & License Integrity

1. **3 Model Local Unverified:**
   Kiểm tra lại bằng `git check-ignore -v`:
   ```text
   .gitignore:66:models/letters/production/uvw_medium_model.joblib    models/letters/production/uvw_medium_model.joblib
   .gitignore:67:models/digits/production/digits_final.joblib        models/digits/production/digits_final.joblib
   .gitignore:68:models/space/production/final_space_model.joblib     models/space/production/final_space_model.joblib
   ```
   **Khẳng định:** 100% không bị stage, không bị sửa đổi, vẫn tồn tại nguyên vẹn trên ổ đĩa máy dev.
2. **Model được Stage:**
   - `models/mediapipe/hand_landmarker.task`: Apache 2.0 (Google LLC) — Đầy đủ quyền phân phối.
   - `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`: SigNN ASL Video (CC0 Public Domain, trạng thái `EXPERIMENTAL_NOT_PRODUCTION`).
   - `models/letters/production/uvw_medium_scaler.joblib`: StandardScaler normalization contract (1.8 KB, không chứa weights học).
3. **Từ điển được Stage:**
   - `data/dictionary/english_words_25k.tsv`: SUBTLEX-US 25k (CC BY-SA 4.0).
   - Đi kèm thông báo bản quyền bắt buộc [data/dictionary/NOTICE.md](file:///<project-root>/data/dictionary/NOTICE.md).

---

## 6. Các vấn đề còn tồn tại & Đánh giá rủi ro

- **Cảnh báo chuyển đổi CRLF/LF của Git trên Windows:**
  Git thông báo `warning: LF will be replaced by CRLF the next time Git touches it` đối với các file text. Đây là cơ chế chuẩn `core.autocrlf` của Git trên Windows, hoàn toàn vô hại và đảm bảo tính tương thích đa nền tảng (Windows/Linux/macOS).
- **Không có bất kỳ blocker hay vi phạm bản quyền/bảo mật nào.**

---

## 7. Trạng thái Git Hiện tại (Git Status)

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
	new file:   data/... (4 files)
	new file:   docs/... (17 files)
	new file:   environment.yml
	new file:   models/... (17 files)
	new file:   reports/... (37 files)
	new file:   requirements.txt
	new file:   results/... (3 files)
	new file:   run.py
	new file:   src/... (25 files)
	new file:   tests/... (99 files)
	new file:   tools/... (59 files)
	new file:   training/... (24 files)
```

---

## 8. Kết luận & Điều kiện dừng (Stop Condition)

Giai đoạn **STAGE 2.1** đã hoàn tất:
- Toàn bộ 396 file hợp lệ đã được đưa vào Git Staging Index sẵn sàng cho commit đầu tiên.
- Không có bất kỳ model unverified, secret hay dữ liệu nhạy cảm nào bị stage.
- **TUYỆT ĐỐI CHƯA CHẠY `git commit` HOẶC `git push`**.

**HỆ THỐNG ĐÃ DỪNG LẠI ĐỂ BẠN XEM XÉT VÀ PHÊ DUYỆT DANH SÁCH STAGED FILES TRƯỚC KHI THỰC HIỆN COMMIT.**

---
*Báo cáo được lập và xác thực bởi Antigravity.*
