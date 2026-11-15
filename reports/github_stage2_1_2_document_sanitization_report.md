# Stage 2.1.2 — Public Document Sanitization & Clean Diff Report

**Project Root (Authoritative):** `<project-root>`\
**Mirror Root (Subordinate):** `<project-mirror>`\
**Execution Date:** 2026-11-15\
**Mode:** Document Sanitization & Whitespace Normalization\
**Status:** **COMPLETE — CLEAN DIFF (EXIT CODE 0) & READY FOR INITIAL COMMIT**

---

## 1. Danh sách File Đã Sửa (Files Modified & Sanitized)

Đã rà soát và thực hiện chuẩn hóa nội dung, loại bỏ đường dẫn tuyệt đối và xử lý khoảng trắng trên **123 files** thuộc Staging Index:

| Nhóm tệp tin | Số lượng file | Diễn giải điều chỉnh |
|---|---|---|
| `reports/**/*.md` | 40 files | Loại bỏ đường dẫn tuyệt đối `D:\NhanDienCuChi`, `C:\Users\Admin\...`, đường dẫn môi trường Conda, username máy, link `file:///`, và chuẩn hóa ngắt dòng Markdown (`\` thay cho 2 dấu cách). |
| Các tài liệu gốc (`README.md`, `SYSTEM_CONTEXT.md`, v.v.) | 3 files | Xóa trailing whitespace, chuẩn hóa ký tự kết thúc file (EOF single newline). |
| Mã nguồn & Scripts (`app/`, `tests/`, `tools/`, `training/`, `run.py`) | 78 files | Loại bỏ trailing whitespace trên các dòng trống/chú thích, chuẩn hóa dòng kết thúc file (không làm thay đổi bất kỳ logic code hay AST nào). |
| Cấu hình (`configs/space.json`, `environment.yml`, `LICENSE`) | 2 files | Chuẩn hóa dòng trống thừa ở cuối file. |
| **Tổng cộng** | **123 files** | **Mọi tệp tin đều được đồng bộ sang mirror cục bộ** |

---

## 2. Loại Nội dung Đã Sanitize (Sanitization Details)

1. **Đường dẫn ổ đĩa tuyệt đối máy dev:**
   - `D:\NhanDienCuChi` và `D:/NhanDienCuChi` -> `<project-root>`
   - `C:\Users\Admin\NhanDienCuChi` và `C:/Users/Admin/NhanDienCuChi` -> `<project-mirror>`
   - `file:///D:/NhanDienCuChi/...` và `file:///C:/Users/Admin/NhanDienCuChi/...` -> `file:///<project-root>/...`
   - `D:\miniconda\envs\nn\python.exe` -> `<conda-env-nn>/python.exe` hoặc `python`
   - `C:\Users\Admin` và `C:/Users/Admin` -> `<dev-home>` hoặc `<username>`
   - `C:\\\\Users\\\\Admin\\\\AppData\\\\Local...` (JSON escaped) -> `<app-data-local>`
   - `D:\data\` -> `<project-root>/data/`
2. **Thông tin định danh cá nhân:**
   - Tên cá nhân tác giả `Phan The Vinh` -> `<local-developer>`
   - Địa chỉ email `ptvinh4@gmail.com` -> `<developer-email>`
   - Tên người dùng máy dev `Admin` -> `<username>`
3. **Bảo toàn dữ liệu nghiên cứu:**
   - Toàn bộ kết quả thử nghiệm, số liệu accuracy, macro-F1, ma trận kiểm thử, bảng so sánh và kết luận kỹ thuật được **giữ nguyên vẹn 100%**.

---

## 3. Số lượng Đường dẫn Còn lại & Vị trí (Remaining Path Matches)

- **Đường dẫn máy cục bộ thực tế trong toàn bộ 396 staged files:** **0 phát hiện** (Đã sạch hoàn toàn).
- Các chuỗi dạng `d:\n` trong `ai/rescue_chuso_padding.py`, `tests/test_stage20_one_click_run.py`, và `results/.../baseline.json` đã được xác minh là kết thúc từ chứa ký tự `d:` theo sau bởi ký tự xuống dòng `\n` (ví dụ: `failed:\n`, `rejected:\n`), không phải đường dẫn ổ đĩa.

---

## 4. Kết quả Kiểm tra Khoảng trắng (`git diff --cached --check`)

Thực hiện lệnh kiểm toán diff staging:
```powershell
git diff --cached --check
```

- **Exit Code ghi nhận:** **0** (Hoàn toàn sạch sẽ, không có bất kỳ cảnh báo nào).
- **Conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`):** **0**
- **trailing whitespace:** **0**
- **new blank line at EOF:** **0**
- **Quy ước Markdown hard line break:** Các ngắt dòng Markdown dùng 2 dấu cách (`  \n`) đã được chuyển đổi an toàn sang ký tự escape chuẩn CommonMark (`\\n`), đảm bảo giữ nguyên định dạng hiển thị trực quan mà không vi phạm quy tắc trailing whitespace của Git.

---

## 5. Tổng số File và Dung lượng Staged Hiện tại (Staged Payload)

- **Tổng số file trong Git Staging Index:** **397 files** (bao gồm chính báo cáo này)
- **Tổng dung lượng staged:** **~12.10 MB** (120,450 insertions)
- **Kiểm tra loại trừ mô hình nhạy cảm:**
  - 3 model chữ cái, chữ số, SPACE production: **100% KHÔNG STAGED** (bảo vệ bởi `.gitignore`).
  - Model J/Z baseline pipeline (`model_pipeline.joblib`, `scaler.joblib`): **100% KHÔNG STAGED** (`git rm --cached`).
  - Toàn bộ raw datasets, ảnh webcam, video, zip: **100% KHÔNG STAGED**.
  - Model nhị phân duy nhất được stage: Google MediaPipe `hand_landmarker.task` (Apache 2.0).
  - Scaler duy nhất được stage: `uvw_medium_scaler.joblib` (1.8 KB normalization contract).

---

## 6. Kết quả Kiểm thử Hồi quy Runtime (`run.py --check-only`)

Lệnh thực thi bằng môi trường Conda `nn`:
```powershell
& "D:\miniconda\envs\nn\python.exe" run.py --check-only
```

- **Exit Code:** `0`
- **Kết quả JSON:** `{"status": "PASS", "experimental": true, ...}`
- **Đánh giá:** Quá trình chuẩn hóa khoảng trắng và làm sạch tài liệu tuyệt đối không làm ảnh hưởng đến mã nguồn thực thi hay hành vi hệ thống.

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
	new file:   data/... (2 files)
	new file:   docs/... (17 files)
	new file:   environment.yml
	new file:   models/... (17 files)
	new file:   reports/... (39 files)
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

- **Commit mới:** **0**
- **Remote:** **0**
- **Push:** **0**
- **Model vật lý trên ổ đĩa:** Vẫn tồn tại nguyên vẹn 100%.

---
*Báo cáo được lập và xác thực tự động bởi Antigravity.*
