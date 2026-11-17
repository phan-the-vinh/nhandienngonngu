# Báo Cáo Kiểm Kê Vị Trí Dataset và Xác Minh Khả Năng Tái Lập Hệ Thống Từ GitHub

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu (NhanDienCuChi)  
**Workspace:** `D:\NhanDienCuChi`  
**GitHub Repository:** [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu) (Branch `main`, Commit [`65c38d6`](https://github.com/phan-the-vinh/nhandienngonngu/commit/65c38d6))  
**Google Drive Dataset:** [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)  
**Thời điểm thực hiện:** 17/11/2026 12:45:00 (UTC+7)  
**Nguyên tắc kiểm toán:** Read-Only tuyệt đối; không sửa code, không di chuyển/xóa dữ liệu, không commit/push.

---

## 1. Bảng Vị Trí Toàn Bộ Dataset Trên Máy Tính

Đã quét toàn diện ổ đĩa `D:\`, bao gồm workspace `D:\NhanDienCuChi`, thư mục sao lưu dữ liệu gốc `D:\NhanDienCuChi_RAW_BACKUP` và bản sao lưu trước khôi phục `D:\NhanDienCuChi_Backup_Before_GitHub_Restore`.

### 1.1. Dữ liệu gốc (Raw Datasets — Ảnh và Video)

| Phân hệ / Nhóm | Đường dẫn tuyệt đối | Số lượng file | Dung lượng | Định dạng | Vai trò | Loại dữ liệu |
|---|---|---:|---:|---|---|:---:|
| **ABC (Chữ cái tĩnh)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\chucai` | 134.467 files | 2,86 GB (2.861.676.652 B) | `.jpg` (65.755)<br>`.png` (67.037)<br>`.jpeg` (1.675) | Ảnh cử chỉ bàn tay 24 chữ cái ASL (A-I, K-Y) | **Dữ liệu gốc** |
| **ABC (Bản nén ZIP)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\ABC.zip` | 1 file | 2,66 GB (2.853.078.103 B) | `.zip` | Gói nén phục vụ lưu trữ / upload Google Drive | **Gốc nén** |
| **123 (Chữ số tĩnh)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\chuso` | 1.000 files | 41,12 MB (43.117.292 B) | `.jpeg` (700)<br>`.png` (300) | Ảnh cử chỉ bàn tay cho 10 chữ số (0–9) | **Dữ liệu gốc** |
| **123 (Bản nén ZIP)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\123.zip` | 1 file | 40,48 MB (42.442.485 B) | `.zip` | Gói nén phục vụ lưu trữ / upload Google Drive | **Gốc nén** |
| **J/Z (Cử chỉ động)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\signnteam_jz` | 712 files | 3,93 GB (4.226.112.284 B) | `.avi` (712 video) | Video cử chỉ chuyển động ký hiệu J và Z | **Dữ liệu gốc** |
| **J/Z (Bản nén ZIP)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\jz.zip` | 1 file | 3,88 GB (4.165.691.386 B) | `.zip` | Gói nén phục vụ lưu trữ / upload Google Drive | **Gốc nén** |
| **SPACE (Dấu cách)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space` | 2.997 files | 36,38 MB (38.148.575 B) | `.jpg` (2.997) | Ảnh lòng bàn tay mở với ngón cái mở rộng | **Dữ liệu gốc** |
| **SPACE (Bản nén ZIP)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space.zip` | 1 file | 36,12 MB (37.879.237 B) | `.zip` | Gói nén phục vụ lưu trữ / upload Google Drive | **Gốc nén** |
| **Trọn bộ Raw ZIP** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset.zip` | 1 file | 6,61 GB (7.102.473.537 B) | `.zip` | Tổng hợp nén toàn bộ 4 nhánh dữ liệu thô | **Gốc nén tổng hợp** |

---

### 1.2. Dữ liệu đã xử lý thành Landmark / Vector đặc trưng (Processed Feature Datasets)

| Tên tập dữ liệu | Đường dẫn tuyệt đối | Số mẫu / Dòng | Dung lượng | Định dạng | Vai trò & Mục đích sử dụng | Loại dữ liệu |
|---|---|---:|---:|---|---|:---:|
| **ABC Clean Frozen v3** | `D:\NhanDienCuChi\ai\dataset\preprocessing_frozen_v3\dataset_chucai_clean_frozen_v3.csv` | 89.415 dòng | 106,07 MB | `.csv` (63 cột đặc trưng) | Tập landmark sạch chuẩn hóa toàn diện của 24 chữ cái tĩnh | **Đã xử lý** (Frozen) |
| **ABC Train Split v1** | `D:\NhanDienCuChi\ai\dataset\train_validation_test_split_v1\train.csv` | 62.591 dòng | 74,40 MB | `.csv` (63 đặc trưng) | Tập huấn luyện cơ sở (70% split) của nhánh ABC | **Đã xử lý** (Split) |
| **ABC Val Split v1** | `D:\NhanDienCuChi\ai\dataset\train_validation_test_split_v1\validation.csv` | 13.412 dòng | 15,83 MB | `.csv` (63 đặc trưng) | Tập kiểm định (15% split) của nhánh ABC | **Đã xử lý** (Split) |
| **ABC Test Split v1** | `D:\NhanDienCuChi\ai\dataset\train_validation_test_split_v1\test.csv` | 13.412 dòng | 15,83 MB | `.csv` (63 đặc trưng) | Tập kiểm thử độc lập (15% split) của nhánh ABC | **Đã xử lý** (Split) |
| **ABC Production Train** | `D:\NhanDienCuChi\ai\dataset\mlp_uvw_training_data_v1\train_uvw_v1.csv` | 62.648 dòng | 77,73 MB | `.csv` (63 đặc trưng + nhãn) | 62.591 mẫu base + 57 mẫu webcam bổ sung huấn luyện mô hình sản xuất `uvw_medium` | **Đã xử lý** (Production) |
| **123 Digits Train Frozen** | `D:\NhanDienCuChi\data\processed\digits\digits_final_train_frozen.csv` | 549 dòng | 649,5 KB | `.csv` (63 đặc trưng + nhãn) | Tập huấn luyện chính thức mô hình chữ số sản xuất `digits_final` | **Đã xử lý** (Frozen) |
| **123 Digits Regression Ref** | `D:\NhanDienCuChi\data\splits\digits\regression_reference.csv` | 250 dòng | 315,1 KB | `.csv` (63 đặc trưng + nhãn) | Tập đối chuẩn hồi quy (25 mẫu/lớp) của nhánh chữ số | **Đã xử lý** (Split) |
| **123 Digits Validation Ref** | `D:\NhanDienCuChi\data\splits\digits\validation_reference.csv` | 100 dòng | 127,0 KB | `.csv` (63 đặc trưng + nhãn) | Tập kiểm định (10 mẫu/lớp) của nhánh chữ số | **Đã xử lý** (Split) |
| **J/Z Dynamic Feature JSONL** | `D:\NhanDienCuChi\data\experiments\dynamic_jz_v1\processed\signnteam_jz_exclude_j40_v1.jsonl` | 640 chuỗi video | 14,11 MB | `.jsonl` (chuỗi 16 frame, 69 dims) | Tập chuỗi đặc trưng động 16 bước thời gian trích xuất từ 640 video sạch | **Đã xử lý** (Sequence) |
| **J/Z Baseline Split Manifest** | `D:\NhanDienCuChi\data\experiments\dynamic_jz_binary_baseline_v1\split_manifest.json` | 640 video (512 train, 64 val, 64 test) | 84,1 KB | `.json` | Phân chia chỉ mục train/val/test chính thức cho mô hình J/Z | **Metadata/Split** |
| **SPACE Negative Training Pool** | Trích xuất hợp nhất từ `train.csv` (ABC) và `digits_final_train_frozen.csv` | 63.140 mẫu âm | N/A (ghép trực tiếp khi train) | Bộ nhớ runtime | 62.591 mẫu ABC + 549 mẫu chữ số đóng vai trò lớp âm `NOT_SPACE` | **Đã xử lý** (Negative) |

---

## 2. Đường Dẫn Dataset Được Code Sử Dụng

### 2.1. Mã nguồn trích xuất dữ liệu gốc (Raw -> Processed)
* **Trích xuất ảnh ABC:** [tools/extract_letters.py](file:///D:/NhanDienCuChi/tools/extract_letters.py)
  * Đọc từ: `dulieu/chucai/` (hoặc `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\chucai`)
  * Ghi ra: `ai/dataset/dataset_chucai_all.csv` và `ai/raw_data/raw_landmarks_chucai_all.csv`
* **Trích xuất ảnh Chữ số:** [tools/extract_digits.py](file:///D:/NhanDienCuChi/tools/extract_digits.py)
  * Đọc từ: `dulieu/chuso/` (hoặc `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\chuso`)
  * Ghi ra: `ai/dataset/kytuso/dataset_chuso_all.csv`
* **Trích xuất video J/Z:** [tools/extract_dynamic.py](file:///D:/NhanDienCuChi/tools/extract_dynamic.py)
  * Đọc từ: `data/raw/dynamic/external/signnteam_jz/SigNN Video Data/` (hoặc `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\signnteam_jz`)
  * Ghi ra: `data/experiments/dynamic_jz_v1/processed/`

### 2.2. Mã nguồn huấn luyện mô hình (Training Pipelines)
* **Huấn luyện ABC sản xuất (`uvw_medium`):** [training/letters/train_uvw_validation.py](file:///D:/NhanDienCuChi/training/letters/train_uvw_validation.py)
  * Đọc từ: `ai/dataset/mlp_uvw_training_data_v1/train_uvw_v1.csv`
  * Ghi ra: `models/letters/production/uvw_medium_model.joblib`
* **Huấn luyện Chữ số sản xuất (`digits_final`):** [training/digits/train_digits.py](file:///D:/NhanDienCuChi/training/digits/train_digits.py)
  * Cấu hình: [configs/digits_training.json](file:///D:/NhanDienCuChi/configs/digits_training.json) dòng 2: `"train_data": "data/processed/digits/digits_final_train_frozen.csv"`
  * Đọc từ: `data/processed/digits/digits_final_train_frozen.csv`
  * Ghi ra: `models/digits/production/digits_final.joblib`
* **Huấn luyện J/Z động (`dynamic_jz_binary_baseline_v1`):** [training/dynamic/train_dynamic.py](file:///D:/NhanDienCuChi/training/dynamic/train_dynamic.py)
  * Cấu hình: [configs/dynamic_training.json](file:///D:/NhanDienCuChi/configs/dynamic_training.json) dòng 3–6:
    * `"input_jsonl": "data/experiments/dynamic_jz_v1/processed/signnteam_jz_exclude_j40_v1.jsonl"`
    * `"output_dir": "data/experiments/dynamic_jz_binary_baseline_v1"`
  * Ghi ra: `model_pipeline.joblib`, `scaler.joblib`, `training_manifest.json`, `split_manifest.json`

### 2.3. Rà soát đường dẫn phụ thuộc máy tính (Hardcoded Paths)
* Toàn bộ mã nguồn sản xuất trong `src/`, `app/`, `training/` đều sử dụng biến neo thư mục gốc:
  `ROOT = Path(__file__).resolve().parents[...]` hoặc `PROJECT_ROOT = Path(...)`.
* Không có bất kỳ đường dẫn tuyệt đối cứng dạng `C:\Users\Admin\...` hay `D:\...` nào bị ghim trong code suy luận runtime. Mã nguồn hoàn toàn di động (portable) khi chạy trên các máy khác.

### 2.4. Những file dữ liệu đang bị Git bỏ qua (`.gitignore`)
Theo chính sách phòng tránh phình to dung lượng và bảo mật bản quyền dữ liệu:
* Thư mục `dulieu/`, `ai/dataset/`, `ai/raw_data/` bị loại trừ hoàn toàn.
* Thư mục `data/*` bị loại trừ (chỉ whitelist `data/dictionary/` và `data/experiments/dynamic_jz_binary_baseline_v1/`).
* Do đó, **toàn bộ file ảnh raw (.jpg, .png), video raw (.avi) và file CSV/JSONL đặc trưng nặng đều KHÔNG nằm trên GitHub**, mà được lưu trữ trên ổ đĩa máy phát triển và Google Drive.

---

## 3. Đối Chiếu Google Drive

**Địa chỉ liên kết:** [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)

### 3.1. Đối chiếu theo bằng chứng hiện có

| Nhóm dữ liệu | Tệp nén chuẩn bị trên máy | Dung lượng | Tình trạng theo thông tin người dùng | Khả năng xác minh từ công cụ AI | Kết luận kiểm toán |
|---|---|---:|:---:|:---:|:---:|
| **ABC (Chữ cái raw)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\ABC.zip` | 2,66 GB | Đã lưu trên Drive | Không truy cập được giao diện web Drive | **Được người dùng xác nhận** *(Cần đối chiếu file size)* |
| **123 (Chữ số raw)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\123.zip` | 40,48 MB | Đã lưu trên Drive | Không truy cập được giao diện web Drive | **Được người dùng xác nhận** *(Cần đối chiếu file size)* |
| **J/Z (Video raw)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\jz.zip` | 3,88 GB | Đã lưu trên Drive | Không truy cập được giao diện web Drive | **Được người dùng xác nhận** *(Cần đối chiếu file size)* |
| **SPACE (Ảnh raw)** | `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space.zip` | 36,12 MB | Đã lưu trên Drive | Không truy cập được giao diện web Drive | **Được người dùng xác nhận** *(Cần đối chiếu file size)* |
| **Tập Landmark đặc trưng (CSV/JSONL)** | Chưa có file ZIP riêng tổng hợp các file `.csv` trong `ai/dataset/` | ~300 MB | Chưa ghi nhận rõ trong tài liệu cũ | Không truy cập được giao diện web Drive | **CHƯA RÕ ĐÃ UPLOAD LÊN DRIVE CHƯA** |

> [!IMPORTANT]
> **Giới hạn xác minh:**
> AI Assistant hoạt động trong sandbox dòng lệnh cục bộ, không có quyền đăng nhập / API vào tài khoản Google Drive cá nhân của người dùng. Do đó, hệ thống không thể tự động liệt kê cây thư mục từ xa trên Drive để kiểm tra mã băm.
> **Khuyến nghị:** Người dùng kiểm tra trực tiếp giao diện Google Drive để xác nhận đã có đủ 4 file `ABC.zip`, `123.zip`, `jz.zip`, `space.zip` (hoặc `ASL-Dataset.zip`).

---

## 4. Xác Minh GitHub (Commit `65c38d6`)

Đã đối soát trực tiếp cây thư mục Git (`git ls-files`):

| Nhánh / Thành phần | Tên file model trên Git | Kích thước | SHA-256 Commit trên Git | Có trên GitHub không? | Nhu cầu tải ngoài? |
|---|---|---:|---|:---:|:---:|
| **ABC (Static ASL)** | `models/letters/production/uvw_medium_model.joblib`<br>`models/letters/production/uvw_medium_scaler.joblib` | 218 KB<br>1,8 KB | `88ea2f270cd7...`<br>`93a0ddf102e4...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Đã có sẵn khi clone) |
| **123 (Static Digits)** | `models/digits/production/digits_final.joblib` | 464 KB | `43c7e87a287d...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Đã có sẵn khi clone) |
| **J/Z (Dynamic Gestures)**| `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib`<br>`data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | 36,6 KB<br>27 KB | `8878b0c0cd04...`<br>`93804d73f4fc...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Đã có sẵn khi clone) |
| **SPACE (Dấu cách)** | `models/space/production/final_space_model.joblib` | 1,3 KB | `453162d8aa02...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Đã có sẵn khi clone) |
| **MediaPipe Landmarker** | `models/mediapipe/hand_landmarker.task` | 7,8 MB | `fbc2a30080c3...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Có sẵn hoặc chạy script tải lại) |
| **Từ điển gợi ý** | `data/dictionary/english_words_25k.tsv` | 291 KB | `986eead866a0...` | **CÓ (Tracked)** | **KHÔNG CẦN TẢI** (Đã có sẵn khi clone) |

### Những model KHÔNG có trên GitHub (Được cố ý bỏ qua theo chính sách):
* Các model fallback cũ: `models/letters/fallback/*.joblib`, `models/digits/fallback/*.joblib`.
* Các candidate thử nghiệm: `models/digits/candidates/*.joblib`, `models/digits/production/mlp_large_v2a.joblib`.
* Caches DTW template: `models/dynamic_jz/dtw_templates/`.
* *Lý do:* Hệ thống sản xuất chỉ chạy các model chính thức (Production). Việc loại bỏ các model thử nghiệm giúp repository gọn gàng, tránh phình to kích thước không cần thiết.

---

## 5. Những Dữ Liệu Còn Thiếu Hoặc Chưa Xác Minh

1. **Tập dữ liệu đặc trưng đã xử lý (CSV/JSONL Feature Files):**
   - Trên máy tính cục bộ: Đang lưu đầy đủ trong `ai/dataset/` và `data/experiments/`.
   - Trên GitHub: Bị `.gitignore` loại trừ.
   - Trên Google Drive: Chưa xác minh được người dùng đã upload các file CSV này lên Drive hay chỉ mới upload các file ảnh/video raw (`ABC.zip`, `123.zip`, `jz.zip`, `space.zip`).
   - *Hệ quả:* Nếu một người khác muốn huấn luyện lại mô hình mà không muốn chạy lại bước trích xuất MediaPipe tốn hàng giờ đồng hồ từ 134.000 ảnh và 700 video, họ sẽ cần các file CSV đặc trưng này.
2. **Quyền truy cập thư mục Google Drive:**
   - Cần đảm bảo link Drive được chia sẻ ở chế độ *"Bất kỳ ai có đường liên kết đều có thể xem"* (Anyone with the link can view).

---

## 6. Hướng Dẫn Tải và Tái Lập Hệ Thống

### Kịch bản 1: Người dùng mới chỉ muốn CHẠY ỨNG DỤNG (Inference Mode — ĐÃ ĐẦY ĐỦ 100%)
Người dùng mới **không cần tải bất kỳ dataset nào từ Google Drive**, vì toàn bộ model và tài nguyên runtime đã nằm sẵn trong Git:

```bash
# 1. Clone repository từ GitHub
git clone https://github.com/phan-the-vinh/nhandienngonngu.git
cd nhandienngonngu

# 2. Tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
.venv\Scripts\activate       # Trên Windows
pip install --upgrade pip
pip install -r requirements.txt

# 3. Kiểm tra tính toàn vẹn hệ thống trước khi mở camera
python run.py --check-only

# 4. Khởi chạy ứng dụng nhận diện thời gian thực (ABC, 123, J/Z, SPACE)
python run.py
```

### Kịch bản 2: Nhà nghiên cứu muốn HUẤN LUYỆN LẠI (Retraining Mode)
Nếu người dùng muốn tự train lại từ dữ liệu gốc:
1. Vào Google Drive: [https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10](https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10)
2. Tải các gói nén raw tương ứng (`ABC.zip`, `123.zip`, `jz.zip`, `space.zip`).
3. Giải nén vào thư mục `dulieu/` hoặc chạy script trích xuất:
   - ABC: `python tools/extract_letters.py`
   - Digits: `python tools/extract_digits.py`
   - J/Z: `python tools/extract_dynamic.py`
4. Chạy script huấn luyện tương ứng:
   - ABC: `python training/letters/train_uvw_validation.py`
   - Digits: `python training/digits/train_digits.py`
   - J/Z: `python training/dynamic/train_dynamic.py`

---

## 7. Kết Luận & Khuyến Nghị Tiếp Theo

1. **GitHub hiện tại đã đủ để chạy hệ thống chưa?**
   - **ĐÃ ĐẦY ĐỦ 100%**. Commit [`65c38d6`](https://github.com/phan-the-vinh/nhandienngonngu/commit/65c38d6) chứa toàn bộ mã nguồn, các model `.joblib`, scaler, MediaPipe task, manifest, từ điển và runner. Một máy tính mới chỉ cần clone và cài pip là chạy được ngay lập tức cả 4 nhánh ABC, 123, J/Z, SPACE.
2. **Dataset đã được lưu trữ an toàn chưa?**
   - Trên máy tính: Đã lưu trữ an toàn, đầy đủ tại `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset` (gồm 4 thư mục mở và 4 file nén ZIP tương ứng).
   - Trên Google Drive: Người dùng đã xác nhận lưu trữ dataset.
3. **Có cần upload bổ sung lên Google Drive không?**
   - **Khuyến nghị upload bổ sung 1 file:** Nén các file feature CSV đã trích xuất (`ai/dataset/mlp_uvw_training_data_v1/train_uvw_v1.csv`, `ai/dataset/train_validation_test_split_v1/`, `data/processed/digits/`, `data/experiments/dynamic_jz_v1/processed/`) thành một file nén nhỏ (`processed_features.zip` ~150 MB) và upload lên Drive. Điều này giúp các nhà nghiên cứu sau này có thể tải trực tiếp vector đặc trưng để retrain trong vài giây mà không cần giải nén hàng gigabyte ảnh raw.
4. **Các việc cần làm tiếp theo:**
   - Kiểm tra quyền truy cập công khai của link Google Drive.
   - Hoàn thành dự án và chuẩn bị bài thuyết trình / demo.

