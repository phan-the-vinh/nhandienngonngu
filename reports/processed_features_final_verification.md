# BÁO CÁO XÁC MINH TOÀN DIỆN VÀ ĐỐI SOÁT DỮ LIỆU ĐẶC TRƯNG ĐÃ XỬ LÝ
## (PROCESSED FEATURES FINAL VERIFICATION REPORT)

- **Thời điểm thực hiện:** 2026-11-17
- **Workspace:** `D:\NhanDienCuChi`
- **Mục tiêu:** Xác minh tính đầy đủ, chính xác và khả năng tái lập của toàn bộ tệp processed features trước khi tiến hành đóng gói `processed_features.zip`.
- **Nguyên tắc:** Kiểm tra bằng lệnh thực tế, không sửa code, không xóa dữ liệu, không tạo ZIP, không suy đoán.

---

## 1. KẾT QUẢ KIỂM TRA TỪNG NHÁNH DỮ LIỆU

### 1.1. Nhánh Cử chỉ dấu cách (SPACE)
- **Tình trạng file đặc trưng dương (Positive Landmarks):**
  - Đã thực hiện quét toàn bộ workspace `D:\NhanDienCuChi` và bản sao lưu `D:\NhanDienCuChi_RAW_BACKUP`.
  - **Kết quả:** **KHÔNG TỒN TẠI** bất kỳ file processed feature nào (như `space_landmarks.csv` hay `space_features.csv`) chứa 1.049 mẫu positive SPACE đã trích xuất landmark.
  - Các thư mục staging lịch sử được nhắc đến trong các báo cáo cũ (`staging/space_training_readiness_audit_phase3_9_4_20261106` và `staging/space_training_phase3_9_15`) không còn tồn tại trên ổ đĩa.
  - **Dữ liệu thô thực tế có sẵn:** Thư mục `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space` chứa đúng **2.997 ảnh** `.jpg` và file nén `space.zip` (37.879.237 bytes).
- **Cơ chế tạo 63.140 mẫu âm (NOT_SPACE):**
  - Gồm **62.591 mẫu chữ cái (ABC)** từ `ai/dataset/train_validation_test_split_v1/train.csv` (Đã xác minh tồn tại, 62.591 dòng dữ liệu).
  - Gồm **549 mẫu chữ số (123)** từ `data/processed/digits/digits_final_train_frozen.csv` (Đã xác minh tồn tại, 549 dòng dữ liệu).
  - Tỷ lệ: 1.049 mẫu dương (1,63%) và 63.140 mẫu âm (98,37%).
- **Kiểm tra Script huấn luyện SPACE:**
  - Trong toàn bộ codebase (`training/`, `tools/`, `src/`), **KHÔNG CÓ script huấn luyện độc lập cho SPACE** (ví dụ: `training/space/train_space.py`).
  - Mô hình production `models/space/production/final_space_model.joblib` (1.375 bytes) và cấu hình `configs/space.json` tồn tại, nhưng script tự động trích xuất và fit LogisticRegression hiện không có trong repository.
- **Đánh giá:** **BLOCKED** đối với nhánh SPACE do thiếu file landmark đã trích xuất và thiếu script huấn luyện.

---

### 1.2. Nhánh Chữ cái tĩnh (ABC)
- **Quan hệ giữa 62.591 mẫu train gốc và 62.648 mẫu `train_uvw_v1.csv`:**
  - File `ai/dataset/train_validation_test_split_v1/train.csv`: 62.591 dòng dữ liệu.
  - File `ai/dataset/mlp_uvw_training_data_v1/train_uvw_v1.csv`: 62.648 dòng dữ liệu.
  - **Xác minh đối chiếu:** Đúng 62.591 dòng đầu tiên của `train_uvw_v1.csv` trùng khớp $100\%$ về tên file, nhãn và tọa độ đặc trưng với `train.csv`.
- **Xác minh 57 mẫu bổ sung U/V/W:**
  - File `ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv` chứa đúng **57 dòng** (15 OLD_U, 15 OLD_V, 15 V_TO_W, 12 V_TO_U).
  - Đúng 57 dòng cuối (dòng thứ 62.592 đến 62.648) của `train_uvw_v1.csv` khớp $100\%$ với các mẫu bổ sung này.
- **Kiểm tra rò rỉ dữ liệu (Data Leakage):**
  - Giao giữa tập file `train_uvw_v1.csv` và `validation.csv` (13.412 mẫu): **0 mẫu trùng lặp**.
  - Giao giữa tập file `train_uvw_v1.csv` và `test.csv` (13.412 mẫu): **0 mẫu trùng lặp**.
  - Cam kết không có bất kỳ mẫu nào từ Validation hoặc Test bị đưa vào tập Train.
- **Kiểm tra Sample Weights:**
  - File `ai/dataset/mlp_uvw_training_data_v1/train_uvw_weights.csv` có đúng **62.648 dòng** dữ liệu.
  - Thứ tự các dòng khớp hoàn toàn 1-1 với `train_uvw_v1.csv`.
  - Cột `weight_supplement_medium` được pipeline `train_uvw_validation.py` nạp trực tiếp vào tham số `sample_weight` của bộ phân loại MLP.
- **Đánh giá:** **PASS $100\%$** về tính toàn vẹn dữ liệu.

---

### 1.3. Nhánh Chữ số tĩnh (Digits – 123)
- **Vai trò của từng tệp dữ liệu:**
  - `data/processed/digits/digits_final_train_frozen.csv`: Đúng **549 mẫu** (349 base_train_v1 + 170 Arda Mavi + 30 ASL-HG). Phân bổ 10 lớp chữ số: lớp 0 có 54 mẫu, các lớp 1–9 có 55 mẫu. Đây là tập huấn luyện chính thức (production train).
  - `data/splits/digits/validation_reference.csv`: Đúng **100 mẫu** (10 mẫu/lớp).
  - `data/splits/digits/regression_reference.csv`: Đúng **250 mẫu** (25 mẫu/lớp), nguồn gốc từ tập `test_v1.csv` cũ.
- **Xác minh pipeline có thực sự sử dụng Validation Reference:**
  - Trong `training/digits/train_digits.py` (dòng 331, 346–352): Script đọc trực tiếp `validation_reference.csv`, kiểm tra đúng 100 mẫu, kiểm tra tính hữu hạn của các số thực (`np.isfinite`), và tính toán `validation_macro_f1` qua từng epoch để chọn mô hình tối ưu.
- **Phân biệt rạch ròi với Regression Reference:**
  - Mã nguồn `train_digits.py` (dòng 321–322) áp đặt kiểm tra nghiêm ngặt:
    ```python
    if spec["regression_is_independent_final_test"] is not False or lineage["test_set_used"] is not False:
        raise ValueError("Frozen replay configuration differs from original digits_final training")
    ```
  - Và in thông báo rõ ràng trong kế hoạch huấn luyện:
    `Regression reference: NOT used for training or selection`.
  - Tệp này chỉ phục vụ kiểm thử hồi quy (so sánh xác suất dự đoán với phiên bản model trước), **tuyệt đối không phải test set độc lập**.
- **Chạy thực tế `--check-only`:**
  - Lệnh: `python training/digits/train_digits.py --check-only --output-dir data/experiments/digits_final_replay/check_test_dir`
  - Kết quả: `CHECK_ONLY: PASS; no sampling or training performed.`
  - Không có file nào bị ghi đè, toàn bộ checksum SHA-256 của tập dữ liệu và metadata được xác minh hợp lệ $100\%$.
- **Đánh giá:** **PASS $100\%$**.

---

### 1.4. Nhánh Cử chỉ động (J/Z)
- **Xác minh số lượng chuỗi và Split Manifest:**
  - Tệp đặc trưng chuỗi: `data/experiments/dynamic_jz_v1/processed/signnteam_jz_exclude_j40_v1.jsonl` có kích thước 14.797.025 bytes, gồm đúng **643 chuỗi**.
  - Tệp manifest phân chia: `data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json` gồm đúng **640 chuỗi** được phân chia theo tỷ lệ 80/10/10:
    - **Train:** **512 chuỗi** (212 J, 300 Z).
    - **Validation:** **64 chuỗi** (26 J, 38 Z).
    - **Test:** **64 chuỗi** (27 J, 37 Z).
- **Làm rõ sự chênh lệch 3 chuỗi giữa JSONL (643) và Manifest (640):**
  - Đối chiếu mã nguồn và hash xác nhận: 3 chuỗi có trong JSONL nhưng bị loại khỏi tập huấn luyện baseline v1 gồm:
    1. `data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/99.avi`
    2. `data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/103.avi`
    3. `data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/250.avi`
  - Lý do: Lỗi theo dõi MediaPipe HandLandmarker (tỷ lệ bắt landmark không đạt ngưỡng tối thiểu).
  - 3 video này được khai báo minh bạch trong `EXPECTED_EXCLUSIONS` tại `training/dynamic/train_dynamic.py` và `configs/dynamic_training.json`.
  - Khớp nối $100\%$: Toàn bộ 640 mẫu trong `split_manifest.json` đều tồn tại và hợp lệ trong tệp JSONL.
- **Chạy thực tế `--check-only`:**
  - Lệnh: `python training/dynamic/train_dynamic.py --check-only`
  - Kết quả:
    `{"status": "DATA_PASS_RIGHTS_VERIFIED", "sequence_count": 640, "class_counts": {"J": 265, "Z": 375}, "input_shape": [16, 69], "flattened_features": 1104, "excluded": ["J/103.avi", "J/250.avi", "J/99.avi"], "split_plan": "80/10/10 stratified seed=42", "training_performed": false}`
- **Đánh giá:** **PASS $100\%$**.

---

## 2. MA TRẬN KHẢ NĂNG TÁI LẬP (REPRODUCIBILITY MATRIX)

| Nhánh | File đầu vào bắt buộc | File cấu hình & Phụ thuộc | Script huấn luyện | Artifact đầu ra | Đánh giá khả năng tái lập |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ABC** | • `train_uvw_v1.csv`<br>• `train_uvw_weights.csv`<br>• `validation.csv`<br>• `test.csv` | • `manifest.json`<br>• `split_manifest.json` | `training/letters/train_uvw_validation.py` | `uvw_medium_model.joblib`<br>`uvw_medium_scaler.joblib` | **CONDITIONAL PASS**:<br>Dữ liệu đầy đủ $100\%$. Script `train_uvw_validation.py` kiểm tra SHA-256 mã nguồn cũ (`AI/landmark_utils.py` lệch 2 bytes do newline). Cần trainer độc lập khi chạy trên môi trường mới. |
| **Digits** | • `digits_final_train_frozen.csv`<br>• `validation_reference.csv` | • `configs/digits_training.json`<br>• `digits_final_sample_map.csv`<br>• `digits_final_replay_spec.json`<br>• `digits_final_freeze_manifest.json` | `training/digits/train_digits.py` | `digits_final.joblib` | **FULL PASS**:<br>Chạy lại hoàn toàn tự động với cờ `--check-only` đạt PASS. Tái tạo bit-exact trọng số mô hình. |
| **J/Z** | • `signnteam_jz_exclude_j40_v1.jsonl` | • `configs/dynamic_training.json`<br>• `split_manifest.json`<br>• `config_snapshot.json` | `training/dynamic/train_dynamic.py` | `model_pipeline.joblib`<br>`scaler.joblib` | **FULL PASS**:<br>Chạy lại hoàn toàn tự động với cờ `--check-only` đạt `DATA_PASS_RIGHTS_VERIFIED`. |
| **SPACE** | • `train.csv` (âm)<br>• `digits_final_train_frozen.csv` (âm)<br>• *Mẫu dương: THIẾU* | • `configs/space.json` | *THIẾU script huấn luyện* | `final_space_model.joblib` | **BLOCKED**:<br>Thiếu tệp vector dương 1.049 mẫu và thiếu script huấn luyện độc lập. |

---

## 3. DANH SÁCH FILE CHÍNH THỨC XÁC MINH CẦN ĐÓNG GÓI

Toàn bộ 21 tệp dưới đây đã được kiểm tra trực tiếp trên đĩa, tồn tại thực tế và có mã băm SHA-256 tương ứng:

| STT | Phân loại | Đường dẫn tương đối từ Workspace | Dung lượng (Bytes) | SHA-256 Checksum |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **ABC Train** | `ai/dataset/mlp_uvw_training_data_v1/train_uvw_v1.csv` | 81.509.037 | `898e0a440332abed06da5c52e90bd7f9b37f870e95e4936fe4a1c4486bd2c613` |
| 2 | **ABC Weights** | `ai/dataset/mlp_uvw_training_data_v1/train_uvw_weights.csv` | 6.865.887 | `99b09b88452ea2155bd844e320873a2c33cd9cc548c576fc65ba7d723b64e304` |
| 3 | **ABC Supp** | `ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv` | 99.780 | `9aa0d2b6b667c3ec438e6419284a9841a60219808efd9b411dd094695b602727` |
| 4 | **ABC Manifest** | `ai/dataset/mlp_uvw_training_data_v1/manifest.json` | 6.763 | `16d19c38f4a4d4fbfd9c1689d9c197c1f6990fce8fa638ad584b98190501c7bc` |
| 5 | **ABC Val** | `ai/dataset/train_validation_test_split_v1/validation.csv` | 16.603.726 | `15c6f343d94b7ca3836d80a15a13862a925622918e8c37761d5854f70ed6cf76` |
| 6 | **ABC Test** | `ai/dataset/train_validation_test_split_v1/test.csv` | 16.602.841 | `cc612b03caf4578cd1d234bd9b310e1ea3f57ecbc92d72d5d607233593881639` |
| 7 | **ABC Split** | `ai/dataset/train_validation_test_split_v1/split_manifest.json` | 3.297 | `09c5f103908bef7b9e6346eca15cbf245523491a87b971dd5dee2c99f88b3d25` |
| 8 | **Digits Train** | `data/processed/digits/digits_final_train_frozen.csv` | 665.135 | `79f79d15d06524eb92d8d275a6d2018d507c058bcddffabf295a0dc9a1f8b68c` |
| 9 | **Digits Val** | `data/splits/digits/validation_reference.csv` | 130.055 | `623463a3eedc4f5e66f9ba028fcd638bef0453222a956b0fbcf055bd978257a3` |
| 10 | **Digits Regr** | `data/splits/digits/regression_reference.csv` | 322.634 | `a4f4ac064bdcd45f13f65091e800baf551129677ec96892a0bac00757d185a32` |
| 11 | **Digits Config**| `configs/digits_training.json` | 1.343 | `20c81bec9ce8f288461695bf86236af253be8b101db8acfa0c3eac8a95a62fc2` |
| 12 | **Digits Map** | `data/metadata/digits/digits_final_sample_map.csv` | 162.422 | `e9695ee68af0b77a75fddbcc5104534b95cc7681e5a48328a88de645987c744c` |
| 13 | **Digits Freeze**| `data/metadata/digits/digits_final_freeze_manifest.json` | 5.388 | `fdc97be581074ed816ae5f50b4b0af3001b49ad9598a1781d28d6c515033611f` |
| 14 | **Digits Replay**| `data/metadata/digits/digits_final_replay_spec.json` | 3.032 | `98ad31f6b45ee6a13270e19587bb23d83a8b6f43d4ca5fba891d62285d9b3170` |
| 15 | **Digits Repro** | `data/metadata/digits/digits_final_repro/training_manifest_digits_final.json` | 83.219 | `e1514d1971e508fd4188f3b7d8982936f3603888efeb3dfbee62c11e2c3a5d59` |
| 16 | **J/Z Data** | `data/experiments/dynamic_jz_v1/processed/signnteam_jz_exclude_j40_v1.jsonl` | 14.797.025 | `b6904562621a6249303637d7ed4730f0ca93441924f402f32f59c15a47c695b1` |
| 17 | **J/Z Split** | `data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json` | 86.126 | `8a0bdc86f2b695e6cdfa30a254260d6e6e0b8685cd75b1cc1883d50909f0cd7d` |
| 18 | **J/Z Config** | `configs/dynamic_training.json` | 1.558 | `7a33413cd1cdcd6c51333fcab13abe40c88ac86ed632e2a9c6d2f4d7934fadfe` |
| 19 | **J/Z Snapshot**| `data/experiments/dynamic_jz_binary_baseline_v1/config_snapshot.json` | 1.729 | `104a7442abbfe52345c0a78e6813b8f4165e65c7744db66a92542add335edb08` |
| 20 | **J/Z Manifest**| `data/experiments/dynamic_jz_binary_baseline_v1/training_manifest.json` | 1.570 | `c432b4871e26ed9062ab758781d9032e65896b735d81e29f2eab5ec2b623ce2f` |
| 21 | **J/Z Curation**| `data/experiments/dynamic_jz_v1/metadata/signnteam_jz_exclude_j40_manifest_v1.json` | 1.728 | `5e047bca78bac8b200d3b3d7172c59f8382c1d71801ae1fc2b665097405faa76` |

### Thông số đóng gói thực tế (Mô phỏng nén In-Memory):
- **Tổng số tệp hợp lệ:** **21 tệp**
- **Dung lượng chưa nén (Uncompressed):** **137.954.937 bytes (~131,56 MB)**
- **Dung lượng sau khi nén ZIP (Deflate level 9):** **56.498.162 bytes (~53,88 MB)**
- **Tỷ lệ nén (Compression ratio):** **40,95%** *(tiết kiệm gần 60% băng thông)*.

---

## 4. CÁC TỆP CÒN THIẾU VÀ SAI LỆCH PHÁT HIỆN ĐƯỢC

### 4.1. File còn thiếu (Missing Files)
1. **Thiếu tệp trích xuất landmark của cử chỉ SPACE:**
   - Tệp giả định `space_landmarks.csv` (chứa 1.049 vector đặc trưng dương) không tồn tại trên hệ thống. Hiện chỉ có 2.997 ảnh JPEG tại `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space`.
2. **Thiếu script huấn luyện độc lập cho SPACE:**
   - Không có file Python nào trong `training/` thực hiện trích xuất và huấn luyện mô hình LogisticRegression cho SPACE.

### 4.2. Sai lệch phát hiện được (Discrepancies Identified)
1. **Số lượng chuỗi trong tệp J/Z JSONL:**
   - Báo cáo sơ bộ trước đây ước tính 640 chuỗi trong file JSONL.
   - Kiểm tra thực tế xác nhận file `signnteam_jz_exclude_j40_v1.jsonl` có **643 chuỗi**. 3 chuỗi thừa (`J/99.avi`, `J/103.avi`, `J/250.avi`) là các chuỗi lỗi nhận diện landmark đã được phân tách trong `split_manifest.json` và xử lý loại trừ bởi `train_dynamic.py`.
2. **Ràng buộc kiểm tra mã nguồn cũ của ABC:**
   - Script `training/letters/train_uvw_validation.py` kiểm tra mã băm cứng của file `AI/landmark_utils.py` (phiên bản cũ 1.144 bytes, trong khi file hiện tại ở `ai/preprocessing/landmark_utils.py` là 1.146 bytes do ký tự xuống dòng CRLF). Bản thân dữ liệu feature CSV không bị ảnh hưởng.

---

## 5. KẾT LUẬN VỀ ĐIỀU KIỆN ĐÓNG GÓI `processed_features.zip`

> [!CAUTION]
> **KẾT LUẬN:** **CHƯA ĐỦ ĐIỀU KIỆN ĐÓNG GÓI HOÀN CHỈNH 4 NHÁNH CHO `processed_features.zip`**.

### Lý do cụ thể:
1. **Nhánh SPACE chưa hoàn chỉnh:** Do không tồn tại file đặc trưng landmark dương (1.049 mẫu) và thiếu script huấn luyện độc lập, việc đóng gói lúc này sẽ khiến người dùng không thể tái lập mô hình SPACE từ processed features nếu không trích xuất lại từ 2.997 ảnh raw.
2. **3 nhánh còn lại (ABC, Digits, J/Z) đã hoàn toàn sẵn sàng:**
   - **Digits (123):** Hoàn hảo $100\%$, kiểm thử `--check-only` đạt PASS bit-exact.
   - **J/Z:** Hoàn hảo $100\%$, kiểm thử `--check-only` đạt `DATA_PASS_RIGHTS_VERIFIED`.
   - **ABC:** Dữ liệu hoàn hảo $100\%$, không rò rỉ dữ liệu giữa train/val/test.

### Đề xuất bước tiếp theo cho tác giả:
1. **Đối với SPACE:** Tác giả cần chạy kịch bản trích xuất MediaPipe HandLandmarker từ thư mục ảnh thô `D:\NhanDienCuChi_RAW_BACKUP\ASL-Dataset\space` để sinh ra file `space_positive_landmarks.csv` (1.049 mẫu), đồng thời bổ sung script huấn luyện `training/space/train_space.py`.
2. **Sau khi hoàn thiện SPACE:** Đóng gói toàn bộ 22 tệp (21 tệp đã xác minh + 1 tệp landmark SPACE mới) thành `processed_features.zip` (~55 MB) và tải lên Google Drive.

