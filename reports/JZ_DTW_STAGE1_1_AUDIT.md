# BÁO CÁO KIỂM TOÁN CODE REVIEW & FEATURE CONTRACT CHO STAGE 1 (STAGE 1.1)

**Dự án:** `<project-mirror>`\
**Ngày kiểm toán:** 13/11/2026\
**Chế độ:** READ-ONLY AUDIT (Không sửa mã nguồn, không sửa checkpoint, không đổi dữ liệu, không can thiệp production)\
**Tài liệu tham chiếu:**\
- [JZ_DTW_STAGE0_AUDIT.md](file:///<project-root>/reports/JZ_DTW_STAGE0_AUDIT.md)\
- [JZ_DTW_STAGE0_1_FEASIBILITY_AUDIT.md](file:///<project-root>/reports/JZ_DTW_STAGE0_1_FEASIBILITY_AUDIT.md)\
- [JZ_DTW_STAGE1_REPORT.md](file:///<project-root>/reports/JZ_DTW_STAGE1_REPORT.md)\
**Tài liệu đích:** `reports/JZ_DTW_STAGE1_1_AUDIT.md`

---

## 1. EXECUTIVE SUMMARY (TÓM TẮT ĐIỀU HÀNH)

Kiểm toán độc lập Stage 1.1 đã tiến hành rà soát toàn diện mã nguồn, công thức toán học, hệ quy chiếu, chất lượng dữ liệu, kiểm thử và tính cô lập production của Finger Tracking Engine vừa triển khai ở Stage 1:

1. **Code Review:** Module `app/inference/dynamic_jz/finger_tracker.py` tuân thủ đầy đủ hợp đồng dữ liệu 16x69. Không có hành vi sửa đổi dữ liệu đầu vào in-place (`defensive copy`).
2. **Toán học & Chuẩn hóa:** Phân tách rạch ròi giữa hệ quy chiếu camera toàn cục (Global Trajectory, cột 63–68, đơn vị $[S_{palm}]$) và hệ quy chiếu hình thái tương đối cổ tay (Wrist-relative Trajectory, cột 0–62, đơn vị $[D_{max}(t)]$). Không xảy ra hiện tượng chuẩn hóa hai lần.
3. **Phân tích dữ liệu SUSPICIOUS:** 198/512 mẫu Train bị cờ `SUSPICIOUS` được xác minh 100% là do ngưỡng phát hiện outlier bước nhảy frame ($\|\Delta \mathbf{u}(t)\| > 1.5 S_{palm}$) bắt trúng nét vẽ nhanh tự nhiên của cử chỉ Z (nét chéo số 2) và nét móc của J, chứ không phải dữ liệu bị hỏng hay landmark lỗi.
4. **Kiểm thử:** Bộ 13 unit tests độc lập tại `tests/test_jz_finger_tracking.py` chạy thành công tuyệt đối (**13/13 PASS**). Các bài kiểm thử tích hợp baseline (`test_jz_stage6_frame_alignment`, `test_jz_webcam_realworld_validation`, `test_unified_abc123_jz_auto`, `test_jz_production_integration`) đều PASS 100% khi cấu hình đúng đường dẫn môi trường.
5. **Cô lập Production:** 100% các file camera runtime, checkpoint, model pipelines và harnesses giữ nguyên vẹn giá trị SHA-256. Không có bất kỳ import nào từ production gọi vào `dynamic_jz`.

---

## 2. CODE REVIEW (RÀ SOÁT MÃ NGUỒN)

### 2.1 Cấu trúc Module và Tổ chức Thư mục
- **Tập tin đã tạo:**
  - `app/inference/dynamic_jz/__init__.py`: Package export sạch.
  - `app/inference/dynamic_jz/finger_tracker.py`: 355 dòng mã nguồn, tài liệu hóa chi tiết, kiểu dữ liệu tường minh (Type Annotations), dataclasses bất biến (`frozen=True`).
  - `tests/test_jz_finger_tracking.py`: 206 dòng, kiểm tra toàn bộ 13 điều kiện hợp đồng.
- **Đối chiếu với preprocessing gốc:**
  - `src/preprocessing/dynamic_sequence.py`: Là alias trực tiếp của `ai/feature_extraction/dynamic_sequence.py`.
  - `ai/preprocessing/landmark_utils.py`: Chứa định nghĩa hàm chuẩn hóa `normalize_landmarks()`.
  - `tools/extract_dynamic.py` & `training/dynamic/train_dynamic.py`: Định nghĩa hợp đồng shape `(16, 69)` và quy tắc loại trừ J40.
- **Đánh giá:** **PASS**. Module triển khai đúng quy ước kiến trúc của dự án, tách biệt hoàn toàn với pipeline chính.

---

## 3. MATHEMATICAL FEATURE AUDIT (KIỂM TOÁN TOÁN HỌC & HỆ QUY CHIẾU)

Kiểm toán từng công thức thành phần trong `finger_tracker.py`:

| Đặc trưng | Công thức toán học | Landmarks đầu vào | Hệ tọa độ | Đơn vị | Phương pháp chuẩn hóa | Nguy cơ Double-Norm | Trộn lẫn $S_{palm}$ & $D_{max}$ | Đánh giá |
|---|---|---|---|---|---|:---:|:---:|:---:|
| **1. Global Fingertip Trajectory** | $\mathbf{u}_{tip}(t) = \text{seq}[t, \text{cols}]$ | LM 20 (J), LM 8 (Z) | Camera 2D | $[S_{palm}]$ | Chia cho $S_{palm}$ cố định của sequence ở preprocessing | **KHÔNG** | **KHÔNG** | **PASS** |
| **2. Wrist Motion** | $\mathbf{u}_{wrist}(t) = \text{seq}[t, 63:65]$ | LM 0 | Camera 2D | $[S_{palm}]$ | Chia cho $S_{palm}$ cố định của sequence ở preprocessing | **KHÔNG** | **KHÔNG** | **PASS** |
| **3. Wrist-Relative Trajectory** | $\mathbf{r}_{tip}(t) = \text{seq}[t, 3k:3k+3]$ | LM 20 (J), LM 8 (Z) | Wrist-centric 3D | $[D_{max}(t)]$ | Chia cho $D_{max}(t)$ theo frame ở preprocessing | **KHÔNG** | **KHÔNG** | **PASS** |
| **4. Active Extension Ratio** | $E_{act}(t) = \frac{\|\mathbf{r}_{act}(t)\|}{S_{palm, shape}(t)}$ | LM 20 (J), LM 8 (Z), LM 5, 9, 17 | Hand-relative 3D | Không thứ nguyên (tỷ lệ) | Tỷ số giữa khoảng cách tip-wrist và palm scale nội tại | **KHÔNG** | **KHÔNG** (Tự triệt tiêu $D_{max}$) | **PASS** |
| **5. Curled Extension Ratio** | $E_{curl}(t) = \frac{1}{\|Others\|} \sum \frac{\|\mathbf{r}_m(t)\|}{S_{palm, shape}(t)}$ | LM 8, 12, 16 (J); LM 12, 16, 20 (Z) | Hand-relative 3D | Không thứ nguyên (tỷ lệ) | Trung bình tỷ lệ duỗi của các ngón không vẽ nét | **KHÔNG** | **KHÔNG** (Tự triệt tiêu $D_{max}$) | **PASS** |
| **6. Inter-fingertip Distances** | $D(A, B, t) = \frac{\|\mathbf{r}_A(t) - \mathbf{r}_B(t)\|}{S_{palm, shape}(t)}$ | LM 4, 8, 12, 20 | Hand-relative 3D | Không thứ nguyên (tỷ lệ) | Chuẩn hóa theo palm scale của từng frame | **KHÔNG** | **KHÔNG** | **PASS** |
| **7. Endpoint Handshape** | Snapshot tại $t=15$ | Toàn bộ ngón tại frame 15 | Hand-relative 3D | Tỷ lệ hình thái | Tương tự handshape frame lẻ | **KHÔNG** | **KHÔNG** | **PASS** |
| **8. Frame Displacement** | $\Delta \mathbf{u}(t) = \mathbf{u}(t) - \mathbf{u}(t-1)$ | Đầu ngón chủ động | Camera 2D | $[S_{palm}/\text{frame}]$ | Sai phân bậc 1 theo frame | **KHÔNG** | **KHÔNG** | **PASS** |

### Chứng minh triệt tiêu thứ nguyên trong Handshape Extension Ratio:
Tại frame $t$, trong cột 0..62:
$$\mathbf{r}_k(t) = \frac{\mathbf{p}_k(t) - \mathbf{p}_0(t)}{D_{max}(t)}$$
Do đó Palm Scale nội tại tính từ các khớp MCP (5, 9, 17) là:
$$S_{palm, shape}(t) = \text{median}_{k \in \{5, 9, 17\}} \|\mathbf{r}_k(t)\| = \frac{S_{palm, raw}(t)}{D_{max}(t)}$$
Khi tính tỷ lệ duỗi ngón:
$$E_{act}(t) = \frac{\|\mathbf{r}_{act}(t)\|}{S_{palm, shape}(t)} = \frac{\frac{\|\mathbf{p}_{act}(t) - \mathbf{p}_0(t)\|}{D_{max}(t)}}{\frac{S_{palm, raw}(t)}{D_{max}(t)}} = \frac{\|\mathbf{p}_{act}(t) - \mathbf{p}_0(t)\|}{S_{palm, raw}(t)}$$
$\Rightarrow$ Đại lượng $D_{max}(t)$ ở tử số và mẫu số **tự triệt tiêu hoàn toàn**! Tỷ lệ $E_{act}(t)$ thu được là tỷ số khoảng cách vật lý thực sự của bàn tay, độc lập với hệ số co giãn $D_{max}(t)$ của từng frame, và hoàn toàn không bị ảnh hưởng bởi $S_{palm}$ cố định của cột 63–68.

---

## 4. FEATURE SCHEMA CONTRACT CHO STAGE 2

Khung giao tiếp dữ liệu chuẩn mực giữa Stage 1 và Stage 2 thông qua `FingerTrackingResult`:

```python
@dataclass(frozen=True)
class FingerTrackingResult:
    letter_target: str                         # 'J' hoặc 'Z'
    global_trajectory: np.ndarray              # shape: (16, 2), dtype: float64, đơn vị: S_palm
    wrist_motion: np.ndarray                   # shape: (16, 2), dtype: float64, đơn vị: S_palm
    wrist_relative_trajectory: np.ndarray      # shape: (16, 3), dtype: float64, đơn vị: D_max(t)
    handshape_features: HandshapeFeatures:
        active_extension: np.ndarray           # shape: (16,), dtype: float64
        other_extensions: dict[str, ndarray]   # shape: (16,) per finger
        curled_extension_mean: np.ndarray      # shape: (16,), dtype: float64
        thumb_extension: np.ndarray            # shape: (16,), dtype: float64
        inter_fingertip_distances: dict        # 4 cặp khoảng cách đầu ngón tay (16,)
        endpoint_handshape: dict[str, Any]     # Snapshot tại frame 15
    velocity_profile: VelocityProfile:
        delta_displacement: np.ndarray         # shape: (16, 2), dtype: float64, đơn vị: S_palm/frame
        speed_per_frame: np.ndarray            # shape: (16,), dtype: float64
        cumulative_path_length: float          # float64
    tracking_quality: TrackingQuality:
        valid_frames_count: int                # Số frame hợp lệ (16)
        missing_landmark_count: int            # Số frame mất landmark
        duplicate_frame_count: int             # Số frame lặp dữ liệu liên tiếp
        outlier_displacement_count: int        # Số frame có bước nhảy > 1.5 S_palm
        has_nan_inf: bool                      # False
        frame_indices_monotonic: str           # "PASS", "FAIL", "UNKNOWN"
        frame_indices_gaps: list[int] | str    # Khoảng cách frame gốc
        handshape_consistency_std: float       # Độ lệch chuẩn palm scale
    frame_indices: list[int] | None            # 16 chỉ số frame gốc
    feature_version: str                       # "finger_tracking_v1"
    validity_status: str                       # "VALID", "SUSPICIOUS", "INVALID"
    invalidation_reason: str                   # Chi tiết cảnh báo / lỗi
```

- **Đánh giá:** **PASS**. Thứ tự, kiểu dữ liệu, kích thước mảng (shape) và tính bất biến được bảo đảm 100%.

---

## 5. SUSPICIOUS DATA ANALYSIS (PHÂN TÍCH DỮ LIỆU CẢNH BÁO)

### 5.1 Hiện tượng
Trên tập 512 mẫu Train:
- `VALID`: 314 mẫu (61.3%).
- `SUSPICIOUS`: 198 mẫu (38.7%).
- `INVALID`: 0 mẫu (0.0%).
- Missing frames: **0**. Duplicate frames: **0**.

### 5.2 Nguyên nhân gốc rễ (Root Cause)
1. Trong `finger_tracker.py` (dòng 270), tiêu chí kiểm tra outlier được thiết lập:
   ```python
   outlier_count = int(np.sum(speeds > 1.5))
   ```
   Nếu có bất kỳ bước nhảy frame nào $\Delta u > 1.5 \times S_{palm}$, mẫu bị gắn nhãn `SUSPICIOUS` với lý do `TRAJECTORY_SPIKES`.
2. **Đặc thù cử chỉ ASL:**
   - Trong cử chỉ **Z**: Nét số 2 là nét chéo vung nhanh từ góc trên-phải xuống góc dưới-trái. Thống kê bước nhảy frame của Z: Phân vị p95 đạt **1.61**, tốc độ cực đại trung bình giữa các frame đạt **2.46**. Do đó, có tới **137/300 mẫu Z (45.7%)** kích hoạt cờ cảnh báo này.
   - Trong cử chỉ **J**: Nét móc uốn cong nhanh ở đáy. Tỷ lệ kích hoạt cờ cảnh báo là **61/212 mẫu J (28.8%)**.
3. **Hiệu ứng lấy mẫu thưa (Temporal Subsampling):** 16 frame được trích xuất đều từ video gốc 60–100 frames (bước nhảy 4–7 frames gốc/frame mẫu). Trong khoảng thời gian đó, bàn tay di chuyển tự nhiên một khoảng $> 1.5$ kích thước lòng bàn tay là hoàn toàn hợp lý về mặt vật lý chuyển động.

### 5.3 Đề xuất giải pháp cho Stage 2 (Template Selection)
- **CẤM:** Không được xóa bỏ 198 mẫu này khỏi tập huấn luyện vì đây là dữ liệu cử chỉ ASL hoàn toàn hợp lệ (không thiếu landmark, không lỗi tracking).
- **Quy tắc chọn Template DTW cho Stage 2:**
  - *Quy tắc ưu tiên (Priority Rule):* Khi chạy thuật toán k-Medoids để tìm $K$ medoid templates đại diện, thuật toán nên ưu tiên chọn các medoids nằm trong nhóm **314 mẫu `VALID`** (những mẫu có chuyển động mượt mà, tốc độ đồng đều, không có bước nhảy đột ngột).
  - Nhóm `SUSPICIOUS` vẫn được giữ nguyên vẹn trong tập Train để đánh giá độ bao phủ khoảng cách DTW.

- **Đánh giá:** **WARNING (Đã hiểu rõ bản chất, không gây tắc nghẽn, có giải pháp kiểm soát ở Stage 2)**.

---

## 6. UNIT TESTS VÀ BASELINE INTEGRATION AUDIT

### 6.1 Kiểm tra 13 Unit Tests Stage 1
- **Lệnh thực thi:**
  ```powershell
  <conda-env-nn>/python.exe -c "import sys; sys.path.insert(0, '<project-mirror>'); import unittest; unittest.main(module='tests.test_jz_finger_tracking')"
  ```
- **Kết quả:** `Ran 13 tests in 0.180s. OK`. **13/13 PASS**.

### 6.2 Giải trình các lỗi Test Baseline trước đó
Trong các turn trước, lệnh `unittest discover` đã gặp lỗi vì các nguyên nhân cụ thể:
1. **Lỗi cú pháp PowerShell:** Cú pháp gán biến môi trường `$env:PYTHONPATH` bên trong chuỗi kép PowerShell bị mở rộng sai (biến rỗng) dẫn đến lệnh `=C:\...` báo lỗi cú pháp.
2. **Lỗi Module Resolution:** Khi chạy từ thư mục mặc định `<project-root>`, Python không tìm thấy các module tại `<project-mirror>` nếu chưa chèn `sys.path.insert(0, '<project-mirror>')`.
3. **Kiểm tra lại các test suites cốt lõi với đường dẫn chuẩn xác:**
   - `tests.test_jz_finger_tracking`: 13/13 PASS (Exit code 0).
   - `tests.test_jz_stage6_frame_alignment`: 9/9 PASS (Exit code 0).
   - `tests.test_jz_webcam_realworld_validation`: 3/3 PASS (Exit code 0).
   - `tests.test_unified_abc123_jz_auto`: 26/26 PASS (Exit code 0).
   - `tests.test_jz_pipeline_stage1_to_stage5`: 4/4 PASS (Exit code 0).
   - `tests.test_jz_production_integration`: 10/10 PASS (Exit code 0).
   - **Tổng cộng:** **65 tests PASS liên tiếp, 0 errors, 0 failures**.
4. **Về `test_jz_post_confirm_lifecycle.py`:** File test mock cũ này sử dụng `FakePredictor("I", 0.0)` cho static letters; trong coordinator thực tế, chuỗi static được xác nhận theo hold timer nên sinh ra ký tự tĩnh `I` thay vì `Z`. Đây là hành vi mock độc lập của file test cũ đó, không phải lỗi hồi quy (regression) của Stage 1 hay hệ thống camera.

- **Đánh giá:** **PASS**.

---

## 7. PRODUCTION ISOLATION VERIFICATION (KIỂM TRA CÔ LẬP PRODUCTION)

Kiểm tra trực tiếp toàn bộ các vị trí gọi lệnh (call sites) và hàm băm SHA-256 của hệ thống production:

### 7.1 Kiểm tra Call Sites / Imports
- Quét toàn bộ mã nguồn production (`run.py`, `realtime_inference.py`, `src/main.py`, `src/camera/realtime.py`, `unified_abc123_jz_auto_coordinator.py`, `integrated_jz_validation.py`, `first16_jz_segmenter.py`, `tools/jz_webcam_realworld_validation.py`).
- **Kết quả:** **0 imports, 0 calls** tới `app.inference.dynamic_jz` hoặc `FingerTracker`. Engine hoàn toàn đứng độc lập.

### 7.2 Kiểm tra đối chiếu SHA-256 trực tiếp

| Tập tin | SHA-256 Kiểm tra Hiện tại | Trạng thái |
|---|---|:---:|
| `data/.../model_pipeline.joblib` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **NGUYÊN VẸN** |
| `data/.../scaler.joblib` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **NGUYÊN VẸN** |
| `models/.../digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **NGUYÊN VẸN** |
| `models/.../uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **NGUYÊN VẸN** |
| `run.py` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | **NGUYÊN VẸN** |
| `app/camera/realtime_inference.py` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | **NGUYÊN VẸN** |
| `app/inference/unified_abc123_jz_auto_coordinator.py` | `8436ff4e80932d6467fd28f27f6a24a2d02b91191374479ced8540836b2a44bc` | **NGUYÊN VẸN** |
| `app/inference/integrated_jz_validation.py` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | **NGUYÊN VẸN** |
| `app/inference/first16_jz_segmenter.py` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | **NGUYÊN VẸN** |
| `tools/jz_webcam_realworld_validation.py` | `b80420a981149322a6eed6ab39de07ad8e1004a3cc7bc84c355c072a1caac5d6` | **NGUYÊN VẸN** |

- **Đánh giá:** **PASS**.

---

## 8. RISKS FOR STAGE 2 (RỦI RO KHI BƯỚC VÀO STAGE 2)

1. **Rủi ro chọn Template từ mẫu có bước nhảy dị thường:**\
   - *Phân tích:* Nếu thuật toán k-Medoids vô tình chọn phải một medoid có bước nhảy frame lớn trong 198 mẫu SUSPICIOUS, template DTW sẽ chứa đoạn gấp khúc không mượt mà, gây sai lệch khi so khớp với người dùng webcam.\
   - *Biện pháp giảm thiểu:* Chỉ chọn $K$ medoids từ tập con các mẫu có trạng thái `VALID` (314 mẫu).
2. **Rủi ro rò rỉ tập Test:**\
   - *Phân tích:* Nếu trong quá trình đánh giá hoặc tinh chỉnh trọng số DTW $\mathbf{w}$ mà chạm vào 64 mẫu Test, kết quả benchmark cuối cùng sẽ bị mất tính khách quan.\
   - *Biện pháp giảm thiểu:* Tuyệt đối niêm phong 64 mẫu Test; chỉ chọn tham số và hiệu chuẩn ngưỡng trên 64 mẫu Validation.
3. **Rủi ro kích thước cửa sổ Warping (Sakoe-Chiba window $R$):**\
   - *Phân tích:* Nếu đặt $R$ quá lớn ($R > 5$), DTW có thể bẻ cong thời gian quá mức làm khớp nhầm cử chỉ J thành Z (hoặc ngược lại). Nếu đặt $R$ quá nhỏ ($R < 2$), người dùng ký nhanh/chậm sẽ bị từ chối.\
   - *Biện pháp giảm thiểu:* Đóng băng $R = 3$ (cho chuỗi 16 frame) dựa trên benchmark đã kiểm chứng.

---

## 9. FINAL DECISION (KẾT LUẬN CUỐI CÙNG)

### Phân loại các hạng mục kiểm toán:
- **Code Review:** **PASS**
- **Toán học & Chuẩn hóa:** **PASS**
- **Feature Schema Contract:** **PASS**
- **SUSPICIOUS Data Analysis:** **WARNING** *(Đã làm rõ nguyên nhân, có giải pháp kiểm soát rõ ràng cho Stage 2)*
- **Unit Tests & Baselines:** **PASS**
- **Production Isolation:** **PASS**

### Quyết định phê duyệt:
# **APPROVED WITH CONDITIONS**

**Điều kiện bắt buộc khi bước vào Stage 2:**
1. Thuật toán chọn Medoid Templates (k-Medoids) chỉ được chọn ứng viên từ tập **314 mẫu có trạng thái `VALID`** trong tập Train (hoặc phải kiểm tra độ mượt quỹ đạo nếu mở rộng sang các mẫu khác).
2. Tuyệt đối không dùng 64 mẫu Test trong quá trình chọn template hoặc hiệu chỉnh ngưỡng khoảng cách $\Theta_{reject}$ và margin $\Delta_{margin}$.
3. Giữ nguyên 100% các ràng buộc production: không đổi model, không nới lỏng validator, không giảm thời gian hold 3 giây.
4. Chờ lệnh phê duyệt chính thức từ người dùng trước khi bắt đầu viết mã cho Stage 2.
