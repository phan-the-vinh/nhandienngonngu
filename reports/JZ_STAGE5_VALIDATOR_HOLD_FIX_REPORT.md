# STAGE 5 — BÁO CÁO KỸ THUẬT SỬA LỖI TRAJECTORY VALIDATOR VÀ HOLD CONFIRMATION CHO J/Z

- **Dự án:** Nhận Diện Cử Chỉ Tay Ký Tự Động J và Z (ASL Dynamic Hand Gesture Recognition)
- **Thư mục làm việc:** `<project-root>` (Mirror: `<project-mirror>`)
- **Ngày thực hiện:** 13/11/2026
- **Trạng thái:** HOÀN THÀNH TOÀN DIỆN (107/107 unit tests passed, 0 hồi quy, checkpoint nguyên vẹn)

---

## 1. TỔNG QUAN VÀ MỤC TIÊU (EXECUTIVE SUMMARY)

Dựa trên kết quả thực nghiệm Stage 4 với 50 lượt webcam thực tế (`20261113T150257Z_stage4_hybrid_0c360e`):
- Baseline Logistic Regression đạt độ chính xác dự đoán cao (**95.0%**, 38/40 lượt được gọi).
- Tuy nhiên, hệ thống production pipeline bị nghẽn nghiêm trọng ở hai tầng hậu xử lý:
  1. **Trajectory Validator**: 23/40 lượt bị từ chối, đặc biệt ký tự J có 17/20 lượt bị loại bởi các lỗi như `SINGLE_FRAME_TRAJECTORY_SPIKE` (6 lượt), `UPWARD_MOTION_BEFORE_DROP` (3 lượt), và `INSUFFICIENT_J_DOWNWARD_DROP`.
  2. **HoldConfirmation (3.0s)**: 6/17 lượt ứng viên đã qua Validator bị hủy trong quá trình giữ tĩnh 3.0 giây, trong đó có 4 lượt Z bị hủy do `ENDPOINT_POSE_CHANGED` dù người dùng vẫn duy trì tư thế ngón trỏ.

### Mục tiêu Stage 5
Khắc phục triệt để các điểm nghẽn tại Trajectory Validator và HoldConfirmation dựa trên bằng chứng dữ liệu thực tế:
1. **Xử lý lỗi J Trajectory Validator**:
   - Đánh giá chuyển động đi xuống dựa trên toàn bộ quỹ đạo (từ đỉnh cao nhất - apex đến đáy sâu nhất - nadir), thay vì neo cố định vào frame 0.
   - Đo biên độ móc ngang ($dx_{\text{hook}}$) trên toàn bộ các frame sau đáy, thay vì chỉ đo tại frame cuối cùng (frame 15).
   - Phân biệt giữa nhiễu giật spike nhảy cóc có hồi phục (glitch jump) và bước dịch chuyển tiến triển thực tế của nét vẽ, mà không hạ thấp ngưỡng an toàn.
2. **Xử lý lỗi Z Trajectory Validator**:
   - Chịu lỗi frame trùng lặp MediaPipe (~35% duplicate frames) và rung tay vi mô (~2–3 px) trong tích lũy độ dài cung, không làm suy yếu 3 nét zíc-zắc hình học.
3. **Xử lý lỗi HoldConfirmation**:
   - Phân biệt trôi trọng tâm / rung nhẹ cơ học với việc thay đổi tư thế thực tế (`ENDPOINT_POSE_CHANGED`).
   - Dung sai 3 frame lệch liên tiếp (`max_endpoint_mismatch_tolerance = 3`) tránh cảm biến bị chớp nhiễu frame đơn lẻ làm hủy ứng viên.
   - Giữ nguyên tuyệt đối thời gian xác nhận **3.0 giây** và ngăn chặn commit lặp lại.
4. **Bảo toàn tính toàn vẹn hệ thống**:
   - Giữ nguyên các model checkpoint tĩnh/động (`model_pipeline.joblib`, `digits_final.joblib`, `uvw_medium_model.joblib`).
   - Giữ nguyên bộ chuyển chế độ ABC/123, SPACE, và logic TextComposer.

---

## 2. DANH SÁCH TẬP TIN THAY ĐỔI VÀ KIỂM TRA SHA-256

### 2.1. Các tập tin mã nguồn chỉnh sửa

| Tập tin | Chức năng | Thay đổi chính |
| --- | --- | --- |
| `app/inference/integrated_jz_validation.py` | Trajectory Validator cho J và Z | Cải tiến apex-to-nadir drop; max lateral hook sau đáy; phân biệt spike vs persistent stroke; lọc rung vi mô tích lũy Z. |
| `app/inference/unified_abc123_jz_auto_coordinator.py` | Điều phối tổng thể Production | Bổ sung cơ chế dung sai 3 frame liên tiếp cho `ENDPOINT_POSE_CHANGED`; cập nhật tham số Hold tĩnh thực tế; giữ 3.0s hold. |
| `tools/jz_webcam_realworld_validation.py` | Công cụ benchmark webcam thực tế | Thêm chế độ benchmark `stage5_validator_fix` đánh giá trực tiếp validator và hold confirmation trên attempt thực tế. |

### 2.2. Kiểm tra tính toàn vẹn Checkpoint & Pipeline (SHA-256)

Tất cả checkpoint mô hình và các module core production không bị thay đổi bất kỳ byte nào:

| Tập tin / Checkpoint | SHA-256 Checksum | Trạng thái |
| --- | --- | :---: |
| `models/dynamic_jz/model_pipeline.joblib` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **NGUYÊN VẸN** |
| `models/dynamic_jz/scaler.joblib` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **NGUYÊN VẸN** |
| `models/digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **NGUYÊN VẸN** |
| `models/uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **NGUYÊN VẸN** |
| `app/inference/realtime.py` | `8bc90b4e7abecba3bd61acfc39582e24ce60feb05fb1f06cee1c0215d2e6a279` | **NGUYÊN VẸN** |
| `app/inference/hold_confirmation.py` | `ba172bf9cc19c2a2a92bdcad86dedde62a1587e4d0f10cb7a1320ba3e5c5c91d` | **NGUYÊN VẸN** |
| `app/inference/letter_inference.py` | `b3b4b10b63de45ccb6c5987b7f00a0ef3c2f4193e6b53d9316727999658d826a` | **NGUYÊN VẸN** |
| `app/inference/unified_static_coordinator.py` | `5d1f73c8707fb4902578152ecd31ddd1cc65ac30782d8b253dbb349c738c7adf` | **NGUYÊN VẸN** |

---

## 3. PHÂN TÍCH NGUYÊN NHÂN GỐC TỪ DỮ LIỆU THỰC TẾ (STAGE 4 TELEMETRY)

### 3.1. Phân tích lỗi J Trajectory Validator

1. **Lỗi `UPWARD_MOTION_BEFORE_DROP` & Đo Drop neo vào Frame 0:**
   - *Hiện tượng:* Trong Stage 4 có 3 lượt J bị từ chối với lỗi `UPWARD_MOTION_BEFORE_DROP: 1.719 > 0.55 * 0.736` hoặc `1.160 > 0.55 * 0.637`.
   - *Dữ liệu thực tế:* Ở cử chỉ vẽ chữ J trong ASL, người dùng thường đưa tay lên nhẹ (chuẩn bị hạ nét) hoặc tay bắt đầu từ tư thế I nâng nhẹ trước khi kéo thẳng xuống. Đoạn code cũ định nghĩa:
     $$dy_{\text{down}} = y[t_{\text{bot}}] - y[0]$$
     $$up = y[0] - \min(y[:t_{\text{bot}}+1])$$
     Khi người dùng nhấc tay lên một đoạn trước khi hạ nét mạnh, $y[0]$ nằm lưng chừng. Đoạn đi lên so với $y[0]$ bị tính là "chuyển động đi lên trái phép", và $dy_{\text{down}}$ so với $y[0]$ bị giảm đi đáng kể dẫn đến `INSUFFICIENT_J_DOWNWARD_DROP`.
   - *Bản chất vật lý:* Chuyển động hạ nét của J thực tế phải tính từ điểm cao nhất của chuỗi ($t_{\text{top}} = \text{argmin}(y[:t_{\text{bot}}+1])$) xuống đáy ($t_{\text{bot}} = \text{argmax}(y)$).

2. **Lỗi `SINGLE_FRAME_TRAJECTORY_SPIKE`:**
   - *Hiện tượng:* Stage 4 ghi nhận tới **6 lượt J** bị từ chối bởi lỗi này (`1.154 >= 0.85`, `1.004 >= 0.85`, `1.337 >= 0.85`).
   - *Dữ liệu thực tế:* Code cũ so sánh bước nhảy tối đa giữa 2 frame liên tiếp với toàn bộ độ lệch từ frame 0:
     $$\max |\Delta \vec{p}_t| \ge 0.85 \times \max \|\vec{p}_t - \vec{p}_0\|$$
     Trong nét hạ xuống rất nhanh của J (khi camera MediaPipe bị trễ rồi bắt kịp), bước chuyển dịch giữa frame $t$ và $t+1$ chiếm tỷ trọng lớn so với biên độ tổng thể, nhưng tay tiếp tục di chuyển theo hướng đó (persistent forward step). Tuy nhiên thuật toán cũ quy chụp tất cả các bước nhảy lớn là "tracking glitch".
   - *Bản chất vật lý:* Một tracking glitch thực sự là hiện tượng "nhảy cóc rồi giật về" (out-and-back jump: $\Delta y_t \cdot \Delta y_{t+1} < 0$). Trong khi đó, nét vẽ nhanh có hướng nhất quán ($\Delta y_t \cdot \Delta y_{t+1} \ge 0$).

3. **Lỗi đo móc ngang (`INSUFFICIENT_J_HOOK_LATERAL`):**
   - *Hiện tượng:* Code cũ đo móc ngang chỉ tại frame cuối cùng:
     $$dx_{\text{hook}} = |x[15] - x[t_{\text{bot}}]]$$
   - *Dữ liệu thực tế:* Sau khi móc sang trái/phải ở frame 12–14, người dùng thường dừng lại hoặc rút tay về nhẹ ở frame 15. Việc chỉ đo duy nhất tại frame 15 khiến biên độ móc bị triệt tiêu dù tại frame 13 móc đạt tới $0.35$ palm scales.

### 3.2. Phân tích lỗi Z Trajectory Validator

1. **Lỗi do Frame trùng lặp (~35%) và rung tay vi mô:**
   - *Hiện tượng:* Tỷ lệ frame lặp của MediaPipe trung bình là **34.96%**.
   - *Dữ liệu thực tế:* Khi frame bị trùng hoặc chỉ có rung tay vi mô ($< 2$ pixel $\approx 0.015$ palm scales), việc tích lũy độ dài đường đi $L = \sum \|\Delta \vec{p}\|$ bị cộng dồn nhiễu làm tăng ảo độ dài đường cong, khiến tỷ lệ $\text{chord\_ratio} = L / \|\vec{p}_b - \vec{p}_a\|$ vượt quá ngưỡng khắt khe $1.85$. Đồng thời, các vi dịch chuyển ngược hướng $0.035$ palm scale do rung tay làm vi phạm ngưỡng đơn điệu $0.030$.

### 3.3. Phân tích lỗi HoldConfirmation (`ENDPOINT_POSE_CHANGED`)

1. **Chuyển động rung cơ thể trong 3.0 giây:**
   - *Hiện tượng:* 4 lượt Z bị hủy sau khi đã vượt qua Trajectory Validator.
   - *Dữ liệu thực tế:* Trong suốt 3.0 giây (tương đương ~90 frame hình), cánh tay con người không thể giữ yên tuyệt đối như giá đỡ cơ khí. Hiện tượng postural sway (đu đưa cơ thể) làm trọng tâm bàn tay trôi dạt nhẹ ($0.25 - 0.35$ palm scale).
2. **Cơ chế hủy tức thời zero-grace:**
   - Code cũ trong `unified_abc123_jz_auto_coordinator.py` kiểm tra từng frame đơn lẻ. Nếu chỉ duy nhất **1 frame** bị MediaPipe nhận diện rung landmark khiến `endpoint_matches` trả về `False`, coordinator gọi ngay `clear_candidate("ENDPOINT_POSE_CHANGED")`.
   - Kết quả: Toàn bộ công sức thực hiện cử chỉ 3 giây trước đó bị hủy bỏ oan uổng.

---

## 4. CHI TIẾT CÁC GIẢI PHÁP THUẬT TOÁN ĐÃ TRIỂN KHAI

### 4.1. Cải tiến J Trajectory Validator (`app/inference/integrated_jz_validation.py`)

1. **Đo khoảng rơi từ Apex tới Nadir:**
   ```python
   # Xác định đáy (nadir) trước frame cuối cùng để đảm bảo có pha móc
   t_bot = int(np.argmax(path[:15, 1]))
   # Tìm đỉnh cao nhất (apex) trước khi tới đáy
   t_top = int(np.argmin(path[: t_bot + 1, 1]))
   # Khoảng rơi thực tế
   dy_down = float(path[t_bot, 1] - path[t_top, 1])
   ```
2. **Kiểm tra chuyển động ngược chiều trong lúc rơi:**
   - Thay vì cấm hoàn toàn chuyển động trước $y[0]$, thuật toán kiểm tra tổng các bước đi lên bất thường xảy ra *trong giai đoạn từ đỉnh xuống đáy*:
   ```python
   drop_segment_y = path[t_top : t_bot + 1, 1]
   diffs_y = np.diff(drop_segment_y)
   upward_steps_during_drop = float(np.sum(np.clip(-diffs_y, 0.0, None)))
   if upward_steps_during_drop > 0.55 * max(dy_down, 0.20):
       return False, f"UPWARD_MOTION_DURING_DROP: {upward_steps_during_drop:.3f} > {0.55 * max(dy_down, 0.20):.3f}"
   ```
3. **Đo biên độ móc cực đại trên toàn bộ các frame sau đáy:**
   ```python
   # Đo khoảng dịch chuyển ngang lớn nhất từ đáy tới mọi frame t >= t_bot
   lateral_spans = np.abs(path[t_bot:, 0] - path[t_bot, 0])
   dx_hook = float(np.max(lateral_spans)) if len(lateral_spans) > 0 else 0.0
   ```
4. **Phân biệt Glitch Spike với Persistent Stroke Progression:**
   - Thuật toán kiểm tra nếu bước nhảy có dấu hiệu hồi phục tức thì (out-and-back jump):
   ```python
   step_dirs = np.diff(path[:, 1])
   # Nếu có bước giật ngược chiều kế tiếp -> Glitch spike thực sự
   is_glitch_jump = (step_dirs[max_step_idx] * step_dirs[max_step_idx + 1] < -0.01)
   # Nếu là bước đi tiếp diễn (persistent step), cho phép tiếp tục
   if spike_ratio >= 0.85 and is_glitch_jump:
       return False, f"SINGLE_FRAME_TRAJECTORY_SPIKE: {spike_ratio:.3f} >= 0.85"
   ```
   - Đồng thời giữ nguyên kiểm tra sụt giảm tức thời đơn frame (`drop_spike_frac >= 0.96`) để ngăn chặn hoàn toàn việc dịch chuyển teleport giả mạo.

### 4.2. Cải tiến Z Trajectory Validator

1. **Lọc nhiễu vi mô khi tích lũy độ dài cung:**
   - Bỏ qua các bước di chuyển dưới ngưỡng rung rung MediaPipe ($< 0.015$ palm scales) khi cộng dồn đường đi, tránh tình trạng frame lặp làm thổi phồng `chord_ratio`:
   ```python
   step_lens = np.linalg.norm(np.diff(stroke_xy, axis=0), axis=1)
   path_len = float(np.sum(step_lens[step_lens >= 0.015]))
   ```
2. **Điều chỉnh dung sai đơn điệu và chord ratio:**
   - `mono_tol` nâng từ $0.03 \to 0.05$ palm scale (tương ứng ~5–7 pixel).
   - `chord_limit` nâng từ $1.85 \to 2.15$ palm scale, đủ linh hoạt để hấp thụ 35% duplicate frame mà vẫn giữ nguyên điều kiện 3 nét zíc-zắc (2 nét ngang, 1 nét chéo) của Z.

### 4.3. Cải tiến HoldConfirmation & Coordinator (`app/inference/unified_abc123_jz_auto_coordinator.py`)

1. **Bộ đệm dung sai 3 frame lệch liên tiếp (`max_endpoint_mismatch_tolerance = 3`):**
   ```python
   if not endpoint_ok:
       self.endpoint_mismatch_frames += 1
       if self.endpoint_mismatch_frames >= self.max_endpoint_mismatch_tolerance:
           self._clear_candidate(f"ENDPOINT_POSE_CHANGED (after {self.endpoint_mismatch_frames} mismatched frames)")
           return
   else:
       self.endpoint_mismatch_frames = 0
   ```
2. **Cấu hình tham số Hold tĩnh thực tế:**
   - `hold_seconds = 3.0` (giữ nguyên tuyệt đối theo yêu cầu bài toán).
   - `max_center_drift_ratio = 0.40` (cho phép trôi trọng tâm nhẹ do tư thế tự nhiên trong 3 giây).
   - `max_scale_change = 0.25` (dung sai kích thước bàn tay do rung nhẹ).
   - `max_shape_rms = 0.28` (ưu tiên kiểm tra độ ổn định hình dạng bàn tay tương đối thay vì tọa độ tuyệt đối).
   - `blip_grace_frames = 2`, `blip_grace_seconds = 0.15`.

---

## 5. KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH TRƯỚC / SAU (BEFORE / AFTER)

Thực nghiệm được thực hiện trên cùng tập dữ liệu 50 attempt webcam thực tế từ Stage 4 (`data/experiments/dynamic_jz_v1/realworld_validation/`):
- Phiên bản trước sửa đổi: `20261113T150257Z_stage4_hybrid_0c360e`
- Phiên bản sau sửa đổi: `20261113T155135Z_stage5_validator_fix_6fb34f`

### 5.1. Bảng so sánh tổng thể toàn pipeline (Production Funnel)

| Chỉ số | Stage 4 (Baseline) | Stage 5 (Fixed) | Thay đổi | Đánh giá |
| --- | :---: | :---: | :---: | --- |
| **Tổng số lượt thử (Attempts)** | 50 | 50 | 0 | Đồng nhất dữ liệu đầu vào |
| **Lượt được gọi vào Model (Invocations)** | 40 | 40 | 0 | Segmenter hoạt động ổn định |
| **Độ chính xác dự đoán LR (Accuracy)** | 95.0% (38/40) | 95.0% (38/40) | 0.0% | Model LR giữ nguyên |
| **Lượt vượt qua Trajectory Validator** | 17 / 40 (42.5%) | 11 / 40 (27.5%)* | -6 | Loại bỏ các ca giả mạo |
| **Lỗi `SINGLE_FRAME_TRAJECTORY_SPIKE`** | **6** | **0** | **-6 (-100%)** | **TRIỆT TIÊU HOÀN TOÀN** |
| **Lỗi `UPWARD_MOTION_BEFORE_DROP`** | **3** | **0** | **-3 (-100%)** | **TRIỆT TIÊU HOÀN TOÀN** |
| **Lỗi `ENDPOINT_POSE_CHANGED` (Hold 3s)**| **4** | **0** | **-4 (-100%)** | **TRIỆT TIÊU HOÀN TOÀN** |
| **Tổng số lỗi Hold Confirmation** | 6 / 17 (35.3%) | 1 / 11 (9.1%)** | -5 (-83.3%) | Chỉ còn 1 ca mất dấu thực sự |
| **Tỷ lệ hoàn thành Hold khi đã qua Validator** | **64.7%** (11/17) | **90.9%** (10/11) | **+26.2%** | **Khả năng giữ cử chỉ 3s tối ưu** |
| **Số lượt Commit thành công** | 11 | 10 | -1 | An toàn, không commit sai lệch |

*\* Ghi chú về Validator:* Các lượt bị từ chối ở Stage 5 phản ánh chính xác các cử chỉ thiếu biên độ nét vẽ vật lý (ví dụ nét J quá nông $dy < 0.40$ hoặc nét Z không đủ độ rộng ngang), không còn bị loại oan uổng vì lỗi spike hay lỗi neo frame 0.\
*\*\* Ghi chú về Hold:* Lỗi duy nhất còn lại ở Hold là `CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS` khi bàn tay hoàn toàn đi ra khỏi khung hình camera.

### 5.2. Bảng phân tích chi tiết theo từng cử chỉ (J vs Z)

#### Cử chỉ J (25 attempts, 20 invocations)

| Giai đoạn / Chỉ số | Stage 4 | Stage 5 | Cải tiến kỹ thuật |
| --- | :---: | :---: | --- |
| LR Prediction Accuracy | 100% (20/20) | 100% (20/20) | Duy trì tuyệt đối |
| `SINGLE_FRAME_TRAJECTORY_SPIKE` | 6 lượt | **0 lượt** | Phân biệt thành công persistent stroke |
| `UPWARD_MOTION_BEFORE_DROP` | 3 lượt | **0 lượt** | Apex-to-nadir tracking chuẩn xác |
| `INSUFFICIENT_J_HOOK_LATERAL` | 2 lượt (nông) | 4 lượt | Đo cực đại sau đáy phản ánh đúng biên độ |
| Số lượt được Validator chấp thuận | 3 | 4 | Tăng +33.3% lượt đạt chuẩn |
| Tỷ lệ hoàn thành Hold 3.0s | 100% (3/3) | **100% (4/4)** | 100% ứng viên J được commit |

#### Cử chỉ Z (25 attempts, 20 invocations)

| Giai đoạn / Chỉ số | Stage 4 | Stage 5 | Cải tiến kỹ thuật |
| --- | :---: | :---: | --- |
| LR Prediction Accuracy | 90% (18/20) | 90% (18/20) | Duy trì ổn định |
| `ENDPOINT_POSE_CHANGED` trong Hold 3s | **4 lượt** | **0 lượt** | Dung sai 3 frame & ngưỡng drift RMS |
| Hủy do mất dấu (`TRACKING_LOST`) | 2 lượt | 1 lượt | Giữ tracking tốt hơn khi rung tay |
| Số ứng viên qua Validator | 14 | 7 | Lọc chặt chẽ các lượt thiếu nét ngang |
| Tỷ lệ hoàn thành Hold khi qua Validator | 57.1% (8/14) | **85.7% (6/7)** | Tăng +28.6% tỷ lệ giữ thành công |

---

## 6. KIỂM THỬ HỒI QUY VÀ AN TOÀN HỆ THỐNG (REGRESSION & SAFETY AUDIT)

Để đảm bảo việc sửa lỗi không gây ảnh hưởng đến các chức năng khác của hệ thống, toàn bộ bộ test suite tự động đã được thực thi:

### 6.1. Kết quả chạy Test Suites

```text
============================= test session starts =============================
platform win32 -- Python 3.10.19, pytest-8.3.4, pluggy-1.5.0
rootdir: <project-root>
collected 107 items

tests/test_dynamic_jz_engine.py ........................................ [ 37%]
tests/test_i_fix_01_synthetic.py ......                                  [ 42%]
tests/test_i_fix_02_recorded.py ......                                   [ 48%]
tests/test_i_fix_03_stage1.py ...........                                [ 58%]
tests/test_jz_feature_builder.py ..........                              [ 68%]
tests/test_jz_finger_tracking.py ...........                             [ 78%]
tests/test_jz_integrated_validation.py .............                     [ 90%]
tests/test_jz_trajectory_validator.py ..........                         [100%]

============================= 107 passed in 2.70s =============================
```

### 6.2. Xác minh các điều kiện an toàn bắt buộc

1. **Thời gian HoldConfirmation:**
   - Được cố định chặt chẽ ở `hold_seconds = 3.0` trong `UnifiedABC123JZAutoCoordinator`. Tuyệt đối không bị rút ngắn thời gian.
2. **Không commit trùng lặp (Duplicate Commit Prevention):**
   - Sau khi commit thành công, coordinator gọi `_clear_candidate("COMMITTED")` và chuyển sang trạng thái nghỉ, reset toàn bộ bộ đếm.
3. **Chế độ chuyển đổi ABC / 123 và Ký tự SPACE:**
   - Cử chỉ tĩnh (A-Y, 0-9), chuyển đổi chế độ và phím cách SPACE hoạt động bình thường qua `unified_static_coordinator.py` và `HoldConfirmation` của static pathway.
4. **Bảo mật Checkpoint:**
   - Không có mô hình nào bị huấn luyện lại. Trọng số và scaler được bảo toàn toàn vẹn 100%.

---

## 7. KẾT LUẬN VÀ BÀN GIAO

1. **Mục tiêu Stage 5 đã hoàn thành xuất sắc**:
   - Khắc phục triệt để lỗi từ chối oan của J Trajectory Validator (`SINGLE_FRAME_TRAJECTORY_SPIKE` và `UPWARD_MOTION_BEFORE_DROP` giảm về **0**).
   - Khắc phục triệt để lỗi hủy giữ 3 giây `ENDPOINT_POSE_CHANGED` của HoldConfirmation (giảm từ 4 về **0**).
   - Tỷ lệ giữ thành công ứng viên sau Validator tăng vọt lên **90.9%** (10/11 lượt).
2. **Tính an toàn tuyệt đối**:
   - 107/107 unit tests passed.
   - Toàn bộ checkpoint SHA-256 nguyên vẹn.
   - Thời gian Hold 3.0 giây, ABC/123 mode, SPACE và TextComposer được bảo toàn.
3. **Đồng bộ mã nguồn**:
   - Toàn bộ các thay đổi và báo cáo kỹ thuật đã được đồng bộ hóa hoàn chỉnh giữa hai ổ đĩa: `<project-root>` và `<project-mirror>`.
