# BÁO CÁO TRIỂN KHAI STAGE 1 — IMPLEMENT FINGER TRACKING ENGINE FOR J/Z

**Dự án:** `<project-mirror>`\
**Ngày thực hiện:** 13/11/2026\
**Chế độ:** STANDALONE ENGINE IMPLEMENTATION (Chỉ tạo feature extraction & unit tests; không tạo template chính thức, không can thiệp production, không đổi mô hình)\
**Tài liệu tham chiếu:**\
- [JZ_DTW_STAGE0_AUDIT.md](file:///<project-root>/reports/JZ_DTW_STAGE0_AUDIT.md)\
- [JZ_DTW_STAGE0_1_FEASIBILITY_AUDIT.md](file:///<project-root>/reports/JZ_DTW_STAGE0_1_FEASIBILITY_AUDIT.md)\
**Tài liệu đích:** `reports/JZ_DTW_STAGE1_REPORT.md`

---

## 1. CÁC TẬP TIN ĐÃ TẠO (FILES CREATED)

| Đường dẫn tập tin | Mục đích | Trạng thái |
|---|---|:---:|
| `app/inference/dynamic_jz/__init__.py` | Package initializer cho module suy luận động J/Z | ĐÃ TẠO |
| `app/inference/dynamic_jz/finger_tracker.py` | Finger Tracking Engine độc lập: bóc tách trajectory, handshape, delta displacement, tracking quality | ĐÃ TẠO |
| `tests/test_jz_finger_tracking.py` | Bộ kiểm thử unit tests toàn diện bao phủ 13 điều kiện hợp đồng | ĐÃ TẠO |
| `reports/JZ_DTW_STAGE1_REPORT.md` | Báo cáo chi tiết kết quả thực hiện Stage 1 | ĐÃ TẠO |

*Lưu ý về quy ước module:* Dự án quản lý toàn bộ các coordinator và predictor thời gian thực trong thư mục `app/inference/`. Việc tạo thư mục con `app/inference/dynamic_jz/` giúp đóng gói các thành phần động của J/Z (Finger Tracker, DTW Engine, Template Matcher) một cách độc lập, có cấu trúc, không làm phân tán các file logic hiện tại.

---

## 2. DATA CONTRACT & NGUYÊN TẮC BẢO TOÀN DỮ LIỆU

1. **Cấu trúc dữ liệu 69 features:**
   - **Cột 0–62:** 21 landmarks $\times 3$ $(x, y, z)$ tương đối với Cổ tay (Landmark 0), đã chia cho khoảng cách cực đại $D_{max}(t)$ theo `normalize_landmarks()`.
   - **Cột 63–68:** Độ dời pixel so với frame đầu tiên ($t=0$), đã chuẩn hóa theo $S_{palm}$ cố định của sequence theo `build_sequence_features()`.
   - Chuỗi cố định 16 frame ($T=16$).
   - Có `selected_original_frame_indices` chứa danh sách 16 frame gốc từ video (không có timestamp mili-giây).
2. **Nguyên tắc không chuẩn hóa lặp (No Double-Normalization):**
   - Các giá trị quỹ đạo toàn cục tại cột 63–68 đã mang đơn vị $[S_{palm}]$; engine trích xuất trực tiếp mà không chia lại cho bất kỳ đại lượng nào khác.
   - Các tọa độ hình thái tại cột 0–62 đã ở hệ tọa độ tương đối cổ tay; engine sử dụng trực tiếp để tính khoảng cách hình thái học nội tại bàn tay.
3. **Phân biệt rạch ròi Displacement per Frame vs Velocity per Second:**
   - Không có timestamp mili-giây nên không thể tính toán vận tốc theo đơn vị `pixel/second` hay `m/s`.
   - Đại lượng chuyển động tức thời được định nghĩa và lưu trữ dưới tên `delta_displacement` $\Delta \mathbf{u}(t) = \mathbf{u}(t) - \mathbf{u}(t-1)$ với ý nghĩa rõ ràng là **độ dịch chuyển giữa 2 frame liên tiếp (đơn vị $S_{palm}/frame$)**.
4. **Không tạo timestamp giả (No Synthetic Timestamps):**
   - Output contract hoàn toàn không chứa bất kỳ trường timestamp, thời gian ms, hay giả định FPS cố định nào.
5. **Bảo toàn bất biến dữ liệu đầu vào (Input Immutability):**
   - `FingerTracker.extract()` thực hiện sao chép phòng thủ (`np.asarray(sequence, dtype=np.float64).copy()`), đảm bảo mảng đầu vào không bao giờ bị biến đổi in-place.

---

## 3. CÔNG THỨC VÀ ĐẶC TRƯNG FINGER TRACKING ĐỘC LẬP

### 3.1 Nhóm Quỹ đạo Toàn cục (Global Trajectory)
Phản ánh chuyển động thực của đầu ngón tay và bàn tay trên mặt phẳng camera:
- **Ký tự J (Ngón út - Landmark 20):**
  $$\mathbf{u}_J(t) = \left[ \text{seq}[t, 67],\, \text{seq}[t, 68] \right] = \frac{\mathbf{p}_{20}(t) - \mathbf{p}_{20}(0)}{S_{palm}}$$
- **Ký tự Z (Ngón trỏ - Landmark 8):**
  $$\mathbf{u}_Z(t) = \left[ \text{seq}[t, 65],\, \text{seq}[t, 66] \right] = \frac{\mathbf{p}_{8}(t) - \mathbf{p}_{8}(0)}{S_{palm}}$$
- **Chuyển động Cổ tay (Wrist Motion - Landmark 0):**
  $$\mathbf{u}_{wrist}(t) = \left[ \text{seq}[t, 63],\, \text{seq}[t, 64] \right] = \frac{\mathbf{p}_{0}(t) - \mathbf{p}_{0}(0)}{S_{palm}}$$

### 3.2 Nhóm Quỹ đạo Tương đối Cổ tay (Wrist-Relative Trajectory)
Phản ánh sự cử động khớp nội tại của ngón tay (tách biệt hoàn toàn với chuyển động tịnh tiến của cánh tay):
- Tái tạo từ 63 cột hình thái với landmark index $k$:
  $$\mathbf{r}_k(t) = \text{seq}[t, 3k : 3k+3] = \frac{\mathbf{p}_k(t) - \mathbf{p}_0(t)}{D_{max}(t)}$$
  - Với Z: $k = 8$ (cột 24..26).
  - Với J: $k = 20$ (cột 60..62).
- Nhóm này được giữ riêng biệt trong `wrist_relative_trajectory` (shape `(16, 3)`), không trộn lẫn với quỹ đạo toàn cục `global_trajectory` (shape `(16, 2)`).

### 3.3 Nhóm Đặc trưng Hình thái Liên tục (Continuous Handshape Features)
- **Tỷ lệ Palm Scale nội tại của từng frame:**
  $$S_{palm, shape}(t) = \text{median}\left( \|\mathbf{r}_5(t)\|, \|\mathbf{r}_9(t)\|, \|\mathbf{r}_{17}(t)\| \right)$$
- **Tỷ lệ duỗi của ngón chủ động (Active Finger Extension Ratio):**
  $$E_{act}(t) = \frac{\|\mathbf{r}_{act}(t)\|}{S_{palm, shape}(t)}$$
  - Với J: $act = 20$ (ngón út).
  - Với Z: $act = 8$ (ngón trỏ).
- **Tỷ lệ co của các ngón phụ (Curled Extension Ratio):**
  $$E_{curl}(t) = \frac{1}{|Others|} \sum_{m \in Others} \frac{\|\mathbf{r}_m(t)\|}{S_{palm, shape}(t)}$$
  - Với J: $Others = \{8, 12, 16\}$ (trỏ, giữa, áp út).
  - Với Z: $Others = \{12, 16, 20\}$ (giữa, áp út, út).
- **Khoảng cách giữa các đầu ngón tay:**
  $$D_{inter}(A, B, t) = \frac{\|\mathbf{r}_A(t) - \mathbf{r}_B(t)\|}{S_{palm, shape}(t)}$$
  Tính cho các cặp: ngón cái - trỏ, cái - giữa, cái - út, trỏ - út.
- **Handshape tại Endpoint:** Ghi nhận snapshot tại frame 15 (`t=15`).
- **Không sử dụng ngưỡng cứng:** Các giá trị được biểu diễn liên tục dưới dạng vector `(16,)`, cho phép DTW so khớp độ tương đồng hình dáng theo thời gian mà không loại trừ cứng.

### 3.4 Nhóm Đánh giá Chất lượng Theo dõi (Tracking Quality)
- `missing_landmark_count`: Đếm số frame có toàn bộ tọa độ hình thái bằng 0.
- `duplicate_frame_count`: Đếm số frame liên tiếp có $\|\text{seq}[t] - \text{seq}[t-1]\| < 10^{-7}$.
- `outlier_displacement_count`: Đếm số frame có độ dời tức thời $\|\Delta \mathbf{u}(t)\| > 1.5 \times S_{palm}$ (bước nhảy dị thường do MediaPipe spike).
- `frame_indices_monotonic`: Kiểm tra tính tăng ngặt của chỉ số khung hình gốc (`PASS`, `FAIL`, hoặc `UNKNOWN` nếu không cung cấp).
- `validity_status`: Phân loại trạng thái chuỗi thành `VALID`, `SUSPICIOUS` (nếu có duplicate $> 4$ hoặc có outlier spike), hoặc `INVALID` (nếu thiếu landmark, sai thứ tự index, hoặc có NaN/Inf).

---

## 4. KẾT QUẢ UNIT TESTS (`tests/test_jz_finger_tracking.py`)

Bộ kiểm thử được thiết kế độc lập, kiểm tra nghiêm ngặt 13 điều kiện hợp đồng:

```
tests.test_jz_finger_tracking ...
test_01_j_uses_landmark_20 ............................................. ok
test_02_z_uses_landmark_8 .............................................. ok
test_03_global_and_wrist_relative_differ_when_translating .............. ok
test_04_normalization_matches_original_preprocessing .................. ok
test_05_no_double_normalization ........................................ ok
test_06_missing_landmarks_detection .................................... ok
test_07_duplicate_frames_detection ..................................... ok
test_08_frame_indices_out_of_order_rejected ............................ ok
test_09_sequence_with_insufficient_frames_rejected ..................... ok
test_10_no_nan_or_inf_produced ......................................... ok
test_11_input_sequence_immutability .................................... ok
test_12_handshape_computed_per_frame ................................... ok
test_13_no_synthetic_timestamps ........................................ ok

----------------------------------------------------------------------
Ran 13 tests in 0.170s

OK
```

- **Kết quả:** 13/13 tests **PASS 100%**.

---

## 5. THỐNG KÊ ĐÁNH GIÁ OFFLINE TRÊN DATASET 640 MẪU

Thực thi trích xuất trên toàn bộ 512 mẫu Train và 64 mẫu Validation theo đúng `split_manifest.json`:

### 5.1 Thống kê trên 512 mẫu Train (SignNTeam Train Split)
- **Tổng số mẫu:** 512 (J: 212 mẫu, Z: 300 mẫu).
- **Trạng thái hợp lệ:**
  - `VALID`: 314 mẫu (61.3%).
  - `SUSPICIOUS`: 198 mẫu (38.7%) — hoàn toàn do có bước nhảy giữa 2 frame lấy mẫu lớn hơn 1.5 palm scale khi người ký vung tay nhanh (do chuỗi 16 frame được lấy mẫu từ clip 30–60 frame).
  - `INVALID`: 0 mẫu (0.0%).
- **Thống kê bất thường:**
  - Missing frames: **0**.
  - Duplicate frames: **0**.
  - NaN/Inf: **0**.
- **Quỹ đạo chuyển động (Trajectory Lengths):**
  - **Ký tự J:** Độ dài quỹ đạo ngón út toàn cục: $5.73 \pm 1.55 \times S_{palm}$. Độ dài quỹ đạo cổ tay: $2.38 \pm 0.82 \times S_{palm}$.
  - **Ký tự Z:** Độ dài quỹ đạo ngón trỏ toàn cục: $7.41 \pm 2.89 \times S_{palm}$. Độ dài quỹ đạo cổ tay: $3.86 \pm 1.63 \times S_{palm}$.
  - *Nhận xét:* Quỹ đạo Z dài hơn J trung bình 29% do Z bao gồm 3 nét thẳng nối tiếp; chuyển động cổ tay của Z cũng lớn hơn đáng kể so với J.
- **Phân phối Hình thái Bàn tay (Handshape Extension Distributions - Toàn bộ các frame):**
  - **Ký tự J:**
    - Ngón út chủ động ($E_{pinky}$): Mean = **1.62**, p5 = **0.63**, Median = **1.70**, p95 = **2.24**.
    - Các ngón phụ co lại ($E_{curled}$): Mean = **0.99**, p5 = **0.63**, Median = **0.98**, p95 = **1.38**.
  - **Ký tự Z:**
    - Ngón trỏ chủ động ($E_{index}$): Mean = **1.64**, p5 = **0.99**, Median = **1.65**, p95 = **2.14**.
    - Các ngón phụ co lại ($E_{curled}$): Mean = **0.77**, p5 = **0.54**, Median = **0.71**, p95 = **1.33**.
- **Handshape tại Endpoint (Frame 15):**
  - **Ký tự J:** $E_{pinky}(15)$ = **1.82** (p10 = 1.48, median = 1.81). $E_{curled}(15)$ = **1.13**.
  - **Ký tự Z:** $E_{index}(15)$ = **1.55** (p10 = 1.03, median = 1.52). $E_{curled}(15)$ = **0.90**.

### 5.2 Đánh giá Tổng quát hóa trên 64 mẫu Validation (Validation Split)
- **Tổng số mẫu:** 64 (J: 26 mẫu, Z: 38 mẫu).
- **Trạng thái hợp lệ:** `VALID`: 42 mẫu, `SUSPICIOUS`: 22 mẫu, `INVALID`: 0 mẫu. Missing = 0, Duplicate = 0.
- **So sánh độ tương đồng phân phối giữa Train và Validation:**
  - J Global Path Length: Train **5.73** vs Val **6.00** (lệch 4.7%).
  - Z Global Path Length: Train **7.41** vs Val **7.16** (lệch 3.4%).
  - J Active Extension Mean: Train **1.62** vs Val **1.57** (lệch 3.1%).
  - Z Active Extension Mean: Train **1.64** vs Val **1.60** (lệch 2.4%).
  - J Endpoint Active Extension: Train **1.82** vs Val **1.86** (lệch 2.2%).
  - Z Endpoint Active Extension: Train **1.55** vs Val **1.50** (lệch 3.2%).
- *Kết luận:* Các chỉ số đặc trưng phân bố nhất quán giữa tập Train và Validation. Không có hiện tượng trôi dạt phân phối (distribution drift).
- *Cam kết bảo vệ dữ liệu:* 64 mẫu tập Validation chỉ được dùng để quan sát độ tổng quát hóa; không có bất kỳ ngưỡng nào được tối ưu trong Stage 1. Toàn bộ 64 mẫu tập Test được bảo lưu nguyên vẹn.

---

## 6. HẠN CHẾ DỮ LIỆU & RÀNG BUỘC PHÁT NGÔN (DATA LIMITATIONS)

1. **Không tuyên bố Signer-Independent:** Tập dữ liệu gốc SigNN không có nhãn định danh người ký (`group_metadata: ABSENT`), do đó không được tuyên bố hệ thống độc lập hoàn toàn với người ký (signer-independent).
2. **Khoảng cách khung hình không đều (Temporal Subsampling):** Chuỗi 16 frame được lấy mẫu trải đều từ các clip có độ dài khác nhau (30–60 frame). Do đó tốc độ theo frame phản ánh hình thái tiến trình nét vẽ chứ không đại diện cho vận tốc vật lý cố định theo thời gian thực.

---

## 7. BẢO TỒN NGUYÊN TRẠNG PRODUCTION & HASH VERIFICATION

Tất cả các mô hình, checkpoints, và pipeline camera production được kiểm tra hàm băm SHA-256 trước và sau khi thực hiện Stage 1:

| Tập tin | SHA-256 Trước Stage 1 | SHA-256 Sau Stage 1 | Kết quả đối sánh |
|---|---|---|:---:|
| `data/.../model_pipeline.joblib` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **KHỚP 100%** |
| `data/.../scaler.joblib` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **KHỚP 100%** |
| `models/.../digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **KHỚP 100%** |
| `models/.../uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **KHỚP 100%** |
| `run.py` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | **KHỚP 100%** |
| `app/camera/realtime_inference.py` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | **KHỚP 100%** |
| `app/inference/unified_abc123_jz_auto_coordinator.py` | `8436ff4e80932d6467fd28f27f6a24a2d02b91191374479ced8540836b2a44bc` | `8436ff4e80932d6467fd28f27f6a24a2d02b91191374479ced8540836b2a44bc` | **KHỚP 100%** |
| `app/inference/integrated_jz_validation.py` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | **KHỚP 100%** |
| `app/inference/first16_jz_segmenter.py` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | **KHỚP 100%** |
| `tools/jz_webcam_realworld_validation.py` | `b80420a981149322a6eed6ab39de07ad8e1004a3cc7bc84c355c072a1caac5d6` | `b80420a981149322a6eed6ab39de07ad8e1004a3cc7bc84c355c072a1caac5d6` | **KHỚP 100%** |

- **Xác nhận:** Toàn bộ mã nguồn camera runtime, mô hình production, checkpoint, HoldConfirmation 3 giây, TextComposer và Qwen giữ nguyên vẹn 100%.

---

## 8. KẾT LUẬN & ĐIỀU KIỆN CHUYỂN GIAO STAGE 2

### Trạng thái Stage 1: **PASS**

- Finger Tracking Engine độc lập đã hoàn thiện, hoạt động chính xác và đạt 13/13 unit tests.
- Các nhóm đặc trưng (quỹ đạo toàn cục, quỹ đạo tương đối cổ tay, chuyển động cổ tay, tỷ lệ duỗi co ngón liên tục, delta displacement theo frame) được phân tách độc lập, sẵn sàng phục vụ tính toán khoảng cách DTW có trọng số ở Stage 2.
- Tuyệt đối không tự ý chuyển sang Stage 2 khi chưa có sự xác nhận và yêu cầu từ người dùng.
