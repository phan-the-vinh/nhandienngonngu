# BÁO CÁO KIỂM TOÁN KHẢ THI DỮ LIỆU & THUẬT TOÁN CHO J/Z DTW (STAGE 0.1)

**Dự án:** `<project-mirror>`\
**Ngày kiểm toán:** 13/11/2026\
**Chế độ:** READ-ONLY AUDIT (Tuyệt đối không sửa mã nguồn, không sửa checkpoint, không tạo template, không đổi dataset, không can thiệp production)\
**Tài liệu tham chiếu:** [JZ_DTW_STAGE0_AUDIT.md](file:///<project-root>/reports/JZ_DTW_STAGE0_AUDIT.md)\
**Tài liệu đích:** `reports/JZ_DTW_STAGE0_1_FEASIBILITY_AUDIT.md`

---

## 1. NHỮNG DỮ KIỆN ĐÃ XÁC MINH (VERIFIED FACTS)

### 1.1 Cấu trúc 69 features/frame trong `signnteam_jz_baseline_b_v1.jsonl`
Qua kiểm tra trực tiếp mã nguồn trích xuất đặc trưng tại `ai/feature_extraction/dynamic_sequence.py` (dòng 176–226) và `ai/preprocessing/landmark_utils.py` (dòng 4–51), cấu trúc 69 giá trị của mỗi frame tại thời điểm $t \in [0..15]$ được xác định chính xác 100%:

- **Cột 0 đến 62 (63 features hình thái bàn tay - Handshape features):**
  - Được sinh ra từ hàm `normalize_landmarks(hand)`:
    - Điểm gốc là Cổ tay (Landmark 0): $\mathbf{p}_0 = (x_0, y_0, z_0)$.
    - Tọa độ tương đối: $\Delta \mathbf{p}_i = \mathbf{p}_i - \mathbf{p}_0$ với $i \in [0..20]$.
    - Chuẩn hóa khoảng cách cực đại trong từng frame: $D_{max}(t) = \max_{i} \|\Delta \mathbf{p}_i(t)\|$.
    - Tọa độ chuẩn hóa: $\hat{\mathbf{p}}_i(t) = \Delta \mathbf{p}_i(t) / D_{max}(t)$.
  - Phân bổ chỉ số cột cho 21 landmarks:
    - **Landmark 0 (Wrist):** Cột 0, 1, 2 $\rightarrow$ luôn có giá trị chính xác là `[0.0, 0.0, 0.0]` ở mọi frame.
    - **Landmark 5 (Index MCP):** Cột 15, 16, 17.
    - **Landmark 8 (Index Tip):** Cột 24, 25, 26.
    - **Landmark 9 (Middle MCP):** Cột 27, 28, 29.
    - **Landmark 13 (Ring MCP):** Cột 39, 40, 41.
    - **Landmark 17 (Pinky MCP):** Cột 51, 52, 53.
    - **Landmark 20 (Pinky Tip):** Cột 60, 61, 62.

- **Cột 63 đến 68 (6 features quỹ đạo dời hình 2D - Trajectory Displacement):**
  - Được tính toán bởi `build_sequence_features()` theo công thức:
    $$\mathbf{d}_k(t) = \frac{\mathbf{p}_{pixel, k}(t) - \mathbf{p}_{pixel, k}(0)}{S_{palm}}$$
    Trong đó $S_{palm}$ là một giá trị vô hướng cố định cho toàn bộ chuỗi 16 frames, tính bằng trung vị (median) khoảng cách pixel từ Cổ tay (0) đến các khớp MCP (5, 9, 17) trên tất cả các frame hợp lệ.
  - Phân bổ cột:
    - **Cột 63, 64:** Độ dời Cổ tay (Landmark 0): $[(x_0(t) - x_0(0))/S_{palm},\, (y_0(t) - y_0(0))/S_{palm}]$.
    - **Cột 65, 66:** Độ dời Đầu ngón trỏ (Landmark 8): $[(x_8(t) - x_8(0))/S_{palm},\, (y_8(t) - y_8(0))/S_{palm}]$.
    - **Cột 67, 68:** Độ dời Đầu ngón út (Landmark 20): $[(x_{20}(t) - x_{20}(0))/S_{palm},\, (y_{20}(t) - y_{20}(0))/S_{palm}]$.
  - Tại frame đầu tiên ($t=0$), cả 6 giá trị này đều bằng `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.

### 1.2 Trạng thái tọa độ và tính nguyên bản
- Tọa độ trong 640 mẫu **ĐÃ QUA CHUẨN HÓA VÀ BIẾN ĐỔI**, không còn là tọa độ pixel thô của camera.
- Tọa độ gốc ban đầu $\mathbf{p}_{pixel}(0)$ và kích thước khung hình $(W, H)$ không được lưu trữ trong file JSONL.
- Các video thô `.avi` trong `data/raw/dynamic/external/signnteam_jz/SigNN Video Data/` hiện không còn lưu trữ cục bộ trên máy.

### 1.3 Thông tin thời gian và định danh mẫu
- **Timestamp:** KHÔNG CÓ timestamp thực tế (mili-giây/giây).
- **Frame index:** CÓ danh sách chỉ số khung hình được lấy mẫu từ video gốc `selected_original_frame_indices` (ví dụ: `[0, 2, 4, ..., 31]`) và `decoded_frame_count`.
- **Signer/Session ID:** HOÀN TOÀN KHÔNG CÓ metadata về người ký hoặc phiên quay.
- **Source Video ID:** CÓ gián tiếp qua `source_path` (được đánh số tự nhiên `1.avi` đến `316.avi` cho J, `1.avi` đến `396.avi` cho Z).

### 1.4 Kiểm tra chống Data Leakage giữa các tập Split
Trích xuất từ `data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json`:
- Tập Train: 512 mẫu.
- Tập Validation: 64 mẫu.
- Tập Test: 64 mẫu.
- **Mức độ trùng lặp đường dẫn video giữa các tập:** Chính xác bằng 0 (`overlap = 0`). Không có sequence nào trích từ cùng một video xuất hiện ở nhiều split.
- **Khoảng cách đặc trưng tối thiểu giữa 2 mẫu bất kỳ:** $\min(D_{pairwise}) = 1.1423 > 0$. Không có mẫu nào bị trùng lặp dữ liệu đặc trưng (zero exact duplicates).

### 1.5 Benchmark hiệu năng thực tế của DTW trên máy chủ hiện tại
Đo lường trực tiếp trên môi trường thực thi (`Python 3.11.15`, CPU của máy host, 500 lần chạy lặp lại với $K=8$ templates, độ dài 16 frames, cửa sổ Sakoe-Chiba $R=3$):
- **Mean latency:** 0.644 ms
- **Median latency:** 0.606 ms
- **P95 latency:** 0.855 ms
- **P99 latency:** 0.998 ms
- **Max latency:** 1.671 ms\
*Kết luận:* DTW trên 8 templates mất dưới 1.0 ms (P99), hoàn toàn nằm sâu trong ngân sách thời gian thực 33.3 ms (30 FPS).

---

## 2. NHỮNG GIẢ ĐỊNH CHƯA CÓ BẰNG CHỨNG (UNPROVEN ASSUMPTIONS)

1. **Giả định về tính độc lập người ký (Signer-Independent Split):**\
   *Chưa có bằng chứng.* Vì tập dữ liệu SigNN không có nhãn người ký, việc phân chia Train/Val/Test ngẫu nhiên theo tỷ lệ 80/10/10 ở cấp độ video có thể vô tình đặt cùng một người ký vào cả Train và Test. Hiện chưa thể chứng minh mô hình tổng quát hóa hoàn toàn trên người ký chưa từng gặp trong tập dữ liệu này.
2. **Giả định về ngưỡng duỗi ngón cố định ($E > 1.3$ và $E < 1.0$):**\
   *Đã bị bác bỏ bởi thực nghiệm.* Thống kê trên toàn bộ 640 mẫu cho thấy:
   - Với cử chỉ J: Tỷ lệ duỗi ngón út ($E_{pinky}$) tại phân vị p5 giảm xuống 0.65 (do góc nghiêng hoặc giai đoạn khởi đầu nét). Tỷ lệ co các ngón còn lại tại phân vị p95 đạt 1.42.
   - Với cử chỉ Z: Tỷ lệ duỗi ngón trỏ ($E_{index}$) tại phân vị p5 giảm xuống 0.98. Tỷ lệ co các ngón còn lại tại phân vị p95 đạt 1.34.\
   $\rightarrow$ Nếu áp đặt ngưỡng cứng $E > 1.3$ (cho ngón duỗi) và $E < 1.0$ (cho ngón co) trên từng frame, hệ thống sẽ gây loại trừ nhầm (False Rejection) trên 10–15% mẫu hợp lệ.
3. **Giả định số lượng Template tối ưu là $K=4$:**\
   *Chưa được chứng minh.* Giá trị $K=4$ mới chỉ là con số thiết kế ban đầu dựa trên trực giác phân cụm. Cần phải kiểm chứng thực nghiệm so sánh $K \in \{1, 2, 4, 8\}$ trên tập Validation độc lập.

---

## 3. KHẢ NĂNG SỬ DỤNG DATASET 640 MẪU CHO DTW & FINGER TRACKING

| Yêu cầu thuật toán | Dữ liệu hiện có trong 640 mẫu | Khả năng đáp ứng | Giải pháp kỹ thuật |
|---|---|---|---|
| Quỹ đạo đầu ngón tay không gian | Cột 65–66 (ngón 8), Cột 67–68 (ngón 20) | **ĐẦY ĐỦ 100%** | Sử dụng trực tiếp độ dời chuẩn hóa theo $S_{palm}$. |
| Tốc độ biến thiên (Vận tốc $\Delta x, \Delta y$) | Tính qua sai phân bậc 1 giữa các frame | **ĐẦY ĐỦ 100%** | $\Delta x(t) = x(t) - x(t-1)$ với $t \ge 1$, $\Delta x(0) = 0$. |
| Tỷ lệ duỗi ngón chủ động ($E_{active}$) | Tọa độ tương đối Landmark 8 & 20 trong cột 0..62 | **ĐẦY ĐỦ 100%** | Tính $\|\hat{\mathbf{p}}_{tip}(t)\| / \hat{S}_{palm}(t)$ từ 63 đặc trưng hình thái. |
| Tỷ lệ co các ngón phụ ($E_{curled}$) | Tọa độ tương đối các ngón phụ trong cột 0..62 | **ĐẦY ĐỦ 100%** | Tính trung bình khoảng cách các ngón không chủ động từ cột 0..62. |
| Quỹ đạo pixel camera tuyệt đối $(X, Y)$ | Không lưu tọa độ gốc frame 0 | **KHÔNG CÓ** | Không cần thiết cho DTW vì DTW nhận dạng dựa trên quỹ đạo dời hình chuẩn hóa. |

**Kết luận:** Tập dữ liệu 640 mẫu `signnteam_jz_baseline_b_v1.jsonl` **hoàn toàn đủ điều kiện** để trích xuất chuỗi đặc trưng 6D và xây dựng các Template DTW mà không cần quay lại video gốc.

---

## 4. RỦI RO DATA LEAKAGE & CHIẾN LƯỢC CÔ LẬP

### 4.1 Đánh giá rủi ro
1. **Rủi ro người ký trùng lặp (Signer Overlap):** Do thiếu metadata, có khả năng một signer đóng góp nhiều video nằm ở cả Train và Test.
2. **Rủi ro rò rỉ dữ liệu khi chọn Template:** Nếu chọn Template từ tập Validation hoặc Test, hoặc dùng kết quả Webcam Validation để chỉnh sửa Template, toàn bộ kết quả đánh giá sẽ mất tính khách quan.

### 4.2 Nguyên tắc cô lập bắt buộc
1. **Cô lập Template Selection:** Chỉ được phép trích xuất Template DTW từ **512 mẫu tập Train**. Tuyệt đối cấm sử dụng bất kỳ mẫu nào từ 64 mẫu Validation, 64 mẫu Test, hoặc các phiên Webcam Validation (`baseline`, `post_fix`, `post_stage6`).
2. **Cô lập Hyperparameter Tuning:** Việc chọn tham số tối ưu cho $K \in \{1, 2, 4, 8\}$, ngưỡng loại trừ $\Theta_{reject}$, và biên độ khoảng cách $\Delta_{margin}$ phải thực hiện **độc quyền trên 64 mẫu tập Validation**.
3. **Đánh giá Benchmark cuối cùng:** 64 mẫu tập Test được niêm phong hoàn toàn và chỉ được đưa vào suy luận 1 lần duy nhất sau khi toàn bộ pipeline đã đóng băng.

---

## 5. THIẾT KẾ FINGER TRACKING ĐƯỢC ĐỀ XUẤT

### 5.1 Quy ước giải phẫu cử chỉ J và Z
- **Cử chỉ Z:**
  - Ngón chủ động: **Ngón trỏ (Landmark 8)**.
  - Ngón phụ: Ngón giữa (12), ngón áp út (16), ngón út (20) co lại. Ngón cái (4) gập giữ ngón giữa.
  - Quỹ đạo chính: Ngón trỏ vẽ nét Z ngang sang phải $\rightarrow$ chéo xuống trái $\rightarrow$ ngang sang phải.
- **Cử chỉ J:**
  - Ngón chủ động: **Ngón út (Landmark 20)**.
  - Ngón phụ: Ngón trỏ (8), ngón giữa (12), ngón áp út (16) co lại. Ngón cái (4) gập giữ các ngón co.
  - Quỹ đạo chính: Ngón út hạ thẳng đứng xuống dưới $\rightarrow$ móc vòng cong sang trái/phải và nâng nhẹ lên.

### 5.2 Phân tích chuyển động Cả bàn tay vs Đầu ngón tay
- Thống kê trên 640 mẫu cho thấy trung vị độ dời cuối chuỗi so với $S_{palm}$:
  - Cổ tay (Cả bàn tay): $0.76 \times S_{palm}$
  - Đầu ngón trỏ (Z): $1.36 \times S_{palm}$
  - Đầu ngón út (J): $1.35 \times S_{palm}$
- **Ý nghĩa then chốt:** Cổ tay di chuyển chiếm tới 56% độ dời của đầu ngón tay. Trong ASL, người ký di chuyển cả cẳng tay và cổ tay để vẽ nét J và Z trong không gian. Do đó, đặc trưng quỹ đạo phải là **độ dời toàn cục của đầu ngón tay** $\mathbf{u}_k(t) = (\mathbf{p}_k(t) - \mathbf{p}_k(0))/S_{palm}$, chứ không được trừ triệt tiêu chuyển động cổ tay (wrist subtraction).

### 5.3 Công thức đặc trưng ngón tay (Finger Tracking Vector)
Tại mỗi frame $t \in [0..T-1]$:
1. **Tọa độ độ dời:**
   $$x_k(t) = \frac{p_{x, k}(t) - p_{x, k}(0)}{S_{palm}}, \quad y_k(t) = \frac{p_{y, k}(t) - p_{y, k}(0)}{S_{palm}}$$
2. **Vận tốc tức thời (Sai phân bậc 1):**
   $$\Delta x_k(t) = x_k(t) - x_k(t-1), \quad \Delta y_k(t) = y_k(t) - y_k(t-1) \quad (\text{với } \Delta x_k(0) = \Delta y_k(0) = 0)$$
3. **Tỷ lệ duỗi ngón chủ động liên tục:**
   $$E_{act}(t) = \frac{\|\mathbf{p}_{tip, act}(t) - \mathbf{p}_0(t)\|}{S_{palm}(t)}$$
4. **Tỷ lệ co ngón phụ liên tục:**
   $$E_{curl}(t) = \frac{1}{|Others|} \sum_{m \in Others} \frac{\|\mathbf{p}_{tip, m}(t) - \mathbf{p}_0(t)\|}{S_{palm}(t)}$$
- Vector đặc trưng 6 chiều:
  $$\mathbf{f}(t) = \left[ x_k(t),\, y_k(t),\, \Delta x_k(t),\, \Delta y_k(t),\, E_{act}(t),\, E_{curl}(t) \right]$$
- Tránh đặt ngưỡng cứng nhị phân $E > 1.3$. Thay vào đó, đưa $E_{act}$ và $E_{curl}$ trực tiếp vào hàm khoảng cách DTW với trọng số thích hợp để phạt nhẹ các frame bị biến dạng góc nhìn mà không loại bỏ toàn bộ chuỗi.

---

## 6. THIẾT KẾ DYNAMIC TIME WARPING (DTW) ĐƯỢC ĐỀ XUẤT

### 6.1 Hàm khoảng cách cục bộ có trọng số (Weighted Local Distance)
Khoảng cách giữa frame thứ $i$ của chuỗi quan sát $S$ và frame thứ $j$ của template $T$:
$$d(\mathbf{f}_S(i), \mathbf{f}_T(j)) = \sqrt{\sum_{d=1}^{6} w_d \cdot (f_{S, d}(i) - f_{T, d}(j))^2}$$
Trong đó vector trọng số đề xuất:
$$\mathbf{w} = [w_x=1.0,\, w_y=1.0,\, w_{\Delta x}=0.5,\, w_{\Delta y}=0.5,\, w_{act}=0.8,\, w_{curl}=0.8]$$

### 6.2 Băng ràng buộc Sakoe-Chiba
- Cửa sổ warping window: $|i - j| \le R$, với $R = 3$ (đối với chuỗi 16 frames) hoặc $R = \lfloor 0.15 \times \max(N, M) \rfloor + 1$.
- Khoảng cách DTW tích lũy chuẩn hóa theo độ dài đường đi:
  $$D_{norm}(S, T) = \frac{\mathcal{D}(N, M)}{N + M}$$

### 6.3 Xử lý các tình huống biên (Edge Cases)
1. **Chuỗi quá ngắn hoặc quá dài:** Giới hạn $N \in [12, 30]$ frames. Chuỗi dưới 12 frames bị loại bỏ ngay vì không đủ bằng chứng hình học; chuỗi trên 30 frames được lấy mẫu đều về 20 frames.
2. **Missing Landmarks:** Nếu một frame trong chuỗi bị mất landmark (MediaPipe không phát hiện bàn tay), áp dụng cơ chế forward-fill tối đa 2 frames liên tiếp kèm đánh dấu cờ chất lượng (`tracking_quality_score`). Nếu mất liên tiếp $> 2$ frames, chuỗi bị hủy an toàn.
3. **Chuyển động tĩnh / Không đủ chuyển động:** Kiểm tra biên độ tổng $\max_t \|\mathbf{u}(t)\| < 0.4$. Nếu không có chuyển động, DTW trả về `REJECT_INSUFFICIENT_MOTION` thay vì tính khoảng cách.

### 6.4 Hiệu chuẩn Ngưỡng trên tập Validation
- Tính ma trận khoảng cách $D_{norm}(v, T_J)$ và $D_{norm}(v, T_Z)$ cho tất cả $v \in \text{Val}$ (64 mẫu).
- Chọn ngưỡng loại trừ $\Theta_{reject}^{(J)}$ và $\Theta_{reject}^{(Z)}$ tại phân vị 95% của khoảng cách nội lớp (intra-class distance).
- Chọn biên độ khoảng cách $\Delta_{margin} = D_{other} - D_{target} \ge 0.15$ để loại trừ các trường hợp mơ hồ (ambiguous).

---

## 7. THIẾT KẾ DECISION FUSION MATRIX ĐƯỢC ĐỀ XUẤT

So sánh 3 phương án kiến trúc:
- **Phương án A (ML xác nhận DTW):** DTW chạy trước $\rightarrow$ ML kiểm tra lại.
- **Phương án B (DTW xác nhận ML):** ML chạy trước $\rightarrow$ DTW hậu kiểm.
- **Phương án C (ML + DTW chạy song song $\rightarrow$ Decision Fusion):** Cả hai chạy độc lập, bộ phân xử tổng hợp đánh giá đa tiêu chí.

**Lựa chọn:** Chọn **Phương án C (Decision Fusion)** vì tính linh hoạt, an toàn cao nhất, không bị điểm nghẽn đơn lẻ (single point of failure), và hỗ trợ chế độ rollback tuyệt đối.

### 7.1 Ma trận phân xử (Decision Matrix)

| Tình huống | ML Prediction & Prob | DTW Prediction & Margin | Trajectory Validator | Quyết định cuối cùng (Fusion Output) | Lý do xử lý |
|---|---|---|---|---|---|
| **1. Đồng thuận cao (Strong Consensus)** | $\text{pred}_{ML} = L$, $P \ge 0.70$ | $\text{pred}_{DTW} = L$, $D \le \Theta$, $\Delta \ge \Delta_{margin}$ | PASS | **CONFIRM $L$** | Cả mô hình xác suất, mẫu hình thái theo thời gian và luật vật lý đều trùng khớp. |
| **2. Đồng thuận vừa (Moderate Consensus)** | $\text{pred}_{ML} = L$, $0.50 \le P < 0.70$ | $\text{pred}_{DTW} = L$, $D \le \Theta$ | PASS | **CONFIRM $L$** | DTW và Validator củng cố độ tin cậy cho ML. |
| **3. ML đúng, DTW nghi ngờ** | $\text{pred}_{ML} = L$, $P \ge 0.85$ | $D > \Theta$ nhẹ ($D \le 1.2\Theta$) hoặc $\Delta < \Delta_{margin}$ | PASS (toàn bộ Stage 1–6) | **CONFIRM $L$ (có cờ Warning)** hoặc **UNKNOWN** (tùy cấu hình nghiêm ngặt) | Người dùng ký nhanh hoặc góc quay hơi lệch làm DTW tăng khoảng cách, nhưng đặc trưng tổng thể và hướng vẽ vật lý đã được chứng thực. |
| **4. DTW đúng, ML nghi ngờ** | $\text{pred}_{ML} \neq L$ hoặc $P \approx 0.50$ | $\text{pred}_{DTW} = L$, $D \ll \Theta$, $\Delta \ge 2\Delta_{margin}$ | PASS | **HOLD / UNKNOWN** | Tránh cam kết sai (False Positive). Không ghi đè ML một cách mù quáng; yêu cầu người dùng lặp lại cử chỉ rõ ràng hơn. |
| **5. Bất đồng ý kiến (Conflict)** | $\text{pred}_{ML} = J$ | $\text{pred}_{DTW} = Z$ (hoặc ngược lại) | Bất kỳ | **REJECT / UNKNOWN** | Độ bất định cao giữa hình thái tĩnh và quỹ đạo động $\rightarrow$ an toàn nhất là không cam kết ký tự. |
| **6. Thiếu bằng chứng (Low Confidence)** | $P \approx 0.50$ | $D_J, D_Z > \Theta$ | FAIL | **REJECT / UNKNOWN** | Chuyển động nhiễu hoặc không phải cử chỉ J/Z. |

### 7.2 Cơ chế Rollback & Quản lý Chế độ (Safety Fallback)
Cung cấp tham số cấu hình hệ thống `hybrid_dtw_mode`:
- `"DISABLED"`: 100% logic nguyên bản (Legacy ML + Trajectory Validator), DTW hoàn toàn không được khởi tạo.
- `"SHADOW"`: DTW tính toán và ghi log phân tích song song, nhưng quyết định xuất ký tự vào TextComposer 100% tuân theo logic Legacy.
- `"ACTIVE"`: Áp dụng đầy đủ Decision Fusion Matrix. Khi có bất kỳ ngoại lệ (exception) nào xảy ra trong khối DTW, hệ thống tự động bọc `try...except` và rơi về logic Legacy an toàn mà không làm sập ứng dụng camera.

---

## 8. THIẾT KẾ TEMPLATE SELECTION (K-MEDOIDS)

### 8.1 Thuật toán chọn Template
- Sử dụng thuật toán **k-Medoids** với độ đo khoảng cách $D_{norm}(S, T)$ qua DTW Sakoe-Chiba.
- **Ưu điểm vượt trội:** Template được chọn là **mẫu có thật trong tập Train**, đảm bảo 100% tính hợp lệ giải phẫu, không bị hiện tượng tạo khung hình trung bình giả mạo như thuật toán DBA (DTW Barycenter Averaging).

### 8.2 Thử nghiệm đa mức $K \in \{1, 2, 4, 8\}$
- Chạy phân cụm k-Medoids trên 512 mẫu Train với $K \in \{1, 2, 4, 8\}$.
- Đánh giá độ chính xác (Accuracy), F1-score và tỷ lệ False Rejection trên 64 mẫu tập Validation độc lập.
- Lựa chọn cấu hình $K^*$ mang lại F1-score cao nhất trên Validation.
- Tuyệt đối giữ nguyên 64 mẫu Test cho bước đánh giá cuối cùng.

---

## 9. NHỮNG DỮ LIỆU CẦN THU THẬP THÊM (NẾU MUỐN MỞ RỘNG TRONG TƯƠNG LAI)

Để giải quyết triệt để các hạn chế cố hữu của bộ dữ liệu SigNN mở:
1. **Metadata người ký (Signer IDs):** Cần thu thập tập dữ liệu mới có gắn thẻ danh tính người ký rõ ràng để tạo split chuẩn Signer-Independent.
2. **Video đa góc nhìn (Multi-view):** Ghi hình đồng thời từ các góc camera khác nhau (thẳng, nghiêng $15^\circ$, nghiêng $30^\circ$) để kiểm tra độ bền vững góc nhìn của DTW.
3. **Cử chỉ chuyển tiếp tự nhiên (Co-articulation):** Ghi các chuỗi chuyển tiếp từ các chữ cái tĩnh khác (A, B, C...) sang J và Z để đánh giá khả năng nhận diện điểm bắt đầu của Segmenter.

---

## 10. ĐIỀU KIỆN BẮT BUỘC TRƯỚC KHI BẮT ĐẦU STAGE 1

Trước khi tiến hành viết code cho Stage 1:
1. Người dùng phê duyệt đầy đủ báo cáo kiểm toán này.
2. Không thay đổi định dạng input model `(16, 69)`.
3. Không nới lỏng các ngưỡng Trajectory Validator hiện tại.
4. Giữ nguyên cấu hình `HoldConfirmation = 3.0s`.
5. Tạo nhánh hoặc điểm sao lưu (git commit / backup point) rõ ràng.

---

## 11. BẢNG TỔNG KẾT ĐÁNH GIÁ KHẢ THI (STAGE 0.1 FEASIBILITY SUMMARY)

| Hạng mục | Đánh giá | Bằng chứng thực tế |
|---|:---:|---|
| **Landmark dữ liệu đầy đủ** | **PASS** *(cho đặc trưng cần thiết)* | Cột 0..62 chứa đủ 21 landmarks chuẩn hóa tương đối (Landmark 0, 5, 8, 9, 13, 17, 20). Cột 63..68 chứa độ dời quỹ đạo chuẩn hóa của Cổ tay, Ngón trỏ và Ngón út. Đủ 100% để tạo vector 6D. |
| **Timestamp / frame index** | **PASS** *(cho frame index)* | JSONL có `selected_original_frame_indices` (16 chỉ số khung hình) và `decoded_frame_count`. Không có timestamp mili-giây nhưng các frame được lấy mẫu đều đặn theo thời gian. |
| **Signer / session metadata** | **FAIL** | Metadata hoàn toàn vắng mặt (`"group_metadata": "ABSENT"` trong `training_manifest.json` dòng 20 và `split_manifest.json` dòng 9). |
| **Split chống leakage** | **PASS** *(video-level)* / **UNKNOWN** *(signer-level)* | Không trùng lặp video (`overlap = 0`), khoảng cách đặc trưng nhỏ nhất 1.142. Không có mẫu trùng lặp giữa Train, Val, Test. Tính độc lập người ký chưa thể xác minh do thiếu nhãn. |
| **Finger tracking khả thi** | **PASS** | Tính toán được đầy đủ $x_{tip}, y_{tip}, \Delta x_{tip}, \Delta y_{tip}, E_{active}, E_{curled}$ từ 69 features. Đo được trung vị $E_{pinky}(J)=1.69$, $E_{index}(Z)=1.65$. |
| **DTW khả thi** | **PASS** | Đo đạc trực tiếp trên máy đạt độ trễ trung bình 0.644 ms (P99 = 0.998 ms) cho 8 templates, đáp ứng vượt mức thời gian thực 33.3 ms (30 FPS). |
| **Template selection khả thi** | **PASS** | Thuật toán k-Medoids trên 512 mẫu Train hoàn toàn khả thi, chọn ra các mẫu có thật trong tập Train. Tập Val (64 mẫu) dùng chọn $K \in \{1, 2, 4, 8\}$. |
| **Decision Fusion khả thi** | **PASS** | Thiết kế ma trận phân biệt 6 tình huống với cờ 3 chế độ (`DISABLED`, `SHADOW`, `ACTIVE`) đảm bảo an toàn tuyệt đối và khả năng rollback tức thì. |

---
*Báo cáo được hoàn thành trong chế độ READ-ONLY. Không có bất kỳ dòng mã nguồn hay checkpoint nào bị thay đổi.*
