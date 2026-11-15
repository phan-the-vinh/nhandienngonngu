# BÁO CÁO KỸ THUẬT STAGE 10 — WRIST-RELATIVE MOTION & HANDSHAPE FUSION

**Dự án:** Hệ Thống Nhận Diện Cử Chỉ Ký Hiệu Động J & Z (Sign Language Dynamic ASL)\
**Tác giả:** Đội ngũ Kỹ sư AI & Edge Computing (DeepMind Antigravity)\
**Thời gian thực hiện:** Ngày 14 Tháng 11 Năm 2026\
**Môi trường thực nghiệm:** Python 3.12 (`nn` conda environment), MediaPipe Hands 0.10, Scikit-Learn 1.6, NumPy 2.2\
**Tập tin Telemetry:** [`reports/data/stage10_wrist_relative_results.json`](file:///<project-root>/reports/data/stage10_wrist_relative_results.json)\
**Công cụ Benchmark Ablation:** [`tools/evaluate_stage10_wrist_relative.py`](file:///<project-root>/tools/evaluate_stage10_wrist_relative.py)\

---

## TỔNG QUAN VÀ MỤC TIÊU STAGE 10

Stage 9 ghi nhận hiện tượng chữ Z bị nhận nhầm thành chữ J (ở cấu hình Trajectory-Only có tới 10/20 lượt Z bị đoán nhầm thành J). Giả thuyết đặt ra là: *Do chuyển động tịnh tiến của toàn bàn tay (whole-hand translation) khiến ngón út bị kéo lao dốc theo trục Y, tạo ra vết dốc và móc giả lập chữ J; do đó nếu trừ khử chuyển động cổ tay (wrist landmark 0) khỏi chuyển động đầu ngón tay thì quỹ đạo sẽ được làm sạch.*

Mục tiêu của **Stage 10** là:
1. **Kiểm tra và xác minh giả thuyết bằng số liệu động học thực nghiệm:** Đo đạc chính xác chuyển động tuyệt đối của cổ tay ($W$), ngón trỏ ($I_8$), ngón út ($P_{20}$), chuyển động tương đối so với cổ tay ($I_8 - W, P_{20} - W$), và chuyển động tương đối so với khớp bàn tay ($I_8 - MCP_5, P_{20} - MCP_{17}$) trên từng attempt thực tế. Phân loại tỷ lệ từng cơ chế: tịnh tiến bàn tay, ngón tay co duỗi, nhiễu tracking, hay quỹ đạo hình thái trùng lặp.
2. **Thực hiện Ablation bắt buộc trên 5 cấu hình độc lập:**
   - *Cấu hình 1 (Absolute Trajectory):* Quỹ đạo tuyệt đối trên mặt phẳng camera (baseline).
   - *Cấu hình 2 (Wrist-Relative Trajectory):* Trừ tọa độ cổ tay $W_0$ khỏi từng đầu ngón tay.
   - *Cấu hình 3 (Local Joint-Relative Trajectory):* Tọa độ đầu ngón tay tương đối so với khớp MCP tương ứng trong không gian 3D.
   - *Cấu hình 4 (Handshape-Only):* Nhận diện thuần túy dựa trên hình dạng bàn tay 21 điểm landmarks.
   - *Cấu hình 5 (Wrist-Relative + Handshape Fusion):* Kết hợp quỹ đạo tương đối cổ tay với hình dạng bàn tay và Soft Handshape Veto.
   - *Tham chiếu (Production LR & Adaptive Stage 8):* Đối chiếu trực tiếp trên cùng tập mẫu.
3. **Đánh giá trên 3 Suite dữ liệu không trộn lẫn mẫu:**
   - *Suite 1 (Clean Validation):* 64 mẫu chuẩn (26 J, 38 Z).
   - *Suite 2 (Synthetic OOD):* 32 mẫu nhiễu động học phi cử chỉ.
   - *Suite 3 (Webcam Validation):* 70 lượt webcam thực tế (25 J, 25 Z, 20 Negative Controls), tách riêng 50 lượt model-invoked và 70 lượt end-to-end.
4. **Đưa ra quyết định dựa trên Tiêu chí Kỹ thuật nghiêm ngặt:** Đánh giá tính khả thi của biểu diễn wrist-relative và đề xuất hướng đi tiếp theo.

---

## 1. PHÂN TÍCH VÀ XÁC MINH GIẢ THUYẾT ĐỘNG HỌC

Không mặc định giả thuyết là đúng, ta tiến hành trích xuất vector động học chi tiết trên toàn bộ 20 lượt webcam chữ Z được kích hoạt (Invoked):

### 1.1. Bảng số liệu động học từng Attempt Z trên Webcam

| Att # | Nhãn GT | Dự đoán LR | Dự đoán Adp | Dự đoán Trajectory | Độ dài Cổ tay ($W$) | Ngón trỏ Tuyệt đối | Ngón trỏ Tương đối Cổ tay | Ngón út Tuyệt đối | Ngón út Tương đối Cổ tay | Tỷ lệ $W / I_8$ | Tỷ lệ $W / P_{20}$ | Phân loại Cơ chế |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | Z | Z | Z | UNKNOWN | 0.75 | 1.01 | 1.38 | 0.76 | 0.49 | 0.74 | 0.98 | FINGER_ARTICULATION |
| **4** | Z | Z | Z | Z | 2.60 | 3.65 | 1.80 | 3.11 | 0.92 | 0.71 | 0.84 | WHOLE_HAND_TRANSLATION |
| **6** | Z | Z | Z | Z | 3.79 | 4.33 | 1.84 | 4.28 | 1.37 | 0.88 | 0.89 | WHOLE_HAND_TRANSLATION |
| **8** | Z | Z | J | J | 1.51 | 2.54 | 2.36 | 3.69 | 4.04 | 0.59 | 0.41 | GENUINE_Z_RESEMBLES_J |
| **10** | Z | J | J | J | 3.25 | 2.77 | 4.16 | 6.95 | 6.65 | 1.17 | 0.47 | WHOLE_HAND_TRANSLATION |
| **12** | Z | Z | Z | Z | 3.58 | 5.68 | 3.21 | 5.01 | 2.84 | 0.63 | 0.71 | WHOLE_HAND_TRANSLATION |
| **14** | Z | Z | Z | UNKNOWN | 0.51 | 1.61 | 1.33 | 2.59 | 2.42 | 0.32 | 0.20 | FINGER_ARTICULATION |
| **16** | Z | Z | UNKNOWN | J | 1.78 | 2.07 | 1.23 | 2.94 | 2.09 | 0.86 | 0.60 | WHOLE_HAND_TRANSLATION |
| **18** | Z | Z | Z | UNKNOWN | 1.22 | 4.49 | 3.98 | 3.66 | 3.32 | 0.27 | 0.33 | FINGER_ARTICULATION |
| **20** | Z | Z | Z | UNKNOWN | 1.29 | 3.06 | 3.00 | 2.45 | 2.58 | 0.42 | 0.53 | FINGER_ARTICULATION |
| **22** | Z | Z | Z | J | 1.65 | 5.68 | 5.49 | 3.52 | 3.37 | 0.29 | 0.47 | GENUINE_Z_RESEMBLES_J |
| **24** | Z | Z | Z | J | 1.04 | 5.01 | 4.87 | 2.15 | 1.91 | 0.21 | 0.48 | GENUINE_Z_RESEMBLES_J |
| **26** | Z | Z | UNKNOWN | Z | 3.06 | 4.09 | 2.01 | 4.19 | 2.89 | 0.75 | 0.73 | WHOLE_HAND_TRANSLATION |
| **28** | Z | Z | Z | J | 1.57 | 2.80 | 1.79 | 1.99 | 1.16 | 0.56 | 0.79 | GENUINE_Z_RESEMBLES_J |
| **30** | Z | J | J | J | 3.25 | 2.77 | 4.16 | 6.95 | 6.65 | 1.17 | 0.47 | WHOLE_HAND_TRANSLATION |
| **32** | Z | Z | Z | Z | 0.59 | 1.63 | 1.84 | 0.67 | 0.36 | 0.36 | 0.88 | FINGER_ARTICULATION |
| **34** | Z | Z | Z | Z | 3.21 | 3.52 | 1.29 | 3.44 | 0.92 | 0.91 | 0.93 | WHOLE_HAND_TRANSLATION |
| **36** | Z | Z | Z | J | 2.49 | 3.99 | 3.16 | 2.83 | 1.65 | 0.62 | 0.88 | WHOLE_HAND_TRANSLATION |
| **38** | Z | Z | Z | J | 2.03 | 2.36 | 1.44 | 2.70 | 2.04 | 0.86 | 0.75 | WHOLE_HAND_TRANSLATION |
| **40** | Z | Z | J | J | 1.82 | 2.85 | 2.21 | 3.45 | 4.04 | 0.64 | 0.53 | WHOLE_HAND_TRANSLATION |

### 1.2. Phân loại 4 nhóm trường hợp trên 20 lượt Webcam Z

```
+----------------------------------------------------------------------------------------------------+
|                         PHÂN BỐ 4 CƠ CHẾ ĐỘNG HỌC TRÊN 20 LƯỢT WEBCAM Z                            |
+------------------------------+------------+------------+-------------------------------------------+
| Nhóm Cơ Chế                  | Số lượng   | Tỷ lệ (%)  | Đặc điểm Động học Chính                   |
+------------------------------+------------+------------+-------------------------------------------+
| 1. WHOLE_HAND_TRANSLATION    | 11 / 20    |   55.0%    | Cổ tay di chuyển 62%–117% độ dài ngón trỏ |
|                              |            |            | Ngón út bị kéo dốc theo cả bàn tay        |
+------------------------------+------------+------------+-------------------------------------------+
| 2. FINGER_ARTICULATION       |  5 / 20    |   25.0%    | Cổ tay di chuyển ít (< 1.3 palm scale)    |
|                              |            |            | Ngón trỏ chủ động uốn vươn vẽ chữ Z       |
+------------------------------+------------+------------+-------------------------------------------+
| 3. GENUINE_Z_RESEMBLES_J     |  4 / 20    |   20.0%    | Nét vẽ Z có góc lượn mềm hoặc ngón trỏ    |
|                              |            |            | chém chéo tạo vết hook thỏa mãn chữ J     |
+------------------------------+------------+------------+-------------------------------------------+
| 4. TRACKING_NOISE_OR_LOST    |  0 / 20    |    0.0%    | Không có bước nhảy teleport (> 2.5 scale) |
+------------------------------+------------+------------+-------------------------------------------+
```

**Kết luận thực nghiệm Section 1:**
1. Có **55.0% (11/20)** các lượt Z thực tế xuất phát từ chuyển động của toàn bộ bàn tay và cẳng tay (`W_Len` đạt $1.78 \to 3.79 S_{\text{palm}}$).
2. Tuy nhiên, bản chất vật lý của chuyển động này **hoàn toàn khác với giả định**: Người dùng vẽ chữ Z bằng cách di chuyển toàn bộ bàn tay trong không gian (air-writing), trong khi ngón trỏ đóng vai trò là "ngòi bút" gắn cứng trên bàn tay.

---

## 2. XÂY DỰNG VÀ ĐỐI CHIẾU CÁC BIỂU DIỄN TỌA ĐỘ

Ta khảo sát 3 biểu diễn động học hình học:

```
A. Absolute (Tuyệt đối Camera-Plane):
   P_idx(t) = L_8(t) - L_8(0)
   P_pky(t) = L_20(t) - L_20(0)

B. Wrist-Relative (Tương đối Cổ tay):
   P_idx,rel(t) = (L_8(t) - L_8(0)) - (L_0(t) - L_0(0)) = [L_8(t) - L_0(t)] - [L_8(0) - L_0(0)]
   P_pky,rel(t) = (L_20(t) - L_20(0)) - (L_0(t) - L_0(0)) = [L_20(t) - L_0(t)] - [L_20(0) - L_0(0)]

C. Local Joint-Relative (Tương đối Khớp MCP trong không gian chuẩn hóa):
   P_idx,joint(t) = p_8(t) - p_5(t)   (Vector từ khớp MCP ngón trỏ đến đầu ngón trỏ)
   P_pky,joint(t) = p_20(t) - p_17(t) (Vector từ khớp MCP ngón út đến đầu ngón út)
```

### Hiện tượng triệt tiêu nét vẽ (Stroke Cancellation) khi trừ chuyển động cổ tay:
Khi phân tích hình học vector:
$$\vec{L}_8(t) = \vec{L}_0(t) + \vec{R}_{\text{hand}}(t) \cdot \vec{r}_{\text{index}}$$

Trong cử chỉ vẽ chữ Z tự nhiên, người dùng giữ ngón trỏ thẳng và vung cẳng tay/cổ tay để tạo thành 3 nét $Z$. Do đó, thành phần tạo nên nét ngang $dx_1$ và $dx_3$ chính là $\vec{L}_0(t)$ (sự dịch chuyển của cổ tay).
Khi ta thực hiện phép trừ cổ tay:
$$\vec{I}_{\text{rel}}(t) = \vec{L}_8(t) - \vec{L}_0(t) = \Delta \vec{R}_{\text{hand}}(t) \cdot \vec{r}_{\text{index}}$$
Kết quả toán học:
- **Biên độ nét ngang bị co cụm nghiêm trọng:** Độ mở rộng ngang $x_{\text{span}}$ của ngón trỏ giảm từ $2.36 S_{\text{palm}}$ xuống chỉ còn $1.50 S_{\text{palm}}$ trên Clean data, và trên webcam giảm từ $0.95 S_{\text{palm}}$ xuống $0.02 S_{\text{palm}}$ (giảm **97.8%**!).
- **Triệt tiêu nét 1 ($dx_1$):** Nét ngang đầu tiên bị co ngắn dưới ngưỡng nhận diện tối thiểu ($0.04 S_{\text{palm}}$), khiến bộ tìm kiếm 3 nét không thể tìm thấy điểm ngoặt hợp lệ.

---

## 3. PHÂN TÍCH RIÊNG J VÀ Z DƯỚI BIỂU DIỄN TƯƠNG ĐỐI

### 3.1. Chữ J (Landmark 20 — Ngón út)
- Khi thực hiện J, người dùng hạ bàn tay xuống rồi móc cổ tay lên.
- Phép trừ cổ tay $\vec{P}_{\text{rel}} = \vec{P}_{\text{abs}} - \vec{W}_{\text{abs}}$ làm giảm độ rơi dọc của ngón út, nhưng do ngón út có sự gập/móc cổ tay (ulnar/radial deviation), thành phần móc ngang $\Delta x_{\text{hook}}$ vẫn còn sót lại một phần.
- Tuy nhiên, độ dài đường đi giảm từ $6.00$ xuống $6.11$ (trên clean) và trên webcam tỷ lệ nhận diện đúng J không tăng mà còn làm một số lượt J bị phạt lỗi Anti-1D Drop.

### 3.2. Chữ Z (Landmark 8 — Ngón trỏ)
- Chữ Z phụ thuộc sống còn vào 3 nét rõ ràng: Ngang $\to$ Chéo $\to$ Ngang ngược chiều.
- Khi người dùng vẽ nét 1 sang phải: Cổ tay di chuyển sang phải $+0.93 S_{\text{palm}}$, ngón trỏ di chuyển sang phải $+0.95 S_{\text{palm}}$.
- Khi trừ cổ tay: $dx_{1,\text{rel}} = 0.95 - 0.93 = \mathbf{0.022 S_{\text{palm}}}$.
- Giá trị $0.022$ nhỏ hơn ngưỡng nét ngang tối thiểu $0.04$, khiến bộ phân tích kết luận: **"Không có nét ngang đầu tiên"**, phạt điểm $s_{\text{rev}} = 0$ và triệt tiêu toàn bộ điểm quỹ đạo $S_{\text{tr}}(Z) = 0.000$!
- Đây là nguyên nhân trực tiếp khiến số lượt nhận diện đúng Z trên webcam bị sụt giảm từ 6 lượt xuống còn **3 lượt** (Recall rơi từ 30% xuống 15%)!

---

## 4. KẾT QUẢ ABLATION BẮT BUỘC TRÊN 3 SUITE DỮ LIỆU

### 4.1. Suite 1 — Clean Validation Split (64 mẫu: 26 J, 38 Z)

| Cấu hình Thử nghiệm | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR | Latency P50 | Latency P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR (Ref)** | **100.0%** | **1.000** | 100.0% | 0.0% | **100.0%** | **100.0%** | 0.0% | 0.459 ms | 0.874 ms |
| **Adaptive Stage 8 (Ref)** | 98.4% | 0.990 | 98.4% | 1.6% | 96.2% | **100.0%** | 0.0% | 1.148 ms | 1.646 ms |
| **1. Absolute Trajectory** | 89.1% | 0.905 | 95.3% | 4.7% | 80.8% | 94.7% | 0.0% | 1.120 ms | 1.499 ms |
| **2. Wrist-Relative Trajectory** | 79.7% | 0.817 | 93.8% | 6.2% | 73.1% | 84.2% | 0.0% | 1.080 ms | 1.487 ms |
| **3. Local Joint-Relative** | 76.6% | 0.828 | 84.4% | 15.6% | 76.9% | 76.3% | 0.0% | 1.080 ms | 1.492 ms |
| **4. Handshape-Only** | 98.4% | 0.990 | 98.4% | 1.6% | 96.2% | **100.0%** | 0.0% | **0.111 ms** | **0.218 ms** |
| **5. Wrist-Relative + Handshape** | 90.6% | 0.949 | 90.6% | 9.4% | 88.5% | 92.1% | 0.0% | 1.240 ms | 1.734 ms |

---

### 4.2. Suite 2 — Synthetic Out-Of-Distribution (OOD) Suite (32 mẫu)

| Cấu hình Thử nghiệm | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | Negative FAR | Đánh giá Chống Nhiễu OOD |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Production LR (Ref)** | 0.0% | 0.000 | 100.0% | 0.0% | **100.0% (32/32)** | **Báo động giả 100%** (không có UNKNOWN) |
| **Adaptive Stage 8 (Ref)** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối hoàn hảo 100% mẫu OOD |
| **1. Absolute Trajectory** | 0.0% | 0.000 | 100.0% | 0.0% | **34.4% (11/32)** | Bị đánh lừa bởi rơi thẳng và xoay tròn |
| **2. Wrist-Relative Trajectory** | 0.0% | 0.000 | 100.0% | 0.0% | **34.4% (11/32)** | Bị đánh lừa tương tự quỹ đạo tuyệt đối |
| **3. Local Joint-Relative** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối tốt do không có handshape hợp lệ |
| **4. Handshape-Only** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối hoàn hảo 100% mẫu OOD |
| **5. Wrist-Relative + Handshape** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối hoàn hảo 100% mẫu OOD |

---

### 4.3. Suite 3 — Authentic Real-World Webcam Attempts (70 lượt)

Đánh giá đối chiếu cùng tập mẫu và cùng mẫu số:

#### Bảng A: Đánh giá Model-Invoked (50 lượt được Motion Segmenter kích hoạt: 20 J, 20 Z, 10 Negative)

| Cấu hình Thử nghiệm | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR | Latency P50 | Latency P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR (Ref)** | **95.0%** | **0.950** | 100.0% | 0.0% | **100.0%** | **90.0%** | 100.0% | 0.457 ms | 0.819 ms |
| **Adaptive Stage 8 (Ref)** | 85.0% | 0.866 | 95.0% | 5.0% | **100.0%** | 70.0% | 100.0% | 1.143 ms | 1.621 ms |
| **1. Absolute Trajectory** | 65.0% | 0.631 | 90.0% | 10.0% | **100.0%** | 30.0% | 90.0% | 1.054 ms | 1.507 ms |
| **2. Wrist-Relative Trajectory** | 57.5% | 0.508 | 90.0% | 10.0% | **100.0%** | **15.0%** | 90.0% | 1.048 ms | 1.431 ms |
| **3. Local Joint-Relative** | 35.0% | 0.381 | 80.0% | 20.0% | 50.0% | 20.0% | 70.0% | 1.104 ms | 1.622 ms |
| **4. Handshape-Only** | 77.5% | 0.831 | 82.5% | 17.5% | **100.0%** | 55.0% | 80.0% | **0.117 ms** | **0.248 ms** |
| **5. Wrist-Relative + Handshape** | 62.5% | 0.676 | 67.5% | 32.5% | **100.0%** | 25.0% | **40.0%** | 1.207 ms | 1.417 ms |

#### Bảng B: Đánh giá End-to-End (Toàn bộ 70 lượt: 25 J, 25 Z, 20 Negative)

| Cấu hình Thử nghiệm | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR (Ref)** | **76.0%** | **0.844** | 80.0% | 20.0% | **80.0%** | **72.0%** | 50.0% |
| **Adaptive Stage 8 (Ref)** | 68.0% | 0.767 | 76.0% | 24.0% | **80.0%** | 56.0% | 50.0% |
| **1. Absolute Trajectory** | 52.0% | 0.557 | 72.0% | 28.0% | **80.0%** | 24.0% | 45.0% |
| **2. Wrist-Relative Trajectory** | 46.0% | 0.452 | 72.0% | 28.0% | **80.0%** | **12.0%** | 45.0% |
| **3. Local Joint-Relative** | 28.0% | 0.335 | 64.0% | 36.0% | 40.0% | 16.0% | 35.0% |
| **4. Handshape-Only** | 62.0% | 0.731 | 66.0% | 34.0% | **80.0%** | 44.0% | 40.0% |
| **5. Wrist-Relative + Handshape** | 50.0% | 0.592 | 54.0% | 46.0% | **80.0%** | 20.0% | **20.0%** |

#### Ma trận Nhầm lẫn Chi tiết trên Webcam Invoked (40 lượt mẫu dương tính J/Z):
- **Production LR:** J: 20 TP, 0 FP; Z: 18 TP, 2 FP (nhận nhầm J), 0 UNK.
- **Adaptive Stage 8:** J: 20 TP, 0 FP; Z: 14 TP, 4 FP (nhận nhầm J), 2 UNK.
- **1. Absolute Trajectory:** J: 20 TP, 0 FP; Z: 6 TP, 10 FP (nhận nhầm J), 4 UNK.
- **2. Wrist-Relative Trajectory:** J: 20 TP, 0 FP; Z: **3 TP, 13 FP (nhận nhầm J)**, 4 UNK.
- **3. Local Joint-Relative:** J: 10 TP, 10 FP (nhận nhầm Z); Z: 4 TP, 8 FP (nhận nhầm J), 8 UNK.
- **4. Handshape-Only:** J: 20 TP, 0 FP; Z: 11 TP, 2 FP (nhận nhầm J), 7 UNK.
- **5. Wrist-Relative + Handshape:** J: 20 TP, 0 FP; Z: 5 TP, 2 FP (nhận nhầm J), 13 UNK.

---

## 5. ĐÁNH GIÁ THEO TIÊU CHÍ QUYẾT ĐỊNH (DECISION CRITERIA)

Theo mục tiêu kỹ thuật đề ra tại Section 7 của đề bài, **chỉ chọn Wrist-Relative Fusion nếu đồng thời thỏa mãn 4 điều kiện**:

| Tiêu chí Kiểm tra | Ngưỡng Yêu Cầu | Kết quả Thực nghiệm Stage 10 | Đánh giá Đạt/Không đạt |
| :--- | :--- | :--- | :---: |
| **1. Z Recall tăng rõ rệt** | Tăng vượt mức 70.0% của Adaptive Stage 8 | Z Recall **tụt dốc từ 70.0% xuống còn 25.0%** | **KHÔNG ĐẠT** (Thất bại nặng nề) |
| **2. J Recall không suy giảm** | Duy trì $\ge 95.0\%$ | J Recall đạt **100.0%** (20/20) | **ĐẠT** |
| **3. Negative FAR không tăng** | Duy trì $\le 50.0\%$ (End-to-End) | Negative FAR giảm xuống **20.0%** | **ĐẠT** |
| **4. Xác nhận trên webcam tuning độc lập** | Phải tăng trên tập webcam chưa dùng tuning | Giảm trên 100% các tập webcam thử nghiệm | **KHÔNG ĐẠT** |

**QUYẾT ĐỊNH CUỐI CÙNG:**
> [!CAUTION]
> **TỪ CHỐI TÍCH HỢP WRIST-RELATIVE FUSION.**
> Kết quả thực nghiệm chứng minh rằng việc trừ khử chuyển động cổ tay (Wrist-Relative Decoupling) là một sai lầm về mặt mô hình hóa vật lý. Nó triệt tiêu chính tín hiệu cử chỉ của chữ Z, làm Z Recall giảm từ 70% xuống 25% (hoặc từ 30% xuống 15% ở bộ Trajectory đơn thuần), đồng thời làm tăng số ca Z bị nhận nhầm thành J từ 10 ca lên 13 ca!

---

## 6. TRẢ LỜI 4 CÂU HỎI NGHIÊN CỨU CỐT LÕI

### Câu hỏi 1: Whole-hand translation có thực sự là nguyên nhân chính không?
**Trả lời:**
- **ĐÚNG LÀ CÓ TỒN TẠI, NHƯNG BẢN CHẤT VẬT LÝ HOÀN TOÀN NGƯỢC VỚI PHÁN ĐOÁN BAN ĐẦU.**
- Dữ liệu thực nghiệm chứng minh trong **55.0%** các lượt chữ Z, cổ tay di chuyển một quãng đường rất lớn ($W_{\text{len}} \in [1.78, 3.79] S_{\text{palm}}$), chiếm từ 62% đến 117% tổng độ dài đường đi của ngón trỏ.
- Tuy nhiên, sự di chuyển của cổ tay **không phải là nhiễu tịnh tiến vô nghĩa**, mà chính là **phương thức cơ học chính để người ký hiệu vẽ chữ Z trong không gian**. Trong ngôn ngữ ký hiệu ASL tự nhiên, người dùng cố định khớp ngón trỏ ở tư thế chỉ tay ('1') và dùng cánh tay/cổ tay để "viết chữ Z trên không khí".
- Vì vậy, chuyển động toàn bàn tay không phải là "nhiễu cần lọc bỏ", mà chính là một phần của tín hiệu cử chỉ!

### Câu hỏi 2: Wrist-relative có cải thiện nhận diện Z không?
**Trả lời:**
- **HOÀN TOÀN KHÔNG. WRIST-RELATIVE LÀM TỒN THƯƠNG NGHIÊM TRỌNG ĐẾN NHẬN DIỆN CHỮ Z.**
- Khi trừ chuyển động cổ tay ($\vec{I}_{\text{rel}} = \vec{I}_{\text{abs}} - \vec{W}_{\text{abs}}$), ta vô tình trừ đi chính nét vẽ chữ Z.
- Hậu quả:
  - Biên độ nét ngang đầu tiên ($dx_1$) bị triệt tiêu từ $0.95$ xuống còn $0.022 S_{\text{palm}}$ (giảm 97.8%), rơi xuống dưới ngưỡng nhận diện nét ngang.
  - Cấu trúc 3 nét của Z bị gãy đổ, biến chữ Z thành một vết rung giật không có đảo chiều rõ ràng.
  - Z Recall trên webcam sụt giảm từ **30.0%** (Absolute) xuống còn **15.0%** (Wrist-Relative), và số ca Z bị nhận nhầm thành J tăng từ **10 ca lên 13 ca**.

### Câu hỏi 3: Kết hợp handshape và trajectory có tốt hơn Adaptive Stage 8 không?
**Trả lời:**
- **KHÔNG TỐT HƠN NẾU DÙNG QUỸ ĐẠO WRIST-RELATIVE.**
- Cấu hình `Wrist-Relative Trajectory + Handshape` chỉ đạt **62.5% Accuracy** và **25.0% Z Recall** trên Webcam Invoked, kém xa mức **85.0% Accuracy** và **70.0% Z Recall** của Adaptive Stage 8.
- Cấu hình này chỉ phát hiện được 5 lượt Z, và thực chất cả 5 lượt này đều được "cứu" bởi nhánh Handshape-Only, trong khi nhánh Wrist-Relative Trajectory cho điểm gần như bằng 0 và không đóng góp được giá trị tích cực nào.

### Câu hỏi 4: Phương án nào đáng tiếp tục?
**Trả lời:**
1. **DỪNG VIỆC SỬ DỤNG BIỂU DIỄN WRIST-RELATIVE:** Không tiếp tục trừ tọa độ cổ tay theo cách tuyến tính đơn giản.
2. **DỪNG VIỆC BỔ SUNG CÁC NGƯỠNG HÌNH HỌC PHÂN ĐOẠN 3 NÉT CỨNG:** Việc đếm số lần đảo chiều hay kiểm tra góc ngoặt hình học là quá giòn (brittle) trên dữ liệu webcam FPS biến thiên.
3. **GIỮ NGUYÊN PRODUCTION LOGISTIC REGRESSION LÀM MODEL SUY LUẬN CHÍNH:** Production LR đạt 95.0% Accuracy và 90.0% Z Recall trên các lượt hợp lệ. LR không bị hiện tượng triệt tiêu nét vẽ vì nó nhìn đồng thời cả 69 đặc trưng (tọa độ 3D bàn tay + độ dịch chuyển tuyệt đối của cổ tay và ngón tay) và tự học trọng số tương quan tối ưu.
4. **HƯỚNG ĐI ĐÚNG ĐẮN CHO GIAI ĐOẠN TIẾP THEO (TEMPORAL SEQUENCE REPRESENTATION):**
   - Thay vì thiết kế các bộ luật heuristic if/else, cần chuyển sang biểu diễn chuỗi thời gian học máy:
     - Huấn luyện một mô hình chuỗi thời gian nhẹ dạng **1D Temporal Convolutional Network (1D-TCN)** hoặc **Bi-GRU / Bi-LSTM nhẹ** (dưới 50K tham số, latency < 2ms) nhận đầu vào toàn bộ chuỗi $16 \times 69$.
     - Mô hình học sâu sẽ tự động trích xuất đặc trưng không gian - thời gian: nó học được rằng khi bàn tay mang hình dạng '1' VÀ cổ tay vẽ đường ziczac thì đó là Z, không cần con người phải bóc tách cổ tay một cách gượng ép.

---

## 7. KIỂM SOÁT HỆ THỐNG VÀ BẢO VỆ PRODUCTION

Tuân thủ tuyệt đối các quy tắc an toàn hệ thống:
1. **Production Pipelines:** Không chỉnh sửa [`src/main.py`](file:///<project-root>/src/main.py) và [`run.py`](file:///<project-root>/run.py).
2. **Model Checkpoints:** Cả 4 checkpoint đã được xác minh SHA-256 nguyên vẹn 100%:
   - `model_pipeline.joblib`: `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` (MATCH)
   - `scaler.joblib`: `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` (MATCH)
   - `digits_final.joblib`: `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` (MATCH)
   - `uvw_medium_model.joblib`: `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` (MATCH)
3. **Các phân hệ cốt lõi:** Chế độ ABC/123, SPACE, HoldConfirmation 3s, TextComposer giữ nguyên trạng.
4. **Đồng bộ hóa song song:** Toàn bộ dữ liệu, test script, telemetry JSON và báo cáo kỹ thuật đã được đồng bộ hóa hoàn toàn sang [`<project-mirror>`](file:///<project-mirror>).

---

## TỔNG KẾT

Stage 10 đã giải quyết triệt để và minh bạch câu hỏi khoa học về chuyển động tương đối cổ tay:
- Bác bỏ hoàn toàn giả thuyết cho rằng việc trừ khử chuyển động cổ tay sẽ làm sạch quỹ đạo chữ Z.
- Cung cấp bằng chứng toán học và cơ sinh học cho thấy chuyển động cổ tay chính là "ngòi bút" vẽ nên chữ Z trong cử chỉ ASL thực tế.
- Khuyến nghị kiên quyết: **Không triển khai Wrist-Relative Fusion vào production**, giữ nguyên Production LR được bảo vệ bởi Motion Segmenter, và định hướng chuyển dịch sang mô hình chuỗi thời gian học máy (Temporal Sequence Learning) cho các giai đoạn nâng cấp dài hạn.
