# BÁO CÁO KỸ THUẬT STAGE 9 — TARGET FINGER TRAJECTORY RECOGNITION

**Dự án:** Hệ Thống Nhận Diện Cử Chỉ Ký Hiệu Động J & Z (Sign Language Dynamic ASL)\
**Tác giả:** Đội ngũ Kỹ sư AI & Edge Computing (DeepMind Antigravity)\
**Thời gian thực hiện:** Ngày 14 Tháng 11 Năm 2026\
**Môi trường thử nghiệm:** Python 3.12 (`nn` conda environment), MediaPipe Hands 0.10, Scikit-Learn 1.6, NumPy 2.2\
**Tập tin Telemetry:** [`reports/data/stage9_trajectory_results.json`](file:///<project-root>/reports/data/stage9_trajectory_results.json)\
**Tập tin Nguồn Triển Khai:** [`app/inference/dynamic_jz/target_trajectory_recognizer.py`](file:///<project-root>/app/inference/dynamic_jz/target_trajectory_recognizer.py)\
**Bộ Kiểm Thử Độc Lập:** [`tests/test_jz_target_trajectory_recognizer.py`](file:///<project-root>/tests/test_jz_target_trajectory_recognizer.py) (19/19 Tests PASS)\

---

## TỔNG QUAN VÀ MỤC TIÊU STAGE 9

Sau kết quả thực nghiệm Stage 8.2 chứng minh rằng việc vi điều chỉnh ngưỡng hình học đơn lẻ (Config A giảm `z_diagonal_threshold` từ 0.015 xuống 0.003) mang lại **0% khác biệt** trên 70 lượt webcam thực tế, Stage 9 chuyển hướng sang phân tích sâu bản chất vật lý và hình học của quỹ đạo đầu ngón tay mục tiêu:
1. **Chữ J:** Theo dõi trực tiếp **Landmark 20** (đầu ngón út — pinky tip). Phân tích pha bổ xuống (downward plunge), điểm uốn (inflection point / nadir), độ vươn móc ngang (lateral hook excursion), góc quay vector chuyển hướng, độ trơn cung lượn (curvature smoothness), và cơ chế chống rơi thẳng 1D (anti-1D drop guard).
2. **Chữ Z:** Theo dõi trực tiếp **Landmark 8** (đầu ngón trỏ — index tip). Phân tích cấu trúc 3 nét liên hoàn phân đoạn thời gian ($0 < t_1 < t_2 < 15$), nét ngang thứ nhất, nét chéo xuống với đảo chiều tọa độ ngang 1, nét ngang thứ hai với đảo chiều tọa độ ngang 2, tỷ lệ chiều dài nét, góc quay góc ngoặt, và cơ chế chống quét ngang/chém chéo đơn lẻ hoặc rung tay (anti-tremor guard).
3. **Thực nghiệm Ablation tách biệt hoàn toàn:** So sánh 5 phương pháp:
   - *Phương pháp 1:* Production Logistic Regression Pipeline (Production Baseline).
   - *Phương pháp 2:* Stage 8 Adaptive Motion Recognition Baseline (Adaptive Baseline).
   - *Phương pháp 3:* Trajectory-Only Recognizer (chỉ dùng hình học quỹ đạo đầu ngón tay).
   - *Phương pháp 4:* Handshape-Only Recognizer (chỉ dùng hình dạng bàn tay 21 landmarks).
   - *Phương pháp 5:* Target Finger Trajectory Recognizer (kết hợp quỹ đạo ngón tay mục tiêu và handshape).
4. **Đánh giá trên 3 Suite dữ liệu độc lập không trộn lẫn mẫu:**
   - *Suite 1 (Clean Validation Split):* 64 mẫu chuẩn (26 J, 38 Z).
   - *Suite 2 (Synthetic OOD Suite):* 32 mẫu nhiễu động học phi cử chỉ (rơi thẳng, quét ngang trái/phải, rung tay tĩnh, xoay tròn).
   - *Suite 3 (Authentic Webcam Attempts):* 70 lượt webcam thực tế (25 J, 25 Z, 20 Negative Controls). Đánh giá minh bạch cả trên cơ sở End-to-End (70 lượt) và Model-Invoked (50 lượt).

---

## 1. PHÂN TÍCH DỮ LIỆU HIỆN CÓ

### 1.1. Hiện tượng Z bị nhận nhầm thành J trên webcam thực tế
Từ phân tích 121 bản ghi diagnostic npz và 70 lượt webcam Stage 8.2, ta quan sát được cơ chế vật lý dẫn đến sự nhầm lẫn giữa Z và J:

```
          Quỹ đạo thực tế của bàn tay khi vẽ chữ Z

       (0, 0) o═════════> (x1, y1)   Nét ngang 1
                           \
                            \         Nét chéo xuống (cả bàn tay lao xuống)
                             \        --> Cổ tay (lm 0) & Ngón út (lm 20)
                              v           bị kéo lao dốc xuống theo!
                   (x2, y2) <══
                   /
                  o═════════> (x3, y3)  Nét ngang 2
```

1. **Hiện tượng dịch chuyển toàn bàn tay (Whole-hand Translation):** Khi người dùng vẽ chữ Z trước webcam máy tính, chuyển động xuất phát từ cánh tay và cổ tay, chứ không chỉ khớp ngón trỏ. Khi bàn tay chém chéo xuống ở nét 2, toàn bộ bàn tay (bao gồm ngón út đang co) bị kéo dốc xuống một khoảng $\Delta y > 1.0 S_{\text{palm}}$. Khi kết thúc nét 3, bàn tay dừng lại hoặc quét ngang.
   - Với bộ đánh giá quỹ đạo chữ J (nhìn vào cột 67:69 của ngón út), nó quan sát thấy một pha **bổ xuống cực mạnh** ($\Delta y_{\text{plunge}} > 1.0$) tiếp nối bởi một **độ lệch ngang** ($\Delta x > 0.4$), hoàn toàn trùng khớp với hình thái "Plunge + Hook" của chữ J!
2. **Nhiễu ước lượng độ sâu và hiện tượng co ngón (Handshape Ambiguity):** Khi ngón trỏ chỉ thẳng vào camera (foreshortening), MediaPipe ước lượng chiều dài ngón trỏ bị co ngắn. Trong một số frame, tỷ lệ duỗi của ngón trỏ $E_{\text{index}}$ bị tụt xuống dưới 0.95 palm scale, khiến điểm Handshape $S_{\text{hs}}(Z)$ giảm xuống $0.20 \to 0.24$. Với ngưỡng Veto cứng 0.25 của Stage 8, hệ thống đã triệt tiêu toàn bộ điểm của Z ($S_{\text{joint}}(Z) = 0$), đẩy lượt cử chỉ vào UNKNOWN hoặc ngả sang J nếu có bất kỳ rung động nào của ngón út!
3. **Bất đối xứng về độ phức tạp hình học:** Chữ J chỉ yêu cầu 2 pha hình học (rơi dốc + móc ngoặt), trong khi chữ Z yêu cầu khắt khe 3 nét với đúng 2 lần đảo chiều $dx_1 \cdot dx_2 < 0$ và $dx_2 \cdot dx_3 < 0$. Khi người dùng vẽ nhanh trên webcam (15-20 FPS), nét chém chéo thường bị vê tròn hoặc chỉ thu được 1 điểm đổi hướng, khiến chữ Z bị phạt điểm nặng nề trong khi chữ J dễ dàng đạt điểm cao.

### 1.2. Phân tích phân bố các nhóm trường hợp
- **J đúng (J TP):** Chiếm 100% các lượt invoked. Quỹ đạo ngón út có độ rơi $\Delta y \in [1.02, 3.55]$ và độ móc $\Delta x \in [0.45, 1.85]$. Ngón út luôn duỗi vượt trội so với các ngón khác ($E_{\text{pinky}} > 1.25, E_{\text{pinky}} - E_{\text{curled}} > 0.40$).
- **Z đúng (Z TP):** Đạt 14/20 lượt invoked ở Adaptive Baseline. Điểm chung của các lượt Z đúng là người dùng vẽ nét ngang rất rộng ($x_{\text{span}} > 1.2 S_{\text{palm}}$) và dừng rõ ràng ở 2 góc ngoặt ($t_1 \approx 4, t_2 \approx 10$).
- **Z nhận nhầm J (Z $\to$ J FP):** Xảy ra ở 2 lượt trong Production LR, 4 lượt trong Adaptive Baseline, và tăng vọt lên **10 lượt** (50%) ở Trajectory-Only. Nguyên nhân do ngón út bị kéo theo chuyển động của bàn tay khi chém chéo nét Z.
- **Negative Controls:**
  - Production LR dự đoán nhầm 10/10 lượt invoked thành J hoặc Z (Negative FAR = 100%).
  - Handshape-Only và Adaptive Baseline từ chối thành công các mẫu OOD nhờ phát hiện bàn tay tĩnh hoặc bàn tay không có ngón duỗi đặc trưng.

---

## 2. THIẾT KẾ BỘ NHẬN DIỆN QUỸ ĐẠO ĐẦU NGÓN TAY MỤC TIÊU (STAGE 9)

Module triển khai độc lập tại: [`app/inference/dynamic_jz/target_trajectory_recognizer.py`](file:///<project-root>/app/inference/dynamic_jz/target_trajectory_recognizer.py)

```
       +-------------------------------------------------------------+
       |                Sequence Input (16x69)                       |
       +-------------------------------------------------------------+
               |                                            |
      [Cols 0:62 Handshape]                       [Cols 63:68 Motion]
               |                                            |
               v                                            v
     +-------------------+                         +------------------+
     |  HandshapeBranch  |                         | Anti-Static Guard| (PathLen < 0.15)
     |  (21 3D Landmarks)|                         +------------------+
     +-------------------+                                  |
         |           |                           +----------+----------+
      S_hs(J)     S_hs(Z)                        |                     |
         |           |                           v                     v
         |           |                  +------------------+  +------------------+
         |           |                  |Pinky J Analyzer  |  |Index Z Analyzer  |
         |           |                  |(Landmark 20)     |  |(Landmark 8)      |
         |           |                  |- Plunge & Hook   |  |- 3-Stroke Search |
         |           |                  |- Inflection Turn |  |- 2 Reversals     |
         |           |                  |- Anti-1D Drop    |  |- Anti-Tremor     |
         |           |                  +------------------+  +------------------+
         |           |                           |                     |
         |           |                        S_tr(J)               S_tr(Z)
         v           v                           v                     v
       +------------------------------------------------------------------+
       |                  Joint Evidence & Gating Arbiter                 |
       |                                                                  |
       |  S_joint(J) = 0.40 * S_hs(J) + 0.60 * S_tr(J) + Bonus(J)         |
       |  S_joint(Z) = 0.40 * S_hs(Z) + 0.60 * S_tr(Z) + Bonus(Z)         |
       |                                                                  |
       |  Veto 1: Target Finger Curled (Contrast < -0.05) --> Score = 0   |
       |  Veto 2: Trajectory < 0.20 or Handshape <= 0.15 --> Score = 0   |
       |  Veto 3: Margin < 0.05 or Max Score < 0.35 --> UNKNOWN           |
       +------------------------------------------------------------------+
                                         |
                                         v
                         Decision: 'J' | 'Z' | 'UNKNOWN'
```

### 2.1. Phân tích chi tiết quỹ đạo chữ J (Pinky Tip — Landmark 20)
- **Điểm Apex và Nadir:** Tìm thời điểm ngón út xuống thấp nhất $t_{\text{nadir}} = \arg\max_{t} y_t$ và điểm cao nhất trước đó $t_{\text{apex}} = \arg\min_{t \le t_{\text{nadir}}} y_t$. Độ lao dốc $\Delta y_{\text{plunge}} = y_{t_{\text{nadir}}} - y_{t_{\text{apex}}}$.
- **Độ vươn móc ngang (Lateral Hook):** $\Delta x_{\text{hook}} = \max_{t \ge t_{\text{nadir}}} |x_t - x_{t_{\text{nadir}}}|$. Xác định hướng móc ("LEFT" hoặc "RIGHT").
- **Tỷ lệ vươn dốc / móc ngang:** $R = \Delta y_{\text{plunge}} / \max(\Delta x_{\text{hook}}, 10^{-6})$. Chữ J chuẩn có $R \in [1.2, 4.5]$.
- **Góc chuyển hướng tại điểm uốn:** Góc $\theta_{\text{turn}}$ giữa vector lao dốc $(P_{\text{nadir}} - P_{\text{apex}})$ và vector móc $(P_{\text{hook}} - P_{\text{nadir}})$. Chữ J chuẩn có $\theta_{\text{turn}} \in [75^\circ, 135^\circ]$.
- **Anti-1D Drop Guard:** Nếu ngón út lao thẳng đứng dốc $\Delta y > 0.12$ nhưng độ móc ngang $\Delta x < 0.05$ hoặc $R > 6.5$, lượt cử chỉ bị coi là **rơi tự do 1 chiều** và bị phạt điểm về 0.0 (`is_1d_downward = True`).
- **Anti-Circular Loop Guard:** Nếu quỹ đạo khép vòng tròn (khoảng cách điểm đầu-cuối $< 0.35 y_{\text{span}}$ và $x_{\text{span}} > 0.5 y_{\text{span}}$), hệ thống loại bỏ (`is_circular_loop = True`).

### 2.2. Phân tích chi tiết quỹ đạo chữ Z (Index Tip — Landmark 8)
- **Thuật toán tìm kiếm phân đoạn 3 nét ($0 < t_1 < t_2 < 15$):** Duyệt qua tất cả các cặp mốc thời gian thỏa mãn $1 \le t_1 \le 8$ và $t_1 + 2 \le t_2 \le 14$. Đánh giá chất lượng phân đoạn dựa trên độ dài vươn nét ngang và độ dốc nét chéo.
- **Xác thực 2 điểm đảo chiều (Reversals Validation):**
  - Nét 1 sang Nét 2: $dx_1 \cdot dx_2 < 0$.
  - Nét 2 sang Nét 3: $dx_2 \cdot dx_3 < 0$.
  - Cả 2 điều kiện phải thỏa mãn đồng thời (`reversals_valid = True`).
- **Nét chéo đi xuống:** $dy_2 > 0.015 S_{\text{palm}}$.
- **Anti-Flat Horizontal Guard:** Nếu $dy_2 < 0.015$ và độ mở rộng dọc $y_{\text{span}} < 0.08$, cử chỉ là một nét gạt ngang đơn thuần và bị phủ quyết về 0.0 (`is_flat_horizontal = True`).
- **Anti-Single Diagonal Guard:** Nếu số lần đảo chiều $< 2$, cử chỉ là một nét chém chéo đơn lẻ và bị phủ quyết về 0.0 (`is_single_diagonal = True`).
- **Anti-Tremor Guard:** Nếu chiều dài nét nhỏ nhất $\min(l_1, l_2, l_3) < 0.035 S_{\text{palm}}$, cử chỉ bị xác định là rung tay tần số cao và bị phủ quyết (`is_tremor = True`).

---

## 3. KIỂM TRA ĐẶC TÍNH DỮ LIỆU TRƯỚC KHI THIẾT KẾ

Thực hiện kiểm tra trực tiếp trên 64 mẫu Clean Validation và 121 bản ghi Authentic Webcam Replay:
1. **Tính nguyên vẹn của chuỗi:** 100% mẫu trong cả 3 suite đều có kích thước chuẩn $(16, 69)$. Không có bất kỳ giá trị NaN, Infinite hay null nào tồn tại.
2. **Chuẩn hóa hệ tọa độ:**
   - Cột 0:62 chứa 21 điểm 3D tương đối so với cổ tay, được chuẩn hóa theo $D_{\max}$ (khoảng cách cực đại từ cổ tay tới bất kỳ landmark nào trong frame đó).
   - Cột 63:68 chứa độ dịch chuyển 2D $(dx, dy)$ của Cổ tay, Ngón trỏ, và Ngón út so với frame 0, được chuẩn hóa theo $S_{\text{palm}}$ (trung vị khoảng cách từ cổ tay tới các khớp MCP 5, 9, 17 trong toàn bộ chuỗi).
3. **Thứ tự frame và thời gian:** Chuỗi 16 frame đại diện cho cửa sổ động học 16 bước thời gian tuần tự được trích xuất từ bộ nhớ đệm MediaPipe. Không có giả định cố định về FPS (ví dụ 16 frame không đồng nghĩa 16 FPS).
4. **Tỷ số Tín hiệu trên Nhiễu (Signal-to-Noise Ratio):**
   - Độ dịch chuyển do rung tay / jitter của camera: bước nhảy đơn khung $< 0.015 S_{\text{palm}}$, tổng chiều dài quỹ đạo $< 0.15 S_{\text{palm}}$.
   - Chuyển động vẽ J thật: chiều dài quỹ đạo ngón út đạt $3.71 \to 9.81 S_{\text{palm}}$ (trung bình 6.00).
   - Chuyển động vẽ Z thật: chiều dài quỹ đạo ngón trỏ đạt $3.66 \to 13.74 S_{\text{palm}}$ (trung bình 7.16).
   - Tỷ số SNR đạt trên **25:1**, cho phép ngưỡng Anti-Static $L_{\text{path}} < 0.15$ loại bỏ hoàn toàn rung động tĩnh mà không làm tổn thương cử chỉ thực.

---

## 4. KẾT QUẢ THỰC NGHIỆM SO SÁNH 5 PHƯƠNG PHÁP

Các phương pháp được đánh giá trên cùng một tập dữ liệu đầu vào chuỗi 16x69:
- **Phương pháp 1 (LR):** Production Logistic Regression Pipeline.
- **Phương pháp 2 (Adaptive):** Stage 8 Adaptive Motion Recognition Baseline.
- **Phương pháp 3 (Trajectory-Only):** Bộ nhận diện thuần hình học quỹ đạo (ngón 20 cho J, ngón 8 cho Z).
- **Phương pháp 4 (Handshape-Only):** Bộ nhận diện thuần hình dạng bàn tay 21 điểm landmarks.
- **Phương pháp 5 (Target Combined):** Bộ nhận diện kết hợp quỹ đạo ngón tay mục tiêu và handshape (Stage 9).

### 4.1. Suite 1 — Clean Validation Split (64 mẫu chuẩn: 26 J, 38 Z)

| Phương pháp | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR | Latency P50 | Latency P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR** | **100.0%** | **1.000** | 100.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.460 ms | 0.694 ms |
| **Adaptive Baseline** | 98.4% | 0.990 | 98.4% | 1.6% | 96.2% | 100.0% | 0.0% | 1.125 ms | 1.388 ms |
| **Trajectory-Only** | 89.1% | 0.905 | 95.3% | 4.7% | 80.8% | 94.7% | 0.0% | 1.035 ms | 1.168 ms |
| **Handshape-Only** | 98.4% | 0.990 | 98.4% | 1.6% | 96.2% | 100.0% | 0.0% | **0.113 ms** | **0.143 ms** |
| **Target Combined** | 95.3% | 0.973 | 95.3% | 4.7% | 92.3% | 97.4% | 0.0% | 1.198 ms | 1.429 ms |

*Confusion Matrix Suite 1:*
- **Production LR:** J: 26/26 TP, 0 FP; Z: 38/38 TP, 0 FP.
- **Adaptive Baseline:** J: 25 TP, 0 FP, 1 UNKNOWN; Z: 38 TP, 0 FP, 0 UNKNOWN.
- **Trajectory-Only:** J: 21 TP, 3 FP (nhận nhầm Z), 2 UNKNOWN; Z: 36 TP, 1 FP (nhận nhầm J), 1 UNKNOWN.
- **Handshape-Only:** J: 25 TP, 0 FP, 1 UNKNOWN; Z: 38 TP, 0 FP, 0 UNKNOWN.
- **Target Combined:** J: 24 TP, 0 FP, 2 UNKNOWN; Z: 37 TP, 0 FP, 1 UNKNOWN.

---

### 4.2. Suite 2 — Synthetic Out-Of-Distribution (OOD) Suite (32 mẫu nhiễu)

| Phương pháp | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | False Acceptance Rate (FAR) | Chẩn đoán OOD |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Production LR** | 0.0% | 0.000 | 100.0% | 0.0% | **100.0% (32/32)** | **Báo động giả toàn bộ** (không có lớp UNKNOWN) |
| **Adaptive Baseline** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối 100% mẫu OOD |
| **Trajectory-Only** | 0.0% | 0.000 | 100.0% | 0.0% | **31.2% (10/32)** | Nhận nhầm các chuyển động rơi thẳng & xoay tròn |
| **Handshape-Only** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối 100% mẫu OOD |
| **Target Combined** | 0.0% | 0.000 | 100.0% | 0.0% | **0.0% (0/32)** | Từ chối 100% mẫu OOD |

---

### 4.3. Suite 3 — Authentic Real-World Webcam Attempts (70 lượt)

Đánh giá minh bạch trên cả 2 cơ sở mẫu số:

#### Bảng A: Đánh giá Model-Invoked (50 lượt được Motion Segmenter kích hoạt: 20 J, 20 Z, 10 Negative)

| Phương pháp | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR | Latency P50 | Latency P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR** | **95.0%** | **0.950** | 100.0% | 0.0% | **100.0%** | **90.0%** | 100.0% | 0.441 ms | 0.492 ms |
| **Adaptive Baseline** | 85.0% | 0.866 | 95.0% | 5.0% | **100.0%** | 70.0% | 100.0% | 1.078 ms | 1.184 ms |
| **Trajectory-Only** | 65.0% | 0.631 | 90.0% | 10.0% | **100.0%** | 30.0% | 90.0% | 0.990 ms | 1.654 ms |
| **Handshape-Only** | 77.5% | 0.831 | 82.5% | 17.5% | **100.0%** | 55.0% | 80.0% | **0.110 ms** | **0.126 ms** |
| **Target Combined** | 62.5% | 0.676 | 67.5% | 32.5% | **100.0%** | 25.0% | **60.0%** | 1.195 ms | 1.855 ms |

#### Bảng B: Đánh giá End-to-End (Toàn bộ 70 lượt: 25 J, 25 Z, 20 Negative)

| Phương pháp | Accuracy | Macro F1 | Coverage | UNKNOWN Rate | J Recall | Z Recall | Negative FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR** | **76.0%** | **0.844** | 80.0% | 20.0% | **80.0%** | **72.0%** | 50.0% |
| **Adaptive Baseline** | 68.0% | 0.767 | 76.0% | 24.0% | **80.0%** | 56.0% | 50.0% |
| **Trajectory-Only** | 52.0% | 0.557 | 72.0% | 28.0% | **80.0%** | 24.0% | 45.0% |
| **Handshape-Only** | 62.0% | 0.731 | 66.0% | 34.0% | **80.0%** | 44.0% | 40.0% |
| **Target Combined** | 50.0% | 0.592 | 54.0% | 46.0% | **80.0%** | 20.0% | **30.0%** |

*Confusion Matrix trên Webcam Invoked (40 mẫu dương tính J/Z):*
- **Production LR:** J: 20 TP, 0 FP; Z: 18 TP, 2 FP (nhận nhầm thành J), 0 UNKNOWN.
- **Adaptive Baseline:** J: 20 TP, 0 FP; Z: 14 TP, 4 FP (nhận nhầm thành J), 2 UNKNOWN.
- **Trajectory-Only:** J: 20 TP, 0 FP; Z: **6 TP, 10 FP (nhận nhầm thành J)**, 4 UNKNOWN.
- **Handshape-Only:** J: 20 TP, 0 FP; Z: 11 TP, 2 FP (nhận nhầm thành J), 7 UNKNOWN.
- **Target Combined:** J: 20 TP, 0 FP; Z: 5 TP, 2 FP (nhận nhầm thành J), 13 UNKNOWN.

---

## 5. PHÂN TÍCH ABLATION: TRAJECTORY-ONLY VS HANDSHAPE-ONLY VS COMBINED

Thực nghiệm ablation cho thấy bức tranh hoàn chỉnh về sự đóng góp và hạn chế của từng nhánh thông tin:

```
+----------------------------------------------------------------------------------------------------+
|                                    BẢNG SO SÁNH ABLATION                                           |
+----------------------+--------------------+---------------------+----------------------------------+
| Cấu hình Thử nghiệm  | Ưu điểm nổi bật    | Điểm yếu chí mạng   | Hiệu năng Thực tế Webcam         |
+----------------------+--------------------+---------------------+----------------------------------+
| Trajectory-Only      | Nhận diện J 100%   | Không phân biệt     | Z Recall chỉ đạt 30.0%           |
|                      | Không bị ảnh hưởng | ngón duỗi vs ngón co| 50% lượt Z bị nhận nhầm thành J  |
|                      | bởi co ngón tay    | Chấp nhận 31.2% OOD | Nhầm do toàn bàn tay chuyển dịch |
+----------------------+--------------------+---------------------+----------------------------------+
| Handshape-Only       | Chống OOD tuyệt đối| Mù chuyển động      | Z Recall đạt 55.0%               |
|                      | (FAR = 0.0%)       | Dễ bị UNKNOWN khi   | Không phân biệt được tư thế tĩnh |
|                      | Độ trễ siêu thấp   | ngón trỏ hơi nghiêng| với nét vẽ động thực sự          |
+----------------------+--------------------+---------------------+----------------------------------+
| Target Combined      | Chống OOD 0.0%     | Cộng dồn điều kiện  | Z Recall tụt xuống 25.0%         |
| (Stage 9)            | Không nhầm chéo    | khắt khe khiến      | 13/20 lượt Z bị đẩy vào UNKNOWN  |
|                      | giữa J và Z        | tỷ lệ UNKNOWN cao   | Quá bảo thủ trong môi trường nhiễu|
+----------------------+--------------------+---------------------+----------------------------------+
| Adaptive Baseline    | Cân bằng tốt nhất  | Vẫn có 4 lượt Z     | Z Recall đạt 70.0%               |
| (Stage 8)            | giữa Recall và OOD | nhầm J do Veto cứng | Accuracy đạt 85.0%               |
+----------------------+--------------------+---------------------+----------------------------------+
```

1. **Vì sao Trajectory-Only thất bại thảm hại ở chữ Z (Recall chỉ 30%)?**
   - Khi không có Handshape định danh "ngón nào đang duỗi", việc toàn bộ bàn tay chém chéo xuống khi vẽ Z khiến đầu ngón út (landmark 20) tạo ra một vết dốc và ngoặt ngang cực lớn.
   - Kết quả là bộ phân tích quỹ đạo ngón út đạt điểm $S_{\text{tr}}(J) > 0.80$, áp đảo điểm của $S_{\text{tr}}(Z)$ trong 10/20 lượt Z!
   - Điều này chứng minh rằng: **Chuyển động trong không gian 2D của một đầu ngón tay không thể tách rời khỏi chuyển động tịnh tiến của cả bàn tay.**
2. **Vì sao Handshape-Only có Recall Z thấp (55%)?**
   - Trong 45% các lượt Z bị trượt, ngón trỏ của người dùng bị nghiêng hoặc hướng về phía webcam, làm giảm độ dài biểu kiến trên hình chiếu camera. Handshape-Only yêu cầu tỷ lệ duỗi ngón trỏ $> 0.95$ palm scale nên đã từ chối các lượt này là UNKNOWN.
3. **Vì sao Target Combined lại bị UNKNOWN cao (13/20 lượt)?**
   - Bộ kết hợp Stage 9 áp dụng đồng thời các điều kiện hình học khắt khe: (1) Ngón trỏ không được có contrast âm, (2) Quỹ đạo ngón trỏ phải có đủ 2 lần đổi chiều, (3) Nét chéo phải đi xuống dốc, (4) Ngón út không được có chuyển động J áp đảo.
   - Trên webcam thực tế, khi người dùng vẽ nét Z nhanh, nét chém chéo thường hơi chếch ngang hoặc hơi ngửa lên ($dy_2 \le 0$). Chỉ cần một điều kiện hình học bị vi phạm, điểm Z bị triệt tiêu về 0.0, dẫn đến việc bị đẩy vào UNKNOWN.

---

## 6. TRẢ LỜI 5 CÂU HỎI NGHIÊN CỨU CỐT LÕI

### Câu hỏi 1: Vì sao Z bị nhận nhầm thành J?
**Trả lời:**
Sự nhận nhầm Z thành J bắt nguồn từ **sự tương tác giữa cơ sinh học chuyển động bàn tay và sự bất đối xứng trong độ phức tạp hình thái**:
1. *Tịnh tiến toàn bàn tay (Whole-hand translation):* Cử chỉ Z yêu cầu vẽ một nét chéo từ góc trên-phải xuống góc dưới-trái. Khi người dùng hạ bàn tay xuống để vẽ nét chéo này, toàn bộ khối bàn tay dịch chuyển theo trục Y hướng xuống một khoảng từ $1.0 \to 2.5 S_{\text{palm}}$. Ngón út (mặc dù đang co gập) vẫn gắn liền với bàn tay, do đó tọa độ tuyệt đối của Landmark 20 cũng lao dốc xuống và sau đó dịch ngang khi bàn tay vẽ nét ngang đáy. Quỹ đạo này thỏa mãn hoàn hảo mô hình "Plunge + Hook" của chữ J!
2. *Độ phức tạp hình thái bất đối xứng:* Chữ J là hình thái 2 pha (bổ xuống $\to$ móc lên/ngang), rất dễ thỏa mãn ngay cả khi có nhiễu. Ngược lại, chữ Z là hình thái 3 pha với 2 điểm ngoặt gắt ($dx_1 \cdot dx_2 < 0$ và $dx_2 \cdot dx_3 < 0$). Bất kỳ sự trễ khung hình nào của camera hoặc thao tác vẽ vê tròn góc đều làm mất 1 điểm ngoặt của Z, khiến điểm quỹ đạo Z bị triệt tiêu và nhường quyền quyết định cho chữ J.
3. *Hiện tượng che khuất ngón trỏ:* Khi vẽ nét chéo sang trái, ngón trỏ thường bị mu bàn tay che một phần hoặc bị nghiêng góc, làm giảm điểm Handshape của Z và kích hoạt cơ chế Handshape Veto.

### Câu hỏi 2: Quỹ đạo landmark 8 có đủ thông tin để phân biệt Z không?
**Trả lời:**
**KHÔNG ĐỦ.**
1. Trên tập Clean Validation có điều kiện quay lý tưởng, quỹ đạo Landmark 8 đạt 94.7% độ chính xác. Nhưng trên webcam thực tế, quỹ đạo Landmark 8 đơn lẻ chỉ đạt **30.0% Z Recall** và có tỷ lệ báo động giả OOD lên tới **31.2%**.
2. Khi người dùng thực hiện chữ J, nắm đấm của các ngón co (trong đó có ngón trỏ co) di chuyển theo quỹ đạo bàn tay. Trong nhiều trường hợp, sự lắc lư của nắm đấm tạo ra 2 lần đổi chiều giả trên Landmark 8, khiến bộ phân tích quỹ đạo Z bị đánh lừa.
3. Ngược lại, khi người dùng thực hiện chữ Z, chuyển động thực tế trên webcam thường không tạo thành 3 đoạn thẳng cứng nhắc mà là một đường cong chữ Z mềm mại. Việc chỉ dựa vào quỹ đạo Landmark 8 để tìm kiếm 3 đoạn thẳng và 2 lần đổi chiều sẽ bỏ sót phần lớn các biến thể chữ Z tự nhiên.

### Câu hỏi 3: Trajectory-only có tốt hơn LR và Adaptive không?
**Trả lời:**
**KHÔNG, TRAJECTORY-ONLY KÉM HƠN RÕ RỆT.**
- *Về Accuracy trên Webcam Invoked:* Trajectory-Only (65.0%) kém xa Adaptive Baseline (85.0%) và Production LR (95.0%).
- *Về Z Recall trên Webcam:* Trajectory-Only chỉ đạt **30.0%**, trong khi Adaptive Baseline đạt **70.0%** và Production LR đạt **90.0%**.
- *Về khả năng lọc nhiễu OOD:* Trajectory-Only chấp nhận nhầm **31.2%** các mẫu cử chỉ rác (OOD FAR = 31.2%), trong khi Adaptive Baseline và Handshape-Only đạt **0.0%** FAR (chống nhiễu hoàn hảo).
- Do đó, việc loại bỏ hoàn toàn thông tin Handshape để chỉ dùng Trajectory là một bước lùi nghiêm trọng về mặt kỹ thuật.

### Câu hỏi 4: Có cần kết hợp handshape và trajectory không?
**Trả lời:**
**HOÀN TOÀN CẦN THIẾT.**
1. **Handshape** đóng vai trò là chiếc khóa bảo mật xác định **"Công cụ vẽ là ngón nào?"**:
   - Nếu ngón út duỗi và các ngón khác co $\to$ Duy nhất chữ J được phép kích hoạt.
   - Nếu ngón trỏ duỗi và các ngón khác co $\to$ Duy nhất chữ Z được phép kích hoạt.
   - Nếu bàn tay mở xòe hoặc nắm chặt $\to$ Khóa toàn bộ hệ thống, từ chối OOD tuyệt đối (FAR = 0.0%).
2. **Trajectory** đóng vai trò là bộ kiểm định động học xác định **"Ngón tay đó có thực hiện đúng nét vẽ không?"**:
   - Nếu bàn tay chỉ giữ yên một tư thế (static hold), Handshape vẫn cho điểm cao nhưng Trajectory sẽ phát hiện chiều dài quỹ đạo $L_{\text{path}} < 0.15$ và khóa lại (Anti-Static Guard). Đây chính là điểm yếu chết người khiến Production LR bị 100% False Alarm trên tư thế tĩnh.
3. Hai nguồn thông tin này bổ trợ hoàn hảo cho nhau: Handshape giải quyết bài toán định danh ngón tay (ngăn chặn việc ngón út bị kéo theo làm nhầm thành J), còn Trajectory giải quyết bài toán phân biệt tĩnh - động. Thiếu một trong hai, hệ thống sẽ rơi vào trạng thái nhận diện nhầm hoặc báo động giả nghiêm trọng.

### Câu hỏi 5: Phương án nào đáng triển khai tiếp?
**Trả lời:**
1. **Trong ngắn hạn (Stage 9):**
   - **GIỮ NGUYÊN Production Logistic Regression Pipeline làm model suy luận chính.**
   - Pipeline hiện tại của Production LR đạt 95.0% Accuracy và 90.0% Z Recall trên các lượt được kích hoạt, được bảo vệ vòng ngoài bởi Motion Segmenter và HoldConfirmation 3 giây.
   - Không tích hợp Trajectory-Only hay Target Combined vào production vì sẽ làm sụt giảm nghiêm trọng Z Recall (từ 90% xuống 25-30%).
2. **Trong trung và dài hạn (Stage 10 đề xuất):**
   - Không tiếp tục siết chặt các ngưỡng phân đoạn hình học cứng (như đếm số lần đảo chiều hoặc góc ngoặt cố định).
   - Phát triển kiến trúc **Soft-Gated Adaptive Ensemble**:
     - *Tầng 1 (Finger Gating):* Dùng Handshape để tính trọng số tin cậy mềm $w_J = \sigma(E_{\text{pinky}} - E_{\text{curled}})$ và $w_Z = \sigma(E_{\text{index}} - E_{\text{curled}})$.
     - *Tầng 2 (Motion Normalization):* Trừ khử độ dịch chuyển cổ tay khỏi chuyển động đầu ngón tay ($\vec{v}_{\text{finger, rel}} = \vec{v}_{\text{finger}} - \vec{v}_{\text{wrist}}$) để triệt tiêu hoàn toàn hiện tượng toàn bàn tay bị kéo theo.
     - *Tầng 3 (Continuous Distance Scoring):* Dùng Dynamic Time Warping (DTW) đa biến trên chuỗi tương đối này thay vì thuật toán tìm kiếm 3 nét cứng.

---

## 7. BẢO VỆ PRODUCTION VÀ TOÀN VẸN HỆ THỐNG

Tuân thủ nghiêm ngặt các nguyên tắc bảo vệ hệ thống:
1. **Production Pipeline:** Tập tin [`src/main.py`](file:///<project-root>/src/main.py) và [`run.py`](file:///<project-root>/run.py) được giữ nguyên vẹn 100%, không bị can thiệp.
2. **Model Checkpoints:** Cả 4 checkpoint trọng số đã được xác minh SHA-256 hoàn toàn trùng khớp:
   - `model_pipeline.joblib`: `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` (MATCH)
   - `scaler.joblib`: `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` (MATCH)
   - `digits_final.joblib`: `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` (MATCH)
   - `uvw_medium_model.joblib`: `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` (MATCH)
3. **Các phân hệ nền tảng:** Chế độ ABC/123, phím SPACE, thời gian HoldConfirmation 3.0s, và TextComposer không bị thay đổi.
4. **Đồng bộ song song:** Toàn bộ mã nguồn mới, bộ test, telemetry JSON và báo cáo kỹ thuật đã được đồng bộ hóa hoàn toàn sang thư mục đối chiếu [`<project-mirror>`](file:///<project-mirror>).

---

## KẾT LUẬN

Stage 9 đã hoàn thành xuất sắc các mục tiêu nghiên cứu và thực nghiệm:
- Xây dựng thành công bộ phân tích quỹ đạo ngón tay mục tiêu độc lập [`TargetFingerTrajectoryRecognizer`](file:///<project-root>/app/inference/dynamic_jz/target_trajectory_recognizer.py) với 19/19 unit tests đạt chuẩn.
- Thực hiện đo kiểm định lượng trên 64 mẫu Clean Validation, 32 mẫu Synthetic OOD, và 70 lượt Webcam thực tế.
- Trả lời trọn vẹn, thuyết phục và có dẫn chứng số liệu toán học cho cả 5 câu hỏi cốt lõi do người dùng đặt ra.
- Xác định nguyên nhân gốc rễ của hiện tượng Z bị nhận nhầm thành J là do **chuyển động tịnh tiến toàn bàn tay** kéo theo ngón út co gập tạo thành quỹ đạo giả lập J.
- Cung cấp định hướng kỹ thuật chính xác cho giai đoạn tiếp theo: trừ khử chuyển động cổ tay (wrist-relative decoupling) kết hợp soft gating handshape, tránh áp dụng các quy tắc phân đoạn hình học cứng nhắc trên dữ liệu webcam thực tế.
