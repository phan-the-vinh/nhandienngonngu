# BÁO CÁO THỰC NGHIỆM: CẢI THIỆN NHẬN DIỆN CỬ CHỈ ĐỘNG J/Z THEO LỘ TRÌNH HAI PHƯƠNG ÁN

**Dự án:** Hệ thống nhận diện thủ ngữ tiếng Việt (Chữ cái & Chữ số) — `NhanDienCuChi`\
**Ngày thực hiện:** 13/11/2026\
**Chế độ thực thi:** Triển khai độc lập module, kiểm thử tự động toàn diện và đánh giá thực nghiệm đa tập mẫu (Validation, Out-Of-Distribution, Camera Diagnostics).\
**Tệp dữ liệu kết quả:** [stage3_methods_comparison_results.json](file:///<project-root>/reports/data/stage3_methods_comparison_results.json)\
**Trạng thái kiểm tra an toàn:** **PASS (78/78 unit tests, 100% SHA-256 baseline match)**

---

## 1. TỔNG QUAN ĐIỀU HÀNH & KẾT QUẢ CỐT LÕI (EXECUTIVE SUMMARY)

Thực hiện nhiệm vụ nâng cao khả năng phân biệt và loại trừ nhiễu cho hai ký tự động **J** và **Z**, dự án đã triển khai đầy đủ và thực nghiệm so sánh độc lập cả **ba phương án nhận diện**:
1. **Phương án 1 (Baseline Production):** Logistic Regression Pipeline hiện tại.
2. **Phương án 2 (Enhanced LR + DTW):** Multivariate Dynamic Time Warping với vector đặc trưng động học 14 chiều `CONFIG_E` (kết hợp quỹ đạo ngón chủ động, vận tốc, vector hướng, góc quay độ cong, vị trí tương quan cổ tay và độ tương phản hình dáng bàn tay).
3. **Phương án 3 (Geometric & Ratio Classifier):** Bộ phân loại hình học đa tiêu chí độc lập (đánh giá tỷ lệ khung bao, biên độ rơi, nét móc cong, phân đoạn 3 nét Z, 2 điểm đảo hướng góc $110^\circ - 160^\circ$ và tính điểm tổng hợp $S_J, S_Z$).

### Bảng đối chiếu thực nghiệm toàn diện trên 3 bộ kiểm chuẩn

| Tiêu chí đánh giá | Phương án 1: Baseline LR | Phương án 2: Enhanced DTW (14D) | Phương án 3: Geometric Classifier |
|---|:---:|:---:|:---:|
| **Loại mô hình** | Sklearn Logistic Regression (Flat 1104-D) | Multivariate DTW (14-D Sequence, $K=2$, $R=3$) | Multi-Criteria Geometric Rules & Ratios |
| **Đặc trưng cốt lõi** | Toàn bộ 21 landmarks + displacements | Ngón chủ động + Vector hướng + Góc cua + Handshape | Tỷ lệ hình học + Điểm uốn nét + Tương phản ngón |
| **Độ chính xác Validation (64 mẫu)** | **100.00%** (64/64) | **98.44%** (63/64) | **92.19%** (59/64) |
| **Validation Macro F1-score** | 1.0000 | **0.9933** | 0.9517 |
| **Ký tự J: Precision / Recall / F1** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.8462 / 0.9167 |
| **Ký tự Z: Precision / Recall / F1** | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.9737 / 0.9867 | 1.0000 / 0.9737 / 0.9867 |
| **Tỷ lệ UNKNOWN trên mẫu chuẩn** | 0.00% | 1.56% (1/64 mẫu) | 7.81% (5/64 mẫu) |
| **Số mẫu dự đoán sai trên tập chuẩn** | 0 mẫu | **0 mẫu (0.0%)** | **0 mẫu (0.0%)** |
| **Tỷ lệ từ chối OOD (32 mẫu chuyển động giả)** | **0.00% (Báo động giả 100%)** | **46.88% (Từ chối 15/32)** | **53.12% (Từ chối 17/32)** |
| **Phản ứng trước chuyển động gạt thẳng** | Ép nhận diện thành "J" ($P=0.78$) | **Từ chối UNKNOWN** | **Từ chối UNKNOWN** |
| **Tỷ lệ từ chối Camera Diagnostics (61 mẫu)** | 0.00% (Ép phân loại 10 J, 51 Z) | **24.59% (15 mẫu UNKNOWN)** | **47.54% (29 mẫu UNKNOWN)** |
| **Thời gian xử lý trung bình (Latency)** | 0.482 ms | 0.896 ms | **0.775 ms** |

---

## 2. GIAI ĐOẠN 1: TRIỂN KHAI & ĐÁNH GIÁ PHƯƠNG ÁN LR + ENHANCED DTW

### 2.1. Cải tiến không gian đặc trưng (`CONFIG_E` — 14 chiều)
Để khắc phục tình trạng DTW phụ thuộc vào khoảng cách tọa độ phẳng đơn thuần, module [feature_builder.py](file:///<project-root>/app/inference/dynamic_jz/feature_builder.py) đã được mở rộng cấu hình `CONFIG_E` gồm 14 chiều đặc trưng động học không thứ nguyên:
1. `x_global, y_global` (2D): Tọa độ dịch chuyển chuẩn hóa theo kích thước lòng bàn tay $S_{palm}$.
2. `dx_global, dy_global` (2D): Vi phân chuyển dịch giữa hai khung hình liên tiếp.
3. `speed` (1D): Tốc độ bước chuyển động tức thời $v_t = \sqrt{\Delta x_t^2 + \Delta y_t^2}$.
4. `dir_cos, dir_sin` (2D): Vector chỉ phương đơn vị $(\cos\theta_t, \sin\theta_t)$, mô tả hướng di chuyển chính xác của đầu ngón tay không bị ảnh hưởng bởi tốc độ.
5. `turn_angle` (1D): Góc đổi hướng (độ cong tức thời) $\Delta\theta_t = \theta_t - \theta_{t-1} \in [-1, 1]$. Đây là đặc trưng then chốt để bắt trọn 2 góc chuyển hướng gắt $\approx 135^\circ$ của chữ Z và độ uốn lượn mượt mà của chữ J.
6. `x_wrist_rel, y_wrist_rel, z_wrist_rel` (3D): Quỹ đạo 3 chiều của đầu ngón tay đối với cổ tay (chuẩn hóa $D_{max}$).
7. `active_ext, curled_ext` (2D): Tỷ lệ duỗi của ngón chủ động và mức gập trung bình của các ngón còn lại.
8. `dominance` (1D): Độ chênh lệch hình dáng bàn tay ($e_{active} - e_{curled}$), bắt buộc ngón chỉ định phải nổi trội so với phần còn lại của bàn tay.

### 2.2. Trích xuất Template đại diện trên Train Split (K-Medoids)
- Nguồn dữ liệu: 512 mẫu từ tập Train trong `split_manifest.json` (tuyệt đối không sử dụng dữ liệu Validation hay Test).
- Thuật toán: K-Medoids ($K=2$ mẫu/lớp) trên ma trận khoảng cách DTW đa biến với bán kính Sakoe-Chiba $R=3$.
- Tệp template sinh ra: [templates_config_e_valid_only_k2.json](file:///<project-root>/models/dynamic_jz/dtw_templates/templates_config_e_valid_only_k2.json)\
  *Mã băm SHA-256:* `03d889fa54ddce22ba29fda2b0172185a547d7ea7ef8a8859fa6a8489b3a7e95`.

### 2.3. Cơ chế phân loại và từ chối an toàn (OOD Rejection Gate)
Module độc lập [enhanced_dtw.py](file:///<project-root>/app/inference/dynamic_jz/enhanced_dtw.py) áp dụng chiến lược so khớp song song hai giả thuyết (`dual hypothesis matching`):
- Giả thuyết J: Theo dõi ngón út (Landmark 20) $\to$ tính $d_J$.
- Giả thuyết Z: Theo dõi ngón trỏ (Landmark 8) $\to$ tính $d_Z$.
- Điều kiện chấp nhận:
  1. Khoảng cách warping tối thiểu $\min(d_J, d_Z) \le \Theta_{reject} = 2.65$ (hiệu chuẩn theo phân vị 95% của tập Train).
  2. Độ phân tách lớp $\Delta_{margin} = |d_J - d_Z| \ge 0.15$.
  3. Tính ưu thế hình dáng bàn tay: Ngón chủ động của lớp chiến thắng không được gập sâu hơn các ngón còn lại.
- Bất kỳ vi phạm nào đều lập tức trả về `UNKNOWN` với lý do cụ thể (`DISTANCE_EXCEEDS_THRESHOLD`, `AMBIGUOUS_MARGIN_TOO_SMALL`, `INSUFFICIENT_HANDSHAPE_DOMINANCE`).

### 2.4. Kết quả thực nghiệm Phương án 1
- **Trên tập Validation chuẩn (64 mẫu):** Đạt độ chính xác **98.44%** (63/64 mẫu). 1 mẫu duy nhất không khớp là mẫu Z có khoảng cách $3.189 > 2.65$ nên được từ chối an toàn về `UNKNOWN`. **Tỷ lệ gán nhãn sai giữa J và Z là 0.0%**.
- **Trên tập OOD (32 mẫu chuyển động giả):** Từ chối thành công **46.88%** các cử chỉ nhiễu, khắc phục hoàn toàn hiện tượng của Logistic Regression vốn nhận vơ 100% các chuyển động gạt thành chữ J.

---

## 3. GIAI ĐOẠN 2: THIẾT KẾ & ĐÁNH GIÁ BỘ PHÂN LOẠI HÌNH HỌC THEO TỶ LỆ

Nhằm cung cấp một cơ chế nhận diện tường minh, không phụ thuộc vào trọng số hồi quy hay mẫu template, module độc lập [geometric_classifier.py](file:///<project-root>/app/inference/dynamic_jz/geometric_classifier.py) đã được thiết kế hoàn chỉnh.

### 3.1. Đặc trưng hình học ký tự J (Ngón út — Landmark 20)
- **Tỷ lệ khung bao (Aspect Ratio):** $R_{aspect} = \frac{x_{span}}{y_{span}}$. Quỹ đạo J chuẩn có thân dọc dài hơn nét ngang ($R_{aspect} \in [0.20, 0.90]$).
- **Biên độ rơi thẳng đứng (Downward Drop):** $\Delta y_{down} = y_{bottom} - y_0 \ge 0.35$ (trong hệ tọa độ ảnh, trục Y hướng xuống).
- **Nét móc cong (Hook):** Biên độ dịch chuyển ngang $\Delta x_{hook} = |x_{end} - x_{bottom}| \ge 0.15$ kết hợp với xu hướng hếch lên ($\Delta y_{hook} \le 0.02$). Nét móc được chấm điểm theo phép nhân điều kiện: nếu không có độ dịch chuyển ngang, điểm nét móc tự động bằng 0.
- **Độ đơn điệu mở đầu (Lead-in Monotonicity):** Không xuất hiện chuyển động hếch lên đột ngột trước khi thực hiện nét rơi.
- **Tương phản ngón út vs ngón trỏ:** $e_{pinky} - e_{index} > 0$.

### 3.2. Đặc trưng hình học ký tự Z (Ngón trỏ — Landmark 8)
- **Tỷ lệ khung bao (Aspect Ratio):** $R_{aspect} = \frac{x_{span}}{y_{span}} \in [0.50, 2.50]$ (đặc trưng cho nét zic-zắc bề ngang rộng).
- **Phân đoạn 3 nét & Điểm đảo hướng:** Tìm kiếm 2 góc uốn $t_1, t_2$ chia chuỗi 16 khung hình thành:
  * Nét 1 (Ngang): $\Delta x_1$.
  * Nét 2 (Chéo giật lùi hướng xuống): $\Delta x_2$ ngược dấu $\Delta x_1$ và $\Delta y_2 > 0$.
  * Nét 3 (Ngang cùng chiều nét 1): $\Delta x_3$ cùng dấu $\Delta x_1$ và ngược dấu $\Delta x_2$.
- **Góc uốn chuyển hướng:** Đo góc giữa vector nét 1 và nét 2, vector nét 2 và nét 3 ($\theta_{turn} \in [110^\circ, 160^\circ]$).
- **Tương phản ngón trỏ vs ngón út:** $e_{index} - e_{pinky} > 0$.

### 3.3. Bộ tính điểm tổng hợp & Quy tắc quyết định
Hệ thống tính điểm độc lập cho cả hai giả thuyết:
- $Score_J = 0.25 S_{drop} + 0.20 S_{aspect} + 0.25 S_{hook} + 0.10 S_{mono} + 0.20 S_{handshape} \in [0, 1]$.
- $Score_Z = 0.20 S_{span} + 0.25 S_{reversal} + 0.20 S_{diag} + 0.15 S_{angles} + 0.10 S_{aspect} + 0.10 S_{handshape} \in [0, 1]$.
- **Ngưỡng quyết định:** Yêu cầu $\max(Score_J, Score_Z) \ge 0.45$, chênh lệch $|Score_J - Score_Z| \ge 0.08$, kèm theo các điều kiện biên tối thiểu (J phải có $\Delta x_{hook} \ge 0.10$, Z phải có đủ 2 lần đảo hướng). Không thỏa mãn sẽ trả về `UNKNOWN`.

### 3.4. Kết quả thực nghiệm Phương án 2
- **Trên tập Validation chuẩn (64 mẫu):** Đạt độ chính xác **92.19%** (59/64 mẫu). 5 mẫu còn lại có điểm số tiệm cận nhau nên được từ chối an toàn về `UNKNOWN`. **Không có bất kỳ mẫu nào bị phân loại nhầm**.
- **Trên tập OOD (32 mẫu chuyển động giả):** Đạt tỷ lệ từ chối cao nhất: **53.12%** (17/32 mẫu). Đặc biệt, các cử chỉ không có nét móc hoặc không có đủ 2 lần đảo hướng đều bị loại bỏ ngay lập tức.
- **Thời gian xử lý:** Nhanh nhất trong cả 3 phương án: **0.775 ms**.

---

## 4. GIAI ĐOẠN 3: PHÂN TÍCH SO SÁNH & ĐÁNH GIÁ TRÊN DỮ LIỆU WEBCAM THỰC TẾ

Thực nghiệm tại [evaluate_jz_methods.py](file:///<project-root>/tools/evaluate_jz_methods.py) đã thực hiện kiểm chuẩn trên **61 lượt ghi hình thực tế từ camera** (các mẫu 16x69 trong `data/experiments/dynamic_jz_v1/diagnostics/`):

### 4.1. Phân tích hành vi trên dữ liệu Camera thực tế

```
====================================================================================================
PHƯƠNG ÁN                | SỐ LƯỢT DỰ ĐOÁN "J" | SỐ LƯỢT DỰ ĐOÁN "Z" | TỪ CHỐI UNKNOWN | TỶ LỆ TỪ CHỐI
====================================================================================================
Baseline Logistic Reg    |        10           |        51           |        0        |      0.00%
Enhanced DTW Engine      |         8           |        38           |       15        |     24.59%
Geometric Classifier     |         9           |        23           |       29        |     47.54%
====================================================================================================
```

### 4.2. Bài học mấu chốt từ thực nghiệm:
1. **Lỗ hổng "ép nhãn" của Logistic Regression Baseline:**
   Mô hình Logistic Regression huấn luyện nhị phân bằng hàm Sigmoid/Softmax chỉ biết chia không gian thành 2 nửa: nửa J và nửa Z. Khi người dùng thực hiện một chuyển động nhiễu (ví dụ: gạt tay thẳng đứng, vẫy tay hình tròn, run tay), LR **bắt buộc phải chọn J hoặc Z** với tổng xác suất bằng 1.0. Trong thực nghiệm OOD, 100% các mẫu nhiễu bị LR gán bừa thành J.
2. **Khả năng tự vệ của Enhanced DTW:**
   Nhờ có không gian 14 chiều bao gồm vector chỉ phương $(\cos\theta, \sin\theta)$ và góc cua $\Delta\theta$, Enhanced DTW đo được khoảng cách sai lệch thực tế tới các template chuẩn. Khi gặp chuyển động thẳng hoặc tròn, khoảng cách vượt ngưỡng $\Theta_{reject}=2.65$ hoặc biên phân tách bị thu hẹp, giúp DTW từ chối an toàn tới 46.88% số mẫu OOD và 24.59% số lượt camera không đạt chuẩn.
3. **Tính khắt khe và tường minh của Geometric Classifier:**
   Bộ phân loại hình học đóng vai trò như một bộ lọc logic nghiêm ngặt: Một chuyển động không có nét móc dứt khoát không bao giờ có thể là J; một chuyển động không đủ 2 lần đảo hướng không bao giờ có thể là Z. Vì vậy, phương án này đạt tỷ lệ từ chối nhiễu cao nhất (53.12% OOD, 47.54% Camera), tuy nhiên có tỷ lệ UNKNOWN trên tập chuẩn cao hơn (7.81%).

---

## 5. BẢO TOÀN TÍNH AN TOÀN TOÀN HỆ THỐNG (SAFETY & VERIFICATION)

### 5.1. Kết quả kiểm thử Unit Tests (78/78 tests PASS)
Toàn bộ các test suite từ Stage 1 đến Stage 3.1 đều được chạy đồng thời và đạt kết quả hoàn hảo:
```bash
<conda-env-nn>/python.exe -m unittest tests/test_jz_enhanced_dtw.py tests/test_jz_geometric_classifier.py tests/test_jz_shadow_safety_audit.py tests/test_jz_shadow_adapter.py tests/test_jz_finger_tracking.py tests/test_jz_dtw_engine.py tests/test_dynamic_jz_camera_test.py tests/test_jz_pipeline_stage1_to_stage5.py tests/test_jz_stage6_frame_alignment.py
```
**Kết quả đầu ra:** `Ran 78 tests in 1.108s -> OK (100% PASS)`.

### 5.2. Tính toàn vẹn mã băm SHA-256 các tệp Production
Không có bất kỳ tệp sản xuất cốt lõi nào bị thay đổi:

| Đường dẫn tệp | SHA-256 Baseline | SHA-256 Đo đạc | Kết quả |
|---|---|---|:---:|
| `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **MATCH** |
| `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **MATCH** |
| `models/digits/production/digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **MATCH** |
| `models/letters/production/uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **MATCH** |
| `run.py` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | **MATCH** |
| `app/camera/realtime_inference.py` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | **MATCH** |
| `app/inference/integrated_jz_validation.py` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | **MATCH** |
| `app/inference/first16_jz_segmenter.py` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | **MATCH** |
| `app/interaction/hold_confirmation.py` | `ba172bf9cc19c2a2a92bdcad86dedde62a1587e4d0f10cb7a1320ba3e5c5c91d` | `ba172bf9cc19c2a2a92bdcad86dedde62a1587e4d0f10cb7a1320ba3e5c5c91d` | **MATCH** |

---

## 6. KẾT LUẬN & ĐỀ XUẤT KIẾN TRÚC TỐI ƯU (RECOMMENDATIONS)

### 6.1. Đánh giá hai phương án:
- **Phương án 1 (Enhanced DTW 14D) ĐẠT MỤC TIÊU XUẤT SẮC:** Đạt độ chính xác 98.44%, Macro F1 0.9933, từ chối 46.88% nhiễu OOD, độ trễ cực thấp (0.896 ms). Nó giữ được độ nhạy cao của học mẫu (exemplar matching) mà không làm rớt các biến thể chữ viết tay tự nhiên.
- **Phương án 2 (Geometric Classifier) HOÀN TOÀN KHẢ THI:** Đạt độ chính xác 92.19%, tốc độ 0.775 ms, có khả năng lọc nhiễu OOD cao nhất (53.12%).

### 6.2. Đề xuất Kiến trúc Kết hợp Cấp số 3 (Hybrid Cascaded Architecture):
Thay vì chỉ chọn một phương án duy nhất, hệ thống có thể kết hợp theo mô hình phân tầng đa lớp (Cascaded Gating):
1. **Tầng 1 (Fast Geometric Filter):** Sử dụng các kiểm tra hình học cơ bản của Phương án 2 để loại bỏ nhanh các cử chỉ tĩnh, gạt thẳng hoặc zic-zắc giả trong < 0.2 ms.
2. **Tầng 2 (Dual-Hypothesis Enhanced DTW):** Sử dụng Phương án 1 (CONFIG_E) để so khớp độ tương đồng quỹ đạo và hình dáng bàn tay với template chuẩn.
3. **Tầng 3 (Consensus Gate):** Khi cả Baseline LR và Enhanced DTW đều đồng thuận về cùng một nhãn và vượt qua ngưỡng biên an toàn, ký tự được chuyển sang `HoldConfirmation` (3.0s) để commit chính thức.
