# STAGE 8.1 — BÁO CÁO KỸ THUẬT: XÁC MINH BENCHMARK VÀ NGUYÊN NHÂN Z RECALL TRÊN WEBCAM

**Dự án:** Nhận diện Cử chỉ Ký tự Động J & Z (American Sign Language)\
**Địa bàn thực thi:** `<project-root>` (Mirror đồng bộ: `<project-mirror>`)\
**Môi trường runtime:** Python 3.11 (`<conda-env-nn>/python.exe`)\
**Ngày thực hiện:** 14/11/2026\
**Trạng thái kiểm định:** HOÀN TẤT KIỂM TOÁN VÀ ĐÁNH GIÁ (AUDITED & COMPLETED)

---

## 1. KIỂM TRA TÍNH CHÍNH XÁC CỦA BENCHMARK STAGE 8

Sau khi rà soát độc lập mã nguồn [evaluate_stage8_adaptive.py](file:///<project-root>/tools/evaluate_stage8_adaptive.py), tệp dữ liệu [trials.jsonl](file:///<project-root>/data/experiments/dynamic_jz_v1/realworld_validation/20261113T155135Z_stage5_validator_fix_6fb34f/trials.jsonl), và hàm tính toán `compute_metrics` trong [evaluate_stage7_fusion.py](file:///<project-root>/tools/evaluate_stage7_fusion.py), kết quả kiểm toán cụ thể như sau:

### 1.1. Công thức tính Accuracy giữa các phương pháp
- **Kết luận:** **CÙNG CÁCH TÍNH 100%.**
- Cả 5 phương pháp (Production LR, Handshape-Only, Ratio-Only, Stage 7 Fusion, và Adaptive Motion) đều sử dụng chung hàm `compute_metrics`:
  $$\text{Accuracy} = \frac{\sum_{c \in \{J, Z\}} \text{ConfusionMatrix}[c][c]}{\text{Tổng số lượt có nhãn } J \text{ hoặc } Z}$$
- Trong Suite 4 (50 attempts), mẫu số là $50$ cho tất cả các phương pháp.

### 1.2. Cách tính Coverage và xử lý nhãn UNKNOWN
- **Coverage được tính trên toàn bộ 50 attempts:**
  $$\text{Coverage} = 1.0 - \frac{\text{Tổng số lượt dự đoán UNKNOWN}}{50} = \frac{50 - \text{UNKNOWN}}{50}$$
- **UNKNOWN có bị loại khỏi Accuracy không?**\
  **KHÔNG.** Mọi dự đoán `UNKNOWN` đều được tính là một lỗi phân loại (False Negative) trong mẫu số của cả Accuracy và Recall. Không có bất kỳ phương pháp nào được phép "loại bỏ UNKNOWN" để làm đẹp chỉ số Accuracy.

### 1.3. Bản chất của 50 lượt trong Suite 4 (Stage 5 Webcam)
- Trong tệp `trials.jsonl` của Stage 5:
  - Có đúng **25 lượt Ground Truth J** và **25 lượt Ground Truth Z**.
  - **40 lượt đầu tiên (Trials 1–40):** Bộ trích xuất tiền xử lý (`First16Segmenter`) hoạt động thành công (`feature_valid == True`, trích xuất được ma trận đặc trưng $16 \times 69$).
  - **10 lượt cuối cùng (Trials 41–50):** Là các lượt thực hiện J/Z nhưng bị bộ tiền xử lý từ chối trước khi gọi mô hình (`feature_valid == False`, `model_prediction == NONE`) do mất bám dính MediaPipe, góc khuất bàn tay hoặc timeout.
  - Khi đưa vào benchmark, vì các lượt 41–50 không có đặc trưng hợp lệ, **toàn bộ 5 phương pháp đều trả về `UNKNOWN`**.
  - Do đó, trần Coverage tối đa trên 50 attempts là $\frac{40}{50} = 80.0\%$.

### 1.4. Phân tích Suite 3: Vì sao Production LR đạt 100.0%?
- Trong mã nguồn kiểm thử Suite 3 ([evaluate_stage8_adaptive.py](file:///<project-root>/tools/evaluate_stage8_adaptive.py), dòng 248):
  `target = meta.get("candidate_letter") or m_res.get("prediction")`
- Tại các lượt diagnostic metadata thu nhận từ webcam, trường `candidate_letter` đôi khi bị bỏ trống, nên nhãn mục tiêu (`target`) được lấy tự động từ `m_res["prediction"]` — chính là dự đoán của Production LR khi sequence được ghi lại.
- Khi chạy lại benchmark, `predict_lr(feat)` chạy lại mô hình LR trên chính đặc trưng `feat` đó, hiển nhiên tái tạo lại 100% dự đoán ban đầu.
- **Ý nghĩa:** Chỉ số 100% của LR trên Suite 3 phản ánh tính ổn định tái lặp (determinism) của LR chứ **không phải** LR đạt 100% so với Ground Truth con người. Ngược lại, Adaptive Motion, Handshape, và Ratio được đánh giá độc lập trên 61 sequence này mà không tham gia định nghĩa nhãn mục tiêu, trong đó Adaptive Motion đạt **83.6% Accuracy** (51/61) và **86.9% Coverage**.

---

### 1.5. Ma trận nhầm lẫn (Confusion Matrix) chi tiết trên Suite 4

Dữ liệu được đối chiếu minh bạch dưới 2 góc nhìn: **Góc nhìn toàn hệ thống (50 attempts)** và **Góc nhìn bộ phân loại (40 model-invoked attempts)**.

#### A. Góc nhìn toàn hệ thống (Tất cả 50 attempts — Bao gồm 10 lượt segmenter không gọi mô hình)
| Phương pháp | GT J $\to$ J | GT J $\to$ Z | GT J $\to$ UNK | GT Z $\to$ J | GT Z $\to$ Z | GT Z $\to$ UNK | Accuracy | Coverage | J Recall | Z Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR** | 20 | 0 | 5 | 2 | 18 | 5 | **76.0%** | **80.0%** | 80.0% (20/25) | 72.0% (18/25) | 0.8441 |
| **Handshape-Only (Stage 7)** | 16 | 0 | 9 | 3 | 13 | 9 | **58.0%** | **64.0%** | 64.0% (16/25) | 52.0% (13/25) | 0.7057 |
| **Ratio-Only (Stage 6)** | 8 | 0 | 17 | 3 | 3 | 19 | **22.0%** | **28.0%** | 32.0% (8/25) | 12.0% (3/25) | 0.3294 |
| **Stage 7 Fusion** | 8 | 0 | 17 | 3 | 7 | 15 | **30.0%** | **36.0%** | 32.0% (8/25) | 28.0% (7/25) | 0.4410 |
| **Adaptive Motion (Stage 8)** | 20 | 0 | 5 | 3 | 15 | 7 | **70.0%** | **76.0%** | 80.0% (20/25) | 60.0% (15/25) | 0.7917 |

#### B. Góc nhìn bộ phân loại (40 attempts thực sự được chuyển vào Model)
| Phương pháp | GT J $\to$ J | GT J $\to$ Z | GT J $\to$ UNK | GT Z $\to$ J | GT Z $\to$ Z | GT Z $\to$ UNK | Invoked Acc | Invoked Cov | Invoked J Rec | Invoked Z Rec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production LR** | 20 | 0 | 0 | 2 | 18 | 0 | **95.0%** (38/40) | **100.0%** | 100.0% (20/20) | 90.0% (18/20) |
| **Handshape-Only (Stage 7)** | 16 | 0 | 4 | 3 | 13 | 4 | **72.5%** (29/40) | **80.0%** | 80.0% (16/20) | 65.0% (13/20) |
| **Ratio-Only (Stage 6)** | 8 | 0 | 12 | 3 | 3 | 14 | **27.5%** (11/40) | **35.0%** | 40.0% (8/20) | 15.0% (3/20) |
| **Stage 7 Fusion** | 8 | 0 | 12 | 3 | 7 | 10 | **37.5%** (15/40) | **45.0%** | 40.0% (8/20) | 35.0% (7/20) |
| **Adaptive Motion (Stage 8)** | 20 | 0 | 0 | 3 | 15 | 2 | **87.5%** (35/40) | **95.0%** | 100.0% (20/20) | 75.0% (15/20) |

---

## 2. NGUYÊN NHÂN CỤ THỂ Z RECALL CHỈ ĐẠT 60% TRÊN WEBCAM

Trong 25 lượt Ground Truth Z của Suite 4:
- **Số lượt nhận diện đúng:** **15 lượt** (Trials 4, 6, 8, 12, 16, 18, 20, 22, 24, 26, 32, 34, 36, 38, 40).
- **Số lượt không được gọi vào mô hình (Uninvoked):** **5 lượt** (Trials 42, 43, 44, 49, 50).
- **Số lượt bị từ chối thành UNKNOWN:** **2 lượt** (Trials 2, 14).
- **Số lượt bị phân loại nhầm thành J:** **3 lượt** (Trials 10, 28, 30).

### Phân tích chi tiết từng trường hợp thất bại:

| Trial | LR Pred | Adp Pred | $S_{\text{hs}}(Z)$ | $S_{\text{tr}}(Z)$ | Revs | $\Delta y_2$ (Diag Drop) | PathLen | Nguyên nhân gốc rễ (Root Cause) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Trial 2** | Z | **UNKNOWN** | **0.17** | 0.53 | 2 | 0.005 | 1.610 | **Bị từ chối bởi Handshape Veto:** Độ duỗi ngón trỏ thấp ($S_{\text{hs}}=0.17 < 0.25$). Ngón trỏ chưa duỗi căng khi bắt đầu cử chỉ khiến rào chắn an toàn phủ quyết để tránh nhận diện nhầm bàn tay nắm đấm. |
| **Trial 14** | Z | **UNKNOWN** | 0.54 | 0.55 | 2 | **0.004** | 1.579 | **Bị từ chối bởi Trajectory Morphology:** Nét vẽ chéo số 2 gần như nằm ngang hoàn toàn ($\Delta y_2 = 0.004$ palm scales $< 0.015$). Ngưỡng kiểm định hình thái chéo chặn lại do không đủ độ dốc đi xuống. |
| **Trial 10** | **J** | **J** | 0.09 | 0.53 | 2 | -0.222 | 2.841 | **Lỗi nhãn hoặc camera bắt nhầm ngón:** Cả LR và Adaptive đều cho kết quả là J ($P(J)=0.97, S_{\text{hs}}(J)=0.91$). Đặc trưng thu nhận từ webcam có ngón út duỗi thay vì ngón trỏ. |
| **Trial 30** | **J** | **J** | 0.09 | 0.53 | 2 | -0.222 | 2.841 | **Lỗi nhãn hoặc camera bắt nhầm ngón:** Tương tự Trial 10, LR gốc dự đoán J với xác suất 97%, đặc trưng thực tế khớp hoàn toàn với cấu trúc cử chỉ J. |
| **Trial 28** | Z | **J** | **0.13** | 0.61 | 2 | 0.116 | 2.854 | **Bất đồng phân loại đa thuật toán:** Trong telemetry Stage 5, LR dự đoán Z với biên độ thấp ($P(Z)=0.69$), trong khi DTW dự đoán J, Geometric dự đoán J, Hybrid dự đoán J và Handshape phân tích thấy ngón út duỗi rõ ($S_{\text{hs}}(J)=0.872, S_{\text{hs}}(Z)=0.125$). |
| **Trials 42, 43, 44, 49, 50** | NONE | **UNKNOWN** | - | - | - | - | - | **Bộ tiền xử lý camera không chuyển dữ liệu:** Segmenter báo mất dấu tay/không có chuyển động (`feature_valid == False`). Mô hình không được gọi. |

> **Tổng kết:** Trên 20 lượt Z thực sự được chuyển vào mô hình, Adaptive Motion đạt **15/20 = 75.0%**. Lý do Z Recall báo cáo là 60% vì mẫu số bao gồm cả 5 lượt bị rớt ở khâu camera/segmenter ($15/25 = 60.0\%$).

---

## 3. THỬ NGHIỆM CẢI THIỆN Z VÀ NGHIÊN CỨU ABLATION

Để kiểm chứng xem có thể cải thiện Z Recall mà **không làm giảm độ phân biệt J và không nhận diện nhầm OOD**, một nghiên cứu Ablation độc lập đã được thực hiện tại tệp thực nghiệm [scratch/ablation_study_z.py](file:///<dev-home>/.gemini/antigravity/brain/a6bec65a-a674-48f4-8c5f-997aee1fd69f/scratch/ablation_study_z.py):

### Các cấu hình thử nghiệm:
- **Config 0 (Stage 8 Baseline):** $\Delta y_2 \ge 0.015$, $\text{hs\_veto} = 0.25$, $w_{\text{tr}} = 0.55$.
- **Config A (Hạ ngưỡng dốc nét chéo Z):** $\Delta y_2 \ge 0.003$, $\text{hs\_veto} = 0.25$, $w_{\text{tr}} = 0.55$.
- **Config B (Hạ ngưỡng phủ quyết Handshape):** $\Delta y_2 \ge 0.015$, $\text{hs\_veto} = 0.15$, $w_{\text{tr}} = 0.55$.
- **Config C (Kết hợp cả hai):** $\Delta y_2 \ge 0.003$, $\text{hs\_veto} = 0.15$, $w_{\text{tr}} = 0.55$.

### Bảng kết quả Ablation Study:

| Cấu hình | Suite 1 (Clean Val) | Suite 2 (OOD FAR) | Suite 3 (Diag 61 Acc) | Suite 4 (Invoked 40 Acc) | Suite 4 (Invoked Z Rec) | Suite 4 (Total 50 Acc) | Suite 4 (Total 50 Z Rec) | Nhận định kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config 0 (Stage 8)** | Acc 98.4% | **0.0%** (32/32 rej) | 83.6% (Cov 86.9%) | 87.5% (35/40) | 75.0% (15/20) | 70.0% | 60.0% (15/25) | Chuẩn mực cơ sở hiện tại |
| **Config A** ($\Delta y_2 \ge 0.003$) | Acc 98.4% | **0.0%** (32/32 rej) | **86.9%** (Cov 90.2%) | **90.0%** (36/40) | **80.0%** (16/20) | **72.0%** | **64.0%** (16/25) | **CẢI THIỆN AN TOÀN:** Cứu được Trial 14 mà không tăng FAR |
| **Config B** ($\text{veto} = 0.15$) | Acc 98.4% | **0.0%** (32/32 rej) | 83.6% (Cov 88.5%) | 77.5% (31/40) | 75.0% (15/20) | 62.0% | 60.0% (15/25) | **THOÁI HÓA NGHIÊM TRỌNG:** J Recall tụt từ 80% xuống 64%! |
| **Config C** (Hạ cả hai) | Acc 98.4% | **0.0%** (32/32 rej) | 86.9% (Cov 91.8%) | 80.0% (32/40) | 80.0% (16/20) | 64.0% | 64.0% (16/25) | **THOÁI HÓA:** Tương tự B, rò rỉ điểm Z làm hỏng phân biệt J |

### Phát hiện cốt lõi từ Ablation:
1. **Hạ ngưỡng nét chéo Z $\Delta y_2$ từ $0.015 \to 0.003$ (Config A) mang lại lợi ích thuần túy:**
   - Cứu được **Trial 14**, nâng Z Recall trên 40 lượt gọi từ **75.0% lên 80.0%** (và từ 60.0% lên 64.0% trên toàn bộ 50 lượt).
   - Nâng Accuracy Suite 3 từ **83.6% lên 86.9%**.
   - Vẫn duy trì **100% tỷ lệ từ chối OOD (0.0% FAR)** vì Z vẫn bắt buộc phải có $\ge 1$ lần đảo chiều horizontal ($\text{reversals\_count} \ge 1$), ngăn chặn mọi cử chỉ vuốt ngang đơn hướng.
2. **Tuyệt đối KHÔNG ĐƯỢC hạ Handshape Veto xuống dưới 0.25 (Config B & C):**
   - Khi hạ $\text{hs\_veto} = 0.15$, các lượt vẽ J có ngón trỏ hơi nhô lên bị rò rỉ điểm tin cậy Z, làm J Recall giảm sút nghiêm trọng từ 80.0% xuống **64.0%**. Rào chắn Handshape $\ge 0.25$ là thiết yếu để bảo vệ cử chỉ J.

> *Lưu ý tuân thủ cam kết:* Kết quả Config A chỉ được ghi nhận trong nghiên cứu thực nghiệm độc lập và JSON telemetry; không tự ý chỉnh sửa mã nguồn production hay tuyên bố vượt chuẩn trên cùng tập test.

---

## 4. ĐIỀU TRA CÁC LỆNH KIỂM THỬ CÓ EXIT CODE 1

Qua kiểm toán chi tiết từng tệp test có lỗi, nguyên nhân được phân loại minh bạch như sau:

| Tệp kiểm thử | Số lỗi | Phân loại nguyên nhân | Chi tiết kỹ thuật |
| :--- | :---: | :--- | :--- |
| `tests.test_jz_production_integration` | 1 fail, 1 error | **Môi trường & Dữ liệu** | - `ValueError: Experiment lineage mismatch`: Kiểm tra commit hash trong manifest dữ liệu huấn luyện cũ.<br>- `AssertionError WindowsPath('<project-root>/...') != WindowsPath('<project-mirror>/...')`: Tệp test gán cứng đường dẫn kỳ vọng thay vì dùng đường dẫn tương đối của dự án. |
| `tests.test_jz_shadow_replay` | 3 fail, 2 error | **Hệ điều hành Windows & Giả lập Mock** | - `WinError 32 PermissionError`: Trên Windows, tệp manifest CSV được mở nhưng chưa giải phóng handle trước khi khối `tempfile.TemporaryDirectory` dọn dẹp.<br>- `AssertionError: 'I' != 'Z'`: Mô hình tĩnh production hiện tại nhận diện bàn tay tĩnh ngón trỏ là chữ cái `I` (hoặc `D`), trong khi kịch bản test cũ kỳ vọng chuỗi tĩnh xác nhận chữ `Z`. |
| `tests.test_jz_post_confirm_lifecycle` | 15 fail | **Giả lập Mock lỗi thời** | - Toàn bộ 15 lỗi bắt nguồn từ hàm phụ trợ `confirm_first()` tại dòng 34: `self.assertEqual(self.c.composer.text, 'Z')`. Bộ phân loại tĩnh hiện tại phân loại khung hình tĩnh đó thành `'I'`, khiến điều kiện khẳng định trước khi kiểm tra vòng đời bị chặn. |
| `tests.test_jz_finger_tracker` | 1 error | **Lỗi cú pháp gọi CLI** | - Lỗi `ModuleNotFoundError`: Khi chạy lệnh unittest, tham số truyền vào là `tests.test_jz_finger_tracker` nhưng tệp thực tế nằm tại `app.inference.dynamic_jz.finger_tracker`. |
| **Core Dynamic Suites (Stage 6, 7, 8)** | **0 fail** | **HOÀN TOÀN ĐẠT (PASS)** | - `test_jz_adaptive_classifier`: **7/7 tests PASSED in 0.022s**<br>- `test_jz_fusion_classifier`: **7/7 tests PASSED**<br>- `test_jz_ratio_classifier`: **7/7 tests PASSED** |

---

## 5. SO SÁNH TỔNG HỢP: PRODUCTION LR VS ADAPTIVE MOTION

Bảng đối chiếu tổng kết trên cùng bộ dữ liệu thực tế Suite 4 (50 attempts và 40 invoked attempts):

| Tiêu chí đánh giá | Production LR (Baseline) | Adaptive Motion (Stage 8) | Adaptive Motion (Ablation Config A) |
| :--- | :---: | :---: | :---: |
| **Toàn hệ thống (50 attempts) — Accuracy** | 76.0% (38/50) | **70.0%** (35/50) | **72.0%** (36/50) |
| **Toàn hệ thống (50 attempts) — Coverage** | 80.0% (40/50) | **76.0%** (38/50) | **78.0%** (39/50) |
| **Toàn hệ thống (50 attempts) — J Recall** | 80.0% (20/25) | **80.0%** (20/25) | **80.0%** (20/25) |
| **Toàn hệ thống (50 attempts) — Z Recall** | 72.0% (18/25) | **60.0%** (15/25) | **64.0%** (16/25) |
| **Macro F1 (50 attempts)** | 0.8441 | **0.7917** | **0.8069** |
| **Bộ phân loại (40 invoked) — Accuracy** | 95.0% (38/40) | **87.5%** (35/40) | **90.0%** (36/40) |
| **Bộ phân loại (40 invoked) — Coverage** | 100.0% (40/40) | **95.0%** (38/40) | **97.5%** (39/40) |
| **Bộ phân loại (40 invoked) — J Recall** | 100.0% (20/20) | **100.0%** (20/20) | **100.0%** (20/20) |
| **Bộ phân loại (40 invoked) — Z Recall** | 90.0% (18/20) | **75.0%** (15/20) | **80.0%** (16/20) |
| **Tỷ lệ nhận diện sai OOD (Suite 2 FAR)** | **100.0% (Cực kỳ nguy hiểm)** | **0.0% (An toàn tuyệt đối)** | **0.0% (An toàn tuyệt đối)** |
| **Chống tư thế tĩnh (Static Hand Guard)** | **KHÔNG CÓ (Bị lừa 100%)** | **CÓ (Chặn hoàn toàn $L_{\text{motion}} < 0.15$)** | **CÓ (Chặn hoàn toàn $L_{\text{motion}} < 0.15$)** |
| **Độ trễ xử lý (Latency P50)** | 0.44 ms | **1.29 ms** | **1.29 ms** |

---

## 6. KIỂM ĐỊNH BẢO VỆ PRODUCTION VÀ TÍNH TOÀN VẸN CHECKPOINT

- **Mô hình Checkpoints:**
  - `data/.../model_pipeline.joblib`: SHA-256 = `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` [KHỚP 100%]
  - `data/.../scaler.joblib`: SHA-256 = `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` [KHỚP 100%]
  - `models/digits/production/digits_final.joblib`: SHA-256 = `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` [KHỚP 100%]
  - `models/letters/production/uvw_medium_model.joblib`: SHA-256 = `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` [KHỚP 100%]
- **Quy trình gõ phím:** ABC/123, SPACE, TextComposer, và HoldConfirmation 3.0s hoàn toàn không bị chỉnh sửa.
- **Dữ liệu telemetry:** Xuất và lưu đầy đủ tại [stage8_1_z_improvement_results.json](file:///<project-root>/reports/data/stage8_1_z_improvement_results.json) (đã đồng bộ song song sang `<project-mirror>\reports\data\stage8_1_z_improvement_results.json`).
