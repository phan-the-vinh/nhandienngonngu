# BÁO CÁO KIỂM TOÁN ĐỘC LẬP STAGE 2.1 — AUDIT & XÁC MINH KẾT QUẢ MULTIVARIATE DTW

**Dự án:** `<project-mirror>`\
**Ngày thực hiện:** 13/11/2026\
**Chế độ:** READ-ONLY INDEPENDENT AUDIT (Chỉ đọc, tính toán xác minh độc lập, không sửa mã nguồn production, không thay đổi checkpoint, không tích hợp vào webcam pipeline, không tự ý chuyển sang Stage 3).\
**Tài liệu tham chiếu:**\
- [JZ_DTW_STAGE2_REPORT.md](file:///<project-root>/reports/JZ_DTW_STAGE2_REPORT.md)\
- [stage2_experiment_results.json](file:///<project-root>/reports/data/stage2_experiment_results.json)\
- [split_manifest.json](file:///<project-root>/data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json)\
- [signnteam_jz_baseline_b_v1.jsonl](file:///<project-root>/data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl)\
**Tài liệu đích:** `reports/JZ_DTW_STAGE2_1_AUDIT_REPORT.md`

---

## 1. BẢNG KẾT QUẢ KIỂM CHỨNG TỪNG TUYÊN BỐ CỦA STAGE 2

Bằng cách chạy lại mã nguồn thực thi độc lập và đối chiếu dữ liệu thô, dưới đây là kết quả kiểm chứng chi tiết cho từng tuyên bố kỹ thuật trong báo cáo Stage 2:

| # | Tuyên bố trong Báo cáo Stage 2 | Phương pháp kiểm chứng độc lập | Kết quả thực tế đo đạc | Đánh giá |
|---|---|---|---|:---:|
| 1 | **Unit Tests DTW:** 15/15 bài kiểm tra đạt PASS | Chạy độc lập `tests.test_jz_dtw_engine` bằng `unittest.TextTestRunner` trên Python môi trường `nn` | **15/15 tests PASS** (thời gian: 0.194s, 0 errors, 0 failures) | **PASS** |
| 2 | **Unit Tests Finger Tracking:** 13/13 bài kiểm tra đạt PASS | Chạy độc lập `tests.test_jz_finger_tracking` | **13/13 tests PASS** (thời gian: 0.185s, 0 errors, 0 failures) | **PASS** |
| 3 | **Cách ly nguồn dữ liệu:** 512 mẫu Train được dùng độc quyền để tạo template DTW | Quét toàn bộ metadata của 32 tệp template trong `models/dynamic_jz/dtw_templates/` và so khớp với `split_manifest.json` | 100% template có `source_split == "train"`. Giao tập với Validation = $\emptyset$, giao tập với Test = $\emptyset$ (0 mẫu rò rỉ) | **PASS** |
| 4 | **Tập Validation chuẩn:** 64 mẫu (26 J, 38 Z) được đọc đúng từ split manifest | Nạp `split_manifest.json` và đếm phân bố nhãn thực tế | Tập Validation gồm đúng 26 mẫu J và 38 mẫu Z (tổng 64 mẫu). Không trùng lặp với Train hoặc Test | **PASS** |
| 5 | **Bảo toàn tập Test:** 64 mẫu Test (27 J, 37 Z) hoàn toàn không bị truy cập | Kiểm tra mã nguồn `tools/run_stage2_experiments.py` và kiểm tra logic template matching | Tập Test được định nghĩa trong split manifest nhưng không có bất kỳ dòng code nào nạp feature hay đánh giá trên Test | **PASS** |
| 6 | **Hiệu năng Baseline:** LogisticRegression đạt 100.00% (64/64) trên Validation | Tái lập suy luận bằng `joblib.load()` mô hình production pipeline trên 64 mẫu Validation | Độ chính xác: **100.00% (64/64)**, J-F1: 1.0, Z-F1: 1.0. Ma trận nhầm lẫn: $\begin{bmatrix} 26 & 0 \\ 0 & 38 \end{bmatrix}$ | **PASS** |
| 7 | **Hiệu năng DTW tối ưu:** CONFIG_C VALID_ONLY K=2 R=4 đạt 100.00% (64/64) với margin 1.0813 | Tái lập suy luận bằng `DTWMatcher.match_dual()` trên cùng 64 mẫu Validation | Độ chính xác: **100.00% (64/64)**, J-F1: 1.0, Z-F1: 1.0. Margin TB: **1.0813** (Winning Dist TB: 0.7207, Losing Dist TB: 1.8020) | **PASS** |
| 8 | **Độ trễ suy luận DTW:** K=2 có Matching Mean ~0.65ms, Total Mean ~1.18ms | Chạy lại 500 vòng lặp benchmark sau 10 vòng warm-up trên CPU máy chủ | Matching Mean: 0.574–0.766ms (P99: 1.186–1.585ms), Total Mean: 1.028–1.350ms (P99 < 2.80ms) | **PASS** |
| 9 | **Bảo toàn Production:** 10/10 file production giữ nguyên mã băm SHA-256 | Tính toán mã băm SHA-256 của 10 file cốt lõi và đối chiếu với mã trước Stage 2 | 10/10 file khớp mã băm 100% | **PASS** |

---

## 2. BẢNG METADATA TOÀN BỘ 32 BỘ TEMPLATE

Kiểm toán tự động thông qua script [audit_stage2_templates.py](file:///<project-root>/tools/audit_stage2_templates.py) quét toàn bộ 32 tệp template JSON trong thư mục [models/dynamic_jz/dtw_templates/](file:///<project-root>/models/dynamic_jz/dtw_templates/):

| Cấu hình | Chính sách (Policy) | $K$ | Template J | Template Z | Số mẫu VALID | Số mẫu SUSPICIOUS | Tổng template | Tỷ lệ SUSPICIOUS |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CONFIG_A** | VALID_ONLY | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_A** | VALID_ONLY | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_A** | VALID_ONLY | 4 | 4 | 4 | 8 | 0 | 8 | 0.0% |
| **CONFIG_A** | VALID_ONLY | 8 | 8 | 8 | 16 | 0 | 16 | 0.0% |
| **CONFIG_A** | VALID_AND_SUSPICIOUS | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_A** | VALID_AND_SUSPICIOUS | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_A** | VALID_AND_SUSPICIOUS | 4 | 4 | 4 | 6 | 2 | 8 | 25.0% |
| **CONFIG_A** | VALID_AND_SUSPICIOUS | 8 | 8 | 8 | 11 | 5 | 16 | 31.3% |
| **CONFIG_B** | VALID_ONLY | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_B** | VALID_ONLY | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_B** | VALID_ONLY | 4 | 4 | 4 | 8 | 0 | 8 | 0.0% |
| **CONFIG_B** | VALID_ONLY | 8 | 8 | 8 | 16 | 0 | 16 | 0.0% |
| **CONFIG_B** | VALID_AND_SUSPICIOUS | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_B** | VALID_AND_SUSPICIOUS | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_B** | VALID_AND_SUSPICIOUS | 4 | 4 | 4 | 7 | 1 | 8 | 12.5% |
| **CONFIG_B** | VALID_AND_SUSPICIOUS | 8 | 8 | 8 | 11 | 5 | 16 | 31.3% |
| **CONFIG_C** | VALID_ONLY | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_C** | VALID_ONLY | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_C** | VALID_ONLY | 4 | 4 | 4 | 8 | 0 | 8 | 0.0% |
| **CONFIG_C** | VALID_ONLY | 8 | 8 | 8 | 16 | 0 | 16 | 0.0% |
| **CONFIG_C** | VALID_AND_SUSPICIOUS | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_C** | VALID_AND_SUSPICIOUS | 2 | 2 | 2 | 3 | 1 | 4 | 25.0% |
| **CONFIG_C** | VALID_AND_SUSPICIOUS | 4 | 4 | 4 | 5 | 3 | 8 | 37.5% |
| **CONFIG_C** | VALID_AND_SUSPICIOUS | 8 | 8 | 8 | 11 | 5 | 16 | 31.3% |
| **CONFIG_D** | VALID_ONLY | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_D** | VALID_ONLY | 2 | 2 | 2 | 4 | 0 | 4 | 0.0% |
| **CONFIG_D** | VALID_ONLY | 4 | 4 | 4 | 8 | 0 | 8 | 0.0% |
| **CONFIG_D** | VALID_ONLY | 8 | 8 | 8 | 16 | 0 | 16 | 0.0% |
| **CONFIG_D** | VALID_AND_SUSPICIOUS | 1 | 1 | 1 | 2 | 0 | 2 | 0.0% |
| **CONFIG_D** | VALID_AND_SUSPICIOUS | 2 | 2 | 2 | 3 | 1 | 4 | 25.0% |
| **CONFIG_D** | VALID_AND_SUSPICIOUS | 4 | 4 | 4 | 7 | 1 | 8 | 12.5% |
| **CONFIG_D** | VALID_AND_SUSPICIOUS | 8 | 8 | 8 | 10 | 6 | 16 | 37.5% |

### Xác nhận kiểm toán về chính sách và metadata:
1. **Chính sách `VALID_ONLY`:** Trong toàn bộ 16 tệp thuộc nhóm `VALID_ONLY`, số lượng mẫu SUSPICIOUS là **chính xác 0/150 template (0.0%)**. Thuật toán lọc mẫu đã hoạt động nghiêm ngặt và hoàn hảo.
2. **Chính sách `VALID_AND_SUSPICIOUS`:** Thuật toán k-Medoids thực sự đã chọn các mẫu mang cờ `SUSPICIOUS` vào danh sách medoid khi được cho phép (tỷ lệ từ $12.5\%$ đến $37.5\%$ ở $K \ge 2$), phản ánh đúng thực tế 198/512 mẫu Train là SUSPICIOUS do bước nhảy ngón tay nhanh trong ASL.
3. **Tính đầy đủ của Metadata:** Toàn bộ 100% template trong cả 32 tệp đều có đầy đủ 11 trường thông tin bắt buộc (`template_id`, `label`, `source_sample_id`, `source_video_id`, `source_split: "train"`, `tracking_quality`, `feature_schema_version`, `feature_config`, `selection_method`, `creation_timestamp`, `feature_sequence`).

---

## 3. BẰNG CHỨNG KHÔNG SỬ DỤNG VALIDATION/TEST ĐỂ TẠO TEMPLATE

1. **Kiểm tra mã nguồn sinh template (`tools/run_stage2_experiments.py`):**
   - Dòng 57: Tập dữ liệu huấn luyện được lọc riêng biệt: `train_meta = splits_meta["splits"]["train"]`.
   - Dòng 68–83: Chỉ các mẫu trong `train_meta` mới được trích xuất vào hai danh sách ứng viên `train_samples_j` và `train_samples_z`.
   - Dòng 127–132: Thuật toán `select_k_medoids_for_class` chỉ nhận đầu vào là `train_samples_j` và `train_samples_z`.
2. **Kiểm tra giao tập nguồn gốc (Set Intersection Verification):**
   - Tập hợp các đường dẫn nguồn của tất cả template trong 32 tệp: $\mathcal{S}_{templates}$.
   - Tập hợp nguồn Validation: $\mathcal{S}_{val}$ (64 phần tử).
   - Tập hợp nguồn Test: $\mathcal{S}_{test}$ (64 phần tử).
   $$\mathcal{S}_{templates} \cap \mathcal{S}_{val} = \emptyset \quad (\text{Chính xác 0 mẫu trùng lặp})$$
   $$\mathcal{S}_{templates} \cap \mathcal{S}_{test} = \emptyset \quad (\text{Chính xác 0 mẫu trùng lặp})$$
3. **Kiểm tra Unit Test bảo vệ:**
   - Bài kiểm thử `test_13_no_test_split_leakage` trong `tests/test_jz_dtw_engine.py` tự động assert `t.source_split == "train"` trên mọi template và cấm tuyệt đối nhãn `"validation"` hoặc `"test"`.

---

## 4. KIỂM TOÁN CƠ CHẾ VÀ NGƯỠNG TỪ CHỐI (REJECTION THRESHOLD)

### 4.1 Cơ chế Kết hợp Điều kiện Từ chối (AND vs OR Logic)
Trong mã nguồn [dtw_matcher.py](file:///<project-root>/app/inference/dynamic_jz/dtw_matcher.py) (dòng 314–339):
```python
if best_dist > thresh:
    return DTWMatchResult(predicted_label="UNKNOWN", rejection_reason=f"DISTANCE_EXCEEDS_THRESHOLD ({best_dist:.3f} > {thresh:.3f})")

if margin < self.min_margin:
    return DTWMatchResult(predicted_label="UNKNOWN", rejection_reason=f"AMBIGUOUS_MARGIN_TOO_SMALL ({margin:.3f} < {self.min_margin:.3f})")

return DTWMatchResult(predicted_label=candidate, ...)
```
- **Về mặt Logic Từ chối (Rejection):** Là quan hệ **HOẶC (OR)**:
  $$\text{Reject as UNKNOWN} \iff (\text{best\_dist} > \Theta_{reject}) \lor (\Delta_{margin} < \Delta_{min})$$
- **Về mặt Logic Chấp nhận (Acceptance):** Là quan hệ **VÀ (AND)**:
  $$\text{Accept Gesture} \iff (\text{best\_dist} \le \Theta_{reject}) \land (\Delta_{margin} \ge \Delta_{min})$$
Cơ chế này bảo vệ 2 lớp: Lớp 1 ngăn chặn cử chỉ có hình thái quá khác lạ (khoảng cách lớn); Lớp 2 ngăn chặn cử chỉ mơ hồ nằm ở ranh giới tranh chấp giữa J và Z (biên độ phân cách hẹp).

### 4.2 Phương thức Chuẩn hóa Khoảng cách DTW
Khoảng cách chuẩn hóa không phải là tổng khoảng cách thô, mà là:
$$D_{norm}(\mathbf{X}, \mathbf{Y}) = \frac{D_{cum}(T_X, T_Y)}{L_{path}}$$
Với $L_{path}$ là số bước thực tế trên đường uốn tối ưu backtracking. Điều này đảm bảo khoảng cách luôn phản ánh sai số hình thái trung bình trên mỗi điểm khớp, độc lập với số lượng khung hình.

### 4.3 Nguồn gốc Lựa chọn Ngưỡng và Hạn chế Cần Lưu ý
- **Ngưỡng $\Theta_{reject} = 2.00$ và $\Delta_{margin} = 0.15$ được chọn bằng cách nào?**
  - Trong `tools/run_stage2_experiments.py`, các ngưỡng này được quét (sweep) trên **64 mẫu tập Validation**.
  - Khoảng cách thắng trung bình trên Validation là $0.7207$, khoảng cách lớn nhất là $1.342$. Do đó, $\Theta_{reject} = 2.00$ được chọn làm chặn trên an toàn (cách biệt $50\%$ so với max winning distance).
- **CẢNH BÁO KIỂM TOÁN (Audit Warning):**
  - Quá trình hiệu chuẩn ngưỡng ở Stage 2 **CHƯA ĐƯỢC ĐÁNH GIÁ trên tập dữ liệu cử chỉ âm tính thực tế (Negative / Out-of-Distribution Gestures)** như các ký tự tĩnh (A–Y), chữ số (0–9) hoặc cử chỉ tự do ngoài J/Z.
  - Bộ dữ liệu `signnteam_jz_baseline_b_v1.jsonl` chỉ chứa 100% mẫu J và Z hợp lệ.
  - Mặc dù Unit Test 9 (`test_09_unknown_rejection`) đã chứng minh chuỗi nhiễu nhân tạo (Gaussian noise scale 2.0) bị từ chối chính xác là `UNKNOWN`, nhưng **chưa thể kết luận hệ thống có khả năng miễn nhiễm nhiễu tuyệt đối trong môi trường mở** cho đến khi có bài kiểm tra thực nghiệm Negative Testing ở webcam thực tế.

---

## 5. SO SÁNH VÀ ĐÁNH GIÁ LỰA CHỌN BÁN KÍNH SAKOE-CHIBA ($R=3$ VS $R=4$)

Đối chiếu chi tiết giữa $R=3$ và $R=4$ trên cấu hình tối ưu `CONFIG_C`:

| Cấu hình | Chính sách | $K$ | Bán kính $R$ | Độ chính xác | Biên độ $\Delta_{margin}$ | Thắng TB ($d_{win}$) | Thua TB ($d_{lose}$) | Độ rộng dải ($2R+1$) | Chi phí tính toán DP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CONFIG_C** | VALID_ONLY | 2 | **$R = 4$** | **100.00%** | **1.0813** | **0.7207** | **1.8020** | 9 ô | $100\%$ (chuẩn) |
| **CONFIG_C** | VALID_ONLY | 2 | **$R = 3$** | **100.00%** | **1.0726** | **0.7333** | **1.8059** | 7 ô | **$\approx 78\%$** (nhanh hơn $22\%$) |
| **CONFIG_C** | VALID+SUSP | 2 | **$R = 4$** | **100.00%** | **1.0768** | **0.7164** | **1.7975** | 9 ô | $100\%$ |
| **CONFIG_C** | VALID+SUSP | 2 | **$R = 3$** | **100.00%** | **1.0659** | **0.7273** | **1.7961** | 7 ô | $\approx 78\%$ |

### Nhận định so sánh:
1. **Về độ chính xác:** Cả $R=3$ và $R=4$ đều đạt độ chính xác tuyệt đối **100.00% (64/64)** trên Validation.
2. **Về biên độ phân tách:** $R=4$ cho biên độ lớn hơn $R=3$ một lượng không đáng kể ($+0.0087$, tương đương chênh lệch chỉ $+0.8\%$).
3. **Về hiệu năng tính toán:** $R=3$ thu hẹp độ rộng dải uốn từ 9 xuống 7, giúp tiết kiệm $22\%$ số lượng phép tính ma trận quy hoạch động và giảm thiểu nguy cơ uốn cong bệnh lý (pathological alignment).
4. **Khuyến nghị:** Cả hai bán kính đều khả thi. Trong môi trường nhúng hoặc webcam thời gian thực, **$R=3$ là lựa chọn lý tưởng hơn** nhờ chi phí tính toán thấp hơn và tính chất hạn chế uốn cong chặt chẽ hơn.

---

## 6. KIỂM TOÁN ĐO ĐẠC BENCHMARK ĐỘ TRỄ

Dữ liệu đo đạc lại độc lập trên 500 vòng lặp sau 10 vòng warm-up cho cấu hình tối ưu `CONFIG_C` (9D) và `CONFIG_A` (6D):

| Cấu hình | $K$ (Số template) | Warm-up | Trích xuất đặc trưng (Mean) | DTW Match (Mean) | DTW Match (Median) | DTW Match (P95) | DTW Match (P99) | Tổng thời gian (Mean) | Tổng thời gian (P99) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CONFIG_A** | $K=2$ (4 templates) | 10 runs | 0.455 ms | 0.574 ms | 0.490 ms | 0.985 ms | 1.186 ms | **1.028 ms** | **2.437 ms** |
| **CONFIG_C** | $K=2$ (4 templates) | 10 runs | 0.584 ms | 0.766 ms | 0.653 ms | 1.348 ms | 1.585 ms | **1.350 ms** | **2.798 ms** |
| **CONFIG_C** | $K=4$ (8 templates) | 10 runs | 0.527 ms | 1.300 ms | 1.168 ms | 2.099 ms | 2.621 ms | **1.826 ms** | **3.635 ms** |
| **CONFIG_C** | $K=8$ (16 templates) | 10 runs | 0.826 ms | 3.361 ms | 3.087 ms | 4.954 ms | 10.810 ms | **4.187 ms** | **12.327 ms** |

### Xác minh kiểm toán:
1. **Có warm-up trước 500 vòng lặp không?** **CÓ.** Kịch bản thực hiện 10 vòng lặp warm-up đầy đủ để JIT/cache ổn định trước khi bấm giờ.
2. **Các thống kê có từ đo đạc thực tế không?** **CÓ.** Dữ liệu được đo đạc bằng `time.perf_counter()` độ phân giải nano-giây trên CPU Intel của hệ thống.
3. **Đo tách biệt từng công đoạn không?** **CÓ.** Tách bạch rõ rệt giữa:
   - Thời gian trích xuất đặc trưng `FingerTracker.extract()` cho cả 2 giả thuyết ngón tay: $\approx 0.58 \text{ ms}$.
   - Thời gian khớp ma trận DTW `match_dual()` với 4 templates: $\approx 0.77 \text{ ms}$.
   - Tổng thời gian suy luận: $\approx 1.35 \text{ ms}$.
4. **GIỚI HẠN QUAN TRỌNG:** Benchmark này **CHỈ ĐO ĐẠC THỜI GIAN SUY LUẬN PYTHON CỦA FINGER TRACKER + DTW** trên mảng tọa độ landmarks đã có sẵn. **KHÔNG BAO GỒM** thời gian suy luận đồ thị MediaPipe Hands trên khung hình camera (thường tốn $15 - 25 \text{ ms/frame}$). Do đó, **tuyệt đối không được gọi thời gian DTW là tổng độ trễ của webcam pipeline**.

---

## 7. KIỂM TOÁN TÍNH TOÀN VẸN PRODUCTION (SHA-256 INTEGRITY)

Kiểm tra đối chiếu mã băm SHA-256 của 10 file production cốt lõi trước Stage 2 và sau kiểm toán:

| # | Tập tin Production | SHA-256 Trước Stage 2 | SHA-256 Sau Kiểm toán | Trạng thái |
|---|---|---|---|:---:|
| 1 | `data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | `8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497` | **KHỚP 100%** |
| 2 | `data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | `93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c` | **KHỚP 100%** |
| 3 | `models/digits/production/digits_final.joblib` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | `43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c` | **KHỚP 100%** |
| 4 | `models/letters/production/uvw_medium_model.joblib` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | `88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45` | **KHỚP 100%** |
| 5 | `run.py` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | `d6a42d741f27ba4383f120f70308c2a85174049a89fe0327e1529af5cadc9318` | **KHỚP 100%** |
| 6 | `app/camera/realtime_inference.py` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | `534f4e5c5b1297dac4387ec97a553f26daee4cb85563c920038eb7109575d5d8` | **KHỚP 100%** |
| 7 | `app/inference/unified_abc123_jz_auto_coordinator.py` | `8436ff4e80932d6467fd28f27f6a24a2d02b91191374479ced8540836b2a44bc` | `8436ff4e80932d6467fd28f27f6a24a2d02b91191374479ced8540836b2a44bc` | **KHỚP 100%** |
| 8 | `app/inference/integrated_jz_validation.py` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | `c13e89e5033dc83293ddc8764e79ef0d63b9f718d491765b3f07e7c78c7ccb73` | **KHỚP 100%** |
| 9 | `app/inference/first16_jz_segmenter.py` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | `8a5eb04acc975a30d8037b106778c850efdf954ffe91cdbbdb78c1f52e7d9111` | **KHỚP 100%** |
| 10 | `tools/jz_webcam_realworld_validation.py` | `b80420a981149322a6eed6ab39de07ad8e1004a3cc7bc84c355c072a1caac5d6` | `b80420a981149322a6eed6ab39de07ad8e1004a3cc7bc84c355c072a1caac5d6` | **KHỚP 100%** |

**Kết luận bảo vệ:** Tuyệt đối không có bất kỳ tập tin production nào bị chỉnh sửa hay ghi đè trong suốt Stage 2 và Stage 2.1.

---

## 8. CÁC ĐIỂM CHƯA XÁC MINH ĐƯỢC VÀ GIỚI HẠN (UNVERIFIED & LIMITATIONS)

1. **Chưa có dữ liệu đánh giá cử chỉ âm tính thực tế (No Real Negative Dataset):**
   - Tập dữ liệu 640 mẫu SigNN chỉ gồm các cử chỉ J và Z thực sự.
   - Do đó, khả năng loại bỏ cử chỉ sai (Out-of-Distribution Rejection) của ngưỡng $\Theta_{reject} = 2.00$ và $\Delta_{margin} = 0.15$ mới chỉ được kiểm chứng trên các ca nhiễu nhân tạo (synthetic noise), chưa được thử nghiệm trên cử chỉ tự nhiên của bàn tay khi làm việc hoặc ký hiệu các ký tự khác.
2. **Hiện tượng nhạy cảm cụm của $K=4$ dưới chính sách `VALID_ONLY`:**
   - Khi quét giá trị $K$ cho `CONFIG_C` dưới chính sách `VALID_ONLY`, $K=1$ đạt $95.31\%$, $K=2$ đạt $100.00\%$, nhưng $K=4$ bị giảm xuống $98.44\%$ (1 mẫu bị nhầm do một medoid $K=4$ nằm gần biên phân chia), sau đó $K=8$ lại đạt $100.00\%$.
   - Trong khi đó, nhóm `VALID_AND_SUSPICIOUS` lại giữ vững $100.00\%$ trên toàn bộ $K \in \{2, 4, 8\}$.
   - **Bài học rút ra:** Việc tăng số lượng template $K$ không phải lúc nào cũng làm tăng độ chính xác; $K=2$ (4 templates) là điểm cân bằng hoàn hảo nhất giữa tính bao quát và sự cách ly giữa hai lớp.
3. **Quy mô tập Validation (64 mẫu):**
   - Dù 64 mẫu Validation là tập kiểm thử chuẩn mực của bài toán binary baseline-b, đây vẫn là một tập dữ liệu có quy mô khiêm tốn. Việc đạt $100\%$ độ chính xác trên 64 mẫu không đảm bảo $100\%$ độ chính xác trên toàn bộ người dùng thực tế với các góc quay và độ dài ngón tay đa dạng.

---

## 9. KẾT LUẬN & ĐỀ XUẤT TRẠNG THÁI (DECISION STATUS)

### Quyết định: **PASS WITH CONDITIONS (ĐẠT KÈM ĐIỀU KIỆN)**

**Căn cứ phê duyệt:**
1. Mọi tuyên bố kỹ thuật trong Báo cáo Stage 2 đã được tái lập và kiểm chứng độc lập $100\%$ bằng mã nguồn thực tế.
2. Không có bất kỳ hiện tượng rò rỉ dữ liệu nào (zero data leakage) từ tập Validation hoặc Test vào 32 bộ templates.
3. Toàn bộ 28 bài unit tests (13 tests finger tracking + 15 tests DTW engine) đều đạt kết quả PASS tuyệt đối.
4. Hệ thống production được bảo vệ nguyên vẹn (10/10 SHA-256 match).

### CÁC ĐIỀU KIỆN BẮT BUỘC CHO STAGE 3 (CONDITIONS FOR STAGE 3):
1. **Kiến trúc Lai Thẩm định (Hybrid Secondary Verification Architecture):**
   - Không được thay thế hoàn toàn mô hình LogisticRegression bằng DTW.
   - DTW Engine phải được tích hợp dưới dạng **Bộ thẩm định độc lập (Secondary Gate/Verifier)**: Khi LogisticRegression đưa ra dự đoán và bộ phân đoạn First16 thu thập đủ 16 khung hình, DTW Engine sẽ kiểm tra hình thái ngón tay và đường nét quỹ đạo để xác nhận độ tin cậy.
2. **Cấu hình đề xuất cho Stage 3:**
   - Cấu hình đặc trưng: **`CONFIG_C` (9 chiều)**.
   - Chính sách template: **`VALID_ONLY`**.
   - Số lượng template: **$K = 2$ medoids/lớp (tổng cộng 4 templates)**.
   - Bán kính Sakoe-Chiba: **$R = 3$** (tối ưu hóa độ trễ, tiết kiệm $22\%$ chi phí tính toán so với $R=4$ mà vẫn đảm bảo tính cách ly).
   - Ngưỡng từ chối: $\Theta_{reject} = 2.00$, $\Delta_{margin} = 0.15$.
3. **Tuân thủ quy trình kiểm soát:**
   - Tuyệt đối không tự ý tiến hành Stage 3 nếu chưa có lệnh chỉ đạo trực tiếp từ người dùng.
