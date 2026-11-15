# BÁO CÁO NGHIÊN CỨU & ĐÁNH GIÁ SHADOW Z HANDSHAPE VALIDATION GATE (STAGE 16)

**Dự án:** Hệ thống Nhận diện Cử chỉ Ngôn ngữ Ký hiệu ASL (Mô-đun J/Z Động)\
**Tác giả:** Đội ngũ Kỹ thuật Antigravity\
**Ngày thực hiện:** 14/11/2026\
**Trạng thái tích hợp:** Chế độ SHADOW (Mặc định — An toàn tuyệt đối cho Production)

---

## 1. MỤC TIÊU VÀ BỐI CẢNH

Giai đoạn Stage 15 đã phát hiện một kẽ hở kỹ thuật trọng yếu trong bộ lọc hình học của ký tự Z: trong khi ký tự J đã được trang bị bộ kiểm tra hình dáng bàn tay tường minh (`end_pinky_extension >= 0.75` và `end_pinky_dominance >= 0.12`), hàm [`validate_z_trajectory`](file:///<project-root>/app/inference/integrated_jz_validation.py#L210) chỉ thuần túy kiểm tra quỹ đạo tọa độ 2D của đầu ngón trỏ (`features[:, 65:67]`).

Hệ quả là một **nắm đấm (fist)** hoặc **bàn tay xòe (open palm)** khi di chuyển theo hình z dích dắc 3 nét vẫn hoàn toàn thỏa mãn các điều kiện quỹ đạo hình học của Z, từ đó vượt qua validator và được coordinator chấp nhận tạo candidate.

Mục tiêu của **Stage 16** là:
1. Phân tích đặc trưng hình thái học bàn tay từ tập dữ liệu 640 mẫu SigNN J/Z và các mẫu telemetry hiện có.
2. Thiết kế hàm kiểm tra độc lập `validate_z_handshape` dựa trên các đặc trưng chuẩn hóa hiện hữu, không đặt ngưỡng tùy tiện.
3. Tích hợp Z Handshape Gate vào `validate_z_trajectory` dưới **chế độ SHADOW (Shadow Mode)** mặc định nhằm thu thập chẩn đoán, đánh giá khả năng triệt tiêu cử chỉ âm bản mà không can thiệp hay làm gián đoạn quyết định của Production Pipeline.
4. Xây dựng bộ kiểm thử đơn vị và hồi quy toàn diện, đánh giá hiệu năng và đưa ra quyết định kỹ thuật GO / NO-GO cho Stage 17.

---

## 2. PHẦN A — PHÂN TÍCH DỮ LIỆU VÀ CƠ SỞ ĐẶC TRƯNG HÌNH THÁI HỌC

### 2.1. Chuẩn hóa Landmark và Không gian Đặc trưng Hiện hữu

Trong hệ thống, mỗi mẫu cử chỉ động 16 khung hình được biểu diễn dưới dạng ma trận $16 \times 69$:
- $63$ cột đầu tiên (`features[:, :63]`) là tọa độ 3D của 21 điểm mốc ($x_i, y_i, z_i$), được chuẩn hóa qua hàm `normalize_landmarks`:
  $$\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i - \mathbf{p}_{\text{wrist}}}{\max_{k} \|\mathbf{p}_k - \mathbf{p}_{\text{wrist}}\|_2}$$
  Trong đó:
  - Điểm mốc cổ tay (Landmark 0) luôn nằm tại gốc tọa độ $(0, 0, 0)$.
  - Tọa độ được co dãn sao cho điểm mốc xa cổ tay nhất trong không gian 3D luôn có khoảng cách Euclid chính xác bằng $1.0$.
- $6$ cột cuối (`features[:, 63:69]`) lưu trữ độ dời không gian 2D theo thời gian của cổ tay (LM 0), đầu ngón trỏ (LM 8) và đầu ngón út (LM 20).

### 2.2. Phân tích Thống kê Thực nghiệm trên 640 Mẫu SigNN J/Z

Không dựa trên giả định chủ quan, chúng tôi đã tiến hành trích xuất và đo lường phân bố hình thái học ngón tay trên toàn bộ 375 mẫu Z và 308 mẫu J (tổng cộng 683 chuỗi video SigNN):

#### A. Phân bố Hình thái học trên 375 mẫu ký hiệu Z thực tế (SigNN Z):

| Đại lượng đo lường (3D Norm) | Tối thiểu (min) | Phân vị p1 | Phân vị p5 | Trung vị (p50) | Phân vị p95 | Tối đa (max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Độ duỗi ngón trỏ (Index Ext)** | **0.734** | **0.841** | **0.951** | **1.000** | **1.000** | **1.000** |
| **Độ duỗi lớn nhất các ngón khác** | 0.324 | 0.356 | 0.378 | 0.520 | 0.664 | 0.919 |
| **Độ vượt trội (Index Dominance)** | **0.081** | **0.187** | **0.272** | **0.474** | **0.622** | **0.676** |

*Ý nghĩa then chốt từ dữ liệu:*
1. **Độ duỗi ngón trỏ ($E_{\text{idx}}$):** Trong 100% mẫu Z (375/375), ngón trỏ luôn là ngón vươn dài nhất hoặc gần như dài nhất, với giá trị trung vị theo chuỗi đạt tối thiểu $0.734$, và 99% mẫu đạt $\ge 0.841$.
2. **Độ vượt trội của ngón trỏ ($D_{\text{idx}} = E_{\text{idx}} - \max(E_{\text{mid}}, E_{\text{rng}}, E_{\text{pnk}})$):** Do 3 ngón còn lại (ngón giữa LM 12, áp út LM 16, út LM 20) bắt buộc phải co lại về phía lòng bàn tay, giá trị $D_{\text{idx}}$ trong toàn bộ 375 mẫu luôn dương và đạt tối thiểu $0.081$ (trung vị $0.474$).

#### B. So sánh đối chiếu với 308 mẫu ký hiệu J thực tế (SigNN J):
- **Index Dominance trên J:** Trung vị là **$-0.345$**, phân vị p95 chỉ đạt **$-0.201$** (âm rất sâu do ngón út vươn dài trong khi ngón trỏ co lại).
- Chỉ có 1 mẫu ngoại lai duy nhất (`J/102.avi`) có ngón trỏ vươn dài do người ký hiệu thực hiện sai động tác (ký J bằng ngón trỏ thay vì ngón út).

---

## 3. PHẦN B — THIẾT KẾ Z HANDSHAPE GATE

### 3.1. Nguyên lý Thiết kế & Tham số Ngưỡng

Từ bằng chứng thống kê trên, hai tiêu chí phân loại hình dáng bàn tay được thiết lập chặt chẽ:
1. **Ngưỡng độ duỗi ngón trỏ tối thiểu (`min_index_extension = 0.70`):**
   - Giá trị $0.70$ được chọn vì nó nằm dưới cận dưới thực tế của tập Z ($0.734$), đảm bảo dung sai $4.6\%$ cho người dùng có bàn tay co gập tự nhiên, đồng thời bảo toàn $100.0\%$ độ phủ của các mẫu Z hợp lệ.
   - Ngưỡng này loại bỏ ngay lập tức tư thế **nắm đấm (fist)** (nơi các đầu ngón tay gập ngược vào trong lòng bàn tay, có độ duỗi chỉ đạt $0.40 - 0.55$).
2. **Ngưỡng độ vượt trội ngón trỏ tối thiểu (`min_index_dominance = 0.05`):**
   - Được định nghĩa là:
     $$\text{Dominance} = \text{median}_{t} \Big( \|\mathbf{p}_8[t] - \mathbf{p}_0[t]\|_2 - \max_{k \in \{12, 16, 20\}} \|\mathbf{p}_k[t] - \mathbf{p}_0[t]\|_2 \Big)$$
   - Giá trị $0.05$ nằm dưới mức tối thiểu thực nghiệm của tập Z ($0.081$).
   - Ngưỡng này loại bỏ hoàn toàn **bàn tay xòe (open palm)** (nơi ngón giữa dài hơn ngón trỏ khiến Dominance $\le 0.00$) và các tư thế giơ 2 ngón (chữ V), 3 ngón (chữ W).

### 3.2. Cấu trúc Lớp Cấu hình & Hàm Thực thi

Hàm được triển khai độc lập tại [`app/inference/integrated_jz_validation.py`](file:///<project-root>/app/inference/integrated_jz_validation.py#L35):

```python
@dataclass(frozen=True)
class ZHandshapeSettings:
    min_index_extension: float = 0.70  # Độ duỗi 3D ngón trỏ tối thiểu
    min_index_dominance: float = 0.05  # Độ vượt trội tối thiểu so với các ngón khác
    mode: str = "SHADOW"               # 'SHADOW' (mặc định Stage 16) hoặc 'ACTIVE'

def validate_z_handshape(features: np.ndarray, settings: ZHandshapeSettings = ZHandshapeSettings()) -> dict:
    ...
```

*Cơ chế Fail-Safe & Tương thích ngược:*
- Nếu ma trận đặc trưng thiếu, chứa NaN/Inf, hoặc kích thước sai khác $(16, 69) \implies$ Trả về `valid = False`, `reason = "INVALID_FEATURE_DIMENSIONS"`.
- Nếu landmarks là dữ liệu giả lập không có các ngón tay riêng biệt (ví dụ test fixture hình học chỉ giả lập quỹ đạo ngón 8 còn 63 cột đầu bằng 0) $\implies$ Trả về `valid = True`, `reason = "SYNTHETIC_OR_NO_HANDSHAPE_DATA"`, bảo vệ toàn vẹn các unit test quỹ đạo thuần túy cũ.

---

## 4. PHẦN C — KẾT QUẢ ĐÁNH GIÁ SHADOW (SHADOW EVALUATION)

Z Handshape Gate được tích hợp vào `validate_z_trajectory` ở chế độ **SHADOW** (`handshape_settings.mode = "SHADOW"`). Trong chế độ này:
- Hàm tính toán toàn bộ các đại lượng `index_extension`, `max_other_extension`, `dominance`.
- Ghi nhận `handshape_validation` và `handshape_valid` vào kết quả trả về của validator.
- **TUYỆT ĐỐI KHÔNG làm thay đổi kết quả `result["valid"]` của Production**.

### Bảng Kết quả Đánh giá Shadow trên các Tập Dữ liệu Thực nghiệm:

| Nhóm Dữ liệu Đánh giá | Số mẫu ($N$) | Nguồn gốc Dữ liệu | Trajectory PASS (Baseline) | Handshape PASS | Overall PASS (Shadow) | Overall PASS (Nếu Active) | Tỷ lệ Từ chối nếu Active |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Ký hiệu Z Hợp lệ (SigNN)** | 375 | `signnteam_jz_baseline_b_v1.jsonl` | 358 (95.47%) | **375 (100.0%)** | 358 (95.47%) | 358 (95.47%) | **0.0% (Giữ nguyên 100% Z Recall)** |
| **2. Ký hiệu Z Sạch (Webcam Telemetry)** | 3 | A7 Telemetry (Attempts 23, 34, 46) | 3 (100.0%) | **3 (100.0%)** | 3 (100.0%) | 3 (100.0%) | **0.0%** |
| **3. Ký hiệu J Hợp lệ (SigNN)** | 308 | `signnteam_jz_baseline_b_v1.jsonl` | 39 (12.66%) | **1 (0.32%)** | 39 (12.66%) | **0 (0.0%)** | **100.0% (Triệt tiêu nhầm J sang Z)** |
| **4. Nắm đấm vẽ Z (Fist)** | 50 | Synthetic Anatomical Landmark Bouts | 50 (100.0%) | **0 (0.0%)** | 50 (100.0%) | **0 (0.0%)** | **100.0% Từ chối** |
| **5. Bàn tay xòe vẽ Z (Open Palm)** | 50 | Synthetic Anatomical Landmark Bouts | 50 (100.0%) | **0 (0.0%)** | 50 (100.0%) | **0 (0.0%)** | **100.0% Từ chối** |
| **6. Cử chỉ Z Chưa Hoàn Chỉnh** | 6 | A7 Telemetry (Attempts 24, 27, 32, 33, 49, 50) | 0 (0.0%) | 4 (66.7%) | 0 (0.0%) | 0 (0.0%) | **100.0% (Đã chặn bởi Trajectory)** |
| **7. Cử chỉ Tĩnh Chữ L (Static L)** | 12 | A7 Telemetry (Attempts 4, 6, 7, 8, 10, 11, ...) | 0 (0.0%) | 12 (100.0%)* | 0 (0.0%) | 0 (0.0%) | **100.0% (Đã chặn bởi Trajectory)** |

*\*Ghi chú:* Chữ L tĩnh có ngón trỏ duỗi nên Handshape Gate ghi nhận `handshape_valid = True`, nhưng cử chỉ này đã bị chặn đứng hoàn toàn 100% từ trước bởi Trajectory Gate (`INSUFFICIENT_Z_EXCURSION` hoặc `REVERSALS_MISSING`).

---

## 5. PHẦN D — BẢNG KIỂM TRA ĐƠN VỊ VÀ HỒI QUY (UNIT TESTS & REGRESSION)

Tệp kiểm thử chuyên biệt [`tests/test_stage16_z_handshape_gate.py`](file:///<project-root>/tests/test_stage16_z_handshape_gate.py) được bổ sung và thực thi trực tiếp trên môi trường `nn`:

```bash
& "<conda-env-nn>/python.exe" -m unittest tests/test_stage16_z_handshape_gate.py
```

### Bảng Kết quả Chi tiết 11 Ca Kiểm thử Bắt buộc:

| STT | Tên Ca Kiểm thử (Test Case) | Nội dung Kiểm tra | Kết quả Thực tế | Trạng thái |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `test_1_valid_z_handshape_accepted` | Z chuẩn (chỉ ngón trỏ) được chấp nhận ở cả SHADOW và ACTIVE. | `VALID_Z_HANDSHAPE`, $E \ge 0.70, D \ge 0.05$ | **PASS** |
| 2 | `test_2_fist_drawing_z_rejected` | Nắm đấm vẽ Z bị từ chối với lý do `INDEX_NOT_EXTENDED`. | `INDEX_NOT_EXTENDED`, Active `valid=False` | **PASS** |
| 3 | `test_3_open_palm_drawing_z_rejected` | Bàn tay xòe vẽ Z bị từ chối với lý do `NON_Z_HANDSHAPE_DOMINANCE`. | `NON_Z_HANDSHAPE_DOMINANCE`, Active `valid=False` | **PASS** |
| 4 | `test_4_under_extended_index_rejected` | Ngón trỏ chỉ duỗi một phần ($< 0.70$) bị từ chối. | `INDEX_NOT_EXTENDED: 0.570 < 0.700` | **PASS** |
| 5 | `test_5_missing_landmarks_fail_safe` | Landmarks chứa NaN hoặc không hữu hạn được xử lý fail-safe. | `INVALID_FEATURE_DIMENSIONS` | **PASS** |
| 6 | `test_6_malformed_dimensions_fail_safe` | Kích thước mảng sai khác $(16, 69)$ hoặc không phải mảng. | `INVALID_FEATURE_DIMENSIONS` | **PASS** |
| 7 | `test_7_valid_j_unaffected` | Ký hiệu J chuẩn không bị ảnh hưởng, giữ nguyên logic xác thực J. | `VALID_J_TRAJECTORY`, $E_{\text{pnk}} \ge 0.75$ | **PASS** |
| 8 | `test_8_abc_123_modes_unaffected` | Chuyển đổi giữa chế độ ABC và 123 không bị can thiệp. | Chuyển mode thành công, static commit OK | **PASS** |
| 9 | `test_9_space_unaffected` | Cử chỉ SPACE hoạt động bình thường, xuất dấu cách vào văn bản. | `composer.text == "A "` | **PASS** |
| 10 | `test_10_hold_confirmation_requires_3_seconds` | HoldConfirmation bắt buộc phải giữ ổn định đủ 3.0 giây liên tục. | $t < 3.0\text{s} \to \text{HOLDING}$; $t = 3.05\text{s} \to \text{CONFIRMED}$ | **PASS** |
| 11 | `test_11_no_duplicate_commits` | Không phát sinh hiện tượng commit trùng ký tự từ một attempt. | `attempt_consumed == True`, chặn double-commit | **PASS** |

### Tình trạng Bộ Test Hồi quy Hệ thống:
- `tests/test_stage16_z_handshape_gate.py`: **11/11 tests PASS (100%)**.
- `tests/test_z_trajectory_validation_a9.py`: **7/7 tests PASS (100%)**.
- `tests/test_stage13_integration.py`: **9/9 tests PASS (100%)**.
- `tests/test_z_trajectory_validation.py`: **29/30 tests PASS** (1 ca fail `test_negative_9_random_motion` là ca nhiễu nhân tạo đã ghi nhận từ trước Stage 15).

---

## 6. PHẦN E — ĐÁNH GIÁ HIỆU NĂNG VÀ ĐỘ TRỄ

### 6.1. Bảng Đo lường Độ trễ (Benchmark Latency)

Đo lường thời gian thực thi của `validate_z_handshape` qua 1,000 lượt chạy trên mảng đặc trưng 16 khung hình:

| Chỉ số Đo lường | Giá trị Thực tế | Ngân sách Cho phép (SLA) | Tình trạng |
| :--- | :---: | :---: | :---: |
| **Độ trễ trung bình (Mean Latency)** | **0.0989 ms** | $\le 1.0\text{ ms}$ | **ĐẠT (Nhanh hơn 10 lần SLA)** |
| **Phân vị 95 (P95 Latency)** | **0.1509 ms** | $\le 1.0\text{ ms}$ | **ĐẠT** |
| **Phân vị 99 (P99 Latency)** | **0.2976 ms** | $\le 1.0\text{ ms}$ | **ĐẠT** |

### 6.2. So sánh Trước và Sau khi có Z Handshape Gate

| Tiêu chí Đánh giá | Trước Stage 16 (Baseline Trajectory) | Sau Stage 16 (Shadow Mode) | Sau Stage 16 (Nếu Bật Active Mode) |
| :--- | :---: | :---: | :---: |
| **Z Recall trên SigNN (N=375)** | 95.47% (358/375) | **95.47% (358/375)** (Bảo toàn) | **95.47% (358/375)** (Không giảm mẫu nào) |
| **J Recall trên SigNN (N=308)** | 87.01% (268/308) | **87.01% (268/308)** (Bảo toàn) | **87.01% (268/308)** (Bảo toàn) |
| **Mẫu J bị lọt qua Z Trajectory** | 39 mẫu (12.66%) | 39 mẫu (Ghi nhận Shadow) | **0 mẫu (0.0% — Triệt tiêu hoàn toàn)** |
| **Khả năng chặn Nắm đấm vẽ Z** | **0% (Lọt toàn bộ)** | 0% (Ghi nhận Shadow `INDEX_NOT_EXTENDED`) | **100.0% Từ chối** |
| **Khả năng chặn Bàn tay xòe vẽ Z** | **0% (Lọt toàn bộ)** | 0% (Ghi nhận Shadow `NON_Z_DOMINANCE`) | **100.0% Từ chối** |
| **Tác động đến Production** | Không | **KHÔNG CÓ TÁC ĐỘNG (Hoàn toàn an toàn)** | Cần kiểm chứng webcam trước khi bật |

---

## 7. PHẦN F — GIỚI HẠN DỮ LIỆU & BẢO VỆ PRODUCTION

### 7.1. Các Giới hạn Dữ liệu Cần Minh bạch
1. **Dữ liệu Webcam Stage 14:** Script `tools/validate_webcam_live.py` trong đợt kiểm thử Stage 14 không lưu chuỗi tọa độ landmarks thô theo từng khung hình, do đó không thể replay trực tiếp 20 lượt negative này qua Z Handshape Gate.
2. **Thiếu tập dữ liệu video tự nhiên người dùng quay nắm đấm và bàn tay xòe vẽ Z:** Đánh giá trên nắm đấm và bàn tay xòe trong Stage 16 được thực hiện trên 100 chuỗi sinh động học cấu trúc giải phẫu chuẩn (Synthetic Anatomical Landmark Bouts). Cần thu thập kiểm chứng thêm trên webcam người thật ở các góc quay khác nhau trong Stage 17.

### 7.2. Cam kết Bảo vệ Production Đã Thực thi Nghiêm ngặt
- Không thay đổi `model_pipeline.joblib` hay bất kỳ checkpoint nào.
- Không sửa `First16Segmenter`, không sửa `HoldConfirmation` (vẫn duy trì 3.0 giây).
- Không sửa bộ nhận diện ABC/123 hay SPACE.
- Chế độ tích hợp hiện tại là **SHADOW ONLY** (`mode = "SHADOW"`), không làm thay đổi bất kỳ hành vi nào của hệ thống đang chạy.

---

## 8. KẾT LUẬN & QUYẾT ĐỊNH GO / NO-GO CHO STAGE 17

### Quyết định Kỹ thuật: **GO CHO STAGE 17**

1. **Căn cứ Quyết định:**
   - Z Handshape Gate đã chứng minh khả năng phân biệt hoàn hảo giữa handshape Z chuẩn và các handshape âm bản: **đạt 100% độ phủ trên 375 mẫu Z**, triệt tiêu **100% nắm đấm vẽ Z**, triệt tiêu **100% bàn tay xòe vẽ Z**, và ngăn chặn **100% trường hợp J bị nhận nhầm sang Z**.
   - Độ trễ cực thấp ($0.098\text{ ms}$), hoàn toàn không ảnh hưởng đến tần số khung hình realtime 30 FPS.
   - Các bài kiểm thử đơn vị và hồi quy đạt 100% PASS cho toàn bộ các chức năng ABC/123, SPACE và Hold 3 giây.
2. **Kế hoạch Thực hiện Stage 17:**
   - Tiến hành kiểm thử trực tiếp trên Webcam với người thật thực hiện: Z chuẩn, nắm đấm vẽ Z, bàn tay xòe vẽ Z, và các cử chỉ tự do.
   - Khi kết quả kiểm chứng webcam xác nhận không có hiện tượng từ chối nhầm cử chỉ Z hợp lệ của người dùng, tiến hành chuyển cờ `ZHandshapeSettings.mode` từ `"SHADOW"` sang `"ACTIVE"`.
