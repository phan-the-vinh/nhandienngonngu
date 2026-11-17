# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN SỐ LIỆU HUẤN LUYỆN ML VÀ PHÂN CHIA DỮ LIỆU POWERPOINT (PHIÊN BẢN V2 ĐÃ XÁC MINH)
## DỰ ÁN: HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU ASL VÀ GỢI Ý VĂN BẢN
**Tập tin kiểm toán gốc tiếng Anh:** `SIC_Capstone_Presentation_English_Final.pptx`  
**Tập tin đối chiếu gốc tiếng Việt:** `Xây Dựng Hệ Thống Nhận Diện Ngôn Ngữ Cử Chỉ Bằng Camera.pptx`  
**Tập tin xuất bản Verified V2 tiếng Anh:** `SIC_Capstone_Presentation_English_Verified_v2.pptx`  
**Tập tin xuất bản Verified V2 tiếng Việt:** `SIC_Capstone_Presentation_Vietnamese_Verified_v2.pptx`  
**Workspace kiểm định:** `D:\NhanDienCuChi`  
**Thời điểm thực hiện:** 16/11/2026  
**Nguyên tắc thực hiện:** Trung thực khoa học tuyệt đối, minh bạch dữ liệu, trích xuất trực tiếp từ checkpoint nhị phân, training manifest, evaluation logs và mã nguồn Scikit-learn/PyTorch.

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM ĐỊNH (EXECUTIVE SUMMARY)

Đợt kiểm định chuyên sâu đã rà soát và xác minh độc lập toàn bộ các số liệu học máy, cấu hình huấn luyện và cấu trúc phân chia dữ liệu được trình bày trên cả hai bộ slide PowerPoint (tiếng Anh và tiếng Việt):

1. **Nhánh ABC (Chữ cái tĩnh – 24 lớp A–Y trừ J/Z):**
   - **Xác minh: CHÍNH XÁC 100%.**
   - Tổng số epoch chạy thực tế: **101 epochs**.
   - Best Epoch: **Epoch 81**, được xác định dựa trên tiêu chí giám sát chính là `VALIDATION v1 macro-F1` (đạt mức cực đại $0.9895$).
   - Stopping Epoch: **Epoch 101**. Thuật toán huấn luyện với tham số kiên nhẫn `external_validation_patience = 20`, sau khi đạt đỉnh tại Epoch 81 đã tiếp tục chạy thêm đúng 20 epoch nữa mà không cải thiện $\ge \text{min\_delta } (0.0001)$, qua đó dừng lại tại Epoch 101 và tự động khôi phục checkpoint Epoch 81.
   - Checkpoint được chọn đóng gói production: Checkpoint tại Epoch 81 (`models/letters/production/uvw_medium_model.joblib`).
   - Số tham số huấn luyện: **6,968 tham số** (khớp chính xác kiến trúc $63 \times 64 + 64 + 64 \times 32 + 32 + 32 \times 24 + 24$).
   - Phân biệt rõ ràng giữa **Best Epoch (81)** và **Stopping Epoch (101)**.

2. **Nhánh 123 (Chữ số tĩnh – 10 lớp 0–9):**
   - **Xác minh: CÓ CƠ CHẾ EARLY STOPPING THỰC TẾ, CẦN TÁCH BẠCH RÕ RÀNG GIỮA BEST EPOCH VÀ STOPPING EPOCH.**
   - Cơ chế Early Stopping: **CÓ THỰC TẾ** (`early_stopping = True` trong Scikit-learn `MLPClassifier`, `validation_fraction = 0.15`, `n_iter_no_change = 40`, `tol = 0.0001`).
   - Tổng số epoch huấn luyện thực tế (Stopping Epoch): **52 epochs**.
   - **Best Epoch thực tế:** Điểm số validation nội bộ cao nhất ($1.0000$) đạt được lần đầu tiên tại **Epoch 11** (và duy trì trần đến Epoch 23).
   - Cơ chế ngắt tại Epoch 52: Do điểm số validation đã đạt trần tuyệt đối $1.0000$ từ Epoch 11, điều kiện để Scikit-learn ghi nhận "có cải thiện" là $\text{val\_score} > 1.0000 + 0.0001 = 1.0001$ (về mặt toán học là không thể vượt qua 100%). Bộ đếm `no_improvement_count` tăng liên tục suốt 41 lượt từ Epoch 12 đến 52 $\implies$ kích hoạt Early Stopping tại Epoch 52 sau đúng $40$ epoch kiên nhẫn.
   - **Hiệu chỉnh bắt buộc:** **TUYỆT ĐỐI KHÔNG** trình bày mốc 52 là "Best Epoch" hay "Optimal Epoch". Mốc 52 là **Stopping Epoch**; mốc tối ưu thực nghiệm là **Epoch 11**.

3. **Nhánh J/Z (Cử chỉ chuyển động – 2 lớp J và Z):**
   - **Xác minh: MÔ HÌNH HỘI TỤ THEO THUẬT TOÁN TỐI ƯU HÓA LBFGS, KHÔNG SỬ DỤNG HUẤN LUYỆN MINI-BATCH THEO EPOCH.**
   - Mô hình sử dụng `LogisticRegression(solver='lbfgs', C=1.0)` với vector đặc trưng chuỗi $16 \text{ frames} \times 69 = 1,104$ chiều.
   - Mốc **22 iterations**: Là số vòng lặp tối ưu hóa ma trận bậc hai (Quasi-Newton L-BFGS) cho đến khi chuẩn gradient rơi xuống dưới ngưỡng dung sai (`tol=1e-4`), **KHÔNG PHẢI** là epoch huấn luyện mạng neural.
   - Trên biểu đồ Slide 11 và các ghi chú kỹ thuật, cần ghi rõ "22 LBFGS iterations" để tránh gây hiểu nhầm về bản chất thuật toán.

4. **Làm rõ Vai trò của Phân chia Dữ liệu Nhánh 123 (Tổng 899 mẫu):**
   - **549 mẫu Train:** Dùng trực tiếp $100\%$ để cập nhật trọng số mạng neural (`train_samples: 549`, gồm 349 mẫu tác giả + 170 Ardamavi + 30 ASL_HG).
   - **100 mẫu Validation (`validation_reference.csv`):**
     - **Cập nhật trọng số (Training / Fitting): KHÔNG.** Không tham gia gradient descent.
     - **Đánh giá & Chọn mô hình (Model Selection / Tuning): CÓ.** Được sử dụng làm benchmark độc lập ngoài để tính toán `validation_macro_f1` và `validation_accuracy` so sánh giữa các kiến trúc ứng viên (`mlp_large_v2a` vs `v2b/c/d/e`), quyết định chọn `mlp_large_v2a` đưa vào production (`candidate_comparison_digits_final.json`).
   - **250 mẫu Regression Reference (`test_v1.csv`):**
     - **Cập nhật trọng số (Training / Fitting): KHÔNG.**
     - **Đánh giá & Chọn mô hình (Model Selection / Tuning): KHÔNG.** (`"test_set_used": false`, `"regression_reference_used_for_training_or_selection": false`).
     - **Bản chất khoa học:** Là tập kiểm thử đối chứng hồi quy đóng băng (frozen baseline reference) từ bộ dữ liệu tác giả ban đầu, dùng để phát hiện xem việc nạp thêm 200 mẫu ngoại lai (Ardamavi/ASL_HG) có gây suy giảm hiệu năng (catastrophic forgetting / regression) trên phân phối ban đầu hay không. Không gọi tập này là "independent test set" độc lập vì không phù hợp với định danh trong metadata.

---

## 2. BẢNG KIỂM TOÁN CHI TIẾT SỐ LIỆU HUẤN LUYỆN

### A. Nhánh ABC (Static Letters A–Y)

| Tiêu chí kỹ thuật | PowerPoint gốc | Giá trị kiểm chứng thực tế | Nguồn bằng chứng xác thực | Trạng thái kiểm định |
|---|:---:|:---:|---|:---:|
| **Tổng số Epoch (Total Epochs)** | 101 | **101** | `models/letters/production/evaluation_summary.json` (`epochs_run: 101`) | **VERIFIED (Chính xác 100%)** |
| **Best Epoch** | 81 | **81** | `models/letters/production/evaluation_summary.json` (`best_epoch: 81`) | **VERIFIED (Chính xác 100%)** |
| **Stopping Epoch** | Không ghi | **101** | `evaluation_summary.json`, `training_manifest.json` | **VERIFIED (Đã bổ sung)** |
| **Best Checkpoint** | Checkpoint 81 | `checkpoints/uvw_medium/state.joblib` $\to$ `uvw_medium_model.joblib` | `training_manifest.json` | **VERIFIED (Chính xác 100%)** |
| **Tiêu chí lựa chọn (Selection Metric)** | Không ghi rõ | **VALIDATION v1 macro-F1** (Tiêu chí phụ: min recall, accuracy) | `training_config.json` (`selection.primary: "VALIDATION v1 macro-F1"`) | **VERIFIED (Đã bổ sung)** |
| **Patience / Min Delta** | Không ghi rõ | `patience = 20`, `min_delta = 0.0001` | `training_config.json` (`external_validation_patience: 20`) | **VERIFIED (Đã bổ sung)** |
| **Lý do dừng (Stop Reason)** | Không ghi rõ | `EXTERNAL_VALIDATION_MACRO_F1_PATIENCE` | `evaluation_summary.json` (`stop_reason`) | **VERIFIED (Đã bổ sung)** |
| **Số tham số huấn luyện (Trainable Params)** | 6,968 | **6,968** | Kiểm tra kích thước tensor trọng số: $63\times 64 + 64 + 64\times 32 + 32 + 32\times 24 + 24$ | **VERIFIED (Chính xác 100%)** |

---

### B. Nhánh 123 (Static Digits 0–9)

| Tiêu chí kỹ thuật | PowerPoint gốc | Giá trị kiểm chứng thực tế | Nguồn bằng chứng xác thực | Trạng thái kiểm định |
|---|:---:|:---:|---|:---:|
| **Tổng số Epoch (Total Epochs)** | 52 | **52** | `models/digits/production/digits_final.joblib` (`mlp.n_iter_: 52`) | **VERIFIED (Chính xác 100%)** |
| **Best Epoch** | 52 (ghi nhầm ở Chart 11) | **11** (lần đầu đạt 1.0000) / **23** (lần cuối đạt 1.0000) | Trích xuất `mlp.validation_scores_` từ `digits_final.joblib` | <span style="color:red; font-weight:bold;">HIỆU CHỈNH: Về Epoch 11</span> |
| **Stopping Epoch** | 52 | **52** | `mlp.n_iter_: 52`, độ dài `loss_curve_`: 52 | **VERIFIED (Chính xác 100%)** |
| **Cơ chế Early Stopping** | Yes | **CÓ THỰC TẾ** | `training_manifest_digits_final.json` (`early_stopping: true`) | **VERIFIED (Chính xác 100%)** |
| **Patience (n_iter_no_change)** | Không ghi rõ | **40 epochs** | `training_manifest_digits_final.json`, đối tượng `MLPClassifier` | **VERIFIED (Đã bổ sung)** |
| **Dung sai cải thiện (Tolerance)** | Không ghi rõ | **0.0001** (`mlp.tol = 0.0001`) | Trích xuất trực tiếp từ model object `digits_final.joblib` | **VERIFIED (Đã bổ sung)** |
| **Tập phân tách Early Stopping** | Không ghi rõ | **15% nội bộ** trích từ 549 mẫu Train (~83 mẫu) | `training_manifest_digits_final.json` (`validation_fraction: 0.15`) | **VERIFIED (Đã bổ sung)** |
| **Số tham số huấn luyện (Trainable Params)** | 18,858 | **18,858** | Kiểm tra tensor trọng số: $63\times 128 + 128 + 128\times 64 + 64 + 64\times 32 + 32 + 32\times 10 + 10$ | **VERIFIED (Chính xác 100%)** |

---

### C. Phân chia Dữ liệu Nhánh 123 (Dataset 123 Split — Tổng 899 mẫu)

*Bảng kiểm toán đã được tách biệt minh bạch giữa việc cập nhật trọng số và việc thẩm định / lựa chọn mô hình:*

| Phân vùng dữ liệu | Số mẫu | Vai trò thực tế trong hệ sinh thái mô hình | Cập nhật trọng số (Training / Fitting) | Đánh giá & Chọn mô hình (Model Selection) | Nguồn bằng chứng xác thực |
|---|:---:|---|:---:|:---:|---|
| **Train** | **549** | Tập huấn luyện chính thức (349 mẫu tác giả + 170 Ardamavi + 30 ASL_HG) | **CÓ** | **KHÔNG** (Dùng trực tiếp để fit trọng số) | `data/processed/digits/digits_final_train_frozen.csv`, `digits_final_freeze_manifest.json` |
| **Validation** | **100** | Tập thẩm định đối chứng ngoài độc lập theo người thực hiện (`validation_reference.csv`, 10 mẫu/lớp) | **KHÔNG** | **CÓ** (Dùng tính `validation_macro_f1` để chọn `mlp_large_v2a` trong candidate comparison) | `candidate_comparison_v2.json`, `candidate_comparison_digits_final.json`, `data/splits/digits/validation_reference.csv` |
| **Regression Reference** | **250** | Tập đối chứng kiểm thử hồi quy đóng băng (Gốc từ `test_v1.csv` ban đầu, dùng phát hiện suy giảm hiệu năng khi nạp data mở) | **KHÔNG** | **KHÔNG** (`"test_set_used": false`, `"regression_reference_used_for_training_or_selection": false`) | `ai/dataset/kytuso/datatrain/split_manifest_v1.json`, `digits_v2a_test_evaluation.json` |
| **Tổng nguồn** | **899** | Toàn bộ các mẫu dữ liệu liên quan đến nhánh số | — | — | $549 + 100 + 250 = 899$ mẫu |

---

## 3. PHÂN TÍCH CHUYÊN SÂU NGUYÊN NHÂN KỸ THUẬT

### 3.1. Bản chất Toán học của Mốc 52 Epoch và Early Stopping Nhánh 123
Trong Scikit-learn `MLPClassifier`, khi bật `early_stopping=True`:
1. Dữ liệu huấn luyện 549 mẫu được tự động trích ra 15% làm tập validation nội bộ ($549 \times 0.15 \approx 83\text{ mẫu}$).
2. Sau mỗi epoch, mô hình tính toán `val_score` (độ chính xác phân loại trên 83 mẫu này).
3. Thuật toán duy trì biến đếm `no_improvement_count`. Điều kiện để coi là "có cải thiện" và thiết lập lại bộ đếm về 0 là:
   $$\text{val\_score} > \text{best\_val\_score} + \text{tol} \quad (\text{với } \text{tol} = 0.0001)$$
4. Diễn biến thực nghiệm trích xuất từ `mlp.validation_scores_`:
   - **Epoch 1:** Val score = $0.5301$
   - **Epoch 5:** Val score = $0.9639$
   - **Epoch 10:** Val score = $0.9880$
   - **Epoch 11:** Val score đạt **$1.0000$** (100% chính xác trên 83 mẫu).
   - **Epoch 12 đến 52:** Do điểm số trần tối đa của độ chính xác là $1.0000$, điều kiện $\text{val\_score} > 1.0000 + 0.0001 = 1.0001$ là **không thể xảy ra**.
   - Dù ở các Epoch 16, 17, 18, 19, 21, 22, 23 mô hình vẫn đạt điểm số $1.0000$, nhưng theo định nghĩa của Scikit-learn, điểm số này không *vượt trội hơn* $1.0000 + \text{tol}$.
   - Do đó, bắt đầu từ Epoch 12, bộ đếm `no_improvement_count` tăng liên tục qua từng epoch:
     $$\text{Epoch 12} \implies \text{count}=1, \quad \text{Epoch 13} \implies \text{count}=2, \quad \dots, \quad \text{Epoch 51} \implies \text{count}=40$$
   - Tại **Epoch 52**, bộ đếm đạt 41 (vượt ngưỡng `n_iter_no_change = 40`), thuật toán Early Stopping chính thức ngắt quá trình huấn luyện.

**Kết luận khoa học:**
- Mốc 52 là **Epoch dừng huấn luyện (Stopping Epoch)** sau khi kiên nhẫn chờ đợi 40 epoch không có sự đột phá.
- Mốc hội tụ tối ưu thực sự (Optimal / Convergence Point) là **Epoch 11** (hoặc mốc duy trì cuối cùng là **Epoch 23**).
- Việc Slide 11 phiên bản cũ trình bày mốc 52 vừa là Total Epochs vừa là Optimal Point đã vô tình đánh đồng hai khái niệm khác nhau. Phiên bản Verified V2 đã chỉnh sửa Series 2 về đúng giá trị thực nghiệm là **Epoch 11**.

---

### 3.2. Bản chất Cơ chế Early Stopping Nhánh ABC
- Nhánh ABC sử dụng hàm huấn luyện tùy biến trong PyTorch/Scikit-learn wrapper với cấu hình `external_validation_patience = 20` và tiêu chí giám sát `VALIDATION v1 macro-F1`.
- Quá trình huấn luyện diễn ra qua 101 epoch.
- Tại **Epoch 81**, mô hình đạt giá trị Macro-F1 cao nhất trên tập Validation độc lập ($0.9895$). Trọng số mô hình tại thời điểm này được lưu thành checkpoint tối ưu (`state.joblib`).
- Thuật toán tiếp tục chạy thêm 20 epoch nữa (từ epoch 82 đến epoch 101) để thăm dò khả năng tối ưu hóa tiếp theo.
- Sau 20 epoch liên tiếp không vượt qua được ngưỡng kỷ lục tại Epoch 81, cơ chế Early Stopping kích hoạt tại **Epoch 101** với lý do `EXTERNAL_VALIDATION_MACRO_F1_PATIENCE`.
- Checkpoint được trích xuất để đóng gói mô hình production chính là checkpoint tại **Epoch 81**.

---

### 3.3. Bản chất Thuật toán Tối ưu hóa Hội tụ Nhánh J/Z
- Nhánh cử chỉ động J/Z sử dụng `LogisticRegression(solver='lbfgs')`.
- Thuật toán L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) là một thuật toán tối ưu hóa bậc hai quasi-Newton tối ưu hóa trên toàn bộ tập dữ liệu (batch optimization).
- L-BFGS không chia nhỏ dữ liệu thành các mini-batch và không tính toán theo chu kỳ "epoch". Thay vào đó, mỗi vòng lặp (iteration) tính toán xấp xỉ ma trận Hessian nghịch đảo và cập nhật hướng tìm kiếm theo gradient.
- Quá trình tối ưu hóa kết thúc sau **22 iterations** khi chuẩn bậc hai của gradient (gradient norm) nhỏ hơn ngưỡng dung sai `tol=1e-4`, đánh dấu trạng thái hội tụ toán học.
- Do đó, mốc 22 là **số lần lặp của bộ giải (solver iterations)**, không phải là epoch huấn luyện. Biểu đồ Chart 11 và các văn bản trên Slide đã được phân biệt rõ ràng: Series 1 đo lường `Total Epochs / LBFGS Iters` và Series 2 đo lường `Optimal Epoch / Solver Convergence`.

---

### 3.4. Bản chất Phân chia Dữ liệu Nhánh 123 (100 Val & 250 Regression Ref)
- **100 mẫu Validation (`validation_reference.csv`):** Được chuẩn bị có chủ đích với 10 mẫu trên mỗi lớp ký tự chữ số, thu thập độc lập để làm bộ thẩm định khách quan ngoài nhằm so sánh độ chính xác và Macro-F1 giữa các họ kiến trúc mô hình (`mlp_small`, `mlp_medium`, `mlp_large_v2a`, `mlp_large_v2b`, v.v.). Nhóm nghiên cứu đã dựa trên kết quả trên 100 mẫu này để lựa chọn `mlp_large_v2a`. Tuy nhiên, 100 mẫu này **tuyệt đối không tham gia cập nhật trọng số (fitting)** qua backpropagation.
- **250 mẫu Regression Reference (`test_v1.csv`):** Được giữ nguyên vẹn từ giai đoạn 1 khi chưa bổ sung dữ liệu Ardamavi/ASL_HG. Mục đích duy nhất của tập này là chạy kiểm thử đối chứng hồi quy để đảm bảo mô hình sau khi học thêm 200 mẫu mở không bị hiện tượng suy thoái trên dữ liệu cũ. Tập này **không dùng để huấn luyện và cũng không dùng để tinh chỉnh hay lựa chọn mô hình**.

---

## 4. CHI TIẾT CÁC HIỆU CHỈNH TRÊN POWERPOINT VERIFIED V2

### 4.1. File Tiếng Anh: `SIC_Capstone_Presentation_English_Verified_v2.pptx`

1. **Slide 7 — TextBox 9 (Dataset Audit Note):**
   - *Trước:* `...Comprises 549 Train + 100 Validation samples (Ardamavi, ASL_HG) plus 250 regression reference samples used strictly for verification, not model training...`
   - *Sau:* `• 123 Branch (899 source samples): Comprises 549 Train (weight fitting) + 100 Validation samples (model evaluation/selection, no weight updates) plus 250 frozen regression reference samples (strictly verification/regression detection, not model training). Uses HandCentric2DAligner + StandardScaler.`
   - *Ý nghĩa:* Làm rõ ranh giới: 549 Train dùng cập nhật trọng số, 100 Val dùng chọn mô hình (không cập nhật trọng số), 250 Ref là đối chứng hồi quy đóng băng.

2. **Slide 8 — TextBox 31 (ABC) & TextBox 36 (123):**
   - *TextBox 31:* Cập nhật thành `• Training: 101 epochs (Best Epoch: 81, Stopping Epoch: 101, Patience: 20).`
   - *TextBox 36:* Cập nhật thành `• Training: 52 epochs (Stopping Epoch: 52, Best Epoch: 11, Patience: 40).`
   - *Ý nghĩa:* Phân biệt dứt khoát Best Epoch với Stopping Epoch cho cả 2 nhánh mạng neural.

3. **Slide 11 — Table 6 (Params & Conv.):**
   - Dòng ABC: `6,968 | 101 ep (Best 81, Stop 101)`
   - Dòng 123: `18,858 | 52 ep (Stop@52, Best@11)`
   - Dòng J/Z: `1,105 | 22 LBFGS iters (conv.)`

4. **Slide 11 — Chart 11 (Verified Training & Convergence Steps):**
   - Series 1: `'Total Epochs / LBFGS Iters'` $\implies (101.0, 52.0, 22.0)$
   - Series 2: Đổi tên thành `'Optimal Epoch / Solver Convergence'` và cập nhật giá trị $\implies \mathbf{(81.0, 11.0, 22.0)}$
   - *Ý nghĩa:* Sửa lỗi mốc 52 bị đánh đồng; thể hiện mạng 123 đạt trần tại Epoch 11 và dừng ở Epoch 52; J/Z đạt hội tụ sau 22 iterations.

5. **Slide 11 — TextBox 10 (Scientific Integrity & Measurement Audit):**
   - Bổ sung chú thích rõ ràng: Phân biệt Epoch huấn luyện mini-batch với LBFGS solver iterations; giải thích cơ chế dừng sớm tại Epoch 52 (Best Ep 11); làm rõ vai trò của 100 Val và 250 Regression Ref.

---

### 4.2. File Tiếng Việt: `SIC_Capstone_Presentation_Vietnamese_Verified_v2.pptx`

1. **Slide 5 — TextBox 16 (Lưu ý kiểm toán phân chia dữ liệu):**
   - *Cập nhật:* `Lưu ý kiểm toán: Nhánh 123 gồm 100 mẫu Validation dùng chọn mô hình (không cập nhật trọng số); 250 mẫu regression reference dùng đối chứng hồi quy, không phải dữ liệu huấn luyện.`
   - *Bảo toàn định dạng:* Giữ nguyên font chữ `Be Vietnam Pro`, cỡ chữ `13.5pt`, định dạng in đậm/in thường đúng từng run.

2. **Slide 6 — TextBox 19 (ABC), TextBox 24 (123), TextBox 29 (J/Z):**
   - *TextBox 19 (ABC):* `Huấn luyện: 101 epochs (Best Epoch: 81, Stopping Epoch: 101, Patience: 20).`
   - *TextBox 24 (123):* `Huấn luyện: 52 epochs (Stopping Epoch: 52, Best Epoch: 11, Patience: 40).`
   - *TextBox 29 (J/Z):* `Tối ưu hóa: Hội tụ sau 22 iterations (LBFGS solver convergence, không phải epoch).`

---

## 5. BÁO CÁO KIỂM THỬ ĐA PHƯƠNG THỨC (MULTI-MODAL QA AUDIT SUITE)

Để đảm bảo chất lượng kỹ thuật cao nhất và tính trung thực khoa học, hệ thống kiểm thử tự động đã phân loại và thực thi nghiêm ngặt 4 nhóm kiểm tra độc lập:

```text
======================================================================
KẾT QUẢ THỰC THI BỘ KIỂM THỬ QA TỰ ĐỘNG (qa_suite_v2.py)
======================================================================
[English V2]    File: SIC_Capstone_Presentation_English_Verified_v2.pptx    | Size: 1,698,712 bytes
[Vietnamese V2] File: SIC_Capstone_Presentation_Vietnamese_Verified_v2.pptx | Size:   137,593 bytes

1. STRUCTURAL CHECKS (Kiểm tra Cấu trúc):
   - English Slides Count: 13/13 verified.
   - Vietnamese Slides Count: 10/10 verified.
   - Shape Presence: TextBox 9, 31, 36, Table 6, Chart 11, TextBox 10, TextBox 16, 19, 24, 29 đầy đủ 100%.
   ==> TRẠNG THÁI: [PASS - 100%]

2. DATA & CONTENT CHECKS (Kiểm tra Dữ liệu & Nội dung):
   - English Slide 7: Vai trò 549 Train, 100 Val, 250 Ref hiển thị chính xác.
   - English Slide 8: ABC (Best 81, Stop 101, Pat 20); 123 (Stop 52, Best 11, Pat 40).
   - English Slide 11 Table: ABC (Best 81, Stop 101); 123 (Stop@52, Best@11); J/Z (22 LBFGS iters conv.).
   - English Slide 11 Chart: Series 1 = (101.0, 52.0, 22.0); Series 2 = (81.0, 11.0, 22.0).
   - Vietnamese Slide 5: Lưu ý kiểm toán 100 Val (chọn mô hình) & 250 Ref (đối chứng hồi quy).
   - Vietnamese Slide 6: ABC (Best 81, Stop 101, Pat 20); 123 (Stop 52, Best 11, Pat 40); J/Z (22 iters solver convergence).
   ==> TRẠNG THÁI: [PASS - 100%]

3. COORDINATE & LAYOUT CHECKS (Kiểm tra Tọa độ & Bố cục):
   - Mọi shape được hiệu chỉnh đều có kích thước dương (width > 0, height > 0).
   - Không có shape nào có tọa độ âm (left >= 0, top >= 0).
   - Tọa độ biên nằm trong khung hiển thị slide, không tràn biên, không đè lấn layout.
   ==> TRẠNG THÁI: [PASS - 100%]

4. VISUAL RENDER CHECKS (Kiểm tra Hiển thị Đồ họa):
   - TRẠNG THÁI: CHƯA THỰC HIỆN (NOT PERFORMED).
   - NGUYÊN NHÂN KỸ THUẬT: Môi trường làm việc Windows CLI không hỗ trợ hiển thị đồ họa trực tiếp (headless CLI) và không cài đặt công cụ xuất hình ảnh máy tính Microsoft Office / LibreOffice headless rendering.
   - KHUYẾN NGHỊ NGHIỆM THU: Người dùng mở trực tiếp hai tập tin .pptx trong Microsoft PowerPoint desktop trên máy tính cá nhân để nghiệm thu trực quan.
```

---

## 6. DANH MỤC TẬP TIN VÀ ĐƯỜNG DẪN XUẤT BẢN

Toàn bộ các tập tin sau đã được khởi tạo, lưu trữ an toàn trong workspace và sao chép sang thư mục `Downloads` của người dùng:

1. **Tập tin PowerPoint tiếng Anh Verified V2:**
   - Workspace: `D:\NhanDienCuChi\SIC_Capstone_Presentation_English_Verified_v2.pptx` (1,698,712 bytes)
   - Downloads: `C:\Users\Admin\Downloads\SIC_Capstone_Presentation_English_Verified_v2.pptx` (1,698,712 bytes)
2. **Tập tin PowerPoint tiếng Việt Verified V2:**
   - Workspace: `D:\NhanDienCuChi\SIC_Capstone_Presentation_Vietnamese_Verified_v2.pptx` (137,593 bytes)
   - Downloads: `C:\Users\Admin\Downloads\SIC_Capstone_Presentation_Vietnamese_Verified_v2.pptx` (137,593 bytes)
3. **Báo cáo kiểm toán dữ liệu đào tạo Markdown:**
   - `D:\NhanDienCuChi\powerpoint_training_data_audit.md`
   - `D:\NhanDienCuChi\reports\powerpoint_training_data_audit.md`
4. **Script kiểm tra QA tự động đa phương thức:**
   - `D:\NhanDienCuChi\scratch\qa_suite_v2.py`
