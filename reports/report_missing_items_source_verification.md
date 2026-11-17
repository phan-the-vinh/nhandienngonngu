# BÁO CÁO XÁC MINH CÁC THÔNG TIN KỸ THUẬT CÒN THIẾU TRONG BÁO CÁO ĐỒ ÁN
## (TECHNICAL AUDIT FOR THESIS REPORT COMPLETION — STAGE 1)

- **Dự án:** Hệ thống Nhận diện Ngôn ngữ Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)
- **Workspace:** `D:\NhanDienCuChi`
- **GitHub Repository:** [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)
- **Ngày thực hiện:** 2026-11-17
- **Mục tiêu:** Cung cấp các thông tin kỹ thuật chính xác, có bằng chứng thực tế từ mã nguồn và dữ liệu trên đĩa để tác giả bổ sung vào báo cáo đồ án (Word/PowerPoint).

---

## BẢNG TỔNG HỢP CÁC VỊ TRÍ CẦN BỔ SUNG TRONG BÁO CÁO ĐỒ ÁN

| STT | Vị trí / Mục trong Báo cáo Word | Thông tin cần bổ sung | Trạng thái xác minh | Bằng chứng mã nguồn / Tệp tin |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Chương Dữ liệu (Phần ABC) | Nguồn gốc chi tiết và giấy phép của 89.472 mẫu | **VERIFIED (Có phân loại rõ)** | [docs/dataset.md](file:///D:/NhanDienCuChi/docs/dataset.md#L42-L46)<br>[reports/stage2_license_and_provenance_review.md](file:///D:/NhanDienCuChi/reports/stage2_license_and_provenance_review.md#L33-L49)<br>[ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv](file:///D:/NhanDienCuChi/ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv) |
| **2** | Chương Phương pháp (Phần Cử chỉ động J/Z) | Công thức, ý nghĩa, đơn vị và thứ tự của 6 đặc trưng bổ sung (Vector 69D/frame) | **VERIFIED (Toán học & Mã nguồn)** | [ai/feature_extraction/dynamic_sequence.py](file:///D:/NhanDienCuChi/ai/feature_extraction/dynamic_sequence.py#L15-L226)<br>[ai/preprocessing/landmark_utils.py](file:///D:/NhanDienCuChi/ai/preprocessing/landmark_utils.py#L4-L51)<br>[training/dynamic/train_dynamic.py](file:///D:/NhanDienCuChi/training/dynamic/train_dynamic.py#L42-L55) |
| **3** | Chương Triển khai & Môi trường | Tên file môi trường và lệnh cài đặt chính xác | **VERIFIED** | [requirements.txt](file:///D:/NhanDienCuChi/requirements.txt)<br>[environment.yml](file:///D:/NhanDienCuChi/environment.yml)<br>[README.md](file:///D:/NhanDienCuChi/README.md#L93-L111) |
| **4** | Chương Kiến trúc Hệ thống | Danh mục, đường dẫn và vai trò của các tệp cấu hình JSON | **VERIFIED** | Thư mục [configs/](file:///D:/NhanDienCuChi/configs/): `letters.json`, `digits.json`, `space.json`, `dynamic.json`, `application.json`, `camera.json` |
| **5** | Chương Thực nghiệm & Thảo luận | Khoảng cách Euclid giữa các cặp hình thái tương đồng (O–0, V–2, W–6) | **CHƯA ĐO (Ghi nhận đúng thực tế)** | Không có script/EDA log đo định lượng trong codebase; đề xuất lập luận hình thái học ASL và thiết kế phân tách mode. |

---

## 1. CHI TIẾT NGUỒN GỐC VÀ GIẤY PHÉP TẬP DỮ LIỆU ABC (89.472 MẪU)

### 1.1. Bảng phân rã thành phần dữ liệu
Tổng số mẫu của tập dữ liệu chữ cái tĩnh ABC là **89.472 mẫu**, bao gồm 89.415 mẫu cơ sở đã được làm sạch và đóng băng tại phiên bản `Frozen v3` cộng với 57 mẫu thu thập bổ sung qua webcam:

| Nhóm dữ liệu | Số lượng mẫu | Tỷ lệ (%) | Giấy phép công bố | Tác giả / Nguồn thu thập | Tình trạng pháp lý |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **1. Kaggle ASL Alphabet** | **45.236** | 50,56% | **GNU General Public License v2.0 (GPL-2.0)** | Akash Nagaraj (2018)<br>URL: `kaggle.com/datasets/grassknoted/asl-alphabet` | **ĐÃ XÁC MINH**. Giấy phép mở GPL-2.0 có điều khoản Copyleft (tranh luận phái sinh cho trọng số mô hình). |
| **2. Color-coded Hand Segmentation** | **41.676** | 46,58% | **LICENSE_UNKNOWN** | Không rõ tác giả / Nguồn ảnh tiền xử lý nhãn màu từ các bộ dữ liệu trực tuyến cũ. | **CHƯA XÁC MINH ĐƯỢC GIẤY PHÉP**. Rủi ro bản quyền cao nhất. |
| **3. Hand Segmentation & Web** | **2.503** | 2,80% | **LICENSE_UNKNOWN** | Không rõ tác giả / Ảnh thu thập phụ trợ (`hand1_c_bot_seg`...). | **CHƯA XÁC MINH ĐƯỢC GIẤY PHÉP**. |
| **4. Mẫu webcam tăng cường U/V/W** | **57** | 0,06% | **Bản quyền thuộc nhóm tác giả** | Tác giả đề tài tự ghi hình qua webcam (15 OLD_U, 15 OLD_V, 15 V_TO_W, 12 V_TO_U). | **NỘI BỘ ĐỀ TÀI**. Dùng tăng cường phân biệt 3 chữ cái dễ nhầm. |
| **TỔNG CỘNG** | **89.472** | **100%** | — | — | — |

### 1.2. Đề xuất câu chữ học thuật đưa vào Báo cáo Word
> *"Tập dữ liệu ký tự tĩnh ABC gồm tổng cộng 89.472 mẫu đặc trưng 63 chiều, được tổng hợp từ các nguồn dữ liệu mở và dữ liệu thực nghiệm nội bộ nhằm bảo đảm tính đa dạng về phông nền, góc quay và cự ly bàn tay. Trong đó: 45.236 mẫu (chiếm 50,56%) kế thừa từ bộ dữ liệu mở ASL Alphabet của Akash Nagaraj (2018) phát hành theo giấy phép GNU GPL-2.0; 44.179 mẫu (chiếm 49,38%, bao gồm 41.676 ảnh phân đoạn nhãn màu và 2.503 ảnh phân đoạn bàn tay phụ trợ) được thu thập từ các kho lưu trữ trực tuyến nhưng chưa xác định được thông tin bản quyền gốc (LICENSE_UNKNOWN); và 57 mẫu (chiếm 0,06%) do nhóm đề tài tự ghi hình trực tiếp qua webcam nhằm huấn luyện tăng cường, giải quyết triệt để hiện tượng nhầm lẫn đặc trưng giữa ba chữ cái có cử chỉ tương đồng là U, V và W."*

---

## 2. SÁU ĐẶC TRƯNG BỔ SUNG CỦA NHÁNH CỬ CHỈ ĐỘNG J/Z (VECTOR 69D/FRAME)

### 2.1. Nguồn gốc mã nguồn và cấu trúc vector
Toàn bộ logic trích xuất đặc trưng chuỗi cử chỉ động được cài đặt tại [ai/feature_extraction/dynamic_sequence.py](file:///D:/NhanDienCuChi/ai/feature_extraction/dynamic_sequence.py) (hàm `build_sequence_features`, dòng 176–226). 

Mỗi cử chỉ động J hoặc Z được biểu diễn bởi một chuỗi gồm **16 frame** thời gian (`SEQUENCE_LENGTH = 16`). Tại mỗi frame $t$ ($t \in [0, 15]$), vector đặc trưng gồm đúng **69 chiều liên tục** (`FEATURE_DIM = 69`):
$$\mathbf{v}(t) = \big[ f_1(t), f_2(t), \dots, f_{63}(t), \quad \Delta X_0(t), \Delta Y_0(t), \Delta X_8(t), \Delta Y_8(t), \Delta X_{20}(t), \Delta Y_{20}(t) \big]$$

### 2.2. Chi tiết 63 đặc trưng tọa độ 3D gốc (Chỉ số 0 đến 62)
- **Nguồn trích xuất:** 21 điểm mốc bàn tay 3D từ MediaPipe Hand Landmarker (`landmark_utils.py` dòng 16–49).
- **Phép tịnh tiến về cổ tay (Wrist-relative centering):** Lấy điểm cổ tay (Landmark 0) làm gốc tọa độ tương đối:
  $$\mathbf{p}'_i(t) = \mathbf{p}_i(t) - \mathbf{p}_0(t) = \big( x_i(t) - x_0(t), \; y_i(t) - y_0(t), \; z_i(t) - z_0(t) \big), \quad \forall i \in [0, 20]$$
- **Chuẩn hóa tỷ lệ kích thước (Max-distance scale invariance):**
  $$d_{\max}(t) = \max_{j \in [0, 20]} \|\mathbf{p}'_j(t)\|_2 = \max_{j} \sqrt{{x'_j(t)}^2 + {y'_j(t)}^2 + {z'_j(t)}^2}$$
  $$\mathbf{p}''_i(t) = \frac{\mathbf{p}'_i(t)}{\max(d_{\max}(t), 1.0)}$$
- **Thành phần:** 63 giá trị thực phẳng $(x''_0, y''_0, z''_0, \dots, x''_{20}, y''_{20}, z''_{20})$. Luôn có $x''_0 = y''_0 = z''_0 = 0.0$.

### 2.3. Chi tiết 6 đặc trưng bổ sung (Chỉ số 63 đến 68)
- **Điểm mốc theo dõi quỹ đạo (`TRAJECTORY_LANDMARKS = (0, 8, 20)`):**
  - **Landmark 0:** Cổ tay (**Wrist**).
  - **Landmark 8:** Đầu ngón trỏ (**Index Finger Tip**) — theo dõi nét vẽ chữ Z.
  - **Landmark 20:** Đầu ngón út (**Pinky Finger Tip**) — theo dõi nét móc chữ J.
- **Tỷ lệ lòng bàn tay cố định trên toàn chuỗi (`hand_scale_px`):**
  - Tính trung vị khoảng cách pixel Euclidean từ cổ tay (Landmark 0) đến 3 khớp gốc ngón tay (`SCALE_LANDMARKS = (5, 9, 17)` gồm Index MCP, Middle MCP, Pinky MCP) trên tất cả 16 frame của chuỗi:
    $$S_{\text{hand}} = \text{median}_{t=0..15} \left( \|\mathbf{p}_k^{\text{pixel}}(t) - \mathbf{p}_0^{\text{pixel}}(t)\|_2 \right), \quad k \in \{5, 9, 17\}$$
- **Công thức tính độ dời tương đối (2D Trajectory Displacement):**
  - Lấy tọa độ pixel tại frame $t$ trừ đi tọa độ pixel tại frame bắt đầu cử chỉ ($t=0$), sau đó chia cho $S_{\text{hand}}$:
    $$\Delta X_k(t) = \frac{X_k^{\text{pixel}}(t) - X_k^{\text{pixel}}(0)}{S_{\text{hand}}}, \quad \Delta Y_k(t) = \frac{Y_k^{\text{pixel}}(t) - Y_k^{\text{pixel}}(0)}{S_{\text{hand}}}, \quad k \in \{0, 8, 20\}$$
- **Thứ tự 6 đặc trưng trong vector 69D tại frame $t$:**
  - **Chiều 64 ($f_{64}$):** $\Delta X_0(t)$ — Độ dời trục X của Cổ tay so với frame 0.
  - **Chiều 65 ($f_{65}$):** $\Delta Y_0(t)$ — Độ dời trục Y của Cổ tay so với frame 0.
  - **Chiều 66 ($f_{66}$):** $\Delta X_8(t)$ — Độ dời trục X của Đầu ngón trỏ so với frame 0.
  - **Chiều 67 ($f_{67}$):** $\Delta Y_8(t)$ — Độ dời trục Y của Đầu ngón trỏ so với frame 0.
  - **Chiều 68 ($f_{68}$):** $\Delta X_{20}(t)$ — Độ dời trục X của Đầu ngón út so với frame 0.
  - **Chiều 69 ($f_{69}$):** $\Delta Y_{20}(t)$ — Độ dời trục Y của Đầu ngón út so với frame 0.

### 2.4. Đơn vị và Ý nghĩa vật lý
- **Đơn vị:** Đại lượng không thứ nguyên (tỷ số pixel trên kích thước bàn tay chuẩn $S_{\text{hand}}$).
- **Ý nghĩa kỹ thuật:** Phép chuẩn hóa 63 landmark tĩnh ở từng frame đều lấy cổ tay làm gốc, khiến toàn bộ thông tin về chuyển động tịnh tiến trong không gian camera (global trajectory) bị triệt tiêu hoàn toàn. Ký hiệu **J** (vẽ nét móc bằng ngón út) và **Z** (vẽ đường zic-zac bằng ngón trỏ) bắt buộc phải có quỹ đạo di chuyển không gian. 6 đặc trưng bổ sung này khôi phục chính xác quỹ đạo chuyển động 2D của cổ tay và 2 ngón tay chủ đạo so với thời điểm khởi đầu cử chỉ, đồng thời triệt tiêu ảnh hưởng của khoảng cách xa/gần nhờ phép chia cho $S_{\text{hand}}$.
- **Vector đầu vào mô hình:** Khi đưa vào bộ phân loại (`train_dynamic.py` dòng 155), chuỗi $[16, 69]$ được làm phẳng theo thứ tự C-contiguous thành một vector **1.104 chiều** ($16 \times 69 = 1104$) xếp tuần tự từ frame 0 đến frame 15.

---

## 3. TÊN FILE MÔI TRƯỜNG VÀ HƯỚNG DẪN CÀI ĐẶT

### 3.1. Các tệp cấu hình môi trường có trong Repository
1. [requirements.txt](file:///D:/NhanDienCuChi/requirements.txt) (182 bytes, 8 dòng): Dành cho công cụ quản lý gói `pip`.
   ```text
   opencv-python>=4.8.0
   mediapipe>=0.10.0
   numpy>=1.24.0
   scikit-learn>=1.3.0
   joblib>=1.3.0
   pandas>=2.0.0
   ```
2. [environment.yml](file:///D:/NhanDienCuChi/environment.yml) (276 bytes, 14 dòng): Dành cho trình quản lý môi trường `conda` (tên môi trường: `nhandiencuchi`, Python 3.10 / 3.11).
3. **Tệp lock:** Không có tệp lock (`poetry.lock`, `Pipfile.lock`).

### 3.2. Câu lệnh cài đặt chính xác cho người dùng clone mới
Căn cứ tài liệu chính thức tại [README.md](file:///D:/NhanDienCuChi/README.md#L93-L111):

**Cách 1: Sử dụng Conda (Khuyến nghị)**
```bash
conda env create -f environment.yml
conda activate nhandiencuchi
```

**Cách 2: Sử dụng Python venv chuẩn**
```bash
python -m venv .venv

# Kích hoạt môi trường:
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt thư viện:
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. DANH MỤC CÁC TỆP CẤU HÌNH THỰC TẾ TRONG CODEBASE

Hệ thống quản lý toàn bộ tham số, mô hình và quy trình thông qua các tệp JSON trong thư mục [configs/](file:///D:/NhanDienCuChi/configs/):

| Nhánh / Thành phần | Tên file & Đường dẫn | Chức năng thực tế trong hệ thống |
| :--- | :--- | :--- |
| **ABC (Static Letters)** | `configs/letters.json` | Cấu hình nhận diện 24 chữ cái tĩnh A–Y (trừ J/Z). Trỏ đến model production `models/letters/production/uvw_medium_model.joblib`, scaler `uvw_medium_scaler.joblib`, danh sách 63 tên đặc trưng `f1..f63`, hàm chuẩn hóa `landmark_utils.py::normalize_landmarks`, và model fallback v1. |
| **Digits (123) - Runtime** | `configs/digits.json` | Cấu hình thực thi camera nhận diện 10 chữ số (0–9). Trỏ đến model production `models/digits/production/digits_final.joblib`, pipeline 3 bước `["HandCentric2DAligner", "StandardScaler", "MLPClassifier"]`, cờ `mirror_before_mediapipe: false`. |
| **Digits (123) - Training**| `configs/digits_training.json` | Cấu hình tái lập huấn luyện (`frozen_replay`). Thiết lập siêu tham số mạng MLP (128, 64, 32), seed=42, nạp `digits_final_train_frozen.csv` (549 mẫu) và `validation_reference.csv` (100 mẫu). |
| **J/Z - Pipeline Schema** | `configs/dynamic.json` | Khai báo schema cử chỉ động `baseline_b_v1` (16 frame, 69 chiều), tham số trích xuất `motion_preview` (ngưỡng phát hiện đỉnh chuyển động, cửa sổ làm mịn trung vị). |
| **J/Z - Training** | `configs/dynamic_training.json` | Cấu hình huấn luyện mô hình phân loại nhị phân J/Z (`StandardScaler` + `LogisticRegression`, $C=1.0$), khai báo 3 video bị loại trừ trong `confirmed_additional_exclusions` (`J/99.avi`, `J/103.avi`, `J/250.avi`). |
| **SPACE Gesture** | `configs/space.json` | Khai báo mô hình LogisticRegression nhị phân `models/space/production/final_space_model.joblib`, ngưỡng phân loại `threshold: 0.889063295339523`, cờ `apply_abc_x_flip: true` để đồng bộ ảnh gương giữa camera ABC và tập dữ liệu huấn luyện. |
| **Hệ thống chung (Inference)**| `configs/application.json` | File điều phối trung tâm. Khai báo đường dẫn gói `models/mediapipe/hand_landmarker.task`, liên kết `letters_config` và `digits_config`, thư mục thử nghiệm J/Z, cờ hiển thị gương `mirror_display: true`. |
| **Camera Phần cứng** | `configs/camera.json` | Cấu hình cổng camera phần cứng (`camera_index: 0`), thuật toán hàng đợi làm mịn dự đoán (`smoothing_window: 7`, `smoothing_min_votes: 4`). |

---

## 5. KHOẢNG CÁCH EUCLID VÀ CƠ SỞ HÌNH THÁI HỌC GIỮA CÁC CẶP CHỮ – SỐ (O–0, V–2, W–6)

### 5.1. Kết quả kiểm tra mã nguồn thực tế
- Đã thực hiện quét toàn bộ codebase (bao gồm các module `ai/`, `tools/`, `training/`, `tests/` và các báo cáo thử nghiệm trước đây).
- **KẾT LUẬN XÁC MINH:** **CHƯA THỰC HIỆN ĐO LƯỜNG ĐỊNH LƯỢNG KHOẢNG CÁCH EUCLID GIỮA CÁC CẶP (O–0), (V–2), (W–6) TRONG CODEBASE**. Không có tệp tin hay nhật ký thực nghiệm nào lưu ma trận khoảng cách hình học số liệu giữa các cặp nhãn này.

### 5.2. Cơ sở hình thái học lý thuyết (ASL Morphology)
Sự tương đồng giữa các cặp ký hiệu này bắt nguồn từ chính quy chuẩn ngôn ngữ ký hiệu Mỹ (ASL), chứ không phải do sai số thuật toán:
1. **Cặp O (chữ cái) và 0 (chữ số):** Cả hai đều có hình thái bàn tay khum tròn, đầu bốn ngón tay chụm lại tiếp xúc với đầu ngón cái tạo thành một hình oval rỗng. Tọa độ tương đối của 21 điểm mốc MediaPipe gần như trùng khít nhau trong không gian đặc trưng 63 chiều.
2. **Cặp V (chữ cái) và 2 (chữ số):** Cả hai đều duỗi thẳng ngón trỏ và ngón giữa tạo thành hình chữ V, trong khi ngón cái gấp đè lên ngón đeo nhẫn và ngón út gập vào lòng bàn tay.
3. **Cặp W (chữ cái) và 6 (chữ số):** Trong quy chuẩn số đếm ASL, số 6 được ký hiệu bằng cách xòe 3 ngón (trỏ, giữa, nhẫn) trong khi ngón cái chạm đầu ngón út; cấu trúc xòe 3 ngón này có sự tương đồng đặc trưng rất cao với chữ cái W.

### 5.3. Quyết định thiết kế kiến trúc hệ thống
Do sự chồng lấn hình thái học tự nhiên quá lớn, nếu gom chung toàn bộ chữ cái (24 lớp) và chữ số (10 lớp) vào cùng một mô hình phân loại 34 lớp duy nhất, mạng nơ-ron sẽ gặp hiện tượng nhập nhằng không gian quyết định (decision boundary ambiguity), dẫn đến độ chính xác suy giảm mạnh trên các cặp này.

Vì vậy, nhóm đề tài đã đưa ra quyết định kiến trúc: **Tách biệt hoàn toàn thành hai chế độ phân loại độc lập:**
- Chế độ chữ cái tĩnh: `--mode letters` (mô hình `uvw_medium_model.joblib`).
- Chế độ chữ số tĩnh: `--mode digits` (mô hình `digits_final.joblib`).

### 5.4. Đề xuất câu chữ học thuật đưa vào Báo cáo Word
> *"Trong ngôn ngữ ký hiệu ASL tiêu chuẩn, tồn tại sự tương đồng hình thái học tự nhiên rất lớn giữa một số ký tự chữ cái và chữ số, tiêu biểu như cặp chữ O và số 0 (cùng tạo hình vòng khép kín bằng các đầu ngón tay), cặp chữ V và số 2 (cùng giơ hai ngón trỏ và giữa hình chữ V), cũng như cặp chữ W và số 6 (cùng xòe ba ngón tay). Sự chồng lấn hình học này dẫn đến việc khoảng cách đặc trưng giữa các cặp ký hiệu liên phân hệ là cực kỳ nhỏ.*
> 
> *Nhận thức rõ thách thức này, thay vì gộp chung toàn bộ 34 ký hiệu vào một không gian phân loại duy nhất (dễ gây nhầm lẫn nghiêm trọng trong các điều kiện ánh sáng và góc quay camera thực tế), đề tài đã lựa chọn giải pháp phân tách kiến trúc thành hai chế độ nhận diện chuyên biệt: Chế độ ký tự chữ cái (Letters mode) và Chế độ ký tự chữ số (Digits mode). Trong phạm vi nghiên cứu của đồ án, quyết định phân tách này được xây dựng trên cơ sở phân tích hình thái học ký hiệu học và kiểm chứng trực tiếp qua giao diện tương tác thực tế; việc đo lường định lượng ma trận khoảng cách Euclidean chính xác giữa các vector đặc trưng của các cặp chữ - số này chưa được đưa vào kịch bản tính toán tự động và được đề xuất như một hướng mở rộng nghiên cứu trong tương lai."*

