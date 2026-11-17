# DANH SÁCH CÔNG VIỆC CUỐI CÙNG HOÀN THIỆN BÁO CÁO ĐỒ ÁN CAPSTONE
## (FINAL THESIS REPORT & PRESENTATION COMPLETION CHECKLIST)

- **Dự án:** Hệ thống Nhận diện Ngôn ngữ Ký hiệu ASL Đa nhánh (ABC, 123, J/Z, SPACE)
- **Tác giả:** Phan Thế Vinh (MSSV: dtc245200309) — Khóa SIC AI
- **Workspace:** `D:\NhanDienCuChi`
- **File Báo cáo Word chính:** `C:\Users\Admin\Downloads\SIC_AI_Capstone_Final_Report_Phan_The_Vinh_Vietnamese_v5.docx`
- **File Slide PowerPoint chính:** `C:\Users\Admin\Downloads\SIC_Capstone_Presentation_Vietnamese_Final_v7_slide7_dataset.pptx`
- **Ngày lập:** 2026-11-17

---

## PHẦN 1: CÁC ĐIỂM KỸ THUẬT ĐÃ CHỐT VÀ ĐỦ BẰNG CHỨNG ĐƯA VÀO BÁO CÁO

Các nội dung dưới đây đã được đối soát thực tế qua mã nguồn và dữ liệu trên đĩa, có bằng chứng dòng mã và tệp tin rõ ràng. Tác giả có thể điền ngay vào các vị trí tương ứng trong file Word và Slide:

### 1. Ngưỡng SPACE thực tế trong Inference Runtime: `0.889`
- **Bằng chứng file cấu hình:** [configs/space.json](file:///D:/NhanDienCuChi/configs/space.json#L7) định nghĩa:
  `"threshold": 0.889063295339523`
- **Bằng chứng mã nguồn thực thi:** [app/inference/space_predictor.py](file:///D:/NhanDienCuChi/app/inference/space_predictor.py#L30-L164):
  - Dòng 30: `DEFAULT_THRESHOLD = 0.889063295339523`
  - Dòng 164: `threshold = float(config.get("threshold", DEFAULT_THRESHOLD))`
  - Dòng 114: `is_space = bool(prob_space >= self.threshold)`
- **Kết luận:** Ngưỡng kích hoạt SPACE thực tế là **`0.889`** (chính xác: `0.889063295339523`). Con số $0.85$ xuất hiện trong một số báo cáo thảo luận sơ bộ cũ chỉ là ước lượng làm tròn thô ban đầu, đã được thay thế chính thức bằng $0.889$.

### 2. Số liệu tập dữ liệu ABC nhất quán: `89.415` vs `89.472`
- **Tập cơ sở làm sạch (Frozen v3):** Đúng **89.415 mẫu** ([final_freeze_manifest.json](file:///D:/NhanDienCuChi/ai/dataset/preprocessing_frozen_v3/final_freeze_manifest.json)).
  - Chia 70/15/15: Train = **62.591**, Val = **13.412**, Test = **13.412** ($62.591 + 13.412 + 13.412 = 89.415$).
- **Tập webcam tăng cường U/V/W:** Đúng **57 mẫu** ([additional_uv_samples.csv](file:///D:/NhanDienCuChi/ai/dataset/mlp_uvw_training_data_v1/additional_uv_samples.csv)).
- **Tổng mẫu vũ trụ ABC (Total Universe):** $89.415 + 57 = \mathbf{89.472\ \text{mẫu}}$.
- **Tập Train thực tế của mô hình `uvw_medium`:** $62.591 + 57 = \mathbf{62.648\ \text{mẫu}}$ ([manifest.json](file:///D:/NhanDienCuChi/ai/dataset/mlp_uvw_training_data_v1/manifest.json)).
- **Nguồn gốc và giấy phép của 89.472 mẫu ABC:**
  1. Kaggle ASL Alphabet (Akash Nagaraj, 2018): **45.236 mẫu** ($50,56\%$) — Giấy phép **GNU GPL-2.0**.
  2. Color-coded Hand Segmentation: **41.676 mẫu** ($46,58\%$) — **`LICENSE_UNKNOWN`** (Chưa xác định nguồn gốc/giấy phép).
  3. Hand Segmentation phụ trợ & Web: **2.503 mẫu** ($2,80\%$) — **`LICENSE_UNKNOWN`**.
  4. Mẫu webcam tăng cường U/V/W: **57 mẫu** ($0,06\%$) — Bản quyền nhóm đề tài tự thu thập nội bộ.

### 3. Công thức trích xuất 69 đặc trưng/frame của J/Z
- **Bằng chứng mã nguồn:** [ai/feature_extraction/dynamic_sequence.py](file:///D:/NhanDienCuChi/ai/feature_extraction/dynamic_sequence.py#L176-L226) (hàm `build_sequence_features`).
- **Cấu trúc vector 69D tại mỗi frame $t$ (chuỗi 16 frame):**
  - **63 đặc trưng đầu ($f_1 \dots f_{63}$):** 21 landmark 3D chuẩn hóa tịnh tiến về cổ tay ($p_i - p_0$) và chia cho cự ly tối đa từ cổ tay ($d_{\max}$) qua `landmark_utils.py::normalize_landmarks`.
  - **6 đặc trưng bổ sung ($f_{64} \dots f_{69}$):** Độ dời quỹ đạo 2D pixel $(\Delta X, \Delta Y)$ của 3 điểm mốc quan trọng so với frame $t=0$:
    - Landmark 0 (Cổ tay - Wrist): $(\Delta X_0, \Delta Y_0)$ (chiều 64, 65).
    - Landmark 8 (Đầu ngón trỏ - Index Tip): $(\Delta X_8, \Delta Y_8)$ (chiều 66, 67) — nét vẽ chữ Z.
    - Landmark 20 (Đầu ngón út - Pinky Tip): $(\Delta X_{20}, \Delta Y_{20})$ (chiều 68, 69) — nét móc chữ J.
  - **Chuẩn hóa tỷ lệ:** Chia cho $S_{\text{hand}}$ (trung vị cự ly pixel từ cổ tay đến 3 khớp MCP 5, 9, 17 trên cả 16 frame).
  - **Vector đầu vào mô hình:** Chuỗi $[16, 69]$ làm phẳng C-order thành vector **1.104 chiều** ($16 \times 69$).

### 4. Ranh giới tập dữ liệu Digits (Nhánh 123)
- **Tập Train chính thức:** **549 mẫu** (`data/processed/digits/digits_final_train_frozen.csv`).
- **Validation Reference:** **100 mẫu** (`data/splits/digits/validation_reference.csv`).
- **Regression Reference:** **250 mẫu** (`data/splits/digits/regression_reference.csv`, lấy từ `test_v1.csv` cũ).
- **Khẳng định tính trung thực:** Nhánh 123 **KHÔNG CÓ tập Test Set độc lập (0%)**. 250 mẫu Regression Reference chỉ dùng để kiểm tra độ lệch dự đoán so với model gốc, `train_digits.py` dòng 321–322 bắt buộc `regression_is_independent_final_test is False`. Tuyệt đối không được coi là Test Set độc lập.

### 5. Khoảng cách Euclid O–0, V–2, W–6
- **Khẳng định tính trung thực:** Codebase **CHƯA CÓ phép đo định lượng bằng script hoặc log EDA**.
- **Cơ sở khoa học:** Sự tương đồng là đặc tính hình thái học tự nhiên của bảng chữ cái và số ASL (cùng chụm tròn, cùng giơ 2 ngón, cùng xòe 3 ngón).
- **Quyết định kiến trúc:** Để tránh hiện tượng nhập nhằng không gian đặc trưng giữa chữ và số, hệ thống đã **phân tách thành hai chế độ độc lập** (`--mode letters` và `--mode digits`).

### 6. Tên tệp cấu hình và môi trường
- **Môi trường:** `requirements.txt` (pip) và `environment.yml` (conda, môi trường `nhandiencuchi`).
- **Cấu hình JSON:**
  - Tổng thể: `configs/application.json`, `configs/camera.json`.
  - ABC: `configs/letters.json`.
  - Digits: `configs/digits.json` (runtime), `configs/digits_training.json` (replay training).
  - J/Z: `configs/dynamic.json` (schema 16x69), `configs/dynamic_training.json` (huấn luyện).
  - SPACE: `configs/space.json`.
- **Liên kết chính thức:**
  - GitHub: `https://github.com/phan-the-vinh/nhandienngonngu`
  - Google Drive Dataset: `https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10`
  - Dataset Doc: `https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md`

---

## PHẦN 2: CÁC NỘI DUNG CẦN NGƯỜI DÙNG CUNG CẤP (ẢNH THỰC TẾ)

Hệ thống AI không tự tạo ảnh giả lập. Tác giả cần tự chụp và chèn vào báo cáo các hình ảnh sau:

1. **Ảnh chân dung cá nhân sinh viên:**
   - **Vị trí trong Word:** Trang bìa phụ hoặc Mục 5 (đoạn P413).
   - **Mô tả:** Ảnh chụp chân dung lịch sự của sinh viên Phan Thế Vinh (MSSV: dtc245200309) để hoàn thiện hồ sơ dự án cá nhân SIC AI.
2. **Ảnh chụp màn hình Giao diện Ứng dụng (Live UI Screenshots):**
   - **Vị trí trong Word:** Mục 3.4 (đoạn P326) và Bảng 3.4 / Bảng A.2 (đoạn P541).
   - **Vị trí trong Slide:** Slide 9 và Slide 10 (minh họa luồng camera và gợi ý từ vựng).
   - **Các ảnh chụp cần chuẩn bị (Chạy ứng dụng `python run.py` bằng webcam thật):**
     - *Ảnh 1:* Giao diện nhận diện chữ cái tĩnh ABC (ví dụ giơ tay chữ 'A' hoặc 'B', hiển thị khung xương 21 landmark và thanh tiến trình giữ 3.0s).
     - *Ảnh 2:* Giao diện nhận diện chữ số (chạy `python run.py --mode digits`, giơ số '3' hoặc '5').
     - *Ảnh 3:* Giao diện nhận diện cử chỉ động J hoặc Z (quỹ đạo nét vẽ trên camera).
     - *Ảnh 4:* Giao diện cử chỉ SPACE (bàn tay mở hướng ngang chèn dấu cách vào Text Composer).
     - *Ảnh 5:* Giao diện gợi ý từ vựng Text Composer (hiển thị 3 từ gợi ý từ từ điển Trie và mô hình ngôn ngữ).

---

## PHẦN 3: CÁC NỘI DUNG CẦN CHẠY THỰC NGHIỆM THỰC TẾ (NẾU MUỐN THAY THẾ SỐ LIỆU ƯỚC LƯỢNG)

Trong Bảng 3.1 của Word và Slide 11 của PowerPoint, các chỉ số đánh giá hiện đang gắn dấu hoa thị (*) kèm nhãn:
`Chỉ số gắn dấu * là ước lượng minh họa chuẩn hóa theo F1 = 2PR/(P+R), không phải kết quả đo thực tế từ log kiểm chứng độc lập`.
Nếu tác giả muốn nâng cấp thành kết quả đo thực nghiệm chính thức, cần chạy các lệnh sau trên máy tính của mình và ghi lại số liệu:

1. **Đo thời gian xử lý thực tế (CPU Latency) và FPS:**
   - **Lệnh thực hiện:** Chạy `python run.py --check-only` và chạy camera thật trong 1 phút, theo dõi FPS hiển thị ở góc trên màn hình OpenCV.
   - **Chỉ số cần ghi nhận:**
     - FPS trung bình trên webcam (ví dụ: ~28–30 FPS).
     - Thời gian trích xuất MediaPipe HandLandmarker (ms/frame, thường khoảng 15–25 ms trên CPU).
     - Thời gian suy luận mô hình MLP ABC/Digits (ms, thường < 1 ms trên CPU).
     - Thời gian trích xuất và suy luận chuỗi J/Z (ms).
2. **Kiểm tra mô hình gợi ý ngôn ngữ Qwen3:4B qua Ollama:**
   - **Điều kiện:** Đã cài đặt Ollama trên máy và tải model bằng lệnh `ollama run qwen:4b` (hoặc `qwen2.5:3b` / `qwen3:4b`).
   - **Ghi chú tính sẵn sàng:** Nếu không chạy Ollama, hệ thống tự động suy thoái mềm (graceful degradation) về từ điển Offline Trie (`english_words_25k.tsv`) với độ trễ tức thì < 0.1 ms.
3. **Đánh giá bộ kiểm thử tự động (Unit / Integration Tests):**
   - **Lệnh thực hiện:**
     ```bash
     python -m unittest tests/test_stage13_integration.py -v
     python -m unittest tests/test_space_integration.py -v
     ```
   - **Chỉ số cần ghi nhận:** Số lượng test case PASS (ví dụ: 10/10 PASS) để đưa vào Bảng 3.4.

---

## PHẦN 4: HƯỚNG DẪN CHỈNH SỬA CHI TIẾT TRONG WORD VÀ POWERPOINT

### 4.1. Chỉnh sửa trong File Word (`SIC_AI_Capstone_Final_Report_Phan_The_Vinh_Vietnamese_v5.docx`)

| Đoạn trong Word | Placeholder hiện tại | Nội dung chính xác thay thế |
| :--- | :--- | :--- |
| **P116 & P145** | `[CẦN XÁC MINH NGUỒN DỮ LIỆU]` (Mục 2.1 & Bảng 2.1) | Điền bảng phân rã nguồn gốc: 45.236 ảnh Kaggle (GPL-2.0), 41.676 ảnh phân đoạn màu (LICENSE_UNKNOWN), 2.503 ảnh phụ trợ (LICENSE_UNKNOWN), 57 ảnh webcam tăng cường U/V/W (Tác giả tự thu). Tổng vũ trụ ABC = 89.472 mẫu. |
| **P139, P549, P550**| `[GITHUB_REPOSITORY_URL]` | Điền: `https://github.com/phan-the-vinh/nhandienngonngu` |
| **P140, P552, P566**| `[GOOGLE_DRIVE_DATASET_URL]` | Điền: `https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10` |
| **P141** | `[GITHUB_DATASET_DOCUMENTATION_URL]` | Điền: `https://github.com/phan-the-vinh/nhandienngonngu/blob/main/docs/dataset.md` |
| **P232 & P523** | `[CẦN XÁC MINH TỪ SOURCE CODE]` (Mục 3.1.2 - 69D J/Z) | Điền: 63 tọa độ 3D tĩnh chuẩn hóa qua `landmark_utils.py` + 6 đặc trưng độ dời 2D pixel $(\Delta X, \Delta Y)$ của Cổ tay (0), Ngón trỏ (8) và Ngón út (20) chia cho kích thước lòng bàn tay $S_{\text{hand}}$ qua `ai/feature_extraction/dynamic_sequence.py`. |
| **P241** | `[CẦN XÁC MINH TỪ SOURCE CODE / EDA LOG]` (Mục 3.1.2 - Euclid) | Điền: Chưa thực hiện đo lường định lượng ma trận khoảng cách Euclid trong codebase; sự tương đồng giữa O–0, V–2, W–6 là đặc tính hình thái học ASL tự nhiên, là cơ sở để hệ thống thiết kế tách biệt 2 chế độ độc lập (`--mode letters` và `--mode digits`). |
| **P326 & P541** | `[CẦN BỔ SUNG: Ảnh chụp màn hình UI thực tế]` (Mục 3.4) | Chèn 2–3 ảnh chụp màn hình ứng dụng webcam thật khi chạy `python run.py`. |
| **P413** | `[CẦN BỔ SUNG: Vị trí đính kèm ảnh cá nhân]` (Mục 5) | Chèn ảnh chụp chân dung của tác giả Phan Thế Vinh. |
| **P551** | `[CẦN XÁC MINH TỪ SOURCE CODE tệp requirements.txt]` | Điền: Sử dụng tệp `requirements.txt` (cho pip) hoặc `environment.yml` (cho conda môi trường `nhandiencuchi`). |
| **P555** | `[CẦN XÁC MINH TỪ SOURCE CODE tên tệp config]` | Điền: Cấu hình hệ thống lưu tại thư mục `configs/` gồm `application.json`, `letters.json`, `digits.json`, `dynamic_training.json`, `space.json`. Ngưỡng SPACE là `0.889`. |
| **P518–P544** | Bảng A.2 (Danh mục thông tin cần bổ sung) | Đổi trạng thái từ `[CẦN XÁC MINH]` sang `ĐÃ XÁC MINH TỪ MÃ NGUỒN VÀ DỮ LIỆU` cho các mục 1, 2, 7; giữ nhãn `ƯỚC LƯỢNG MINH HỌA / CHỜ CHẠY THỰC NGHIỆM ĐỘC LẬP` cho mục 3, 4, 5. |

---

### 4.2. Chỉnh sửa và Đối chiếu trong Slide PowerPoint (`SIC_Capstone_Presentation_Vietnamese_Final_v7_slide7_dataset.pptx`)

1. **Slide 7 (Kiểm toán Dữ liệu):**
   - Đã chuẩn xác $100\%$: ABC = 89.472 mẫu (Train 62.648, Val 13.412, Test 13.412); 123 = 899 mẫu (Training Pool 549, Ext. Val 100, Test 0, Regression Ref 250); J/Z = 640 mẫu (Train 512, Val 64, Test 64).
   - **Giữ nguyên nội dung Slide 7**, không cần chỉnh sửa số liệu.
2. **Slide 8 & Slide 9 (Pipeline & Giữ 3.0s):**
   - Đã đầy đủ sơ đồ 5 bước và cơ chế giữ 3.0s chống lặp.
   - Có thể chèn thêm 1 ảnh nhỏ chụp giao diện camera thực tế vào góc slide để tăng tính thuyết phục.
3. **Slide 11 (Bảng tổng hợp chỉ số đánh giá):**
   - Đã có đầy đủ ghi chú hoa thị (*) giải thích số liệu ước lượng minh họa chuẩn hóa theo $F_1 = 2PR/(P+R)$.
   - Ghi rõ ngưỡng SPACE là **`0.889`** trong phần ghi chú về chức năng Text Spacing.
4. **Slide 12 (Tổng kết & Hướng phát triển):**
   - Đã ghi rõ thành quả 3 mô hình ASL, cơ chế 3.0s, Trie + Qwen3:4B.
   - Phần hạn chế: Nhấn mạnh việc cần mở rộng thu thập tập Test Set độc lập cho nhánh Digits và đo đạc khoảng cách hình thái học Euclid trong giai đoạn nghiên cứu tiếp theo.
