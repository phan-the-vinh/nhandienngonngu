# BÁO CÁO KIỂM THỬ VÀ NGHIỆM THU ĐỒ HỌA SLIDE 7 — LINE CHART
## DỰ ÁN: HỆ THỐNG NHẬN DIỆN KÝ HIỆU ASL VÀ GỢI Ý VĂN BẢN (SIC CAPSTONE)
**Tập tin nguồn:** `SIC_Capstone_Presentation_English_Final_v2.pptx`  
**Tập tin xuất bản PPTX:** `SIC_Capstone_Presentation_English_Final_v3.pptx`  
**Tập tin xuất bản PDF:** `SIC_Capstone_Presentation_English_Final_v3.pdf`  
**Workspace thực thi:** `D:\NhanDienCuChi`  
**Thời điểm thực hiện:** 16/11/2026 09:50:00 UTC+7  

---

## 1. TỔNG QUAN NHIỆM VỤ (EXECUTIVE SUMMARY)

Thực hiện yêu cầu tái cấu trúc trực quan hóa dữ liệu trên **Slide 7** (*Dataset Accounting & Editable Train / Validation / Test Split Chart*):
1. **Thay thế biểu đồ thanh xếp chồng cũ bằng Biểu đồ Đường có điểm mốc (Native Editable Line Chart with Markers):**
   - Sử dụng định dạng biểu đồ gốc của Microsoft PowerPoint (`LINE_MARKERS` / 65), tuyệt đối không dùng hình ảnh tĩnh, ảnh chụp màn hình hay vector vẽ thủ công.
   - Người dùng hoàn toàn có thể nhấp chuột phải và chọn "Edit Data" để chỉnh sửa dữ liệu thông qua giao diện Excel nhúng của PowerPoint.
2. **Triển khai phương án ngữ nghĩa đã được người dùng phê duyệt (Option B):**
   - Biểu diễn tỷ lệ phần trăm chuẩn hóa (% của tổng số mẫu từng nhánh: J/Z 640 chuỗi, 123 899 mẫu, ABC 89,472 mẫu).
   - Thiết lập các đường riêng biệt với màu sắc chuẩn nhận diện Samsung, đường nét rõ ràng và hình dạng điểm mốc phân biệt (Circle, Square, Triangle, Diamond).
   - Xử lý giá trị Not Applicable (N/A) thành khoảng trống/khoảng ngắt (blank/gap) đối với Regression Reference ở nhánh J/Z và ABC; bảo toàn giá trị $0.0\%$ thực tế đối với Test Set của nhánh 123.
3. **Bảo tồn toàn vẹn dữ liệu khoa học:**
   - Giữ nguyên vẹn 100% nội dung giải trình kiểm toán dữ liệu tại hộp văn bản `TextBox 9`, sửa triệt để lỗi hiển thị ký tự gạch dưới `_` trong `ASL_HG` và `test_v1.csv`.
   - Bổ sung ghi chú rõ ràng về mẫu số chuẩn hóa (denominators) ngay trên slide.
   - Không làm thay đổi bất kỳ slide nào khác (Slides 1–6 và 8–13 giữ nguyên vẹn 100%).

---

## 2. BẢNG ĐỐI CHIẾU XÁC MINH SỐ LIỆU GỐC (SOURCE DATA AUDIT MATRIX)

Trước khi thực hiện chỉnh sửa, toàn bộ các số liệu đã được đối soát chéo với các manifest và tệp nguồn trong dự án:

| Nhánh mô hình | Training | Validation | Test | Regression Reference | Tổng số mẫu ghi nhận | Nguồn đối soát xác thực |
|---|---:|---:|---:|---:|---:|---|
| **J/Z (Chuyển động)** | **512** *(80.0%)* | **64** *(10.0%)* | **64** *(10.0%)* | *Không áp dụng (N/A)* | **640 chuỗi** *(J=265, Z=375)* | `data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json` |
| **123 (Chữ số tĩnh)** | **549** *(61.1%)* | **100** *(11.1%)* | **0** *(0.0%)* | **250** *(27.8%)* | **899 mẫu** | `data/metadata/digits/digits_final_freeze_manifest.json` |
| **ABC (Chữ cái tĩnh)**| **62,648** *(70.0%)*| **13,412** *(15.0%)*| **13,412** *(15.0%)*| *Không áp dụng (N/A)* | **89,472 mẫu** | `models/letters/production/evaluation_summary.json` |

### Các điểm xác thực cốt lõi:
- **Nhánh J/Z:** Tổng 640 chuỗi video SigNN; tỷ lệ phân bổ phân tầng $80\% - 10\% - 10\%$ ($512 - 64 - 64$).
- **Nhánh 123:** Tổng 899 mẫu gồm 549 Training Pool ($349\text{ custom} + 170\text{ Ardamavi} + 30\text{ ASL\_HG}$); 100 mẫu External Validation độc lập (10 mẫu/lớp); 0 mẫu Test set cập nhật; 250 mẫu Regression Reference trích từ `test_v1.csv` ban đầu.
- **Ranh giới phân tách:** 100 mẫu External Validation hoàn toàn độc lập với tập $15\%$ Internal Validation (~83 mẫu) do Scikit-learn tự trích xuất để kiểm tra Early Stopping; 250 mẫu Regression Reference là đối chứng hồi quy đóng băng, hoàn toàn không dùng để huấn luyện hay tinh chỉnh.
- **Nhánh ABC:** Tổng 89,472 mẫu trích xuất từ MediaPipe landmark split v1; phân chia $70\% - 15\% - 15\%$ ($62,648 - 13,412 - 13,412$).

---

## 3. THIẾT KẾ VÀ ĐẶC TẢ KỸ THUẬT BIỂU ĐỒ LINE CHART (OPTION B)

### 3.1. Cấu hình Biểu đồ gốc PowerPoint (Chart 10)
- **Loại biểu đồ:** Native PowerPoint Line with Markers (`XL_CHART_TYPE.LINE_MARKERS` / mã định danh enum `65`).
- **Tiêu đề biểu đồ:** `Dataset Split Proportions (% of Branch Total) — Editable Line Chart` (Font: `Samsung Sharp Sans`, 13pt, Bold, màu Navy `#0A235A`).
- **Trục hoành (Category X-Axis):**
  - Category 1: `J/Z Dynamic\n(640 seqs)`
  - Category 2: `123 Static\n(899 samples)`
  - Category 3: `ABC Static\n(89,472 samples)`
  - Font: `Calibri`, 9.5pt, Bold.
- **Trục tung (Value Y-Axis):**
  - Tiêu đề trục: `Percentage of Total Accounting (%)` (Font: `Calibri`, 9.5pt, Bold).
  - Thang đo: Tối thiểu $0.0$ ($0\%$), Tối đa $1.0$ ($100\%$), Bước nhảy chính (Major Unit): $0.2$ ($20\%$).
  - Định dạng hiển thị số: `0%` (`0%`, `20%`, `40%`, `60%`, `80%`, `100%`).
  - Đường lưới chính (Major Gridlines): Bật hiển thị dạng đường ngang thanh mảnh.
- **Chú giải (Legend):**
  - Vị trí: Đặt phía trên biểu đồ (`TOP`), không chèn đè lên vùng vẽ (`include_in_layout = False`).
  - Font: `Calibri`, 9.5pt.

### 3.2. Đặc tả 4 chuỗi dữ liệu (Data Series) và Xử lý N/A
| Tên chuỗi (Series Name) | J/Z Dynamic (640) | 123 Static (899) | ABC Static (89,472) | Màu sắc đường & mốc | Kiểu điểm mốc (Marker) | Kích thước mốc | Độ dày nét |
|---|:---:|:---:|:---:|---|:---:|:---:|:---:|
| **Training Pool** | **80.0%** *(0.800)* | **61.1%** *(0.611)* | **70.0%** *(0.700)* | Samsung Navy (`#0F4BA0`) | Tròn (`CIRCLE`) | 8 pt | 2.5 pt |
| **External Validation** | **10.0%** *(0.100)* | **11.1%** *(0.111)* | **15.0%** *(0.150)* | Cyan / Teal (`#0096A5`) | Vuông (`SQUARE`) | 8 pt | 2.25 pt |
| **Test Set** | **10.0%** *(0.100)* | **0.0%** *(0.000)* | **15.0%** *(0.150)* | Slate / Indigo (`#505A82`) | Tam giác (`TRIANGLE`) | 8 pt | 2.25 pt |
| **Regression Reference* (123)** | *N/A (Blank)* | **27.8%** *(0.278)* | *N/A (Blank)* | Coral Amber (`#DC5F1E`) | Thoi (`DIAMOND`) | 10 pt | 2.0 pt |

*Cơ chế Missing-Value:* Trong bảng tính Excel nhúng của PowerPoint, ô J/Z và ABC của Regression Reference được để trống (`None` / blank cell). PowerPoint tự động coi đây là điểm ngắt (gap), chỉ hiển thị duy nhất 1 điểm mốc hình thoi màu cam nổi bật tại vị trí $27.8\%$ của nhánh 123 mà không nối đường giả tạo sang các nhánh khác. Ngược lại, Test Set của nhánh 123 là số $0.0\%$ thực tế và được vẽ chính xác tại vạch đáy $0\%$.

---

## 4. BẢO TỒN VÀ TINH CHỈNH NỘI DUNG GIẢI TRÌNH (TEXTBOX 9)

Hộp văn bản `TextBox 9` được giữ nguyên vẹn tất cả nội dung kỹ thuật, đồng thời được tối ưu hóa hiển thị:
1. **Khắc phục lỗi phông chữ:** Thay thế phông hiển thị để ký tự gạch dưới `_` trong `ASL_HG` và `test_v1.csv` hiển thị hoàn hảo, triệt tiêu hoàn toàn hiện tượng ký tự rác (missing glyph box) của phông chữ cũ.
2. **Căn chỉnh layout thẻ chứa (Cards):**
   - Thẻ biểu đồ trên (`Rounded Rectangle 6`): `top = 1,720,000`, `height = 3,050,000`.
   - Thẻ văn bản dưới (`Rounded Rectangle 8`): `top = 4,850,000`, `height = 1,380,000`.
   - Toàn bộ 5 đoạn văn bản nằm trọn vẹn bên trong đường viền vàng kim, có đệm lề trên/dưới/trái/phải cân đối, hoàn toàn không bị tràn biên hay đè lấn chân trang.
3. **Bổ sung ghi chú chuẩn hóa mẫu số:**
   `• Line Chart Normalization Denominators: Percentages plotted above use total accounted samples per branch: J/Z (640 seqs), 123 (899 samples), ABC (89,472 samples). Regression Ref is unique to 123 (N/A for J/Z & ABC).`

---

## 5. KẾT QUẢ THỰC THI BỘ KIỂM TOÁN QA 11 TIÊU CHÍ (qa_suite.py)

| STT | Hạng mục kiểm tra | Tiêu chuẩn đánh giá | Kết quả thực tế | Trạng thái |
|:---:|---|---|---|:---:|
| **1** | **Mở lại tập tin PPTX đã xuất** | Nạp thành công qua thư viện `pptx` và đối tượng COM PowerPoint | Nạp mượt mà, không gặp lỗi cấu trúc XML | **PASS** |
| **2** | **Tổng số Slide trong bài thuyết trình** | Phải chính xác 13 slides | `13/13 slides` | **PASS** |
| **3** | **Kiểm tra loại biểu đồ Slide 7** | Phải là biểu đồ đường nguyên bản (`LINE_MARKERS` / 65) | `Chart 10`: `LINE_MARKERS (65)` | **PASS** |
| **4** | **Kiểm tra dữ liệu nhúng (Workbook)** | Danh mục và các chuỗi phải hiển thị đúng tên và cấu trúc | Đầy đủ 3 Categories và 4 Series chuẩn | **PASS** |
| **5** | **Kiểm tra giá trị từng điểm mốc** | Khớp chính xác với bảng tỷ lệ % đã phê duyệt | Train (0.8, 0.611, 0.7); Val (0.1, 0.111, 0.15); Test (0.1, 0.0, 0.15); Ref (None, 0.278, None) | **PASS** |
| **6** | **Xác minh tính trung thực N/A vs. Zero** | N/A phải là ô trống (gap); Zero phải ở vạch 0 | Ref J/Z=None, Ref ABC=None; Test 123=0.0% | **PASS** |
| **7** | **Xuất và kiểm tra tập tin PDF** | Tập tin PDF được tạo lập hoàn chỉnh, dung lượng hợp lệ | `SIC_Capstone_Presentation_English_Final_v3.pdf` (1,255,494 bytes) | **PASS** |
| **8** | **Nghiệm thu hiển thị đồ họa Slide 7** | Xuất ảnh PNG độ phân giải cao và thẩm định trực quan | Hình ảnh sắc nét, màu sắc chuẩn Samsung, nhãn rõ ràng | **PASS** |
| **9** | **Kiểm tra tràn viền và lỗi định dạng** | Không cắt xén văn bản, không lỗi font, không tràn biên | TextBox 9 nằm gọn trong thẻ viền vàng, font chữ chuẩn xác | **PASS** |
| **10**| **Kiểm tra bảo toàn các slide khác** | Slides 1–6 và 8–13 phải nguyên vẹn 100% so với Final v2 | Khớp 100% từng shape, vị trí, kích thước và nội dung text | **PASS** |
| **11**| **Kiểm tra nhận diện thương hiệu & Bản quyền** | Header Unit 02, Slide 7 Title, Slide 13 Copyright nguyên vẹn | Tất cả thành phần thương hiệu được bảo toàn nguyên vẹn | **PASS** |

---

## 6. DANH MỤC TẬP TIN XUẤT BẢN VÀ ĐƯỜNG DẪN TRUY CẬP

Tất cả các tập tin bàn giao đã được khởi tạo trong workspace và sao chép đầy đủ sang thư mục `Downloads` của người dùng:

1. **Tập tin PowerPoint Final V3 (chứa Line Chart có thể chỉnh sửa):**
   - Workspace: [SIC_Capstone_Presentation_English_Final_v3.pptx](file:///D:/NhanDienCuChi/SIC_Capstone_Presentation_English_Final_v3.pptx) *(1,757,285 bytes)*
   - Downloads: [C:/Users/Admin/Downloads/SIC_Capstone_Presentation_English_Final_v3.pptx](file:///C:/Users/Admin/Downloads/SIC_Capstone_Presentation_English_Final_v3.pptx)
2. **Tập tin PDF xuất bản toàn bộ bài thuyết trình Final V3:**
   - Workspace: [SIC_Capstone_Presentation_English_Final_v3.pdf](file:///D:/NhanDienCuChi/SIC_Capstone_Presentation_English_Final_v3.pdf) *(1,255,494 bytes)*
   - Downloads: [C:/Users/Admin/Downloads/SIC_Capstone_Presentation_English_Final_v3.pdf](file:///C:/Users/Admin/Downloads/SIC_Capstone_Presentation_English_Final_v3.pdf)
3. **Ảnh nghiệm thu trực quan Slide 7 (Slide7.PNG):**
   - [scratch/final_v3_png/Slide7.PNG](file:///D:/NhanDienCuChi/scratch/final_v3_png/Slide7.PNG)
4. **Báo cáo kiểm thử QA Markdown:**
   - [SIC_Capstone_Slide7_LineChart_QA_Report.md](file:///D:/NhanDienCuChi/SIC_Capstone_Slide7_LineChart_QA_Report.md)
   - [reports/SIC_Capstone_Slide7_LineChart_QA_Report.md](file:///D:/NhanDienCuChi/reports/SIC_Capstone_Slide7_LineChart_QA_Report.md)

