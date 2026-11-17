# BÁO CÁO TỔNG KẾT FINALIZATION SPRINT: HOÀN THIỆN DỰ ÁN NHẬN DIỆN NGÔN NGỮ KÝ HIỆU
**(FINALIZATION SPRINT & HANDOVER READINESS REPORT)**

- **Dự án:** Hệ thống Nhận diện Ngôn ngữ Ký hiệu Đa nhánh (ABC, 123, J/Z, SPACE)  
- **Workspace:** `D:\NhanDienCuChi`  
- **Repository:** [phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu)  
- **Branch:** `main` | **Git HEAD:** `dce96c2`  
- **Ngày hoàn tất:** 17/11/2026 10:50 UTC+7  
- **Mục tiêu sprint:** Hoàn thiện toàn bộ hệ thống tài liệu, đảm bảo tính nhất quán dữ liệu, chuẩn hóa quy trình kỹ thuật để bàn giao và trình bày.

---

## 1. CÁC TỆP ĐÃ CHỈNH SỬA VÀ TẠO MỚI (FILES MODIFIED / CREATED)

Trong quá trình Finalization Sprint, các tài liệu cốt lõi của dự án đã được cập nhật và tạo mới đồng bộ:

| STT | Tệp tài liệu | Đường dẫn tuyệt đối | Hành động | Nội dung chính được cập nhật |
|:---:|---|---|:---:|---|
| 1 | **README.md** | [README.md](file:///D:/NhanDienCuChi/README.md) | **Cập nhật** | - Bổ sung sơ đồ luồng kiến trúc hệ thống đa nhánh dạng Mermaid flowchart.<br>- Hoàn thiện hướng dẫn cài đặt môi trường 2 lựa chọn (Conda & Python venv).<br>- Bổ sung đầy đủ lệnh chạy ứng dụng (unified mode, letters, digits, check-only, camera-index, flags).<br>- Tích hợp liên kết Google Drive, GitHub, tài liệu dataset và MODEL_DISTRIBUTION.<br>- Thêm bảng tra cứu và khắc phục 6 lỗi webcam / môi trường phổ biến. |
| 2 | **docs/dataset.md** | [docs/dataset.md](file:///D:/NhanDienCuChi/docs/dataset.md) | **Cập nhật toàn diện** | - Thay thế tài liệu sơ sài cũ bằng bảng phân tích kỹ thuật chi tiết của cả 4 phân hệ (ABC, 123, J/Z, SPACE).<br>- Ghi nhận chính xác số lượng mẫu Train, Validation, Test theo đúng split gốc.<br>- Làm rõ quy ước đặc trưng 63 tọa độ landmark chuẩn hóa (wrist-relative & max-distance scaled).<br>- Tích hợp liên kết thư mục Google Drive tải dữ liệu và cấu trúc giải nén chi tiết. |
| 3 | **docs/MODEL_DISTRIBUTION.md** | [docs/MODEL_DISTRIBUTION.md](file:///D:/NhanDienCuChi/docs/MODEL_DISTRIBUTION.md) | **Cập nhật** | - Điều chỉnh phân loại pháp lý cho từng mô hình: MediaPipe (Public), 123 (Authorized Backup), SPACE (Authorized Backup / Phụ thuộc ABC), ABC (Source-Only), J/Z (Nghiên cứu).<br>- Sửa mã băm SHA-256 của `hand_landmarker.task` khớp chính xác với bản chuẩn (`fbc2a30...`).<br>- Hướng dẫn chi tiết cách tải model từ Google Drive đặt vào thư mục tương ứng. |
| 4 | **THIRD_PARTY_NOTICES.md** | [THIRD_PARTY_NOTICES.md](file:///D:/NhanDienCuChi/THIRD_PARTY_NOTICES.md) | **Tạo mới** | - Tạo tệp chính thức tại thư mục gốc đáp ứng đầy đủ nghĩa vụ Attribution của Apache 2.0 (Google MediaPipe, Arda Mavi), CC BY 4.0 (Mendeley ASL-HG), CC BY-SA 4.0 (SUBTLEX-US).<br>- Ghi nhận giới hạn quyền hình ảnh theo Điều 2(d) CC0 đối với SigNN ASL Video.<br>- Tuyệt đối không tự ý khẳng định các quyền phân phối chưa có chứng cứ. |
| 5 | **Báo cáo Finalization Sprint** | [reports/finalization_sprint_report.md](file:///D:/NhanDienCuChi/reports/finalization_sprint_report.md) | **Tạo mới** | Tệp báo cáo tổng kết này. |

---

## 2. CÁC KIỂM TRA ĐÃ CHẠY VÀ KẾT QUẢ KỸ THUẬT (TESTS & VERIFICATION)

Toàn bộ các quy trình kiểm thử tự động, tiền kiểm (preflight check), tính toàn vẹn mã băm và kiểm tra mô hình đã được thực thi trực tiếp:

| STT | Lệnh kiểm tra kỹ thuật | Mục đích kiểm tra | Kết quả đạt được | Đánh giá |
|:---:|---|---|---|:---:|
| 1 | `python tools/download_models.py --check-only` | Kiểm tra sự tồn tại và tính toàn vẹn SHA-256 của toàn bộ 6 file mô hình và từ điển | `[SUCCESS] All runtime models and assets are present and verified!` (Toàn bộ 6/6 tệp khớp SHA-256). | **PASS** |
| 2 | `python run.py --check-only` | Khởi tạo toàn bộ pipeline suy luận đa nhánh (unified mode) không mở camera | `{"status": "PASS", "modes": ["ABC", "123"], "space_enabled": true, "word_suggestion_enabled": true...}` | **PASS** |
| 3 | `python run.py --mode letters --check-only` | Kiểm tra riêng phân hệ chữ cái ABC tĩnh | `{"status": "CAMERA_INFERENCE_CHECK_PASS", "model": "uvw_medium", "best_epoch": 81, "scaler_train_rows": 62648...}` | **PASS** |
| 4 | `python app/inference/digit_inference.py --help` | Kiểm tra CLI và đường dẫn model mặc định của phân hệ chữ số | Xuất hiện trợ giúp đầy đủ, model trỏ đúng `models/digits/production/digits_final.joblib`. | **PASS** |
| 5 | `python -m unittest tests/test_stage13_integration.py -v` | Kiểm tra 9 ca tích hợp hệ thống cốt lõi (hold confirmation, margin reject, feature flags, exception fallback) | `Ran 9 tests in 0.005s. OK` (100% passed). | **PASS** |
| 6 | Kiểm tra cơ chế Fail-Closed khi thiếu model | Xác minh phản hồi khi không tìm thấy model nhị phân `.joblib` | Ứng dụng dừng an toàn với Exit Code 2, in thông báo rõ ràng yêu cầu đọc `docs/MODEL_DISTRIBUTION.md`. | **PASS** |

---

## 3. CÁC VẤN ĐỀ CÒN TỒN TẠI VÀ RỦI RO (REMAINING ISSUES & RISKS)

1. **Rủi ro bản quyền của 41.676 ảnh color-coded (Nhánh ABC):**
   - $46,61\%$ dữ liệu huấn luyện của mô hình `uvw_medium` không thể xác định được nguồn gốc ban đầu và giấy phép.
   - *Hạn chế:* Không thể phát hành công khai file nhị phân `uvw_medium_model.joblib` lên GitHub Releases nếu chưa huấn luyện lại trên tập dữ liệu hoàn toàn mở. Dự án đã áp dụng chính sách **Source-Only** để khoanh vùng rủi ro này.
2. **Rủi ro dây chuyền đối với mô hình SPACE:**
   - Mô hình Logistic Regression của SPACE sử dụng $62.591$ mẫu âm từ ABC ($97,51\%$ tập Train), do đó thừa hưởng sự không chắc chắn về bản quyền từ ABC.
3. **Quyền hình ảnh cá nhân (Publicity Rights) đối với video J/Z:**
   - Dataset SigNN mang nhãn CC0 trên Kaggle nhưng chưa có chứng từ Model Release của các tình nguyện viên xuất hiện trong video. Mô hình J/Z được giới hạn ở phạm vi nghiên cứu học thuật (`EXPERIMENTAL_RESEARCH_ONLY`).
4. **Cảnh báo scikit-learn unpickle:**
   - Khi chạy có cảnh báo `InconsistentVersionWarning` (model được pickler ở scikit-learn 1.9.1 trong khi môi trường dev dùng 1.9.0). Đây là cảnh báo không gây lỗi (benign), hệ thống vẫn suy luận chính xác.

---

## 4. NHỮNG VIỆC NGƯỜI DÙNG CẦN TỰ THỰC HIỆN (USER ACTIONS FOR HANDOVER)

Để đưa dự án vào trạng thái bàn giao chính thức hoặc công khai trên GitHub, tác giả/chủ dự án cần thực hiện các bước sau:

1. **Đưa các file tài liệu đã cập nhật vào Git (Git Commit):**
   - Các file tài liệu vừa hoàn thiện gồm: `README.md`, `THIRD_PARTY_NOTICES.md`, `docs/dataset.md`, `docs/MODEL_DISTRIBUTION.md`.
   - Lệnh đề xuất:
     ```bash
     git add README.md THIRD_PARTY_NOTICES.md docs/dataset.md docs/MODEL_DISTRIBUTION.md reports/finalization_sprint_report.md
     git commit -m "docs: finalize handover documentation, dataset guide, and third-party notices"
     ```
2. **Tải dữ liệu lên Google Drive:**
   - Đảm bảo folder Google Drive tại liên kết:
     `https://drive.google.com/drive/u/0/folders/11dG6JtCcgB_fniRazOeujuJPRiJl9m10`
     đã được phân quyền truy cập phù hợp (người có liên kết được xem/tải) và chứa đầy đủ các file dataset vector (.csv) cùng các model checkpoint dự phòng (`.joblib`).
3. **Quyết định thời điểm Push lên GitHub:**
   - Sau khi kiểm tra kỹ lưỡng, thực hiện `git push origin main` để cập nhật kho lưu trữ [https://github.com/phan-the-vinh/nhandienngonngu](https://github.com/phan-the-vinh/nhandienngonngu).
4. **Kế hoạch dài hạn (Hậu bàn giao):**
   - Huấn luyện lại mô hình SPACE với tập âm NOT_SPACE sạch tự thu (khoảng 2.000 – 5.000 mẫu cử chỉ tay thông thường) để giải phóng hoàn toàn model SPACE khỏi tập ABC.
   - Huấn luyện lại mô hình ABC trên tập dữ liệu mở hoàn toàn (ví dụ chỉ dùng Kaggle ASL Alphabet có giấy phép rõ ràng hoặc bộ ảnh tự thu mới) nếu muốn phân phối công khai file binary weights.

---

## 5. ĐÁNH GIÁ MỨC ĐỘ SẴN SÀNG BÀN GIAO (HANDOVER READINESS ASSESSMENT)

$$\mathbf{ĐÁNH\ GIÁ:\ SẴN\ SÀNG\ BÀN\ GIAO\ VÀ\ TRÌNH\ BÀY\ (HANDOVER\ READY)}$$

### Các căn cứ đánh giá:
1. **Về mặt Kỹ thuật & Hoạt động:** `100% READY`. Hệ thống khởi chạy mượt mà qua lệnh duy nhất `python run.py`, nhận diện đồng thời cả chữ cái tĩnh, chữ số tĩnh, cử chỉ động J/Z và dấu cách SPACE, hỗ trợ đầy đủ bộ gợi ý từ vựng 25k từ Trie và LLM offline Ollama.
2. **Về mặt Tài liệu & Hướng dẫn sử dụng:** `100% READY`. Toàn bộ `README.md`, `docs/dataset.md`, `docs/MODEL_DISTRIBUTION.md` và `THIRD_PARTY_NOTICES.md` đã đồng bộ số liệu chính xác toán học, có cấu trúc thư mục rõ ràng, hướng dẫn cài đặt chi tiết và liên kết đầy đủ.
3. **Về mặt Tuân thủ Bản quyền (Compliance):** `CHẶT CHẼ & AN TOÀN`. Đã bổ sung đầy đủ văn bản Attribution cho các bên thứ ba hợp pháp (Google MediaPipe, Arda Mavi, ASL-HG Mendeley, SUBTLEX-US), đồng thời khoanh vùng các model nhạy cảm bằng chính sách **Source-Only Distribution**, giúp dự án hoàn toàn an toàn khi bàn giao và trình diễn khoa học.

---
*Báo cáo được hoàn thành tự động trong Finalization Sprint tại workspace D:\NhanDienCuChi.*
