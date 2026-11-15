# Thiết kế model v2: P ngang và bảo toàn 24 lớp ASL

Ngày: 2026-09-23. Trạng thái: **DESIGN_ONLY — chưa thu thập, chưa triển khai, chưa huấn luyện**.

Khuyến nghị: thử bổ sung dữ liệu có kiểm soát trước với nguyên 63 đặc trưng; so sánh riêng các biểu diễn kết hợp đặc trưng hình dạng với đặc trưng hướng. Không có bảo đảm cải thiện trước khi đánh giá. Các quota và tiêu chí bên dưới là đề xuất để đóng băng trước thử nghiệm, không phải kết quả đo hay thay đổi đã áp dụng cho v1.

Alphabet convention: Standard ASL manual alphabet A-Z; J and Z handled as dynamic gestures; M and N use standard ASL forms.

24 lớp: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.

## 1. Căn cứ thực tế và mức độ kết luận

Đã đọc code `AI/landmark_utils.py`, `AI/static_asl_camera.py`, `AI/train_compare_mlp_validation.py`; các manifest/cấu hình/summary huấn luyện; audit debug và phân tích P/Q. Hash của model/scaler/mapping khớp training manifest; hash của các báo cáo và 144 file debug khớp manifest hiện có. Danh sách 174 file nguồn được ghi trong `evidence_manifest.json` để kiểm tra trước/sau việc viết tài liệu.

| V1 | Cấu hình đã lưu |
|---|---|
| Model chính thức | Baseline, best epoch 143; kết thúc epoch 163 |
| MLP | 63 → 64 → 32 → 24; ReLU, Adam |
| Huấn luyện | batch 256; LR 0.001; alpha 0.0001; tối đa 250 epoch; seed 42 |
| Early stopping | VALIDATION bên ngoài; patience 20; min delta macro-F1 0.0001; không random-split nội bộ |
| Scaler | StandardScaler fit trên 62.591 mẫu TRAIN |
| VALIDATION đã lưu | accuracy 0.9908291083; macro-F1 0.9898618487 |
| Camera | mirror trước MediaPipe theo mặc định; image-normalized landmarks; giữ thứ tự 0–20, xyz |
| Làm mượt | trung bình xác suất 7 frame, tối thiểu 4 vote đồng ý; không thay model |

Hàm chuẩn hóa thực tế: `u_i = (p_i - p_0) / max_j ||p_j - p_0||`, flatten thành 63 số. Camera kiểm tra NaN/Inf và bàn tay suy biến trước khi gọi hàm chung. Công thức loại vị trí cổ tay và một hệ số scale chung, không căn chỉnh hướng, không chuẩn hóa handedness. Pipeline này được giữ nguyên.

| Điều kiện debug | Đúng theo ảnh | Đúng khi mỗi nhóm ứng viên có tổng trọng số bằng nhau |
|---|---:|---:|
| P/front | 16/16 = 100% | 100%, 14 đơn vị |
| P/horizontal | 6/16 = 37.5% | 53.70%, 9 đơn vị |
| Q/horizontal | 15/15 = 100% | 100%, 13 đơn vị |

47 ảnh là dữ liệu chẩn đoán **đã được quan sát**. Có 7 nhóm pHash ứng viên chứa 18 ảnh, không có file raw trùng hash. Các đơn vị này không được coi là 36 lần thực hiện độc lập: thiếu signer/session/bout metadata và pHash toàn ảnh có thể phản ánh nền giống nhau. Toàn bộ 47 mẫu tiếp tục được giữ nguyên, nằm ngoài train/dev/test chính thức của thử nghiệm v2; chỉ dùng làm regression chẩn đoán sau lựa chọn, không làm tiêu chí chọn seed/model.

Audit trước ghi nhận nhãn và correspondence của 47 mẫu phù hợp. Đây là **không phát hiện lỗi bằng quan sát overlay**, không chứng minh tọa độ z hoặc mọi khớp là ground truth. Không audit lại ảnh ở bước thiết kế này.

Khoảng cách centroid trong không gian scaler TRAIN: P/front–P/horizontal 10.73; P/horizontal–Q/horizontal 7.85; P/front–Q/horizontal 14.66. Khi cân bằng nhóm, ba số là 11.08, 7.75, 14.57. 15/16 P ngang gần centroid Q ngang hơn centroid P/front. Tuy nhiên kNN trên TRAIN P/Q vẫn có đa số hàng xóm P cho 11/16 mẫu tại k=15. Do đó khoảng cách centroid không chứng minh P/Q không thể phân biệt, cũng không chứng minh thiếu góc nhìn là nguyên nhân duy nhất.

TRAIN P có 2.290 ảnh/1.563 nhóm, Q có 2.341 ảnh/1.548 nhóm theo báo cáo đã lưu; không có cột góc nhìn. Mẫu minh họa cũ có cả dáng nhìn ngang, nên không kết luận TRAIN hoàn toàn thiếu P ngang. Giả thuyết cần thử: thiếu độ phủ người/góc/điều kiện camera, representation nhạy với hướng/chiếu phối cảnh và ranh giới MLP. Không dùng confidence 99% để xác nhận nhãn.

Các thư mục frozen/split hiện không cho phép liệt kê trong sandbox. Nhiệm vụ thiết kế không cần mở chúng: các số liệu legacy được dẫn từ manifest/báo cáo đọc được, không tuyên bố đã tái kiểm tra nội dung các CSV này. TEST v1 không được mở hay đánh giá lại.

## 2. Góc camera khác hướng có ý nghĩa ASL

Thay vị trí camera trong khi người ký giữ cùng ký hiệu là thay góc quan sát; thay hướng tay so với cơ thể/người đối diện có thể đổi chữ. P/K và Q/G dùng hình dạng tay gần nhau nhưng phân biệt bằng hướng; H/U cũng cần được bảo vệ. Tham chiếu: [ASL University, fingerspelling](https://www.lifeprint.com/asl101/pages-layout/fingerspelling.htm).

Không định nghĩa “P ngang” bằng cách yêu cầu xoay K hoặc P một số độ cố định. Người thông thạo ASL xác nhận cả handshape và hướng ký hiệu so với cơ thể/người đối diện; ghi riêng `camera_view`, `hand_orientation_body` và `image_presentation`. Hướng projected trên ảnh không tự quyết định nhãn. Với bàn tay đổi hướng thành K/G, phải ghi nhận ký hiệu thật qua review hoặc giữ unresolved; không kế thừa nhãn P/Q của prompt.

Các tọa độ MediaPipe image-landmarks có x theo chiều rộng ảnh, y theo chiều cao và z là ước lượng độ sâu tương đối; khác với output world-landmarks. Vì vậy Euclidean distance và căn chỉnh trên 63 số hiện tại là mô tả hình học xấp xỉ, không phải phép đo 3D hiệu chuẩn. Nguồn: [Google Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python). Metadata hiện không có world-landmarks, camera extrinsics hoặc trục cơ thể. Không thể luôn phân biệt camera xoay với tay xoay chỉ từ một bàn tay; giới hạn deployment ban đầu là camera được giữ thẳng, có reference cơ thể trong clip review.

Không dùng augmentation xoay toàn ảnh/landmark, lật nhãn hoặc random 3D rotation mặc định. Một phép xoay số học không tái tạo occlusion của góc camera thật. Nếu sau này thử augmentation phải có thử nghiệm riêng, semantic approval theo lớp và version mới; không thuộc ma trận chính.

## 3. Biểu diễn ứng viên, giữ tín hiệu hướng

R0 — **63 số nguyên bản**: lựa chọn đầu tiên cùng dữ liệu bổ sung; chi phí triển khai thấp nhất.

R1 — **global + hình dạng, 83 số**: giữ nguyên 63 số; nối 10 cosine góc gập (thumb: 1–2–3, 2–3–4; các ngón khác: m–m+1–m+2–m+3 được tách thành hai triple liên tiếp, m=5,9,13,17) và 10 khoảng cách giữa các cặp đầu ngón {4,8,12,16,20}. Tính trên u_i, không fit thống kê từ dev/test. Đây là descriptor xấp xỉ; không gọi là góc giải phẫu đo chính xác. Đoạn xương bằng 0 phải báo feature invalid; chỉ tập TRAIN của các arm retrain dùng cùng eligibility mask để so sánh công bằng, báo số không được dùng và vẫn giữ nguồn. Đánh giá end-to-end giữ toàn bộ bout có nhãn rõ: R1 invalid được chấm NO_PREDICTION riêng cho R1, không làm mất hay biến thành sai kết quả REF/R0 hợp lệ. Conditional evaluation trên common-valid samples chỉ là phụ.

R2 — **global + local + hướng, 133 số**, ưu tiên thấp hơn: từ u_i, đặt e_x theo u_5−u_17; Gram–Schmidt vector u_9 theo e_x để lấy e_y; e_z=e_x×e_y. Local_i=(u_i·e_x,u_i·e_y,u_i·e_z). Ghép 63 global + 63 local + 6 thành phần của e_x/e_y + 1 cờ basis-valid. Giữ dấu, thứ tự index/pinky và mirror metadata; không dùng trị tuyệt đối để xóa orientation. Ngưỡng số học cố định 1e-8 trên norm vector đã chuẩn hóa chỉ chống chia cho 0, không phải quality threshold v1. Basis suy biến: local và basis đặt 0, valid=0, giữ nguyên global; không loại ảnh chỉ cho một arm. Tỷ lệ suy biến được báo riêng.

R1/R2 được tính từ f1…f63 đã lưu, không cần chạy lại extraction/MediaPipe cho legacy. Kết quả phải là artefact v2 riêng, không ghi lại cột frozen. World-landmarks/body-relative orientation là hướng mở rộng sau, vì legacy thiếu đầu vào tương ứng; không trộn world coordinates vào checkpoint vốn học image coordinates.

R2 không hứa hẹn bất biến góc nhìn vật lý. Việc giữ global và basis giúp model còn khả năng dùng hướng; không tự bảo đảm model sẽ dùng đúng, nên các gate P/K, Q/G, H/U là bắt buộc. Không có arm local-only hoặc invariant-only được phép phát hành.

## 4. Kế hoạch thu thập đề xuất

**40 người × 2 phiên khác ngày = 80 phiên; 9.600 bouts, tối đa 28.800 frame chọn trước.** Chốt 20 người TRAIN mới, 8 người DEV, 12 người TEST mới trước thu thập (seed phân bổ 42). Giữ cả hai phiên và hai tay của một người cùng partition. Không yêu cầu người không thuận tay tạo tư thế thiếu tự nhiên; cố gắng có ≥25% người thuận tay trái và phân bổ giữa các cohort, ghi rõ thiếu hụt thực tế.

Mỗi phiên:

- P, Q, K, G: mỗi lớp 3 điều kiện × 5 bouts = 15 bouts/lớp, tổng 60.
- 20 lớp còn lại: mỗi lớp 3 bouts, tổng 60. Một bout frontal, một oblique-left, một oblique-right; M/N/T/H/U được review cùng mức với các lớp khác.
- Tổng 120 bouts/phiên. Một bout là thả tay về nghỉ rồi tạo ký hiệu lại, giữ khoảng 1.5 giây; nhấn lưu thêm trong cùng lần giữ tay không tạo bout mới. Các lần quay đa camera đồng thời cùng `gesture_bout_id` và `multi_view_event_id`.

Ba điều kiện trọng tâm: FRONT, HORIZONTAL_LEFT, HORIZONTAL_RIGHT. Đây là **stratum do người review xác nhận**, không phải nhãn auto từ góc. FRONT có 5 bouts camera frontal với cách trình bày chính diện tham chiếu đã duyệt. Riêng mỗi stratum HORIZONTAL_LEFT/RIGHT có 5 bouts: 3 bouts camera frontal với cách trình bày ngang còn đúng ASL và 2 bouts camera oblique khoảng 30–45° cùng phía, tay giữ hướng ký hiệu phù hợp. Nếu cách trình bày đó không hợp lệ cho K/G, dùng K/G chuẩn ở cùng vị trí camera và ghi hình dạng projected thật; không ép K/G thành dáng P/Q để đủ quota. Phân tích P ngang phải lọc `image_presentation=horizontal` đã xác nhận, không chỉ lọc tên điều kiện. Góc là hướng dẫn rig, không phải rule phân loại.

Reviewer ASL phải duyệt ví dụ chuẩn cho từng điều kiện trong pilot của TRAIN trước thu thập DEV/TEST. Nếu không có đủ mẫu P/Q ngang hợp lệ dưới thiết kế, sửa protocol có version trước khi mở cohort đánh giá; không tự đổi nhãn để đủ số.

Phiên 1: ánh sáng trong nhà ổn định, nền đơn giản. Phiên 2: nền/ánh sáng/khoảng cách khác trong điều kiện sử dụng, camera vẫn upright. Đề xuất khoảng cách 0.5–1.0m, tối thiểu hai thiết bị được phân bố vào mọi cohort; không để device/lighting đồng nghĩa với label hoặc partition. Ngẫu nhiên thứ tự P/Q/K/G, không thu cả P trong một điều kiện ánh sáng và Q trong điều kiện khác. Giữ camera reference rộng gồm cổ tay/cẳng tay/phần thân trên khi được đồng ý; thông tin này phục vụ review, không là feature MLP trong ma trận chính.

Lưu clip liên tục để xem tạo/giữ/thả tay. Chọn tối đa 3 frame/bout ở 0.25, 0.75, 1.25 giây tính từ `hold_start_ms` được reviewer xác định không xem model prediction. Không chọn ảnh theo confidence. TRAIN tối đa 14.400 frame/4.800 bouts; DEV 5.760/1.920; TEST 8.640/2.880. Quota P+Q là 2.400 bouts/7.200 frame; cả K/G thêm 2.400 bouts. Chi tiết tính toán: `collection_plan.csv`.

TEST mới dự kiến có 240 P-horizontal và 240 Q-horizontal bouts, mỗi lớp 120 frontal; K/G mỗi lớp 360 bouts; mỗi lớp khác 72 bouts. Đây là ngân sách khởi đầu, **không là chứng minh đủ statistical power** cho margin 1–2 điểm phần trăm. Ưu tiên tăng số người độc lập hơn tăng frame cùng bout.

## 5. Metadata, review và ranh giới dữ liệu

`metadata_dictionary.csv` mô tả field, tính bắt buộc, giá trị null và validation. Phải phân biệt `prompt_label` với `verified_asl_label`; cả reviewer và phép kiểm tra landmark không nhìn prediction lúc gán nhãn. Hai reviewer độc lập cho P/Q/K/G và mọi bất đồng; lưu adjudication. UNKNOWN/UNCERTAIN tiếp tục lưu nhưng chưa đưa vào corpus supervised. Không đổi hoặc xóa ảnh nguồn.

Hierarchy: signer → session → bout → camera/frame. `group_id` cho collection mới bao gồm component của cùng bout, nguồn capture và duplicate đã xác minh; partition được quyết định ở **signer**, mạnh hơn group. Duplicate cross-signer cần điều tra copy/metadata; nếu xác nhận cùng nguồn, cùng component và không được xuất hiện qua partition. Cross-label near-match chỉ đánh dấu; không tự merge nhãn, không tự loại.

Legacy TRAIN được giữ partition/group-map hiện có. D0 = legacy TRAIN; D1 = legacy TRAIN + new TRAIN signers. Legacy VALIDATION chỉ là regression dev đã biết; legacy TEST và 47 debug không thuộc nguồn train/selection. Near-duplicate checks sau này chỉ đọc new-vs-legacy TRAIN để chống reuse, không cần đọc TEST v1 để thiết kế v2.

Không có signer metadata cho phần internet legacy: signer-disjoint trong collection mới **không chứng minh** người TEST chưa xuất hiện trong legacy. Ghi nhận provenance và việc người mới đã đóng góp/đăng dữ liệu ASL trước đây; câu trả lời người tham gia không tự đủ chứng minh tuyệt đối. Chỉ tuyên bố “signer-disjoint trong collection mới” trừ khi có provenance xác minh. Nếu cần claim subject-independent mạnh, thiết kế thêm arm new-only với toàn bộ provenance signer kiểm soát được, được chốt trước test; không coi v1 legacy là baseline có cùng bảo đảm đó.

Nếu không đạt 40 người, có thể làm pilot trên TRAIN/DEV người ít hơn nhưng không kết luận tổng quát hóa người mới. Không chuyển người/bout của DEV/TEST sang TRAIN để cải thiện số liệu.

## 6. Ma trận thực nghiệm và lựa chọn

Thực hiện sau phê duyệt, không chạy trong nhiệm vụ này:

| Arm | Dữ liệu | Representation | Mục đích |
|---|---|---|---|
| REF | checkpoint v1 epoch 143 | R0 | baseline inference cố định |
| E00 | D0 | R0 | kiểm soát việc retrain/randomness |
| E10 | D1 | R0 | chỉ bổ sung dữ liệu, ưu tiên đầu |
| E01 | D0 | R1 | chỉ thêm descriptor hình dạng |
| E11 | D1 | R1 | dữ liệu + descriptor |
| E02 | D0 | R2 | local+global, thí nghiệm phụ |
| E12 | D1 | R2 | dữ liệu + local+global, thí nghiệm phụ |

So sánh E10−E00, E01−E00, E11−E10 và tương tác (E11−E01)−(E10−E00) trên cùng DEV. Không so sánh E10 với REF rồi quy toàn bộ khác biệt cho dữ liệu, vì đã thay quá trình chọn checkpoint.

Giữ hidden 64/32, activation/optimizer/batch/LR/alpha/max epochs/patience như v1. R0/R1/R2 có 6.968/8.248/11.448 tham số do input dimension khác; báo chênh lệch capacity và latency, không gọi là mô hình hoàn toàn cùng kích thước. Giữ mọi lớp, không class weighting/sampling mới ở ma trận chính để tránh trộn tác động. Ba frame cố định/bout hạn chế overweight trong data mới; báo phân bố class/group và ảnh hưởng legacy redundancy. Nếu muốn class/bout weighting, đăng ký arm riêng và áp dụng cùng công thức cho mọi representation, không âm thầm bật cho E11.

Seeds 42,43,44 cho mỗi arm (18 run nếu làm đủ 6 arm). Một scaler riêng fit **training partition của chính arm**, dùng đúng scaler đó cho DEV; không fit trên TEST. Các cột global scaler thống nhất giữa representation trên cùng D nếu dùng cùng row set. Giữ feature order/version, mirror và extractor tương thích, hash tất cả artefact.

Chọn checkpoint bằng macro-F1 24 lớp trên new DEV có trọng số đều theo signer/session/bout; không sklearn internal validation. Lưu best checkpoint và `best_epoch` riêng; patience tính bằng số epoch không cải thiện >0.0001 so với anchor early-stop, không dùng số epoch cuối làm best. Anchor độc lập là semantics v2 cần version/test, khác cách so với previous-best trong script v1; không sửa lịch sử hoặc script v1. Tie-break macro-F1, minimum recall, accuracy; log đầy đủ để kiểm tra.

Mỗi arm phải qua guard DEV trước, gồm legacy VALIDATION regression đã biết. So sánh trung bình/độ lệch giữa 3 seed; không chọn seed tốt nhất. Nếu đủ điều kiện, chọn arm có mean DEV macro-F1 cao nhất, tie ≤0.0001 ưu tiên mean P-horizontal recall rồi representation đơn giản hơn. Checkpoint release của arm là **seed 42 đã đăng ký trước**, phải tự qua gate DEV; nếu seed 42 fail thì arm không đạt, không thay seed. Không refit với DEV sau selection trong protocol này.

## 7. Đánh giá camera, gate và ý nghĩa thống kê

Chi tiết máy đọc được trong `evaluation_protocol.json`. Đơn vị chính là một bout tại **frame giữa hold (0.75s)**, không phải 3 frame như ba lần thử độc lập. Khi frame tại mốc thiếu landmark/prediction, tính sai cho end-to-end; vẫn báo separately conditional classifier trên phần hợp lệ. Không loại lỗi MediaPipe khỏi mẫu số end-to-end. Nhãn semantic không xác định thì không có ground truth để chấm accuracy; báo số/bout và tỷ lệ unresolved riêng, không coi là prediction đúng.

Recall chính là trung bình đều theo signer, rồi session, rồi bout trong từng lớp/stratum được chấm. Với macro-F1, mỗi bout có trọng số `1 / (số signer × số session của signer × số bout đã xác nhận trong session)` trên toàn cohort, **không thêm class balancing** vào confusion matrix. Ma trận có 24 dòng nhãn thật × 25 cột dự đoán (24 chữ + NO_PREDICTION); cột cuối tính vào false negatives cho từng lớp. Lấy trung bình F1 của đúng 24 lớp. Không dùng API chỉ liệt kê 24 label rồi âm thầm bỏ outcome thất bại. Báo kèm pooled counts, per-signer/session và frame-level phụ; nếu một signer thiếu stratum/lớp cần đánh giá, báo quota thiếu và không âm thầm thay mẫu số để pass gate.

So sánh v1 và v2 **trên cùng** các bout của DEV và TEST mới. Độ bất định: paired signer-cluster bootstrap 10.000 lần, giữ nguyên sessions/bouts trong người; seed 20260923. CI từ 240 bouts không được tính như 240 người độc lập. Nếu đồng thời khẳng định nhiều lớp không suy giảm, dùng simultaneous 95% bounds với bootstrap max-statistic đã đóng băng, không lấy 24 CI riêng để suy thành bảo đảm chung. Khi không có lỗi/không có paired discordance hoặc mọi signer có cùng kết quả, bootstrap có thể cho CI zero-width giả tạo: không dùng CI này để thông qua margin hẹp. Nếu chưa có phương pháp boundary-valid bảo toàn clustering được đăng ký trước test, metric/gate đó là **INCONCLUSIVE**, báo counts và tăng kế hoạch số người dựa trên DEV. Không đổi phương pháp CI sau khi xem TEST.

Gate đề xuất, chốt trước training, áp dụng cả DEV và final test:

- P-horizontal recall ≥95%, lower 95% CI ≥90%, và lower 95% CI của chênh lệch paired v2−v1 >0.
- Q-horizontal recall ≥95%, point recall không thấp hơn v1, lower 95% CI của chênh lệch >−2 điểm phần trăm.
- Macro-F1 24 lớp point không thấp hơn v1, lower 95% CI chênh lệch >−1 điểm phần trăm.
- Recall từng lớp và P/front point không thấp hơn v1. Với từng lớp dùng simultaneous lower bound chênh lệch >−2 điểm phần trăm; nếu thiếu dữ liệu thì INCONCLUSIVE. Không dùng guard cũ “cho giảm 10 điểm” của v1 cho quyết định này.
- Confusion P→K, K→P, Q→G, G→Q, P→Q, Q→P, H→U, U→H point rate không tăng; simultaneous upper CI của mức tăng <2 điểm phần trăm. Chấm trên mọi bout thuộc original label tương ứng.

Margin 1/2 điểm là **độ bất định cho kiểm định non-inferiority**, không cho phép point estimate Q hoặc các lớp giảm mà vẫn gọi “không suy giảm”. Không thể chứng minh suy giảm đúng bằng 0 trong population bằng bộ dữ liệu hữu hạn. FAIL hoặc CI quá rộng → không release/INCONCLUSIVE, không hạ gate sau khi xem TEST. Dùng variability của TRAIN/DEV để định cỡ cohort trước mở TEST; thu thêm người chưa quan sát nếu cần. Không tiếp tục lấy mẫu đến khi CI TEST vừa qua ngưỡng.

Camera replay cùng clip: báo instantaneous, smoother 7/4 giữ nguyên, switch/bout, thời gian từ hold_start đến stable đúng, tỷ lệ không ổn định/không có tay, p50/p95 end-to-end latency và FPS trên máy mục tiêu. Xóa trạng thái khi tay biến mất; camera regression test yêu cầu không giữ chữ cũ sau frame không-tay đã phát hiện. UI target dự kiến p95 xử lý ≤100ms và stable đúng trong ≤1s ở ≥95% bouts hợp lệ; đo máy trước, chốt budget trên DEV, không chỉnh smoother theo TEST. Không chọn model chỉ vì chữ Stable ít nhấp nháy nhưng sai ổn định.

## 8. Trình tự và điểm dừng

1. Duyệt thiết kế, prototype examples ASL trên TRAIN pilot và khóa protocol/metadata/cohort IDs. Chưa thu thập DEV/TEST để nhìn kết quả model.
2. Thu và review TRAIN/DEV với người/phiên độc lập; TEST do bên review tách biệt thu/kiểm tra chất lượng mà không lộ prediction/nhãn để tuning.
3. Đóng băng manifests mới, feature schemas, quotas, rule xử lý missing/unresolved, splits và gate. Kiểm tra union groups/identity/hash; xác nhận đủ precision theo DEV.
4. Chạy ablations trên TRAIN/DEV. Sửa bug/thiết kế còn được phép trước khi TEST mở, nhưng phải version và rerun so sánh công bằng; giữ v1 nguyên trạng.
5. Khóa một arm/checkpoint seed42/scaler/label map/smoother/protocol. Đánh giá v1 reference và candidate đúng một lượt trên TEST mới.
6. Pass mọi gate mới cân nhắc release riêng; fail/inconclusive thì giữ v1, báo kết quả, không chọn model khác theo TEST. Chu kỳ chỉnh model sau khi đã xem TEST cần cohort đánh giá mới chưa quan sát.

Không thu data, fit scaler, retrain, chạy MediaPipe, chia lại dataset hay thay model/công thức normalize trong bước tài liệu này. Không dùng TEST v1 để chọn kiến trúc, threshold, augmentation, seed hoặc hyperparameter.

## Tệp đi kèm

- `collection_plan.csv`: quota bouts/frames từng cohort và block lớp.
- `metadata_dictionary.csv`: schema thu thập và lineage/review.
- `experiment_matrix.csv`: ma trận đã đề xuất, chưa thực thi.
- `evaluation_protocol.json`: đơn vị đánh giá, gate, bootstrap, policy selection.
- `evidence_manifest.json`: snapshot nguồn đã đọc và phạm vi xác minh.
- `document_integrity.json`: kết quả đối chiếu tài liệu, phép tính quota và hash trước/sau.
