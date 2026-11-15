# BÁO CÁO STAGE 20: CẤU HÌNH KHỞI ĐỘNG HỆ THỐNG MỘT CHẠM (ONE-CLICK APPLICATION STARTUP)

**Ngày thực hiện:** 14/11/2026\
**Môi trường:** Conda `nn` (`<conda-env-nn>/python.exe`)\
**Workspace:** `<project-root>` (Primary) & `<project-mirror>` (Mirror)\
**Trạng thái:** HOÀN THÀNH — TOÀN BỘ KIỂM THỬ ĐẠT (PASS 49/49 TESTS)

---

## 1. TỔNG QUAN VÀ MỤC TIÊU

Mục tiêu của Stage 20 là cho phép người dùng khởi động toàn bộ hệ thống nhận diện thời gian thực thông qua nút **"Run Python File"** trực tiếp trên file [run.py](file:///<project-root>/run.py) trong VS Code, không cần phải mở PowerShell hay gõ các cờ dòng lệnh thủ công.

Khi nhấn Run trong [run.py](file:///<project-root>/run.py), hệ thống tự động:
1. Mở webcam và khởi động luồng nhận diện thời gian thực.
2. Bật tính năng nhận diện cử chỉ **SPACE** bằng cấu hình tương đương `--enable-space`.
3. Bật đầy đủ nhận diện **J/Z** với xác nhận commit (`--enable-experimental-jz`).
4. Khởi tạo và kích hoạt hệ thống **gợi ý từ (Offline Word Suggestion)** và **gợi ý câu (Sentence Generation)**.
5. Đảm bảo VS Code luôn thực thi bằng Python trong môi trường Conda `nn`, tuyệt đối không dùng `base`.
6. Giữ nguyên giao diện UI và luồng production hiện tại.

---

## 2. TRUY VẾT VÀ PHÂN TÍCH LUỒNG KHỞI ĐỘNG (CALL STACK)

Luồng kích hoạt từ [run.py](file:///<project-root>/run.py):
```
[User clicks "Run Python File" on run.py]
  │
  ▼
run.py (sys.argv = ['run.py'])
  │ calls main()
  ▼
app/camera/realtime_inference.py:main()
  │
  ├─► mode: "unified" (default)
  ├─► passthrough args: forwards ['--enable-experimental-jz']
  │
  ▼
src/main.py:main()
  │
  ▼
src/camera/realtime.py:main()
  │
  ▼
app/inference/unified_abc123_jz_auto_experiment.py:main()
  │
  ├─► J/Z Commits: enable_experimental_jz = True
  ├─► SPACE Recognition:
  │     load_space_predictor(override_enable=args.enable_space)
  │     args.enable_space is None -> reads configs/space.json ("enable_space": true)
  │     SpacePredictor initialized successfully with threshold 0.889
  │
  ├─► Word & Sentence Suggestions:
  │     enable_word_suggestion = True (default)
  │     OfflineDictionary (25,000 words loaded from data/dictionary/english_words_25k.tsv)
  │     QwenRanker + SentenceGenerator initialized
  │     SuggestionManager attached to TextComposer.on_change event
  │
  ├─► OpenCV VideoCapture(camera_index=0)
  │
  ▼
[Interactive Webcam Recognition Loop Running]
```

### Chi tiết phân tích tham số:
1. **Chế độ nhận diện (`--mode`):**
   - Mặc định là `unified` (tích hợp ABC + 123 + J/Z + SPACE).
2. **Kích hoạt J/Z (`--enable-experimental-jz`):**
   - Được dispatcher trong [realtime_inference.py](file:///<project-root>/app/camera/realtime_inference.py#L50-L51) tự động gắn vào `passthrough` khi chạy mode `unified` (trừ khi có `--disable-experimental-jz`).
3. **Kích hoạt SPACE (`--enable-space`):**
   - Trước Stage 20: [configs/space.json](file:///<project-root>/configs/space.json) có `"enable_space": false`, dẫn đến việc nếu không gõ `--enable-space`, cử chỉ SPACE bị tắt.
   - Stage 20: Đã cập nhật [configs/space.json](file:///<project-root>/configs/space.json) thành `"enable_space": true`.
   - Kết quả: Khi nhấn Run không cờ, `args.enable_space` là `None`, `load_space_predictor()` đọc cấu hình thấy `true` và kích hoạt đầy đủ `SpacePredictor`. Người dùng vẫn có thể tắt nếu muốn bằng `--disable-space`.
4. **Kích hoạt gợi ý từ ngữ (`--enable-word-suggestion`):**
   - Đã mặc định là `True` trong `unified_abc123_jz_auto_experiment.py`. Sử dụng từ điển 25,000 từ đã được tích hợp trong Stage 19.

---

## 3. CẤU HÌNH MÔI TRƯỜNG VS CODE (CONDA `nn`)

Để đảm bảo khi nhấn nút **"Run Python File"** (biểu tượng Play góc trên bên phải editor) hoặc nhấn phím **F5**, VS Code sử dụng chính xác môi trường Conda `nn`, các file cấu hình workspace đã được thiết lập:

### 3.1. File `.vscode/settings.json`
Đường dẫn: [settings.json](file:///<project-root>/.vscode/settings.json) (và mirror tại `<project-mirror>\.vscode\settings.json`):
```json
{
    "python.defaultInterpreterPath": "<conda-env-nn>/python.exe",
    "python.terminal.activateEnvironment": true
}
```
- `"python.defaultInterpreterPath"`: Trỏ thẳng đến Python 3.10 trong Conda `nn`, ngăn chặn VS Code chọn nhầm môi trường `base` hoặc Python hệ thống.
- `"python.terminal.activateEnvironment": true`: Tự động kích hoạt môi trường `nn` khi mở Terminal mới trong VS Code.

### 3.2. File `.vscode/launch.json`
Đường dẫn: [launch.json](file:///<project-root>/.vscode/launch.json) (và mirror tại `<project-mirror>\.vscode\launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run NhanDienCuChi (Webcam + SPACE + J/Z + Suggestions)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Check-Only Preflight (run.py --check-only)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "args": [
                "--check-only"
            ],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```
Cho phép người dùng nhấn **F5** hoặc chọn **Run Without Debugging** (Ctrl+F5) trực tiếp từ bảng điều khiển Run & Debug.

---

## 4. KẾT QUẢ XÁC MINH HỆ THỐNG

### 4.1. Kết quả thực thi `python run.py --check-only` (Mặc định không cờ)
```json
{
  "status": "PASS",
  "experimental": true,
  "camera_opened": false,
  "mediapipe_executed": false,
  "files_written": 0,
  "modes": ["ABC", "123"],
  "word_suggestion_enabled": true,
  "word_suggestion_initialized": true,
  "space_enabled": true,
  "space_model": "<app-data-local>\\staging\\space_final_candidate_phase3_9_8_20261106\\final_space_model.joblib",
  "camera_count_when_running": 1,
  "landmarker_count_when_running": 2,
  "letters_model": "uvw_medium",
  "letters_epoch": 81,
  "digits_model": "models/digits/production/digits_final.joblib",
  "dynamic_model_sha256": "8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497",
  "dynamic_feature_shape": [16, 69],
  "dynamic_implementation": "AutoJzSession + First16Segmenter / latest motion gate",
  "dynamic_pipeline": "IMAGE/raw/no-flip/independent of static VIDEO",
  "static_pipeline": "baseline VIDEO/mirrored ABC, raw 123; mirrored display always",
  "hold_seconds": 3.0,
  "jz_hold_requires_observed_endpoint": true,
  "jz_commits_enabled": true,
  "candidate_validation": "single coherent motion run + capture-bound endpoint; semantic UNVERIFIED",
  "geometry_gate": "SHADOW_ONLY_NOT_RECALL_VALIDATED",
  "jz_reject_option_enabled": true,
  "jz_margin_threshold": 0.1,
  "mirror_display": true
}
```

### 4.2. Bảng đối chiếu các tính năng khi khởi động mặc định:
| Thành phần | Yêu cầu | Trạng thái mặc định | Cơ chế kích hoạt |
|---|---|---|---|
| **Webcam Recognition** | Mở camera 0, nhận diện thời gian thực | **BẬT** | `run.py` -> `realtime_inference` -> `UnifiedAutoCoordinator` |
| **Cử chỉ SPACE** | Bật tương đương `--enable-space` | **BẬT (`space_enabled: true`)** | `configs/space.json` (`"enable_space": true`) |
| **Nhận diện J/Z** | Kích hoạt commit J/Z | **BẬT (`jz_commits_enabled: true`)** | Dispatcher forwards `--enable-experimental-jz` |
| **Gợi ý từ ngữ** | Tải từ điển, hiển thị ứng viên từ | **BẬT (`word_suggestion_initialized: true`)** | `args.enable_word_suggestion=True` + 25k word dictionary |
| **Gợi ý câu** | Sinh câu hoàn chỉnh từ ngữ cảnh | **BẬT** | `SentenceGenerator` tích hợp sẵn trong `SuggestionManager` |
| **Môi trường Python** | Conda `nn`, không dùng `base` | **CHÍNH XÁC** | `.vscode/settings.json` (`<conda-env-nn>/python.exe`) |

### 4.3. Kiểm thử hồi quy tự động (Regression Test Suite)
Bộ kiểm thử tự động gồm 49 test cases đã được thực thi và vượt qua 100%:
- [tests/test_stage20_one_click_run.py](file:///<project-root>/tests/test_stage20_one_click_run.py): **6/6 PASS**
  1. `test_vscode_settings_interpreter_configured_to_nn`: PASS
  2. `test_vscode_launch_json_configured`: PASS
  3. `test_run_py_default_check_only_enables_space_jz_and_suggestions`: PASS
  4. `test_run_py_explicit_disable_space`: PASS
  5. `test_run_py_explicit_disable_jz`: PASS
  6. `test_run_py_explicit_disable_word_suggestion`: PASS
- [tests/test_stage19_suggestion_integration.py](file:///<project-root>/tests/test_stage19_suggestion_integration.py): **8/8 PASS**
- [tests/test_stage18_production_jz_integration.py](file:///<project-root>/tests/test_stage18_production_jz_integration.py): **7/7 PASS**
- [tests/test_jz_production_integration.py](file:///<project-root>/tests/test_jz_production_integration.py): **13/13 PASS**
- `tests.test_src_architecture.EntrypointTests`: **15/15 PASS**

Tổng cộng: **49/49 tests PASSED (100% OK)** trong 26.2 giây.

---

## 5. HƯỚNG DẪN SỬ DỤNG CHO NGƯỜI DÙNG

Để chạy hệ thống nhận diện từ bây giờ:
1. Mở file [run.py](file:///<project-root>/run.py) trong VS Code.
2. Nhấn vào nút **Run Python File** (biểu tượng tam giác Play ở góc trên cùng bên phải).
   *(Hoặc nhấn phím **F5** để chạy)*.
3. Ứng dụng sẽ tự động mở cửa sổ webcam với đầy đủ:
   - Nhận diện 24 ký tự tĩnh (A-Y trừ J, Z) và 10 chữ số (0-9).
   - Nhận diện cử chỉ động **J** và **Z** kèm xác nhận Hold.
   - Nhận diện cử chỉ ngửa lòng bàn tay để chèn **dấu cách (SPACE)**.
   - Hiển thị thanh gợi ý từ ngữ và gợi ý câu trên màn hình.
   - Nhấn phím số (1-3) hoặc click chuột vào gợi ý để chọn từ nhanh.
   - Nhấn **ESC** hoặc phím **Q** để thoát chương trình.
