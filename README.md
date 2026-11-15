# NhanDienCuChi: Real-Time Hand Gesture Sign Language Recognition System

A real-time American Sign Language (ASL) and digit recognition system powered by Google MediaPipe Hand Landmarker, scikit-learn classifiers, and local language intelligence.

---

## Overview

**NhanDienCuChi** translates hand gestures captured via a standard webcam into text characters and sentences in real time. The system combines geometric landmark normalization, spatial-temporal trajectory analysis, and machine learning to achieve robust character input.

> [!NOTE]
> **Source-Only Open Source Distribution:**
> This repository is published under a **Source-Only** distribution policy. It includes all source code, MediaPipe landmark processors, feature normalization algorithms, dynamic J/Z models (CC0), word dictionaries (CC BY-SA 4.0), reproduction scripts, and test suites. Pre-trained weights for the static letter, digit, and space classifiers are excluded from Git tracking due to upstream dataset licensing boundaries.
> For details on providing your own checkpoints or training new models, please read **[MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md)**.

### Supported Gestures
- **Static ASL Letters (24 classes):** `A–I`, `K–Y` recognized using calibrated Multi-Layer Perceptron (MLP) classifiers.
- **Static Digits (10 classes):** `0–9` recognized with dedicated spatial geometric models.
- **Dynamic Letters (2 classes):**
  - **`J`**: Curved pinky-hook motion trajectory.
  - **`Z`**: Three-stroke zig-zag motion with index finger extension.
  - Features motion segmentation (First16Segmenter), wrist-relative feature normalization, and ratio-gated trajectory validation.
- **SPACE Gesture:** Open-palm static posture with thumb extension to separate words naturally.

### Language Intelligence
- **Hold-to-Commit:** Temporal smoothing window (~3 seconds hold confirmation) prevents false triggers during hand transitions.
- **Offline Word Suggestions:** Instant prefix search (<1 ms) powered by a 25,000-word Trie dictionary (`data/dictionary/english_words_25k.tsv`).
- **Contextual Sentence Completion:** Local LLM ranking and completion powered by [Ollama](https://ollama.com) (`qwen3:4b`), functioning completely offline without sending data to external APIs.

---

## Project Structure

```
├── app/               # Application runtime and real-time inference pipelines
│   ├── camera/        # Camera loops and unified multi-modal coordinator
│   └── inference/     # Specialized predictors (letters, digits, dynamic J/Z, SPACE)
├── src/               # Public architectural APIs and clean adapters
├── ai/                # Hand landmark utilities, geometric features, normalization
├── models/            # Production model weights and MediaPipe task bundles
│   ├── mediapipe/     # Google MediaPipe hand_landmarker.task
│   ├── letters/       # ASL letter MLP models (uvw_medium, v1)
│   ├── digits/        # Digit production models (digits_final)
│   └── space/         # Calibrated SPACE gesture classifier
├── data/
│   ├── dictionary/    # 25,000 English word dictionary for offline Trie lookup
│   └── experiments/   # Dynamic J/Z binary baseline pipeline
├── configs/           # JSON configurations (application, space, etc.)
├── tools/             # Utilities (model downloader, camera diagnostic tools)
├── tests/             # Unit and integration test suite
├── run.py             # Primary entry point (one-click launch)
├── requirements.txt   # Pip dependencies
└── environment.yml    # Conda environment definition
```

---

## Quick Start

### 1. Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Version 3.10 or 3.11.
- **Webcam:** Any standard USB or built-in webcam (720p @ 30 FPS recommended).
- *(Optional)* [Ollama](https://ollama.com) with the `qwen3:4b` model installed if local LLM sentence generation is desired:
  ```bash
  ollama pull qwen3:4b
  ```

### 2. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate nhandiencuchi
```

#### Option B: Using Python venv
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Download and Verification
Download the official Google MediaPipe hand landmark model and verify local checkpoints:

```bash
# Check integrity of existing models
python tools/download_models.py --check-only

# Download missing MediaPipe Hand Landmarker task bundle
python tools/download_models.py
```

> [!TIP]
> On a clean clone, pre-trained weights for static letters, digits, and space must either be placed in `models/` or trained locally using the scripts in `training/`. See **[MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md)** for path requirements.

### 4. Run the Application

#### One-Click Launch (Default Unified Mode)
Simply run `run.py` or click **Run Python File** in VS Code:
```bash
python run.py
```
This automatically launches:
- Static ASL (`A–Y` except `J`/`Z`) + Digits (`0–9`) recognition
- Dynamic `J` and `Z` recognition
- `SPACE` gesture recognition
- Offline Trie word suggestions + local Ollama completion

#### Non-Camera Preflight Check
Verify that all model artifacts, configurations, and inference pipelines initialize correctly without accessing the webcam:
```bash
python run.py --check-only
```

#### Dedicated Recognition Modes
```bash
# Letters only
python run.py --mode letters

# Digits only
python run.py --mode digits

# Specify a different camera index
python run.py --camera-index 1
```

#### Command-Line Options
| Option | Description |
|---|---|
| `--mode {unified,letters,digits}` | Operating mode (default: `unified`) |
| `--camera-index <int>` | OpenCV camera capture index (default: `0`) |
| `--check-only` | Validate all components and exit without opening camera |
| `--disable-experimental-jz` | Disable dynamic J/Z gesture commits (diagnostic-only) |
| `--disable-space` | Disable the SPACE gesture classifier |
| `--disable-word-suggestion` | Disable the Trie word suggestion bar |
| `--ollama-model <name>` | Select alternative local Ollama model (default: `qwen3:4b`) |

---

## Operating Instructions & Gesture Tips

1. **Camera Framing:** Position the webcam approximately 50–80 cm away with good ambient lighting. Avoid harsh backlighting.
2. **Holding for Commitment:** When signing a static letter or digit, hold the gesture steady inside the frame. A green confirmation ring will fill up (~3 seconds) before committing the character to the composition buffer.
3. **Inserting a Space:** Form an open palm gesture facing the camera to trigger the SPACE classifier.
4. **Dynamic J & Z:**
   - **Letter J:** Extend the pinky finger and sweep downwards and upwards in a smooth 'J' shape curve.
   - **Letter Z:** Extend the index finger and draw a three-stroke 'Z' pattern in the air.
5. **Word Suggestions:** As characters accumulate, candidate words from the Trie dictionary appear on the UI.
6. **Exit:** Press `q` or `ESC` in the OpenCV display window to gracefully exit the application.

---

## Verification & Testing

Execute the automated regression and unit test suite:

```bash
python -m unittest discover -s tests -q
```

---

## Troubleshooting

- **Webcam Not Detected (`camera_index 0 failed to open`):**
  - Verify your webcam permissions in OS settings.
  - Try alternative camera indices: `python run.py --camera-index 1`.
  - Close other software currently accessing the camera (Zoom, Teams, Discord, etc.).
- **Missing Model Files:**
  - For MediaPipe hand landmark bundle: run `python tools/download_models.py` to restore `models/mediapipe/hand_landmarker.task`.
  - For letter/digit/space weights: refer to **[MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md)** for instructions on providing checkpoints or training replacements.
- **Ollama Connection Warning:**
  - If Ollama is not installed or running, the application gracefully degrades to fast offline Trie dictionary suggestions without raising an exception.

---

## License & Attribution

- **MediaPipe Hand Landmarker:** Developed by Google, distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- **Core Dependencies:** OpenCV, scikit-learn, NumPy, and pandas are licensed under their respective open-source licenses (BSD / Apache / MIT).
- **NhanDienCuChi:** Distributed under the MIT License. See [LICENSE](LICENSE) for details.
