"""Validated artifact adapters. No model loading at import; no promotion/fit."""
from dataclasses import dataclass
import json
from pathlib import Path
from src import ROOT

APPLICATION_CONFIG = ROOT / "configs/application.json"

def project_file(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing/out-of-project artifact: {value}")
    return path

def load_configuration(path=APPLICATION_CONFIG):
    config = json.loads(project_file(path).read_text(encoding="utf-8"))
    # Stage B/C intentionally pins the already hash-validated loaders. Alternate
    # locations need a separately approved manifest/path migration, not ignored overrides.
    expected = {"schema_version":1, "letters_config":"configs/letters.json",
        "digits_config":"configs/digits.json",
        "dynamic_experiment_dir":"data/experiments/dynamic_jz_binary_baseline_v1",
        "mediapipe_task":"models/mediapipe/hand_landmarker.task",
        "jz_commits_default":False, "mirror_display":True}
    if any(config.get(key) != value for key,value in expected.items()):
        raise ValueError("Unsupported runtime contract/path migration; existing model contracts must be preserved")
    for key in ("letters_config","digits_config","mediapipe_task"):
        project_file(config[key])
    for filename in ("model_pipeline.joblib","scaler.joblib","training_manifest.json",
                     "config_snapshot.json","split_manifest.json"):
        project_file(str(Path(config["dynamic_experiment_dir"]) / filename))
    return config

def load_letters(model_name="uvw_medium"):
    letter_model = ROOT / "models" / "letters" / "production" / "uvw_medium_model.joblib"
    if not letter_model.is_file():
        raise FileNotFoundError(
            f"Static ASL letter model not found at '{letter_model}'. "
            "This public repository is distributed source-only without unverified pre-trained model weights. "
            "Please see docs/MODEL_DISTRIBUTION.md for details."
        )
    from app.inference.letter_inference import load_official_artifacts, _validate_deployment_config
    _validate_deployment_config()
    return load_official_artifacts(model_name)

def load_digits():
    digit_model = ROOT / "models" / "digits" / "production" / "digits_final.joblib"
    if not digit_model.is_file():
        raise FileNotFoundError(
            f"Static digit model not found at '{digit_model}'. "
            "This public repository is distributed source-only without unverified pre-trained model weights. "
            "Please see docs/MODEL_DISTRIBUTION.md for details."
        )
    from app.inference.unified_static_coordinator import load_digit_model
    return load_digit_model()

def load_dynamic():
    from app.inference.dynamic_jz_camera_test import load_experiment_model
    return load_experiment_model()

@dataclass(frozen=True)
class ModelBundle:
    letters: object
    digits: object
    dynamic: object
    dynamic_sha256: str
    config: dict

    def camera_inputs(self):
        from src.recognition.letters import StaticAslInference
        from src.recognition.digits import DigitPredictor
        return (self.letters, StaticAslInference(self.letters),
                DigitPredictor(self.digits), self.dynamic, self.dynamic_sha256)

def load_models(config_path=APPLICATION_CONFIG):
    config=load_configuration(config_path)
    letters=load_letters()
    digits=load_digits()
    dynamic,digest=load_dynamic()
    return ModelBundle(letters,digits,dynamic,digest,config)
