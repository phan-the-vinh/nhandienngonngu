"""Unified camera entry through existing lifecycle and the new A-Z facade."""
import argparse
from src.model import APPLICATION_CONFIG, load_configuration, load_models
from src.recognition.alphabet import AlphabetRecognizer

def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--app-config",default=str(APPLICATION_CONFIG))
    args,forwarded=parser.parse_known_args(argv)
    from app.inference import unified_abc123_jz_auto_experiment as existing
    try:
        load_configuration(args.app_config)
    except (OSError,ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return existing.main(forwarded,coordinator_factory=AlphabetRecognizer,
                         artifact_loader=lambda:load_models(args.app_config).camera_inputs())
