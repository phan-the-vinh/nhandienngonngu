"""Model-only prediction API. No camera/UI; no comparison of cross-model scores."""
from src.model import load_models
from src.recognition.letters import StaticAslInference
from src.recognition.digits import DigitPredictor
from src.recognition.dynamic_jz import predict_attempt
from src.recognition.alphabet import AlphabetRecognizer

class PredictionService:
    def __init__(self,bundle):
        self.bundle=bundle
        self.letter_predictor=StaticAslInference(bundle.letters)
        self.digit_predictor=DigitPredictor(bundle.digits)

    @classmethod
    def load(cls,config_path=None):
        return cls(load_models() if config_path is None else load_models(config_path))

    def letters(self,mirrored_hand):
        return self.letter_predictor.predict_landmarks(mirrored_hand)

    def digits(self,raw_hand):
        return self.digit_predictor.predict_landmarks(raw_hand)

    def dynamic_jz(self,valid_entries,total_frames,*,detection_error=""):
        # The caller supplies a complete raw/no-flip bout. No arbitrary window
        # is declared a valid J/Z gesture just because the binary model predicts it.
        return predict_attempt(valid_entries,total_frames,self.bundle.dynamic,detection_error)

    def alphabet(self,**options):
        return AlphabetRecognizer.from_models(self.bundle,**options)
