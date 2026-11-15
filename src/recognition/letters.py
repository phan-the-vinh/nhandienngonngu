"""The existing 24-class production letter predictor, unchanged."""
from app.inference.letter_inference import StaticAslInference, Prediction, STATIC_LABELS, features_from_landmarks
__all__=["StaticAslInference","Prediction","STATIC_LABELS","features_from_landmarks"]
