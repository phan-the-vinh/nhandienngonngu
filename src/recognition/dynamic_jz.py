"""Experimental binary J/Z API; a segmented bout is not semantic J/Z proof."""
from app.inference.dynamic_jz_camera_test import predict_attempt
from app.inference.dynamic_jz_auto_camera_experiment import AutoJzSession
DYNAMIC_LABELS=("J","Z")
FEATURE_SHAPE=(16,69)
FLATTENED_FEATURE_COUNT=1104
__all__=["AutoJzSession","predict_attempt","DYNAMIC_LABELS","FEATURE_SHAPE","FLATTENED_FEATURE_COUNT"]
