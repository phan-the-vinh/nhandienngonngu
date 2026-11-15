"""Dynamic J/Z inference package."""

from __future__ import annotations

from app.inference.dynamic_jz.finger_tracker import (
    FingerTracker,
    FingerTrackingResult,
    FingerTrackingError,
    extract_finger_tracking,
)
from app.inference.dynamic_jz.feature_builder import (
    build_dtw_features,
    FEATURE_CONFIGURATIONS,
)
from app.inference.dynamic_jz.dtw_engine import (
    dtw_multivariate,
    DTWError,
)
from app.inference.dynamic_jz.template_store import (
    DTWTemplate,
    TemplateStore,
    select_k_medoids_for_class,
)
from app.inference.dynamic_jz.dtw_matcher import (
    DTWMatcher,
    DTWMatchResult,
)
from app.inference.dynamic_jz.shadow_adapter import (
    JzShadowAdapter,
    JzShadowResult,
    JzShadowTelemetryRecord,
    ShadowModelProxy,
)
from app.inference.dynamic_jz.ratio_classifier import (
    RatioBasedJZClassifier,
    RatioFeaturesJ,
    RatioFeaturesZ,
    RatioClassificationResult,
)
from app.inference.dynamic_jz.fusion_classifier import (
    HandshapeTrajectoryFusionClassifier,
    HandshapeBranch,
    TrajectoryBranch,
    HandshapeMetrics,
    HandshapeResult,
    TrajectoryMetricsJ,
    TrajectoryMetricsZ,
    TrajectoryResult,
    FusionClassificationResult,
)
from app.inference.dynamic_jz.adaptive_classifier import (
    AdaptiveMotionJZClassifier,
    AdaptiveMotionMetrics,
    AdaptiveClassificationResult,
)
from app.inference.dynamic_jz.target_trajectory_recognizer import (
    TargetPinkyJAnalyzer,
    TargetIndexZAnalyzer,
    TargetPinkyJMetrics,
    TargetIndexZMetrics,
    TargetTrajectoryResult,
    TrajectoryOnlyJZRecognizer,
    HandshapeOnlyJZRecognizer,
    TargetFingerTrajectoryRecognizer,
)
from app.inference.dynamic_jz.stage13_margin_wrapper import (
    ProbabilityMarginModelWrapper,
)

__all__ = [
    "FingerTracker",
    "FingerTrackingResult",
    "FingerTrackingError",
    "extract_finger_tracking",
    "build_dtw_features",
    "FEATURE_CONFIGURATIONS",
    "dtw_multivariate",
    "DTWError",
    "DTWTemplate",
    "TemplateStore",
    "select_k_medoids_for_class",
    "DTWMatcher",
    "DTWMatchResult",
    "JzShadowAdapter",
    "JzShadowResult",
    "JzShadowTelemetryRecord",
    "ShadowModelProxy",
    "RatioBasedJZClassifier",
    "RatioFeaturesJ",
    "RatioFeaturesZ",
    "RatioClassificationResult",
    "HandshapeTrajectoryFusionClassifier",
    "HandshapeBranch",
    "TrajectoryBranch",
    "HandshapeMetrics",
    "HandshapeResult",
    "TrajectoryMetricsJ",
    "TrajectoryMetricsZ",
    "TrajectoryResult",
    "FusionClassificationResult",
    "AdaptiveMotionJZClassifier",
    "AdaptiveMotionMetrics",
    "AdaptiveClassificationResult",
    "TargetPinkyJAnalyzer",
    "TargetIndexZAnalyzer",
    "TargetPinkyJMetrics",
    "TargetIndexZMetrics",
    "TargetTrajectoryResult",
    "TrajectoryOnlyJZRecognizer",
    "HandshapeOnlyJZRecognizer",
    "TargetFingerTrajectoryRecognizer",
    "ProbabilityMarginModelWrapper",
]
