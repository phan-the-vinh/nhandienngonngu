"""A-Z facade over the verified coordinator, NOT a trained 26-class model."""
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator

class AlphabetRecognizer(UnifiedAutoCoordinator):
    """Inherits arbitration/HOLD/diagnostics exactly; J/Z commits opt-in only."""
    alphabet=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    static_labels=tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
    dynamic_labels=("J","Z")
    is_26_class_classifier=False
    dynamic_status="EXPERIMENTAL_NOT_PRODUCTION"

    @classmethod
    def from_models(cls,bundle,**options):
        _,letters,digits,dynamic,_=bundle.camera_inputs()
        return cls(letters,digits,dynamic,**options)
