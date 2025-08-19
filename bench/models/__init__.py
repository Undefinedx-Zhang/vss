from .segmentation import build_seg_model

def build_model(name: str, num_classes: int):
    """Build segmentation model"""
    return build_seg_model(name, num_classes)


