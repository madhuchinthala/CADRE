"""
Tests for VLA Backbone (PART 1)
"""

import sys
sys.path.insert(0, ".")


def test_backbone_class_exists():
    """Test that VLABackbone class can be imported."""
    from src.models.vla_backbone import VLABackbone
    assert VLABackbone is not None


def test_backbone_default_args():
    """Test that VLABackbone has correct default arguments."""
    import inspect
    from src.models.vla_backbone import VLABackbone

    sig = inspect.signature(VLABackbone.__init__)
    params = sig.parameters

    assert "model_path" in params
    assert "dtype" in params
    assert "device_map" in params
    assert "gradient_checkpointing" in params

    assert params["model_path"].default == "checkpoints/llava-v1.5-7b"
    assert params["dtype"].default == "float16"


def test_backbone_methods():
    """Test that VLABackbone has all required methods."""
    from src.models.vla_backbone import VLABackbone

    required_methods = [
        "get_model",
        "get_processor",
        "get_vision_encoder",
        "get_language_model",
        "encode_image",
        "verify",
    ]

    for method_name in required_methods:
        assert hasattr(VLABackbone, method_name), f"Missing method: {method_name}"


if __name__ == "__main__":
    test_backbone_class_exists()
    test_backbone_default_args()
    test_backbone_methods()
    print("✅ All backbone tests passed!")
