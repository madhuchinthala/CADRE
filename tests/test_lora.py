"""
Tests for LoRA Adapter (PART 2)
"""

import sys
sys.path.insert(0, ".")


def test_lora_class_exists():
    """Test that LoRAAdapterManager class can be imported."""
    from src.adapters.lora_adapter import LoRAAdapterManager
    assert LoRAAdapterManager is not None


def test_lora_methods():
    """Test that LoRAAdapterManager has all required methods."""
    from src.adapters.lora_adapter import LoRAAdapterManager

    required_methods = [
        "inject_lora",
        "save_adapter",
        "load_adapter",
        "list_available_adapters",
    ]

    for method_name in required_methods:
        assert hasattr(LoRAAdapterManager, method_name), f"Missing method: {method_name}"


def test_lora_config_loading():
    """Test that LoRA config YAML has correct structure."""
    import yaml

    with open("configs/lora_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "lora" in config
    lora = config["lora"]

    assert "rank" in lora
    assert "alpha" in lora
    assert "dropout" in lora
    assert "target_modules" in lora
    assert "bias" in lora

    assert lora["rank"] == 16
    assert lora["alpha"] == 32
    assert "q_proj" in lora["target_modules"]
    assert "v_proj" in lora["target_modules"]


if __name__ == "__main__":
    test_lora_class_exists()
    test_lora_methods()
    test_lora_config_loading()
    print("✅ All LoRA tests passed!")
