"""
Tests for EWC (PART 3)
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn


def test_ewc_class_exists():
    """Test that EWC class can be imported."""
    from src.continual.ewc import EWC
    assert EWC is not None


def test_ewc_initialization():
    """Test EWC initializes with correct defaults."""
    from src.continual.ewc import EWC

    ewc = EWC()
    assert ewc.lambda_ewc == 5000.0
    assert ewc.fisher_n_samples == 2000
    assert ewc.variant == "online_ewc"
    assert ewc.gamma == 0.95
    assert ewc.lora_only is True


def test_ewc_penalty_zero_initially():
    """Test that EWC penalty is 0 before any Fisher computation."""
    from src.continual.ewc import EWC

    ewc = EWC()

    # Create a simple model
    model = nn.Linear(10, 5)

    penalty = ewc.penalty(model)
    assert penalty.item() == 0.0, f"Expected 0.0, got {penalty.item()}"


def test_ewc_save_load(tmp_path=None):
    """Test EWC state save and load."""
    import tempfile
    from src.continual.ewc import EWC

    ewc = EWC(variant="ewc")

    # Create dummy Fisher data
    ewc.fisher_matrices["test_domain"] = {
        "weight": torch.randn(5, 10)
    }
    ewc.optimal_params["test_domain"] = {
        "weight": torch.randn(5, 10)
    }

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        ewc.save(tmpdir, "test_domain")
        assert (torch.load(f"{tmpdir}/ewc_test_domain.pt", weights_only=False) is not None)

        # Load into new EWC
        ewc2 = EWC(variant="ewc")
        ewc2.load(tmpdir, "test_domain")
        assert "test_domain" in ewc2.fisher_matrices


if __name__ == "__main__":
    test_ewc_class_exists()
    test_ewc_initialization()
    test_ewc_penalty_zero_initially()
    test_ewc_save_load()
    print("✅ All EWC tests passed!")
