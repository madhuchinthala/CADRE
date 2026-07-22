"""
Tests for Domain Router (PART 5)
"""

import sys
sys.path.insert(0, ".")

import torch


def test_router_class_exists():
    """Test that DomainRouter class can be imported."""
    from src.router.domain_router import DomainRouter
    assert DomainRouter is not None


def test_router_forward():
    """Test router forward pass with dummy input."""
    from src.router.domain_router import DomainRouter

    router = DomainRouter(
        input_dim=1024,
        hidden_dims=[512, 256],
        num_domains=4,
    )

    # Dummy input: batch of 8 visual features
    x = torch.randn(8, 1024)
    logits = router(x)

    assert logits.shape == (8, 4), f"Expected (8, 4), got {logits.shape}"


def test_router_route():
    """Test router routing with confidence."""
    from src.router.domain_router import DomainRouter

    router = DomainRouter(
        input_dim=1024,
        num_domains=4,
        confidence_threshold=0.7,
    )

    x = torch.randn(4, 1024)
    domain_names, confidences = router.route(x)

    assert len(domain_names) == 4
    assert confidences.shape == (4,)
    assert all(isinstance(d, str) for d in domain_names)


def test_router_param_count():
    """Test that router is lightweight."""
    from src.router.domain_router import DomainRouter

    router = DomainRouter(
        input_dim=1024,
        hidden_dims=[512, 256],
        num_domains=4,
    )

    total_params = sum(p.numel() for p in router.parameters())
    # Should be < 1M parameters (lightweight classifier)
    assert total_params < 1_000_000, f"Router too large: {total_params:,} params"


def test_router_config():
    """Test router config YAML."""
    import yaml

    with open("configs/router_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "router" in config
    r = config["router"]

    assert r["input_dim"] == 1024
    assert r["num_domains"] == 4
    assert len(r["domain_labels"]) == 4


if __name__ == "__main__":
    test_router_class_exists()
    test_router_forward()
    test_router_route()
    test_router_param_count()
    test_router_config()
    print("✅ All router tests passed!")
