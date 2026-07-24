"""
PART 1: VLA Backbone — Load and Freeze LLaVA-v1.5-7B
=====================================================
Downloads (or loads from disk) the pretrained LLaVA-v1.5-7B model,
freezes ALL backbone parameters, and exposes the vision encoder
and language model for downstream adapter injection.

FIX (this file): the processor must have `patch_size` and
`vision_feature_select_strategy` set explicitly, or transformers will
NOT expand the <image> placeholder token into per-patch tokens during
tokenization. Without this, input_ids end up with 0 image tokens while
pixel_values still has N images, causing:

    ValueError: The input provided to the model are wrong. The number of
    image tokens is 0 while the number of image given to the model is N.
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

logger = logging.getLogger(__name__)


class VLABackbone:
    """
    Wraps the frozen LLaVA-v1.5-7B model as the VLA backbone.

    After initialization:
    - All 7B parameters are frozen (requires_grad = False)
    - The model is in eval mode for the backbone
    - Vision encoder outputs and LLM hidden states are accessible
      for LoRA injection and downstream heads
    - self.processor.patch_size and
      self.processor.vision_feature_select_strategy are set so that
      calling self.processor(text=..., images=...) correctly expands
      the <image> placeholder into the right number of tokens.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/llava-v1.5-7b",
        dtype: str = "float16",
        device_map: str = "auto",
        gradient_checkpointing: bool = True,
    ):
        self.model_path = model_path
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map

        logger.info(f"Loading VLA backbone from: {model_path}")

        # ── Load processor (tokenizer + image processor) ──
        self.processor = AutoProcessor.from_pretrained(model_path)

        # ── Load model ──
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        # ── FIX: enable automatic <image> token expansion ──
        # Newer `transformers` versions require these two attributes on the
        # processor to correctly expand the <image> placeholder into
        # `num_patches` tokens matching the vision tower's output. Without
        # this, input_ids will contain 0 image tokens and the forward pass
        # will raise a ValueError about mismatched image token counts.
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )
        logger.info(
            f"Processor image-token expansion enabled: "
            f"patch_size={self.processor.patch_size}, "
            f"vision_feature_select_strategy={self.processor.vision_feature_select_strategy}"
        )

        # Number of image tokens each image will expand to — useful for
        # dataset/collate code that needs to know this ahead of time.
        image_size = self.model.config.vision_config.image_size
        patches_per_side = image_size // self.processor.patch_size
        num_patches = patches_per_side ** 2
        if self.processor.vision_feature_select_strategy == "full":
            num_patches += 1  # includes CLS token
        self.num_image_tokens = num_patches
        logger.info(f"Each image expands to {self.num_image_tokens} tokens")

        # ── Freeze ALL parameters ──
        self._freeze_backbone()

        # ── Enable gradient checkpointing for memory efficiency ──
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # ── Report stats ──
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        logger.info(f"Total parameters:  {total_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable parameters: {total_params - frozen_params:,}")
        logger.info("✅ Backbone loaded and frozen successfully.")

    def _freeze_backbone(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All backbone parameters frozen (requires_grad=False)")

    def get_model(self):
        """Return the underlying HuggingFace model for LoRA injection."""
        return self.model

    def get_processor(self):
        """Return the processor (tokenizer + image processor)."""
        return self.processor

    def get_vision_encoder(self):
        """Return the vision tower (CLIP ViT)."""
        return self.model.vision_tower

    def get_language_model(self):
        """Return the language model (LLaMA)."""
        return self.model.language_model

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from an image.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W]

        Returns:
            Vision features [B, num_patches, hidden_dim]
        """
        with torch.no_grad():
            vision_outputs = self.model.vision_tower(
                pixel_values, output_hidden_states=True
            )
        return vision_outputs.last_hidden_state

    def build_prompt(self, text: str) -> str:
        """
        Prefix a raw instruction/prompt with the required <image> placeholder
        token so the processor knows where to splice in image features.

        Use this when constructing dataset prompts, e.g.:
            prompt = backbone.build_prompt("Describe the driving scene.")
        """
        image_token = getattr(self.processor, "image_token", "<image>")
        return f"{image_token}\n{text}"

    def verify(self):
        """Run a quick sanity check to ensure the model is loaded correctly."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"\n{'='*60}")
        print(f"  VLA Backbone Verification")
        print(f"{'='*60}")
        print(f"  Model path:        {self.model_path}")
        print(f"  Model type:        {type(self.model).__name__}")
        print(f"  Dtype:             {self.dtype}")
        print(f"  Total params:      {total:,}")
        print(f"  Trainable params:  {trainable:,}")
        print(f"  Frozen params:     {total - trainable:,}")
        print(f"  Frozen ratio:      {(total - trainable) / total * 100:.2f}%")
        print(f"  Patch size:        {self.processor.patch_size}")
        print(f"  Vision select:     {self.processor.vision_feature_select_strategy}")
        print(f"  Tokens per image:  {self.num_image_tokens}")
        print(f"{'='*60}")

        assert trainable == 0, f"Expected 0 trainable params, got {trainable}"
        print("  ✅ All checks passed!")
        print(f"{'='*60}\n")


# ── CLI Entry Point ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Load and freeze VLA backbone")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/llava-v1.5-7b",
        help="Path to pretrained LLaVA model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--verify", action="store_true", help="Run verification")
    args = parser.parse_args()

    backbone = VLABackbone(model_path=args.model_path, dtype=args.dtype)

    if args.verify:
        backbone.verify()