# debug_image.py
from src.models.vla_backbone import VLABackbone
from src.data.bdd100k_dataset import BDD100KDataset
from PIL import Image
import torch

# Adjust model_path if different in configs/base_config.yaml
backbone = VLABackbone(model_path="checkpoints/llava-v1.5-7b", dtype="float16", gradient_checkpointing=False)
proc = backbone.get_processor()

print("processor.image_processor.size:", getattr(getattr(proc, "image_processor", None), "size", None))
print("processor.feature_extractor.size:", getattr(getattr(proc, "feature_extractor", None), "size", None))

ds = BDD100KDataset(root="data/bdd100k", split="train", processor=proc)
print("dataset length:", len(ds))
if len(ds) == 0:
    raise SystemExit("No images found under data/bdd100k; point dataset to sample image and retry.")

item = ds[0]
pv = item["pixel_values"]
print("dataset returned pixel_values type:", type(pv), "shape (if tensor):", getattr(pv, "shape", None))

# Also directly call the processor on the same PIL image to compare:
img_path = ds.files[0]
img = Image.open(img_path).convert("RGB")
proc_out = proc(images=img, return_tensors="pt")
print("processor output pixel_values shape:", proc_out["pixel_values"].shape)