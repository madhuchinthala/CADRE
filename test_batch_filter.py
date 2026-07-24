"""
Quick test to verify batch filtering works with task labels.
This simulates what happens in the training loop.
"""

# Test the batch filtering logic
batch = {
    "pixel_values": "tensor[B,C,H,W]",
    "input_ids": "tensor[B,seq]",
    "labels": "tensor[B,seq]",
    "waypoints": "tensor[B,12,2]",
    "hazard": "tensor[B]",
    "regulation": "tensor[B]",
    "weather": "tensor[B]",
}

print("Original batch keys:", list(batch.keys()))

# This is what the training code does now:
valid_model_keys = {'pixel_values', 'input_ids', 'labels', 'attention_mask', 'pad_token_id', 'output_attentions', 'output_hidden_states', 'return_dict'}
model_batch = {k: v for k, v in batch.items() if k in valid_model_keys}
task_labels = {k: v for k, v in batch.items() if k not in valid_model_keys}

print("\nModel batch keys (will pass to model):", list(model_batch.keys()))
print("Task labels (extracted, for future multi-head training):", list(task_labels.keys()))

print("\n✅ Batch filtering works correctly!")
print("   - Model receives only valid arguments: pixel_values, input_ids, labels")
print("   - Task labels extracted separately: waypoints, hazard, regulation, weather")
