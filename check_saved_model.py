
import torch

path = r"models\saved\prithvi_finetuned.pth"

checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict):
    state = checkpoint
else:
    raise TypeError("Unexpected checkpoint format")

print("Saved checkpoint:", path)
print("Number of parameter/buffer tensors:", len(state))

total = 0
for key, value in state.items():
    if isinstance(value, torch.Tensor):
        total += value.numel()

print("Total elements in saved tensors:", f"{total:,}")
print("Note: this count may include buffers and does not identify")
print("which parameters were trainable or frozen during training.")
