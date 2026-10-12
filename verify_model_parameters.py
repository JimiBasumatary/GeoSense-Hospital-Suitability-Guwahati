
import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=6,
    classes=5
)

checkpoint = torch.load(
    r"models\saved\prithvi_finetuned.pth",
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state = checkpoint["model_state_dict"]
else:
    state = checkpoint

model.load_state_dict(state, strict=True)

frozen_tensors = 0

for name, parameter in model.encoder.named_parameters():
    if "layer1" in name or "layer2" in name:
        parameter.requires_grad = True
    else:
        parameter.requires_grad = False
        frozen_tensors += 1

for parameter in model.decoder.parameters():
    parameter.requires_grad = True

for parameter in model.segmentation_head.parameters():
    parameter.requires_grad = True

total = sum(p.numel() for p in model.parameters())
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
frozen = total - trainable

print("Model: ResNet-50 U-Net")
print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")
print("Frozen parameters:", f"{frozen:,}")
print("Frozen encoder parameter tensors:", frozen_tensors)
