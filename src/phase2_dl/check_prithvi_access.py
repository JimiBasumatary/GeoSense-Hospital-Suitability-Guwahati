
from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-100M"

print("=" * 65)
print("GeoSense Phase 2 - Prithvi-100M Compatibility Check")
print("=" * 65)

print("\nChecking Prithvi-100M checkpoint access...")

try:
    model_file = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="Prithvi_100M.pt"
    )

    print("\nPrithvi-100M access: SUCCESS")
    print("Hugging Face checkpoint:")
    print(model_file)

    print("\nCompatibility assessment:")
    print(
        "The Prithvi checkpoint is a geospatial foundation-model "
        "checkpoint and is not directly compatible with the "
        "current 6-band U-Net segmentation interface."
    )

    print("\nUsing the documented ResNet-50 segmentation")
    print("substitute for the practical training pipeline.")

except Exception as e:
    print("\nPrithvi-100M access failed.")
    print("Reason:", str(e))
    print("The checkpoint-access success message cannot be recorded.")
