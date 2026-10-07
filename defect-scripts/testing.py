import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_inference(image_path, model_path):
    set_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and set strict evaluation mode
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    img_resized = raw_image.resize((512, 512), Image.BILINEAR)

    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )

    # Convert to probabilities
    probs = torch.sigmoid(upsampled).squeeze().cpu().numpy()

    print(f"Logits Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"Raw Probabilities Range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Plot raw probabilities without dynamic min-max stretching
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(raw_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(probs, cmap="jet", vmin=0.0, vmax=0.2)  # Fixed range prevents visual flipping
    axes[1].set_title("Absolute Probability Map (Fixed 0-0.2 Scale)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=".")
    args = parser.parse_args()

    run_inference(args.image, args.model)