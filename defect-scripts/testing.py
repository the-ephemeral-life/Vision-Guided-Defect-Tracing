import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation

def run_inference(image_path, model_path, percentile=98.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}...")

    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load image and resize to 512x512
    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    img_resized = raw_image.resize((512, 512), Image.BILINEAR)

    # Convert to Tensor [0.0, 1.0] matching training normalization
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits  # shape: (1, 1, h, w)

    # Upsample logits to original image dimensions
    upsampled = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )

    # Convert to probability and INVERT (0 = crack -> 1 = crack)
    raw_probs = torch.sigmoid(upsampled).squeeze().cpu().numpy()
    crack_probs = 1.0 - raw_probs

    # Contrast Stretching (Min-Max Normalization on inverted probabilities)
    p_min, p_max = crack_probs.min(), crack_probs.max()
    probs_contrast = (crack_probs - p_min) / (p_max - p_min + 1e-8)

    # Percentile Thresholding (isolates top highest crack confidence pixels)
    adaptive_thresh = np.percentile(crack_probs, percentile)
    mask = (crack_probs >= adaptive_thresh).astype(np.uint8)

    print("\n--- INFERENCE RESULTS ---")
    print(f"Crack Probability Range: [{p_min:.4f}, {p_max:.4f}]")
    print(f"Crack Threshold ({percentile}th percentile): {adaptive_thresh:.4f}")
    print(f"Detected Crack Pixels: {mask.sum()} ({mask.sum()/mask.size*100:.2f}% of image)")

    # Red overlay on detected crack
    img_np = np.array(raw_image)
    overlay = img_np.copy()
    overlay[mask > 0] = (img_np[mask > 0] * 0.2 + np.array([255, 0, 0]) * 0.8).astype(np.uint8)

    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(raw_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(probs_contrast, cmap="jet")
    axes[1].set_title("Crack Probability Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Segmentation Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer Local Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default=".", help="Path to model directory")
    parser.add_argument("--percentile", type=float, default=98.0, help="Percentile threshold (e.g. 98.0 selects top 2% crack pixels)")
    args = parser.parse_args()

    run_inference(args.image, args.model, args.percentile)