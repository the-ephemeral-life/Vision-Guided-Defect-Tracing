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

def run_inference(image_path, model_path, ratio=0.5):
    set_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model in strict evaluation mode
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load and resize image
    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    img_resized = raw_image.resize((512, 512), Image.BILINEAR)

    # Tensor conversion [0.0, 1.0] matching training setup
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits

    # Upsample to full original resolution
    upsampled = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )

    probs = torch.sigmoid(upsampled).squeeze().cpu().numpy()

    p_min, p_max = probs.min(), probs.max()

    # Dynamic thresholding based on relative signal peak
    # Ratio = 0.5 picks the midpoint between p_min and p_max
    threshold = p_min + ratio * (p_max - p_min)
    mask = (probs > threshold).astype(np.uint8)

    print("\n--- INFERENCE SUMMARY ---")
    print(f"Probability Range: [{p_min:.4f}, {p_max:.4f}]")
    print(f"Applied Threshold: {threshold:.4f} (Ratio: {ratio})")
    print(f"Detected Crack Pixels: {mask.sum()} ({mask.sum() / mask.size * 100:.2f}% of image)")

    # Red overlay on top of input image
    img_np = np.array(raw_image)
    overlay = img_np.copy()
    overlay[mask > 0] = (img_np[mask > 0] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(raw_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(probs, cmap="jet", vmin=p_min, vmax=p_max)
    axes[1].set_title("Probability Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Binary Mask (Thresh={threshold:.4f})")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Segmentation Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic SegFormer Testing")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default=".", help="Path to model directory")
    parser.add_argument("--ratio", type=float, default=0.5, help="Relative threshold ratio between min and max prob (0.0 to 1.0)")
    args = parser.parse_args()

    run_inference(args.image, args.model, args.ratio)