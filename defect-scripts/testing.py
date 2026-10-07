import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

def run_inference(image_path, model_path, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}...")

    # Load model and feature extractor from your downloaded best/ folder
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size
    inputs = processor(images=raw_image, return_tensors="pt").to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_classes, h, w)

    # Upsample logits to match original image dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )

    # Process mask depending on output channels
    if upsampled_logits.shape[1] == 1:
        probs = torch.sigmoid(upsampled_logits).squeeze().cpu().numpy()
        mask = (probs > threshold).astype(np.uint8)
    else:
        probs = torch.softmax(upsampled_logits, dim=1).squeeze().cpu().numpy()
        pred = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()
        mask = (pred == 1).astype(np.uint8) if upsampled_logits.shape[1] == 2 else pred

    # Create semi-transparent red overlay on defects
    img_np = np.array(raw_image)
    overlay = img_np.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    overlay[mask > 0] = (img_np[mask > 0] * 0.4 + red_color * 0.6).astype(np.uint8)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(raw_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer Local Test Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to test image file")
    parser.add_argument("--model", type=str, default="./best", help="Path to unzipped 'best' model directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 to 1.0)")
    args = parser.parse_args()

    run_inference(args.image, args.model, args.threshold)