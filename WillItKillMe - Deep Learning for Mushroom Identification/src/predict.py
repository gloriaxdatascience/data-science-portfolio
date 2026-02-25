import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ───────────────────────────────────────────────────────────────
# 1. Device
# ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────────
# 2. ImageNet normalization (same as training)
# ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

predict_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ───────────────────────────────────────────────────────────────
# 3. Load model architecture
# ───────────────────────────────────────────────────────────────
def load_model(weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ───────────────────────────────────────────────────────────────
# 4. Predict a single image
# ───────────────────────────────────────────────────────────────
CLASS_NAMES = ["edible", "poisonous"]

def predict_image(model, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = predict_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return CLASS_NAMES[pred_idx], float(probs[pred_idx])

# ───────────────────────────────────────────────────────────────
# 5. CLI
# ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mushroom classifier prediction script")
    parser.add_argument("input", type=str,
                        help="Path to an image or a folder of images")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model .pth file")
    args = parser.parse_args()

    input_path = Path(args.input)
    weights_path = Path(args.weights)

    model = load_model(weights_path)

    if input_path.is_file():
        label, prob = predict_image(model, input_path)
        print(f"{input_path.name}: {label} ({prob:.2f})")

    elif input_path.is_dir():
        for img_file in sorted(input_path.glob("*")):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                label, prob = predict_image(model, img_file)
                print(f"{img_file.name}: {label} ({prob:.2f})")
    else:
        raise ValueError("Input must be a file or directory")

if __name__ == "__main__":
    main()
