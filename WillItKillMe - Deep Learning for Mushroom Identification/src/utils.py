import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cv2
from PIL import Image

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
def get_data_paths(base_dir=".."):
    base = Path(base_dir).resolve()
    data = base / "data" / "raw"
    return {
        "train": data / "train",
        "validate": data / "validate",
        "test": data / "test",
    }

# Grad-CAM
def get_gradcam_overlay(model, input_tensor, target_layer, device):
    """
    Returns a PIL Image with the Grad-CAM heatmap overlaid on the input.
    
    Args:
        model:          your PyTorch model
        input_tensor:   preprocessed image tensor, shape (1, 3, H, W)
        target_layer:   the layer to hook, e.g. model.layer4[-1] for ResNet
        device:         torch.device
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    fh.remove()
    bh.remove()

    grads = gradients[0].cpu().detach()        # (1, C, H, W)
    acts  = activations[0].cpu().detach()      # (1, C, H, W)

    weights = grads.mean(dim=(2, 3), keepdim=True)  # global average pool
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().numpy()

    # Normalise to [0, 255]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    # Resize to input image size
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Reconstruct original image from tensor for blending
    img_np = input_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = (img_np * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)