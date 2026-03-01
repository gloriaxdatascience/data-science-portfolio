import gradio as gr
import torch
import os
from PIL import Image
# Import your own local logic
from models import load_resnet18  
from utils import get_gradcam_overlay 
from data import get_transforms      

os.environ["GRADIO_DISABLE_COMPRESSION"] = "1"

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_resnet18() 
model.load_state_dict(torch.load("../models/resnet18_mushroom_best.pth", map_location=device))
model.eval()

def predict_and_explain(inp_img):
    if inp_img is None:
        return "Please upload an image", None

    transform = get_transforms()
    img_tensor = transform(inp_img).unsqueeze(0).to(device)

    # Ensure gradients can flow
    img_tensor.requires_grad_(True)

    # 1. Forward pass WITH gradients
    output = model(img_tensor)

    # 2. Grad-CAM overlay (uses the same forward pass + backward)
    overlay_img = get_gradcam_overlay(model, img_tensor, model.layer4[-1], device)

    # 3. Softmax + prediction
    probs = torch.nn.functional.softmax(output, dim=1)[0]
    conf, pred = torch.max(probs, 0)

    label = "POISONOUS" if pred.item() == 1 else "EDIBLE"
    result_text = f"Prediction: {label}\nConfidence: {conf.item():.1%}"

    return result_text, overlay_img

# 2. Build the Layout (Top to Bottom)
with gr.Blocks(theme=gr.themes.Default()) as demo:
    # --- Top Section ---
    gr.Markdown("# Mushroom Safety Classifier")
    gr.Markdown("""
    These metrics are computed on a **held-out test set (282 images)** that was never used during training or model selection.  
    They represent the model’s final real-world generalization performance.

    ⚠️ Note: While overall accuracy is ~91%, false negatives (poisonous predicted as edible) remain a critical safety concern.
    """)
    
    # --- Upload & Output Section ---
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Section")
            run_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column():
            txt_output = gr.Textbox(label="Result")
            img_output = gr.Image(label="Grad-CAM Overlay")

    # --- Bottom Section (Static Plots) ---
    with gr.Accordion("Technical Performance (Confusion Matrix & ROC)", open=False):
        with gr.Row():
            # Loading the static PNGs you saved in Step 1
            gr.Image("test_confusion_matrix.png", label="Confusion Matrix(Test, Final Evaluation)")
            gr.Image("roc_curve.png", label="ROC Curve")

    run_btn.click(fn=predict_and_explain, inputs=img_input, outputs=[txt_output, img_output])

# 3. Launch Locally
if __name__ == "__main__":
    demo.launch(
        inbrowser=True, 
        quiet=True, 
        show_error=True
    )