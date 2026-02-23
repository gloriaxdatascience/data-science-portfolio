# 🍄 Mushroom Classification — Edible vs. Poisonous  
Deep Learning Project (ResNet18 Transfer Learning)

# Deep Learning for Mushroom Identification

## Problem Statement
Can we automatically classify mushroom images as **edible** or **poisonous** using convolutional neural networks, with a target validation accuracy of at least 85%?

## Business Relevance
Incorrect mushroom identification can lead to severe poisoning incidents and even fatalities.  
An image-based classifier can support hobby foragers, emergency triage tools, and educational apps by flagging potentially poisonous mushrooms and visual patterns that are hard for non-experts to distinguish.

## Dataset
- Source: Course-provided mushroom image dataset (Agaricus and Amanita species).
- Structure: Pre-split into `train`, `validate`, and `test` folders with class subdirectories (`edible/`, `poisonous/`).  
- Size:
  - Train: 2,256 images (edible + poisonous).
  - Validation: 282 images.
  - Test: 282 images.[file:1]
- Image type: Color photos of mushrooms in natural conditions with varying lighting, orientation, and background.

## Methods Used
- Data loading
  - Custom `SafeImageFolder` with a robust `winsafe_loader` to handle Windows path issues and unreadable images.[file:1]
- Data preprocessing and augmentation
  - Resizing to 128×128 and 224×224.
  - Random horizontal flip, rotation, color jitter, random resized crop, and normalization with ImageNet mean/std.[file:1]
- Models
  - Baseline CNN:
    - 3 conv blocks (Conv–BatchNorm–ReLU–MaxPool) followed by a fully connected head.[file:1]
  - Transfer Learning:
    - ResNet18 with ImageNet weights, frozen backbone in Phase A.
    - Replaced `fc` with Dropout → Linear(512→256) → ReLU → Dropout → Linear(256→2).[file:1]
- Training strategy
  - Unified `train_model` loop with:
    - `copy.deepcopy` best-checkpoint saving by validation accuracy.
    - StepLR scheduler.
    - Optional fine-tuning phase.
  - Two-phase training for ResNet18:
    - Phase A: Head-only training with frozen backbone.
    - Phase B: Unfreeze `layer4` + `fc` at a configurable `FINETUNE_EPOCH` with differential learning rates (backbone LR = 1e-4, head LR = 1e-3).[file:1]
- Evaluation
  - Accuracy on validation and test sets.
  - Confusion matrices for best epoch and test set.
  - Classification reports (precision, recall, F1).
  - Confidence-based error analysis:
    - Misclassification grids.
    - Confidence histograms (correct vs wrong).
    - Precision–recall vs threshold with a safety-focused threshold for the poisonous class.
    - ROC curve and AUC for detecting poisonous mushrooms.[file:1]

## Results
- Baseline CNN
  - Best validation accuracy: ~74%.[file:1]
  - Demonstrates that the task is learnable but leaves substantial room for improvement.
- ResNet18 Transfer Learning
  - Best validation accuracy: ~95% (target 85% exceeded).![file:1]
  - Test accuracy: ~91% on a held-out test set.[file:1]
  - Test classification report (approximate):
    - Edible: precision ~0.89, recall ~0.95.
    - Poisonous: precision ~0.93, recall ~0.86.
  - ROC AUC for poisonous detection: ~0.97.[file:1]

- Safety-oriented metric
  - Precision–recall analysis for the poisonous class.
  - Chosen safety threshold (example): a threshold that reaches ≥95% recall on poisonous mushrooms at the cost of more false positives (edible predicted as poisonous).[file:1]

## Visuals
- EDA
  - Bar plots of class distribution per split (train/validate/test) to confirm balanced classes.[file:1]
  - Sample grids of mushrooms per class and per split.
- Training curves
  - Loss and accuracy curves for baseline CNN and ResNet18.
  - Marked fine-tuning start epoch and 85% target accuracy line.[file:1]
- Evaluation plots
  - Confusion matrices for validation and test.
  - Misclassification grids highlighting:
    - False negatives (poisonous predicted as edible) in **red**.
    - False positives (edible predicted as poisonous) in **orange**.[file:1]
  - Confidence histograms and ROC curve for the poisonous class.

## Lessons Learned
- Transfer learning from a pretrained ResNet18 significantly outperforms a small CNN when data is limited.
- Two-phase training (frozen backbone then selective unfreezing) helps stabilize training and avoid catastrophic forgetting.
- Heavy augmentation is important to reduce overfitting on relatively small mushroom datasets.
- Safety-focused evaluation (false negatives vs false positives, high-confidence errors) is more informative than accuracy alone for this type of problem.[file:1]

## Failures and Limitations
- The model still produces **high-confidence** errors, including some false negatives (poisonous predicted as edible), which are dangerous in real-world use.[file:1]
- Backgrounds, lighting, and occlusion can mislead the model.
- Dataset is limited in species diversity and real-world variability; results may not generalize to all mushroom types or camera conditions.

## Future Improvements
- Data
  - Expand the dataset with more species and real user photos.
  - Add explicit “unknown / not sure” class for out-of-distribution mushrooms.
- Modeling
  - Experiment with more advanced architectures (e.g., ResNet50, EfficientNet, or vision transformers) and ensembling.
  - Use class-weighted loss or focal loss to penalize false negatives more strongly.
- Evaluation & Deployment
  - Calibrate probabilities and expose an abstain option when confidence is low.
  - Integrate with a simple web or mobile demo to test usability.
  - Add unit tests for data loaders and a small CLI script to score new images using `models/resnet18_mushroom_best.pth`.[file:1]

## Repository Structure

```text
.
├── notebooks/
│   └── 01_final_model.ipynb      # EDA, baseline CNN, ResNet18, evaluation
├── src/
│   ├── data.py                   # SafeImageFolder, path helpers
│   ├── transforms.py             # Train/val/test transforms
│   ├── models.py                 # BaselineCNN, ResNet18 factory
│   ├── train.py                  # Training loop, plotting
│   ├── evaluate.py               # Confusion matrix, ROC, PR, misclassifications
│   └── utils.py                  # Seeding, misc helpers
├── models/
│   └── resnet18_mushroom_best.pth  # Saved best ResNet18 weights
└── README.md

