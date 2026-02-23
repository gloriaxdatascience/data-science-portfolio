# src/train.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Universal Training Loop
# ---------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, num_epochs=15, finetune_epoch=None,
                finetune_layers=('layer4', 'fc'), finetune_lr=1e-4,
                device='cpu'):

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    best_preds, best_labels = [], []

    for epoch in range(num_epochs):

        # ---------------- Fine-tuning switch ----------------
        if finetune_epoch and epoch == finetune_epoch:
            print(f"\n=== Epoch {epoch+1}: Unfreezing {finetune_layers} ===\n")
            for name, param in model.named_parameters():
                if any(layer in name for layer in finetune_layers):
                    param.requires_grad = True

            backbone_params = [p for n, p in model.named_parameters()
                               if p.requires_grad and 'fc' not in n]
            head_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'fc' in n]

            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': finetune_lr},
                {'params': head_params,     'lr': finetune_lr * 10},
            ])

        # ---------------- Training ----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100 * correct / total

        # ---------------- Validation ----------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        epoch_preds, epoch_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += imgs.size(0)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / total
        epoch_val_acc = 100 * correct / total

        if scheduler:
            scheduler.step()

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.2f}%  |  "
              f"Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2f}%")

        # ---------------- Best checkpoint ----------------
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_preds, best_labels = epoch_preds[:], epoch_labels[:]
            print(f"  ★ New best model saved (Val Acc: {best_val_acc:.2f}%)")

    model.load_state_dict(best_weights)
    return model, history, best_preds, best_labels


# ---------------------------------------------------------------------
# Plot Curves
# ---------------------------------------------------------------------
def plot_curves(history, title='Training Curves', finetune_epoch=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Val Loss')
    ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'o-', label='Val Acc')
    ax2.axhline(y=85, color='red', linestyle='--', label='85% target')
    if finetune_epoch is not None:
        ax2.axvline(x=finetune_epoch + 1, color='orange',
                    linestyle='--', label='Fine-tune start')
    ax2.set(title='Accuracy (%)', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.legend(); ax2.grid(True)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
