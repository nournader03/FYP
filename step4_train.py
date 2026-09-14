import os
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             f1_score, classification_report, roc_curve,
                             precision_recall_curve, average_precision_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# CONFIG
SUBJECT_MAP_PATH = Path("data/processed/subject_map.json")
SPLIT_PATH = Path("data/processed/master_split.json")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE       = 8
ACCUM_STEPS      = 4        # gradient accumulation -> effective batch = 32
NUM_WORKERS      = 0        # 0 for Windows
EPOCHS           = 25
PATIENCE         = 7        # early stopping on val AUC
LR_BACKBONE      = 1e-5
LR_HEAD          = 1e-4
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTHING  = 0.1
MAX_GRAD_NORM    = 1.0
WARMUP_EPOCHS    = 3
RANDOM_SEED      = 42
IMG_SIZE         = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- AUGMENTATION --------------------------------------------------------------
def get_albu_train():
    """Albumentations transforms applied before ToTensor."""
    return A.Compose([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
    ])


TRAIN_TFM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TFM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Inception V3 requires 299x299 input
TRAIN_TFM_299 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TFM_299 = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- DATASET -------------------------------------------------------------------
class FCDSliceDataset(Dataset):
    """
    Slice-level dataset. Returns individual slices with subject ID for
    subject-level aggregation during evaluation.
    Labels are subject-level (FCD=1, Control=0) - same for all slices of a subject.
    """
    def __init__(self, subject_ids, subject_map, transform=None, albu_transform=None):
        self.transform       = transform
        self.albu_transform  = albu_transform
        self.samples         = []   # (slice_path, label_int, subj_id)

        for subj_id in subject_ids:
            info        = subject_map[subj_id]
            label_int   = info["label_int"]
            slice_paths = info.get("slice_paths", [])
            for sp in slice_paths:
                self.samples.append((sp, label_int, subj_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subj_id = self.samples[idx]

        # Load 16-bit PNG -> float32 [0,1] -> RGB
        img_arr = np.array(Image.open(path)).astype(np.float32) / 65535.0
        img_arr = np.clip(img_arr, 0, 1)

        # Albumentations (on float32 HxW array)
        if self.albu_transform is not None:
            augmented = self.albu_transform(image=img_arr)
            img_arr   = augmented["image"]

        # Convert to uint8 RGB PIL for torchvision transforms
        img_uint8 = (img_arr * 255).astype(np.uint8)
        pil_img   = Image.fromarray(img_uint8).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        return tensor, label, subj_id


# --- MODEL DEFINITIONS ---------------------------------------------------------
# Final model lineup:
#   efficientnet_b0   - lightweight baseline (5M params)
#   efficientnet_v2s  - modern stronger EfficientNet (21M params)
#   convnext_tiny     - best small-dataset CNN 2022-2025 (28M params)
#   densenet121       - dense connections suit subtle lesion detection (8M params)

def build_model(model_name: str) -> nn.Module:
    """Build pretrained model with replaced classification head."""
    if model_name == "efficientnet_b0":
        model    = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, 2)

    elif model_name == "efficientnet_v2s":
        model    = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, 2)

    elif model_name == "convnext_tiny":
        model    = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_feats, 2)

    elif model_name == "densenet121":
        model    = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, 2)

    # -- timm models ------------------------------------------------------------
    elif model_name == "resnet50v2":
        # ResNet50-D: ResNet50 with V2 improvements (pre-activation, anti-alias)
        model = timm.create_model("resnet50d", pretrained=True, num_classes=2)
        # Bump BN epsilon for AMP stability on 2GB VRAM (default 1e-5 can cause div-by-zero in fp16)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3

    elif model_name == "xception":
        # Original Xception - depthwise separable convolutions
        model = timm.create_model("xception", pretrained=True, num_classes=2)

    elif model_name == "inception_v3":
        # Inception V3 - requires 299x299 input (handled in dataset)
        model = timm.create_model("inception_v3", pretrained=True, num_classes=2)
        # Bump BN epsilon for AMP stability - Inception has BN throughout all branches
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose: efficientnet_b0, efficientnet_v2s, convnext_tiny, densenet121, "
            f"resnet50v2, xception, inception_v3"
        )

    return model.to(DEVICE)


def get_param_groups(model, model_name: str) -> list:
    """
    Differential learning rates - model-specific LRs for stability:
      efficientnet_b0/v2s : backbone=1e-5  head=1e-4  (default)
      convnext_tiny        : backbone=5e-6  head=5e-5
      densenet121          : backbone=1e-7  head=1e-6  (very conservative - dense gradient paths)
      resnet50v2           : backbone=5e-6  head=5e-5  (AMP instability on low-VRAM)
      xception             : backbone=5e-6  head=5e-5  (depthwise layers sensitive to high LR)
      inception_v3         : backbone=5e-6  head=5e-5
    All layers trainable from epoch 1 - no frozen phase.
    """
    if model_name in ("efficientnet_b0", "efficientnet_v2s"):
        head_params     = list(model.classifier.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    elif model_name == "convnext_tiny":
        head_params     = list(model.classifier.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        print(f"    ConvNeXt-Tiny LRs: backbone=5e-6  head=5e-5")
        return [
            {"params": backbone_params, "lr": 5e-6},
            {"params": head_params,     "lr": 5e-5},
        ]

    elif model_name == "densenet121":
        head_params     = list(model.classifier.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        # DenseNet121 prone to NaN - use very conservative LRs
        # Dense connections create long gradient paths that explode at higher LRs
        print(f"    DenseNet121 LRs: backbone=1e-7  head=1e-6")
        return [
            {"params": backbone_params, "lr": 1e-7},
            {"params": head_params,     "lr": 1e-6},
        ]

    # -- timm models - use timm's native head/backbone split -----------------
    elif model_name == "resnet50v2":
        # timm ResNet50-D - head is model.fc
        head_params     = list(model.fc.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        # Conservative LRs - ResNet50D prone to NaN under AMP on low-VRAM GPUs
        print(f"    ResNet50V2 LRs: backbone=5e-6  head=5e-5")
        return [
            {"params": backbone_params, "lr": 5e-6},
            {"params": head_params,     "lr": 5e-5},
        ]

    elif model_name == "xception":
        # timm Xception - head is model.fc
        head_params     = list(model.fc.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        # Conservative LRs - depthwise separable layers sensitive to high LR under AMP
        print(f"    Xception LRs: backbone=5e-6  head=5e-5")
        return [
            {"params": backbone_params, "lr": 5e-6},
            {"params": head_params,     "lr": 5e-5},
        ]

    elif model_name == "inception_v3":
        # timm Inception V3 - head is model.fc
        head_params     = list(model.fc.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        print(f"    InceptionV3 LRs: backbone=1e-7  head=1e-6")
        return [
            {"params": backbone_params, "lr": 1e-7},
            {"params": head_params,     "lr": 1e-6},
        ]


# --- CLASS WEIGHTS -------------------------------------------------------------
def compute_class_weights(subject_ids: list, subject_map: dict) -> torch.Tensor:
    """
    Data-driven class weights from actual slice counts in training fold.
    Inverse frequency weighting - never hardcoded.
    """
    counts = defaultdict(int)
    for subj_id in subject_ids:
        info      = subject_map[subj_id]
        label_int = info["label_int"]
        n_slices  = len(info.get("slice_paths", []))
        counts[label_int] += n_slices

    total    = sum(counts.values())
    n_classes = 2
    weights  = []
    for cls in range(n_classes):
        if counts[cls] > 0:
            weights.append(total / (n_classes * counts[cls]))
        else:
            weights.append(1.0)

    print(f"    Class weights -> Control: {weights[0]:.4f}  FCD: {weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# --- SCHEDULER -----------------------------------------------------------------
def get_scheduler(optimizer, n_epochs: int):
    """Linear warmup for WARMUP_EPOCHS, then CosineAnnealingLR."""
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        return 1.0

    warmup    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - WARMUP_EPOCHS, eta_min=1e-7
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )


# --- SUBJECT-LEVEL PREDICTION --------------------------------------------------
def subject_level_predictions(all_probs: dict) -> tuple:
    """
    Average slice-level probabilities per subject -> subject-level probability.
    Returns arrays of subj_ids, subject_probs (FCD prob), subject_labels.
    """
    subj_ids, subj_probs, subj_labels = [], [], []
    for subj_id, (probs_list, label) in all_probs.items():
        avg_prob = float(np.mean(probs_list))
        subj_ids.append(subj_id)
        subj_probs.append(avg_prob)
        subj_labels.append(label)
    return subj_ids, np.array(subj_probs), np.array(subj_labels)


# --- THRESHOLD CALIBRATION -----------------------------------------------------
def calibrate_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Sweep thresholds 0.05->0.95, compute metrics at each.
    Returns screening threshold (sensitivity>=0.95) and balanced threshold (max Youden's J).
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    results    = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1   = f1_score(labels, preds, zero_division=0)
        j    = sens + spec - 1
        results.append({"t": t, "sens": sens, "spec": spec, "f1": f1, "j": j})

    if not results:
        return {"screening_threshold": 0.5, "balanced_threshold": 0.5, "sweep": []}

    # Screening: lowest t where sensitivity >= 0.95
    # If no threshold achieves sens >= 0.95, use the threshold with highest sensitivity
    # (i.e. the lowest threshold in the sweep) and flag it clearly
    screening_candidates = [r for r in results if r["sens"] >= 0.95]
    if screening_candidates:
        screening_t = screening_candidates[0]["t"]
        screening_achieved = True
    else:
        # Fall back to lowest threshold = highest achievable sensitivity
        screening_t = results[0]["t"]
        screening_achieved = False

    # Balanced: max Youden's J
    balanced_t = max(results, key=lambda r: r["j"])["t"]

    return {
        "screening_threshold":          float(screening_t),
        "screening_sensitivity_achieved": screening_achieved,
        "balanced_threshold":           float(balanced_t),
        "sweep":                        results,
    }


# --- SUBJECT-LEVEL METRICS -----------------------------------------------------
def compute_subject_metrics(probs: np.ndarray, labels: np.ndarray,
                            threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens = spec = 0.0

    return {"auc": auc, "acc": acc, "f1": f1,
            "sensitivity": sens, "specificity": spec,
            "threshold": threshold}


# --- TRAIN ONE EPOCH -----------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, use_amp=True):
    model.train()
    total_loss   = 0.0
    nan_batches  = 0
    optimizer.zero_grad()

    # Keep a snapshot of last known good weights for NaN rollback
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for step, (imgs, labels, _) in enumerate(loader):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        amp_ctx = torch.cuda.amp.autocast() if use_amp else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            logits = model(imgs)
            loss   = criterion(logits, labels) / ACCUM_STEPS

        # Check logits AND loss for NaN/Inf - logit NaN means weights are already corrupt
        if torch.isnan(logits).any() or torch.isinf(logits).any() or \
           torch.isnan(loss)         or torch.isinf(loss):
            nan_batches += 1
            print(f"    WARNING: NaN/Inf at step {step+1} - rolling back to last good weights")
            model.load_state_dict({k: v.to(DEVICE) for k, v in last_good_state.items()})
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            valid_gradients = all(
                p.grad is None or not torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if valid_gradients:
                if use_amp:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
                last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                print(f"    WARNING: NaN gradients at step {step+1} - skipping update")
            if use_amp:
                scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

    if nan_batches > 0:
        print(f"    WARNING: Epoch had {nan_batches} NaN batch(es) - weights rolled back each time")

    return total_loss / max(len(loader) - nan_batches, 1)


# --- EVALUATE ONE EPOCH --------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, subject_map, use_amp=True):
    model.eval()
    total_loss  = 0.0
    all_probs   = defaultdict(lambda: ([], None))

    for imgs, labels, subj_ids in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        amp_ctx = torch.cuda.amp.autocast() if use_amp else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            logits = model(imgs)
            loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        # Skip batch if model output is still NaN (shouldn't happen after rollback, but guard anyway)
        if np.isnan(probs).any():
            continue
        total_loss += loss.item()

        for prob, label, subj_id in zip(probs, labels.cpu().numpy(), subj_ids):
            existing_probs, _ = all_probs[subj_id]
            existing_probs.append(float(prob))
            all_probs[subj_id] = (existing_probs, int(label))

    _, subj_probs, subj_labels = subject_level_predictions(dict(all_probs))
    auc = roc_auc_score(subj_labels, subj_probs) if len(np.unique(subj_labels)) > 1 else 0.0

    return total_loss / len(loader), auc, dict(all_probs)


# --- PLOTS ---------------------------------------------------------------------
def plot_training_curves(train_losses, val_losses, val_aucs, fold, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold} - Loss"); ax1.legend()

    ax2.plot(epochs, val_aucs, label="Val AUC", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_title(f"Fold {fold} - Val AUC"); ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"fold{fold}_training_curves.png", dpi=100)
    plt.close()


def plot_confusion_matrix(labels, preds, fold, threshold, out_dir):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "FCD"])
    ax.set_yticklabels(["Control", "FCD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold} Confusion Matrix (t={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / f"fold{fold}_confusion_matrix.png", dpi=100)
    plt.close()


def plot_roc_curve(labels, probs, fold, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc          = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"Fold {fold} ROC Curve (Subject Level)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"fold{fold}_roc_curve.png", dpi=100)
    plt.close()


def plot_threshold_sweep(sweep, fold, out_dir):
    if not sweep:
        return
    ts    = [r["t"]    for r in sweep]
    sens  = [r["sens"] for r in sweep]
    spec  = [r["spec"] for r in sweep]
    f1s   = [r["f1"]   for r in sweep]
    js    = [r["j"]    for r in sweep]

    plt.figure(figsize=(8, 5))
    plt.plot(ts, sens, label="Sensitivity", marker="o")
    plt.plot(ts, spec, label="Specificity", marker="s")
    plt.plot(ts, f1s,  label="F1",          marker="^")
    plt.plot(ts, js,   label="Youden's J",  marker="D")
    plt.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="Sensitivity=0.95")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title(f"Fold {fold} Threshold Sweep (Subject Level)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"fold{fold}_threshold_sweep.png", dpi=100)
    plt.close()


# --- TRAIN ONE FOLD ------------------------------------------------------------
def train_fold(model_name, fold_idx, fold_data, subject_map, out_dir):
    print(f"\n  {'='*50}")
    print(f"  Fold {fold_idx + 1} / 5")
    print(f"  {'='*50}")

    train_ids = fold_data["train"]
    val_ids   = fold_data["val"]
    print(f"  Train subjects: {len(train_ids)}  |  Val subjects: {len(val_ids)}")

    # -- Datasets & loaders --------------------------------------------------
    # Inception V3 requires 299x299 input
    train_tfm = TRAIN_TFM_299 if model_name == "inception_v3" else TRAIN_TFM
    val_tfm   = VAL_TFM_299   if model_name == "inception_v3" else VAL_TFM

    train_ds = FCDSliceDataset(train_ids, subject_map,
                               transform=train_tfm,
                               albu_transform=get_albu_train())
    val_ds   = FCDSliceDataset(val_ids, subject_map,
                               transform=val_tfm,
                               albu_transform=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train slices: {len(train_ds)}  |  Val slices: {len(val_ds)}")

    # -- Model ---------------------------------------------------------------
    model       = build_model(model_name)
    param_groups = get_param_groups(model, model_name)
    optimizer   = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler   = get_scheduler(optimizer, EPOCHS)
    # Conservative scaler for models prone to NaN under AMP on low-VRAM GPUs
    # EfficientNet/ConvNeXt use default to match already-obtained results exactly
    # Inception V3 disables AMP entirely - per-step NaN even at 1e-7 LR under fp16
    use_amp = model_name != "inception_v3"
    if model_name in ("resnet50v2", "densenet121", "xception"):
        scaler = torch.cuda.amp.GradScaler(init_scale=4096, growth_interval=200)
    elif use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None  # not used when AMP is disabled
        print(f"    WARNING: AMP disabled for {model_name} - running in full fp32")

    # -- Class weights -------------------------------------------------------
    class_weights = compute_class_weights(train_ids, subject_map)
    criterion     = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=LABEL_SMOOTHING)

    # -- Training loop --------------------------------------------------------
    best_auc        = 0.0
    best_epoch      = 0
    patience_counter = 0
    best_state      = None

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, scaler, use_amp)
        val_loss, val_auc, _   = evaluate(model, val_loader, criterion, subject_map, use_amp)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}"
              + (" <- best" if val_auc > best_auc else ""))

        if val_auc > best_auc:
            best_auc     = val_auc
            best_epoch   = epoch
            patience_counter = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    # -- Restore best weights -------------------------------------------------
    model.load_state_dict(best_state)

    # -- Save checkpoint ------------------------------------------------------
    ckpt_path = out_dir / f"fold{fold_idx + 1}_best.pt"
    torch.save({
        "model":       model_name,
        "fold":        fold_idx + 1,
        "best_epoch":  best_epoch,
        "best_val_auc": best_auc,
        "state_dict":  best_state,
    }, ckpt_path)
    print(f"  Saved checkpoint -> {ckpt_path}")

    # -- Final val evaluation with best weights --------------------------------
    _, _, val_all_probs = evaluate(model, val_loader, criterion, subject_map)
    _, subj_probs, subj_labels = subject_level_predictions(val_all_probs)

    # Threshold calibration on val set
    calib    = calibrate_threshold(subj_probs, subj_labels)
    screen_t = calib["screening_threshold"]
    bal_t    = calib["balanced_threshold"]

    metrics_screen  = compute_subject_metrics(subj_probs, subj_labels, screen_t)
    metrics_balanced = compute_subject_metrics(subj_probs, subj_labels, bal_t)

    screen_achieved = calib.get("screening_sensitivity_achieved", True)
    screen_note = "" if screen_achieved else "  WARNING: sens>=0.95 not achieved, using lowest threshold"
    print(f"\n  Subject-level results (fold {fold_idx + 1}):")
    print(f"  Screening  threshold={screen_t:.2f}: "
          f"AUC={metrics_screen['auc']:.3f}  "
          f"Sens={metrics_screen['sensitivity']:.3f}  "
          f"Spec={metrics_screen['specificity']:.3f}  "
          f"F1={metrics_screen['f1']:.3f}{screen_note}")
    print(f"  Balanced   threshold={bal_t:.2f}: "
          f"AUC={metrics_balanced['auc']:.3f}  "
          f"Sens={metrics_balanced['sensitivity']:.3f}  "
          f"Spec={metrics_balanced['specificity']:.3f}  "
          f"F1={metrics_balanced['f1']:.3f}")

    # Classification report
    preds_bal = (subj_probs >= bal_t).astype(int)
    print(f"\n  Classification report (balanced threshold):")
    print(classification_report(subj_labels, preds_bal,
                                target_names=["Control", "FCD"],
                                labels=[0, 1],
                                zero_division=0))

    # -- Plots ----------------------------------------------------------------
    results_fold_dir = RESULTS_DIR / model_name
    results_fold_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(train_losses, val_losses, val_aucs,
                         fold_idx + 1, results_fold_dir)
    plot_confusion_matrix(subj_labels, preds_bal,
                          fold_idx + 1, bal_t, results_fold_dir)
    plot_roc_curve(subj_labels, subj_probs, fold_idx + 1, results_fold_dir)
    plot_threshold_sweep(calib["sweep"], fold_idx + 1, results_fold_dir)

    fold_result = {
        "fold":              fold_idx + 1,
        "best_epoch":        best_epoch,
        "best_val_auc":      best_auc,
        "screening":         metrics_screen,
        "balanced":          metrics_balanced,
        "screening_threshold": screen_t,
        "balanced_threshold":  bal_t,
        "train_losses":      train_losses,
        "val_losses":        val_losses,
        "val_aucs":          val_aucs,
    }

    return fold_result, val_all_probs


# --- MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["efficientnet_b0", "efficientnet_v2s", "convnext_tiny", "densenet121", "resnet50v2", "xception", "inception_v3"],
                        help="Model to train: efficientnet_b0 | efficientnet_v2s | convnext_tiny | densenet121 | resnet50v2 | xception | inception_v3")
    args = parser.parse_args()
    model_name = args.model

    print("\n" + "=" * 60)
    print(f"  FCD PROJECT - STEP 4: TRAINING ({model_name.upper()})")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -- Load data -----------------------------------------------------------
    with open(SUBJECT_MAP_PATH) as f:
        subject_map = json.load(f)

    with open(SPLIT_PATH) as f:
        master_split = json.load(f)

    folds         = master_split["folds"]
    test_subjects = master_split["test_subjects"]
    print(f"  Loaded {len(subject_map)} subjects")
    print(f"  Test set locked: {len(test_subjects)} subjects (not touched)")
    print(f"  Training with {len(folds)}-fold CV\n")

    # -- Output dirs ---------------------------------------------------------
    model_out_dir   = MODELS_DIR / model_name
    results_out_dir = RESULTS_DIR / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    results_out_dir.mkdir(parents=True, exist_ok=True)

    # -- 5-fold CV -----------------------------------------------------------
    all_fold_results = []
    oof_probs        = {}   # subject_id -> avg prob across folds it appeared in val

    for fold_idx, fold_data in enumerate(folds):
        ckpt_path = model_out_dir / f"fold{fold_idx + 1}_best.pt"

        # -- Resume logic - skip already completed folds ----------------------
        if ckpt_path.exists():
            print(f"\n  Fold {fold_idx + 1}: checkpoint already exists - loading and skipping")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            # Rebuild val predictions from checkpoint for OOF accumulation
            model = build_model(model_name)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            val_ids  = fold_data["val"]
            val_ds   = FCDSliceDataset(val_ids, subject_map, transform=VAL_TFM)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
            class_weights = compute_class_weights(fold_data["train"], subject_map)
            criterion     = nn.CrossEntropyLoss(weight=class_weights,
                                                label_smoothing=LABEL_SMOOTHING)
            _, _, val_all_probs = evaluate(model, val_loader, criterion, subject_map)
            _, subj_probs, subj_labels = subject_level_predictions(val_all_probs)

            calib     = calibrate_threshold(subj_probs, subj_labels)
            screen_t  = calib["screening_threshold"]
            bal_t     = calib["balanced_threshold"]
            metrics_screen   = compute_subject_metrics(subj_probs, subj_labels, screen_t)
            metrics_balanced = compute_subject_metrics(subj_probs, subj_labels, bal_t)

            fold_result = {
                "fold":               fold_idx + 1,
                "best_epoch":         ckpt.get("best_epoch", -1),
                "best_val_auc":       ckpt.get("best_val_auc", 0.0),
                "screening":          metrics_screen,
                "balanced":           metrics_balanced,
                "screening_threshold": screen_t,
                "balanced_threshold":  bal_t,
                "train_losses":       [],
                "val_losses":         [],
                "val_aucs":           [],
            }
            all_fold_results.append(fold_result)

            for subj_id, (probs_list, label) in val_all_probs.items():
                avg_prob = float(np.mean(probs_list))
                oof_probs[subj_id] = {"prob": avg_prob, "label": label}

            print(f"  Fold {fold_idx + 1}: loaded - val AUC={ckpt.get('best_val_auc', 0.0):.4f}")
            del model
            torch.cuda.empty_cache()
            continue

        fold_result, val_probs = train_fold(
            model_name, fold_idx, fold_data, subject_map, model_out_dir
        )
        all_fold_results.append(fold_result)

        # Accumulate OOF predictions
        for subj_id, (probs_list, label) in val_probs.items():
            avg_prob = float(np.mean(probs_list))
            oof_probs[subj_id] = {"prob": avg_prob, "label": label}

    # -- OOF summary ---------------------------------------------------------
    oof_subj_ids  = list(oof_probs.keys())
    oof_subj_prbs = np.array([oof_probs[s]["prob"]  for s in oof_subj_ids])
    oof_subj_lbls = np.array([oof_probs[s]["label"] for s in oof_subj_ids])

    oof_auc = roc_auc_score(oof_subj_lbls, oof_subj_prbs) \
              if len(np.unique(oof_subj_lbls)) > 1 else 0.0

    # Save OOF predictions for ensemble stacking
    np.save(model_out_dir / "oof_subj_ids.npy",  np.array(oof_subj_ids))
    np.save(model_out_dir / "oof_probs.npy",     oof_subj_prbs)
    np.save(model_out_dir / "oof_labels.npy",    oof_subj_lbls)
    print(f"\n  OOF AUC ({model_name}): {oof_auc:.4f}")
    print(f"  Saved OOF predictions -> {model_out_dir}/oof_*.npy")

    # -- Cross-fold summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  5-FOLD CV SUMMARY - {model_name.upper()}")
    print("=" * 60)

    for metric_key, threshold_key in [("screening", "screening_threshold"),
                                       ("balanced",  "balanced_threshold")]:
        aucs  = [r[metric_key]["auc"]         for r in all_fold_results]
        senss = [r[metric_key]["sensitivity"]  for r in all_fold_results]
        specs = [r[metric_key]["specificity"]  for r in all_fold_results]
        f1s   = [r[metric_key]["f1"]           for r in all_fold_results]
        ts    = [r[threshold_key]              for r in all_fold_results]

        label = "SCREENING (sens>=0.95)" if metric_key == "screening" else "BALANCED (max Youden's J)"
        print(f"\n  {label}  threshold={np.mean(ts):.2f}+/-{np.std(ts):.2f}")
        print(f"  {'Metric':<15} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*35}")
        for name, vals in [("AUC", aucs), ("Sensitivity", senss),
                            ("Specificity", specs), ("F1", f1s)]:
            print(f"  {name:<15} {np.mean(vals):>8.3f} {np.std(vals):>8.3f}")

    print(f"\n  OOF AUC: {oof_auc:.4f}")

    # -- Save all fold metrics ------------------------------------------------
    metrics_path = results_out_dir / "fold_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model":            model_name,
            "oof_auc":          oof_auc,
            "fold_results":     all_fold_results,
        }, f, indent=2)
    print(f"\n  Saved fold metrics -> {metrics_path}")

    # -- Aggregated confusion matrix ------------------------------------------
    # Plot OOF confusion matrix at mean balanced threshold
    mean_bal_t    = np.mean([r["balanced_threshold"] for r in all_fold_results])
    oof_preds_bal = (oof_subj_prbs >= mean_bal_t).astype(int)
    plot_confusion_matrix(oof_subj_lbls, oof_preds_bal,
                          "OOF", mean_bal_t, results_out_dir)

    # OOF ROC curve
    plot_roc_curve(oof_subj_lbls, oof_subj_prbs, "OOF", results_out_dir)

    print(f"\n  [OK] Training complete for {model_name}")
    print(f"  Checkpoints -> {model_out_dir}/")
    print(f"  Results     -> {results_out_dir}/")
    print(f"  Next models: efficientnet_v2s -> convnext_tiny -> densenet121")
    print(f"  After all 4: run step5_ensemble.py")
    print("=" * 60 + "\n")