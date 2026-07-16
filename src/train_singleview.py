"""
train_singleview.py
-------------------
Single-view EfficientNet-B5 baseline for binary mammogram classification
on the VinDr-Mammo dataset.

Each image (CC or MLO) is treated as an independent sample. Labels are
assigned at the breast level using the breast_birads column:
  BI-RADS 1, 2  ->  0  (routine)
  BI-RADS 3, 4, 5  ->  1  (needs attention)

Preprocessing:
  Otsu background crop + CLAHE + aspect-ratio resize + zero pad to 1024x1024

Augmentation:
  - Horizontal and vertical flip (each with 50% probability)
  - Random rotation up to +/-10 degrees (50% probability)
  Brightness augmentation is excluded because mammogram intensities
  encode tissue density which carries diagnostic significance.

Training:
  Single-phase differential learning rates from epoch 1.
  Backbone lr=2e-5, classification head lr=2e-4.
  Cosine annealing schedule, early stopping on validation AUC.

Usage:
    python train_singleview.py \
        --csv_path   /path/to/VinDr/breast-level_annotations.csv \
        --img_dir    /path/to/VinDr/images_png \
        --output_dir ./results/sv_f0 \
        --fold       0
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score)
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    def _load_effb5():
        return efficientnet_b5(
            weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
except ImportError:
    from torchvision.models import efficientnet_b5
    def _load_effb5():
        return efficientnet_b5(pretrained=True)


# Binary label mapping for BI-RADS scores
BIRADS_TO_LABEL = {
    'BI-RADS 1': 0, 'BI-RADS 2': 0,
    'BI-RADS 3': 1, 'BI-RADS 4': 1, 'BI-RADS 5': 1,
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def crop_breast(img):
    """
    Remove the black background from a mammogram image.

    Uses Otsu thresholding to binarize the image, then finds the
    largest connected component (the breast tissue) and crops to
    its bounding box with a 2% margin. Returns the original image
    if cropping fails or reduces the image below 10% of its original size.

    Args:
        img: Grayscale mammogram image as a NumPy array.

    Returns:
        Cropped grayscale image with background removed.
    """
    try:
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        if num_labels < 2:
            return img
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest, cv2.CC_STAT_LEFT]
        y = stats[largest, cv2.CC_STAT_TOP]
        w = stats[largest, cv2.CC_STAT_WIDTH]
        h = stats[largest, cv2.CC_STAT_HEIGHT]
        mx = max(1, int(0.02 * img.shape[1]))
        my = max(1, int(0.02 * img.shape[0]))
        x1, y1 = max(0, x-mx), max(0, y-my)
        x2, y2 = min(img.shape[1], x+w+mx), min(img.shape[0], y+h+my)
        cropped = img[y1:y2, x1:x2]
        return cropped if cropped.size >= 0.1 * img.size else img
    except Exception:
        return img


def load_and_preprocess(img_path, img_size=1024):
    """
    Load a mammogram and apply the full preprocessing pipeline.

    Steps:
        1. Load image in grayscale.
        2. Remove black background with Otsu thresholding.
        3. Enhance local contrast with CLAHE (clip=2.0, tile=8x8).
        4. Resize preserving aspect ratio to fit within img_size x img_size.
        5. Zero-pad to exactly img_size x img_size.
        6. Stack to 3 channels and normalize to [0, 1].

    Args:
        img_path: Path to the PNG image file.
        img_size: Target square size in pixels (default 1024).

    Returns:
        Float32 array of shape (3, img_size, img_size) in [0, 1].

    Raises:
        FileNotFoundError: If the image cannot be read from img_path.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    img    = crop_breast(img)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img    = clahe.apply(img)
    h, w   = img.shape[:2]
    scale  = img_size / max(h, w)
    new_h, new_w = int(h*scale), int(w*scale)
    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h  = img_size - new_h
    pad_w  = img_size - new_w
    img    = cv2.copyMakeBorder(img, pad_h//2, pad_h-pad_h//2,
                                 pad_w//2, pad_w-pad_w//2,
                                 cv2.BORDER_CONSTANT, value=0)
    return np.stack([img]*3, axis=0).astype(np.float32) / 255.0


def augment(img):
    """
    Apply random augmentation to a single mammogram image.

    Applies horizontal flip, vertical flip, and random rotation
    independently with 50% probability each. Brightness is not
    augmented as mammogram intensity values carry diagnostic information
    about tissue density.

    Args:
        img: Float32 array of shape (3, H, W).

    Returns:
        Augmented float32 array of shape (3, H, W).
    """
    if np.random.random() < 0.5:
        img = np.ascontiguousarray(img[:, :, ::-1])
    if np.random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1, :])
    img = np.transpose(img, (1, 2, 0))
    h, w = img.shape[:2]
    if np.random.random() < 0.5:
        angle = np.random.uniform(-10, 10)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SingleViewDataset(Dataset):
    """
    PyTorch Dataset for single-view mammogram classification.

    Each item is one image (CC or MLO) treated as an independent sample.
    Labels are assigned at the breast level from the breast_birads column.

    Args:
        df:       DataFrame with columns: study_id, image_id, label.
        img_dir:  Root directory containing images at
                  {img_dir}/{study_id}/{image_id}.png.
        img_size: Target image size in pixels (default 1024).
        do_aug:   Whether to apply data augmentation.
    """

    def __init__(self, df, img_dir, img_size=1024, do_aug=False):
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size= img_size
        self.do_aug  = do_aug
        n_pos = int((df['label']==1).sum())
        n_neg = int((df['label']==0).sum())
        print(f"SingleViewDataset: {len(df)} images | "
              f"pos={n_pos} neg={n_neg}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (image_tensor, label) where image_tensor is float32
            of shape (3, H, W) and label is a scalar float32.
        """
        row  = self.df.iloc[idx]
        path = os.path.join(
            self.img_dir, row['study_id'], f"{row['image_id']}.png")
        img  = load_and_preprocess(path, self.img_size)
        if self.do_aug:
            img = augment(img)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(img)), label


def build_dataframe(csv_path):
    """
    Load the VinDr-Mammo CSV and apply binary label mapping.

    Uses the official train/test split from the dataset.

    Args:
        csv_path: Path to breast-level_annotations.csv.

    Returns:
        Tuple of (train_df, test_df) DataFrames.
    """
    df = pd.read_csv(csv_path)
    df['label'] = df['breast_birads'].map(BIRADS_TO_LABEL)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    train_df = df[df['split']=='training'].reset_index(drop=True)
    test_df  = df[df['split']=='test'].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df


def get_cv_splits(train_df, n_splits=5, random_state=42):
    """
    Create stratified cross-validation splits at the study level.

    Splitting at the study level ensures that all images from the same
    patient always appear in the same fold, preventing patient-level
    data leakage between training and validation sets.

    Args:
        train_df:     Training DataFrame with study_id and label columns.
        n_splits:     Number of CV folds (default 5).
        random_state: Random seed for reproducibility.

    Returns:
        List of (fold_train_df, fold_val_df) tuples, one per fold.
    """
    study_labels = (train_df.groupby('study_id')['label']
                    .max().reset_index())
    study_ids = study_labels['study_id'].values
    labels    = study_labels['label'].values
    skf       = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    splits    = []
    for tr_idx, val_idx in skf.split(study_ids, labels):
        tr_s  = set(study_ids[tr_idx])
        val_s = set(study_ids[val_idx])
        splits.append((
            train_df[train_df['study_id'].isin(tr_s)].reset_index(drop=True),
            train_df[train_df['study_id'].isin(val_s)].reset_index(drop=True)
        ))
    print(f"Created {n_splits} study-level stratified folds")
    return splits


# ── Model ─────────────────────────────────────────────────────────────────────

class EfficientNetB5(nn.Module):
    """
    Single-view classifier built on EfficientNet-B5.

    Uses the EfficientNet-B5 feature extractor pretrained on ImageNet,
    followed by global average pooling, dropout, and a single linear
    output node for binary classification.

    Args:
        dropout: Dropout probability before the classification head (default 0.4).
    """

    def __init__(self, dropout=0.4):
        super().__init__()
        backbone      = _load_effb5()
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(2048, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logit tensor of shape (B, 1). Apply sigmoid for probability.
        """
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(self.dropout(x))


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Single-view mammogram classification on VinDr-Mammo')
    p.add_argument('--csv_path',   type=str, required=True,
                   help='Path to breast-level_annotations.csv')
    p.add_argument('--img_dir',    type=str, required=True,
                   help='Root directory of VinDr-Mammo PNG images')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save model checkpoints and results')
    p.add_argument('--fold',       type=int, default=0,
                   help='Cross-validation fold index (0-4)')
    p.add_argument('--n_folds',    type=int, default=5,
                   help='Total number of CV folds')
    p.add_argument('--img_size',   type=int, default=1024,
                   help='Target image size in pixels')
    p.add_argument('--epochs',     type=int, default=200,
                   help='Maximum number of training epochs')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Training batch size')
    p.add_argument('--lr_base',    type=float, default=2e-5,
                   help='Learning rate for the backbone')
    p.add_argument('--lr_head',    type=float, default=2e-4,
                   help='Learning rate for the classification head')
    p.add_argument('--patience',   type=int, default=20,
                   help='Early stopping patience in epochs')
    p.add_argument('--workers',    type=int, default=4,
                   help='Number of DataLoader worker processes')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Single-view EfficientNet-B5 baseline")
    print(f"  Fold   : {args.fold}/{args.n_folds-1}")
    print(f"  Device : {device}")
    print("=" * 60)

    # Build data splits
    train_df, test_df = build_dataframe(args.csv_path)
    folds             = get_cv_splits(train_df, args.n_folds)
    fold_train, fold_val = folds[args.fold]

    # WeightedRandomSampler ensures each batch sees roughly equal class counts
    labels  = fold_train['label'].values
    n_pos   = (labels==1).sum()
    n_neg   = (labels==0).sum()
    weights = np.where(labels==1, 1.0/n_pos, 1.0/n_neg)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights), replacement=True)

    # pos_weight scales the loss to compensate for class imbalance
    pw      = torch.tensor([n_neg/n_pos], dtype=torch.float32).to(device)

    kw = dict(num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(
        SingleViewDataset(fold_train, args.img_dir,
                          args.img_size, do_aug=True),
        batch_size=args.batch_size, sampler=sampler, **kw)
    val_loader   = DataLoader(
        SingleViewDataset(fold_val, args.img_dir,
                          args.img_size, do_aug=False),
        batch_size=args.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(
        SingleViewDataset(test_df, args.img_dir,
                          args.img_size, do_aug=False),
        batch_size=args.batch_size, shuffle=False, **kw)

    model     = EfficientNetB5().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Differential learning rates: backbone trained at a lower rate to
    # preserve ImageNet features, while the classification head adapts faster
    fc_ids      = set(id(p) for p in model.fc.parameters())
    head_params = [p for p in model.parameters() if id(p) in fc_ids]
    base_params = [p for p in model.parameters() if id(p) not in fc_ids]
    optimizer   = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr_base},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=1e-4)
    scaler    = GradScaler()

    # Cosine annealing gradually reduces the learning rate to near-zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    best_auc   = 0.0
    best_epoch = 0
    ckpt       = os.path.join(args.output_dir, 'best_model.pt')
    history    = []

    for epoch in range(args.epochs):
        # Training pass
        model.train()
        total = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs).squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / len(train_loader)

        # Validation pass
        model.eval()
        probs, labs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                p = torch.sigmoid(model(imgs.to(device)).squeeze(1))
                probs.extend(p.cpu().numpy())
                labs.extend(labels.numpy())
        val_auc = roc_auc_score(np.array(labs), np.array(probs))
        scheduler.step()

        history.append({'epoch': epoch+1,
                        'loss': train_loss, 'val_auc': val_auc})
        print(f"  Epoch {epoch+1:03d}/{args.epochs} | "
              f"loss={train_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt)
            print(f"    -> Best AUC: {best_auc:.4f}")

        if (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Evaluation on test set ────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Select classification threshold by maximizing F1 on validation set
    val_probs, val_labs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            p = torch.sigmoid(model(imgs.to(device)).squeeze(1))
            val_probs.extend(p.cpu().numpy())
            val_labs.extend(labels.numpy())
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(np.array(val_labs),
                      (np.array(val_probs)>=t).astype(int),
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # Evaluate on the official test set using the selected threshold
    test_probs, test_labs = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            p = torch.sigmoid(model(imgs.to(device)).squeeze(1))
            test_probs.extend(p.cpu().numpy())
            test_labs.extend(labels.numpy())
    y_true = np.array(test_labs)
    y_prob = np.array(test_probs)
    y_pred = (y_prob >= best_t).astype(int)

    results = {
        'fold':         args.fold,
        'threshold':    best_t,
        'auc':          float(roc_auc_score(y_true, y_prob)),
        'f1':           float(f1_score(y_true, y_pred, zero_division=0)),
        'precision':    float(precision_score(y_true, y_pred,
                                              zero_division=0)),
        'recall':       float(recall_score(y_true, y_pred,
                                           zero_division=0)),
        'accuracy':     float(accuracy_score(y_true, y_pred)),
        'best_val_auc': float(best_auc),
        'best_epoch':   best_epoch + 1,
    }

    print("\nTest results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
