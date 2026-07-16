"""
train_resize_experiment.py
--------------------------
Dual-view mammogram classification on VinDr-Mammo using a Siamese
EfficientNet-B5 backbone with optional supervised contrastive learning.

Each breast is represented by a paired CC and MLO image. Both views
are encoded through the same backbone, and their embeddings are averaged
before the classification head. An optional projection head enables
contrastive regularization during training and is discarded at inference.

Label mapping (BI-RADS binary classification):
  BI-RADS 1, 2  ->  0  (routine, no follow-up required)
  BI-RADS 3, 4, 5  ->  1  (needs attention, follow-up indicated)

Resize strategies (see dataset_vindr_resize.py for details):
  a: Direct resize to 1024x1024 (distorts aspect ratio)
  b: Aspect-ratio resize + zero pad to 1024x1024 (default)
  c: Aspect-ratio resize to 614x1024, no padding

Usage:
    python train_resize_experiment.py \
        --csv_path   /path/to/VinDr/breast-level_annotations.csv \
        --img_dir    /path/to/VinDr/images_png \
        --output_dir ./results/dualview_f0 \
        --strategy   b \
        --fold       0 \
        --single_phase \
        --no_contrastive
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_vindr_resize import (VinDrPairedDataset,
                                   build_breast_dataframe,
                                   get_cv_splits)

try:
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    def _load_effb5():
        return efficientnet_b5(
            weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
except ImportError:
    from torchvision.models import efficientnet_b5
    def _load_effb5():
        return efficientnet_b5(pretrained=True)


# ── Model ─────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Maps the 2048-dimensional backbone embedding to a lower-dimensional
    space (default 128-dim) where the contrastive loss operates.
    This head is used only during training and discarded at inference,
    following the design in SimCLR (Chen et al., 2020) and SupCon
    (Khosla et al., 2020).

    Architecture: Linear -> BatchNorm -> ReLU -> Linear

    Args:
        in_dim:   Input dimension (default 2048, matching EfficientNet-B5).
        proj_dim: Output projection dimension (default 128).
    """

    def __init__(self, in_dim=2048, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, proj_dim))

    def forward(self, x):
        return self.net(x)


class MultiViewBaseline(nn.Module):
    """
    Dual-view mammogram classifier using a Siamese EfficientNet-B5 backbone.

    Both CC and MLO images are processed through the same backbone with
    shared weights. The resulting embeddings are averaged and passed to
    a binary classification head.

    A separate projection head is included for contrastive regularization
    during training. It takes the same embeddings as the classifier but
    maps them to a lower-dimensional space. This separation ensures the
    contrastive objective does not distort the features used for classification.
    The projection head is discarded at inference — only the classifier is used.

    Args:
        proj_dim: Projection head output dimension (default 128).
        dropout:  Dropout probability before the classification head (default 0.4).
    """

    def __init__(self, proj_dim=128, dropout=0.4):
        super().__init__()
        backbone      = _load_effb5()
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(p=dropout)
        self.projector = ProjectionHead(2048, proj_dim)
        self.fc        = nn.Linear(2048, 1)
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone parameters for initial training warmup."""
        for p in self.features.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"MultiViewBaseline: {total:,} total | "
              f"{trainable:,} trainable")

    def unfreeze(self):
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
        print(f"MultiViewBaseline: unfrozen "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def encode(self, x):
        """
        Extract a 2048-dimensional embedding from a single view.

        Args:
            x: Image tensor of shape (B, 3, H, W).

        Returns:
            Embedding tensor of shape (B, 2048).
        """
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.dropout(x)

    def forward(self, cc, mlo):
        """
        Forward pass for a paired CC+MLO input.

        Args:
            cc:  CC view tensor of shape (B, 3, H, W).
            mlo: MLO view tensor of shape (B, 3, H, W).

        Returns:
            Tuple of (logits, z_cc, z_mlo):
              logits: Classification logits of shape (B, 1).
              z_cc:   CC projection embeddings of shape (B, proj_dim).
              z_mlo:  MLO projection embeddings of shape (B, proj_dim).
        """
        emb_cc  = self.encode(cc)
        emb_mlo = self.encode(mlo)
        z_cc    = self.projector(emb_cc)
        z_mlo   = self.projector(emb_mlo)
        fused   = (emb_cc + emb_mlo) / 2.0  # Average the two view embeddings
        logits  = self.fc(fused)
        return logits, z_cc, z_mlo


# ── Contrastive losses ────────────────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor embedding, positives are defined as:
      - The paired view (CC paired with its MLO, and vice versa).
      - Any other embedding in the batch with the same class label.

    All remaining embeddings are negatives. The loss pulls positives
    closer and pushes negatives apart in the normalized embedding space.

    The denominator is computed using logsumexp for numerical stability,
    and torch.where is used instead of mask multiplication to avoid
    NaN values that arise from multiplying -inf by zero.

    Args:
        temperature: Softmax temperature scaling factor (default 0.07).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_cc, z_mlo, labels):
        """
        Args:
            z_cc:   CC projection embeddings of shape (B, D).
            z_mlo:  MLO projection embeddings of shape (B, D).
            labels: Binary class labels of shape (B,).

        Returns:
            Scalar contrastive loss value.
        """
        B      = z_cc.shape[0]
        device = z_cc.device

        # Normalize embeddings to unit sphere before computing similarity
        z_cc  = F.normalize(z_cc,  dim=1)
        z_mlo = F.normalize(z_mlo, dim=1)
        z     = torch.cat([z_cc, z_mlo], dim=0)       # (2B, D)
        labs  = torch.cat([labels, labels], dim=0)    # (2B,)

        sim   = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Positive mask: paired view (CC<->MLO) OR same class label
        pair_mask  = torch.zeros(2*B, 2*B, device=device, dtype=torch.bool)
        idx        = torch.arange(B, device=device)
        pair_mask[idx, idx+B] = True
        pair_mask[idx+B, idx] = True

        label_mask = (labs.unsqueeze(0) == labs.unsqueeze(1))
        self_mask  = torch.eye(2*B, device=device, dtype=torch.bool)
        pos_mask   = (pair_mask | label_mask) & ~self_mask

        # Compute log-softmax, excluding self-similarity from denominator
        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_prob    = sim_no_self - torch.logsumexp(
            sim_no_self, dim=1, keepdim=True)

        # Average loss over all positive pairs for each anchor
        n_pos      = pos_mask.sum(dim=1).float().clamp(min=1)
        loss_per   = -(torch.where(pos_mask,
                                   log_prob,
                                   torch.zeros_like(log_prob))
                       .sum(dim=1)) / n_pos
        return loss_per.mean()


class MemoryBankSupCon(nn.Module):
    """
    Supervised Contrastive Loss with a memory bank of past embeddings.

    With small batch sizes and severe class imbalance, most batches
    contain very few positive samples, which limits the contrastive
    signal. A memory bank stores embeddings from recent batches and
    includes them in the loss computation, effectively increasing the
    number of available positives and negatives without requiring a
    larger batch size.

    The memory bank operates as a fixed-size queue. New embeddings
    are added at the current pointer position and the oldest embeddings
    are overwritten when the bank is full. The bank is updated after
    each forward pass and is not updated through backpropagation.

    References:
        - Khosla et al., Supervised Contrastive Learning, NeurIPS 2020.
        - He et al., Momentum Contrast for Unsupervised Visual
          Representation Learning, CVPR 2020.

    Args:
        bank_size:   Number of embeddings to store (default 512).
        proj_dim:    Embedding dimension (default 128).
        temperature: Softmax temperature (default 0.07).
    """

    def __init__(self, bank_size=512, proj_dim=128, temperature=0.07):
        super().__init__()
        self.bank_size   = bank_size
        self.temperature = temperature

        # Initialize the bank with random unit vectors and zero labels
        self.register_buffer('bank_z',
            F.normalize(torch.randn(bank_size, proj_dim), dim=1))
        self.register_buffer('bank_labels',
            torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer('ptr',
            torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor, labels: torch.Tensor):
        """
        Add new embeddings to the memory bank, overwriting the oldest.

        Args:
            z:      New embeddings of shape (N, D).
            labels: Corresponding labels of shape (N,).
        """
        B    = z.shape[0]
        ptr  = int(self.ptr)

        if ptr + B > self.bank_size:
            # Wrap around the end of the queue
            first = self.bank_size - ptr
            self.bank_z[ptr:]      = z[:first]
            self.bank_labels[ptr:] = labels[:first].long()
            self.bank_z[:B-first]      = z[first:]
            self.bank_labels[:B-first] = labels[first:].long()
        else:
            self.bank_z[ptr:ptr+B]      = z
            self.bank_labels[ptr:ptr+B] = labels.long()

        self.ptr[0] = (ptr + B) % self.bank_size

    def forward(self, z_cc: torch.Tensor,
                z_mlo: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss using the current batch
        and all embeddings stored in the memory bank.

        Args:
            z_cc:   CC projection embeddings of shape (B, D).
            z_mlo:  MLO projection embeddings of shape (B, D).
            labels: Binary class labels of shape (B,).

        Returns:
            Scalar contrastive loss value.
        """
        B      = z_cc.shape[0]
        device = z_cc.device

        z_cc  = F.normalize(z_cc,  dim=1)
        z_mlo = F.normalize(z_mlo, dim=1)

        # Combine current batch embeddings
        z_cur    = torch.cat([z_cc, z_mlo], dim=0)           # (2B, D)
        labs_cur = torch.cat([labels, labels], dim=0)        # (2B,)

        # Combine current batch with memory bank for larger contrast set
        z_all    = torch.cat([z_cur, self.bank_z.clone()], dim=0)
        labs_all = torch.cat([labs_cur,
                               self.bank_labels.clone().float()], dim=0)

        N = z_all.shape[0]  # 2B + bank_size

        # Only current batch embeddings serve as anchors
        sim = torch.matmul(z_cur, z_all.T) / self.temperature  # (2B, N)

        # Self-similarity mask: prevent each anchor from matching itself
        self_mask = torch.zeros(2*B, N, device=device, dtype=torch.bool)
        self_mask[:, :2*B] = torch.eye(2*B, device=device, dtype=torch.bool)

        # Positive mask: paired view OR same class label
        pair_mask  = torch.zeros(2*B, N, device=device, dtype=torch.bool)
        idx        = torch.arange(B, device=device)
        pair_mask[idx, idx+B]   = True
        pair_mask[idx+B, idx]   = True

        label_mask = (labs_cur.unsqueeze(1) == labs_all.unsqueeze(0))
        pos_mask   = (pair_mask | label_mask) & ~self_mask

        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_prob    = sim_no_self - torch.logsumexp(
            sim_no_self, dim=1, keepdim=True)

        n_pos    = pos_mask.sum(dim=1).float().clamp(min=1)
        loss_per = -(torch.where(pos_mask,
                                 log_prob,
                                 torch.zeros_like(log_prob))
                     .sum(dim=1)) / n_pos

        # Update the memory bank with current batch embeddings
        self._enqueue(z_cur.detach(), labs_cur.detach())

        return loss_per.mean()


# ── Training utilities ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Dual-view mammogram classification on VinDr-Mammo')
    p.add_argument('--csv_path',   type=str, required=True,
                   help='Path to breast-level_annotations.csv')
    p.add_argument('--img_dir',    type=str, required=True,
                   help='Root directory of VinDr-Mammo PNG images')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save model checkpoints and results')
    p.add_argument('--strategy',   type=str, required=True,
                   choices=['a', 'b', 'c'],
                   help='Image resize strategy (a=squish, b=pad, c=rect)')
    p.add_argument('--fold',       type=int, default=0,
                   help='Cross-validation fold index (0-4)')
    p.add_argument('--epochs',     type=int, default=200,
                   help='Maximum number of training epochs')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Training batch size')
    p.add_argument('--lr_base',    type=float, default=2e-5,
                   help='Learning rate for the backbone')
    p.add_argument('--lr_head',    type=float, default=2e-4,
                   help='Learning rate for classification and projection heads')
    p.add_argument('--patience',   type=int, default=20,
                   help='Early stopping patience in epochs')
    p.add_argument('--workers',    type=int, default=4,
                   help='Number of DataLoader worker processes')
    p.add_argument('--temperature',type=float, default=0.07,
                   help='Temperature for contrastive loss')
    p.add_argument('--lambda_con', type=float, default=0.1,
                   help='Weight for the contrastive loss term')
    p.add_argument('--no_contrastive', action='store_true',
                   help='Disable contrastive loss, use classification only')
    p.add_argument('--single_phase', action='store_true',
                   help='Train all layers from epoch 1 with differential LRs')
    p.add_argument('--memory_bank', action='store_true',
                   help='Use memory bank extension for contrastive loss')
    p.add_argument('--bank_size',  type=int, default=512,
                   help='Number of embeddings in the memory bank')
    return p.parse_args()


def make_weighted_sampler(df):
    """
    Create a WeightedRandomSampler that balances class frequency per batch.

    Each sample is assigned a weight inversely proportional to its class
    frequency, so minority class samples are drawn more often.

    Args:
        df: DataFrame with a 'label' column.

    Returns:
        WeightedRandomSampler instance.
    """
    labels  = df['label'].values
    n_pos   = (labels == 1).sum()
    n_neg   = (labels == 0).sum()
    weights = np.where(labels == 1, 1.0/n_pos, 1.0/n_neg)
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights), replacement=True)


def build_phase1_optimizer(model, lr_head):
    """
    Optimizer for phase 1: only trains non-frozen parameters (heads).

    Args:
        model:   The MultiViewBaseline model with frozen backbone.
        lr_head: Learning rate for the trainable head parameters.

    Returns:
        Adam optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr_head, weight_decay=1e-4)


def build_phase2_optimizer(model, lr_base, lr_head):
    """
    Optimizer for full fine-tuning with differential learning rates.

    The backbone is trained at a lower rate to preserve the ImageNet
    features, while the classification and projection heads adapt faster.

    Args:
        model:   The MultiViewBaseline model (all parameters trainable).
        lr_base: Learning rate for the backbone.
        lr_head: Learning rate for the heads.

    Returns:
        Adam optimizer with per-parameter-group learning rates.
    """
    model.unfreeze()
    fc_ids      = set(id(p) for p in model.fc.parameters())
    proj_ids    = set(id(p) for p in model.projector.parameters())
    head_ids    = fc_ids | proj_ids
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.Adam([
        {'params': base_params, 'lr': lr_base},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=1e-4)


def train_one_epoch(model, loader, optimizer, cls_crit,
                    con_crit, lam, scaler, device,
                    no_contrastive=False):
    """
    Run one training epoch.

    Computes classification loss and optionally contrastive loss.
    Uses automatic mixed precision (AMP) for faster training on GPU.

    Args:
        model:          The MultiViewBaseline model.
        loader:         Training DataLoader.
        optimizer:      Optimizer instance.
        cls_crit:       Binary cross-entropy loss with pos_weight.
        con_crit:       Contrastive loss instance.
        lam:            Contrastive loss weight.
        scaler:         AMP gradient scaler.
        device:         Compute device (cuda or cpu).
        no_contrastive: If True, only the classification loss is used.

    Returns:
        Tuple of (mean_total_loss, mean_cls_loss, mean_con_loss).
    """
    model.train()
    tot = cls_t = con_t = n = 0
    for cc, mlo, labels in loader:
        cc, mlo, labels = (cc.to(device), mlo.to(device),
                           labels.to(device))
        optimizer.zero_grad()
        with autocast():
            logits, z_cc, z_mlo = model(cc, mlo)
            cls_loss = cls_crit(logits.squeeze(1), labels)
            if no_contrastive:
                loss     = cls_loss
                con_loss = torch.tensor(0.0)
            else:
                con_loss = con_crit(z_cc, z_mlo, labels)
                loss     = cls_loss + lam * con_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot   += loss.item()
        cls_t += cls_loss.item()
        con_t += con_loss.item()
        n     += 1
    return tot/n, cls_t/n, con_t/n


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate the model on a data split and compute AUC.

    Args:
        model:  The MultiViewBaseline model.
        loader: DataLoader (validation or test).
        device: Compute device.

    Returns:
        Tuple of (auc, y_prob, y_true) where y_prob and y_true are
        NumPy arrays of predicted probabilities and ground-truth labels.
    """
    model.eval()
    probs, labs = [], []
    for cc, mlo, labels in loader:
        cc, mlo = cc.to(device), mlo.to(device)
        logits, _, _ = model(cc, mlo)
        probs.extend(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
        labs.extend(labels.numpy())
    y_true = np.array(labs)
    y_prob = np.array(probs)
    return roc_auc_score(y_true, y_prob), y_prob, y_true


def find_threshold(y_true, y_prob):
    """
    Find the classification threshold that maximizes F1-score.

    Searches thresholds in [0.05, 0.95] with step 0.05.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold as a float.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (y_prob>=t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Strategy : {args.strategy.upper()}")
    print(f"  Fold     : {args.fold}")
    print(f"  Device   : {device}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, test_df = build_breast_dataframe(args.csv_path)
    folds             = get_cv_splits(train_df)
    fold_train, fold_val = folds[args.fold]

    sampler = make_weighted_sampler(fold_train)
    n_neg   = int((fold_train['label']==0).sum())
    n_pos   = int((fold_train['label']==1).sum())

    # pos_weight scales the loss to compensate for class imbalance
    pw      = torch.tensor([n_neg/n_pos], dtype=torch.float32).to(device)

    kw = dict(num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(
        VinDrPairedDataset(fold_train, args.img_dir,
                           args.strategy, augment=True),
        batch_size=args.batch_size, sampler=sampler, **kw)
    val_loader   = DataLoader(
        VinDrPairedDataset(fold_val, args.img_dir,
                           args.strategy, augment=False),
        batch_size=args.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(
        VinDrPairedDataset(test_df, args.img_dir,
                           args.strategy, augment=False),
        batch_size=args.batch_size, shuffle=False, **kw)

    # ── Model & Loss ──────────────────────────────────────────────────────────
    model     = MultiViewBaseline().to(device)
    cls_crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    if args.memory_bank:
        con_crit = MemoryBankSupCon(
            bank_size=args.bank_size,
            temperature=args.temperature).to(device)
        print(f"Using memory bank contrastive loss "
              f"(bank_size={args.bank_size})")
    else:
        con_crit = SupervisedContrastiveLoss(
            temperature=args.temperature).to(device)

    scaler = GradScaler()

    # Single-phase: unfreeze all layers from epoch 1 with differential LRs
    if args.single_phase:
        model.unfreeze()
        optimizer = build_phase2_optimizer(
            model, args.lr_base, args.lr_head)
        phase = 1
        print("Single-phase training — all layers from epoch 1")
    else:
        optimizer = build_phase1_optimizer(model, args.lr_head)
        phase = 1

    # Cosine annealing gradually reduces LR to near-zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    best_auc   = 0.0
    best_epoch = 0
    ckpt       = os.path.join(args.output_dir, 'best_model.pt')
    history    = []

    for epoch in range(args.epochs):
        # Two-phase mode: unfreeze backbone after 10 warmup epochs
        if not args.single_phase and epoch == 10 and phase == 1:
            print("\n>>> Phase 2: full fine-tuning <<<\n")
            optimizer = build_phase2_optimizer(
                model, args.lr_base, args.lr_head)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-7)
            phase = 2

        lam              = args.lambda_con
        tot, cls_l, con_l = train_one_epoch(
            model, train_loader, optimizer,
            cls_crit, con_crit, lam, scaler, device,
            no_contrastive=args.no_contrastive)
        val_auc, _, _    = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({'epoch': epoch+1, 'phase': phase,
                        'lambda': lam, 'total': tot,
                        'cls': cls_l, 'con': con_l,
                        'val_auc': val_auc})

        print(f"  [{phase}] Epoch {epoch+1:03d}/{args.epochs} | "
              f"λ={lam:.1f} | "
              f"tot={tot:.4f} cls={cls_l:.4f} con={con_l:.4f} | "
              f"val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt)
            print(f"    -> Best AUC: {best_auc:.4f}")

        if (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Evaluation on test set ────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Select threshold by maximizing F1 on the validation set
    _, y_prob_val, y_true_val = evaluate(model, val_loader, device)
    threshold = find_threshold(y_true_val, y_prob_val)
    print(f"Best threshold (val): {threshold:.2f}")

    # Evaluate on the official test set using the selected threshold
    test_auc, y_prob_test, y_true_test = evaluate(
        model, test_loader, device)
    y_pred = (y_prob_test >= threshold).astype(int)

    results = {
        'strategy':     args.strategy,
        'fold':         args.fold,
        'threshold':    threshold,
        'auc':          float(test_auc),
        'f1':           float(f1_score(y_true_test, y_pred,
                                       zero_division=0)),
        'precision':    float(precision_score(y_true_test, y_pred,
                                              zero_division=0)),
        'recall':       float(recall_score(y_true_test, y_pred,
                                           zero_division=0)),
        'accuracy':     float(accuracy_score(y_true_test, y_pred)),
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

    # Save training curves
    eps = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, [h['cls'] for h in history], label='classification')
    ax1.plot(eps, [h['con'] for h in history], label='contrastive')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'Strategy {args.strategy.upper()} — Training Losses')
    ax1.legend()
    ax2.plot(eps, [h['val_auc'] for h in history])
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'curves.png'), dpi=150)
    plt.close()
    print(f"\nSaved to {args.output_dir}/")


if __name__ == '__main__':
    main()
