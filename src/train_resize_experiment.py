"""
train_resize_experiment.py
--------------------------
Ablation study comparing three resize strategies for VinDr-Mammo
multi-view contrastive classification.

Fixed across all experiments:
  - Model: EfficientNet-B5 baseline, average fusion
  - Loss: SupCon (numerically stable logsumexp version)
  - Label split: BI-RADS 1+2 vs 3+4+5
  - Fold: 0
  - OTSU background crop + CLAHE preprocessing
  - Synchronized CC+MLO augmentation
  - Lambda warmup: 0.1 -> 0.3 -> 0.5
  - Two-phase training

Variable:
  --strategy a | b | c

Usage:
    python train_resize_experiment.py \
        --csv_path   /gpfs/.../VinDr/breast-level_annotations.csv \
        --img_dir    /gpfs/.../VinDr/images_png \
        --output_dir /gpfs/.../results/resize_a \
        --strategy   a
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
    Siamese EfficientNet-B5 with average fusion.
    Separate projection head for contrastive loss (discarded at inference).
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
        for p in self.features.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"MultiViewBaseline: {total:,} total | "
              f"{trainable:,} trainable")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        print(f"MultiViewBaseline: unfrozen "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.dropout(x)

    def forward(self, cc, mlo):
        emb_cc  = self.encode(cc)
        emb_mlo = self.encode(mlo)
        z_cc    = self.projector(emb_cc)
        z_mlo   = self.projector(emb_mlo)
        fused   = (emb_cc + emb_mlo) / 2.0
        logits  = self.fc(fused)
        return logits, z_cc, z_mlo


# ── Contrastive loss (numerically stable) ─────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Numerically stable SupCon loss using logsumexp.
    Previous version used torch.exp() directly which caused NaN
    with temperature=0.07 due to overflow.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_cc, z_mlo, labels):
        B      = z_cc.shape[0]
        device = z_cc.device

        z_cc  = F.normalize(z_cc,  dim=1)
        z_mlo = F.normalize(z_mlo, dim=1)
        z     = torch.cat([z_cc, z_mlo], dim=0)
        labs  = torch.cat([labels, labels], dim=0)

        sim   = torch.matmul(z, z.T) / self.temperature

        pair_mask  = torch.zeros(2*B, 2*B, device=device, dtype=torch.bool)
        idx        = torch.arange(B, device=device)
        pair_mask[idx, idx+B] = True
        pair_mask[idx+B, idx] = True

        label_mask = (labs.unsqueeze(0) == labs.unsqueeze(1))
        self_mask  = torch.eye(2*B, device=device, dtype=torch.bool)
        pos_mask   = (pair_mask | label_mask) & ~self_mask

        # Numerically stable: use logsumexp instead of log(sum(exp))
        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_prob    = sim_no_self - torch.logsumexp(
            sim_no_self, dim=1, keepdim=True)

        n_pos      = pos_mask.sum(dim=1).float().clamp(min=1)
        loss_per   = -(torch.where(pos_mask,
                                   log_prob,
                                   torch.zeros_like(log_prob))
                       .sum(dim=1)) / n_pos
        return loss_per.mean()


class MemoryBankSupCon(nn.Module):
    """
    Supervised Contrastive Loss with Memory Bank.

    Problem with standard SupCon on small datasets:
    With batch size 8 and 9.6% positives, each batch has ~1 positive.
    Not enough to compute meaningful contrastive gradients.

    Solution (MoCo-style memory bank, He et al. CVPR 2020):
    Store embeddings from recent batches in a queue.
    Contrastive loss computed on current batch + memory bank.
    This gives ~49 positives per step (512 bank * 9.6%) instead of ~1.

    Memory bank size 512: stores last 64 batches (batch_size=8).
    Large enough for meaningful positives, small enough to stay fresh.

    Reference:
    - Khosla et al., Supervised Contrastive Learning, NeurIPS 2020
    - He et al., Momentum Contrast for Unsupervised Visual Representation
      Learning, CVPR 2020
    """

    def __init__(self, bank_size=512, proj_dim=128, temperature=0.07):
        super().__init__()
        self.bank_size   = bank_size
        self.temperature = temperature

        # Initialize queues with random unit vectors
        self.register_buffer('bank_z',
            F.normalize(torch.randn(bank_size, proj_dim), dim=1))
        self.register_buffer('bank_labels',
            torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer('ptr',
            torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor, labels: torch.Tensor):
        """Add current batch to memory bank, remove oldest."""
        B    = z.shape[0]
        ptr  = int(self.ptr)

        # Handle wrap-around
        if ptr + B > self.bank_size:
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
        B      = z_cc.shape[0]
        device = z_cc.device

        z_cc  = F.normalize(z_cc,  dim=1)
        z_mlo = F.normalize(z_mlo, dim=1)

        # Current batch: 2B embeddings
        z_cur    = torch.cat([z_cc, z_mlo], dim=0)           # (2B, D)
        labs_cur = torch.cat([labels, labels], dim=0)        # (2B,)

        # Combine with memory bank
        z_all    = torch.cat([z_cur, self.bank_z.clone()], dim=0)    # (2B+K, D)
        labs_all = torch.cat([labs_cur,
                               self.bank_labels.clone().float()],
                              dim=0)                          # (2B+K,)

        N = z_all.shape[0]  # 2B + K

        # Similarity matrix — only current batch as anchors
        sim = torch.matmul(z_cur, z_all.T) / self.temperature  # (2B, 2B+K)

        # Self-mask — prevent anchor from matching itself
        self_mask = torch.zeros(2*B, N, device=device, dtype=torch.bool)
        self_mask[:, :2*B] = torch.eye(2*B, device=device, dtype=torch.bool)

        # Positive mask: paired view + same class label
        pair_mask  = torch.zeros(2*B, N, device=device, dtype=torch.bool)
        idx        = torch.arange(B, device=device)
        pair_mask[idx, idx+B]   = True
        pair_mask[idx+B, idx]   = True

        label_mask = (labs_cur.unsqueeze(1) == labs_all.unsqueeze(0))
        pos_mask   = (pair_mask | label_mask) & ~self_mask

        # Numerically stable log softmax
        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_prob    = sim_no_self - torch.logsumexp(
            sim_no_self, dim=1, keepdim=True)

        n_pos    = pos_mask.sum(dim=1).float().clamp(min=1)
        loss_per = -(torch.where(pos_mask,
                                 log_prob,
                                 torch.zeros_like(log_prob))
                     .sum(dim=1)) / n_pos

        # Update memory bank with current batch (no gradient)
        self._enqueue(z_cur.detach(), labs_cur.detach())

        return loss_per.mean()


# ── Training utilities ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path',   type=str, required=True)
    p.add_argument('--img_dir',    type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--strategy',   type=str, required=True,
                   choices=['a', 'b', 'c'])
    p.add_argument('--fold',       type=int, default=0)
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr_base',    type=float, default=2e-5)
    p.add_argument('--lr_head',    type=float, default=2e-4)
    p.add_argument('--patience',   type=int, default=15)
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--temperature',type=float, default=0.07)
    p.add_argument('--lambda_con',  type=float, default=0.5,
                   help='Weight for contrastive loss (default 0.5)')
    p.add_argument('--no_contrastive', action='store_true',
                   help='Disable contrastive loss — classification only')
    p.add_argument('--single_phase', action='store_true',
                   help='Single phase training — no frozen backbone phase')
    p.add_argument('--memory_bank', action='store_true',
                   help='Use memory bank for contrastive loss')
    p.add_argument('--bank_size',  type=int, default=512,
                   help='Memory bank size (default 512)')
    return p.parse_args()


def get_lambda(epoch):
    if epoch < 10:  return 0.1
    elif epoch < 20: return 0.3
    else:            return 0.5


def make_weighted_sampler(df):
    labels  = df['label'].values
    n_pos   = (labels==1).sum()
    n_neg   = (labels==0).sum()
    weights = np.where(labels==1, 1.0/n_pos, 1.0/n_neg)
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights), replacement=True)


def build_phase1_optimizer(model, lr_head):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr_head, weight_decay=1e-4)


def build_phase2_optimizer(model, lr_base, lr_head):
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

    scaler    = GradScaler()

    # Single phase: unfreeze everything from epoch 1
    if args.single_phase:
        model.unfreeze()
        optimizer = build_phase2_optimizer(
            model, args.lr_base, args.lr_head)
        phase = 1
        print("Single-phase training — all layers from epoch 1")
    else:
        optimizer = build_phase1_optimizer(model, args.lr_head)
        phase = 1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    best_auc   = 0.0
    best_epoch = 0
    ckpt       = os.path.join(args.output_dir, 'best_model.pt')
    history    = []

    for epoch in range(args.epochs):
        # Switch to phase 2 only if not single phase
        if not args.single_phase and epoch == 10 and phase == 1:
            print("\n>>> Phase 2 <<<\n")
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

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    _, y_prob_val, y_true_val = evaluate(model, val_loader, device)
    threshold = find_threshold(y_true_val, y_prob_val)
    print(f"Best threshold (val): {threshold:.2f}")

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

    # Loss curve
    eps = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, [h['cls'] for h in history], label='cls')
    ax1.plot(eps, [h['con'] for h in history], label='con')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'Strategy {args.strategy.upper()} — Losses')
    ax1.legend()
    ax2.plot(eps, [h['val_auc'] for h in history])
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
    ax2.set_title('Val AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'curves.png'), dpi=150)
    plt.close()
    print(f"\nSaved to {args.output_dir}/")


if __name__ == '__main__':
    main()
