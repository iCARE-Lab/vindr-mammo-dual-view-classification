"""
dataset_vindr_resize.py
-----------------------
Paired CC+MLO dataset for VinDr-Mammo with three resize strategies.

Strategy A: 1024x1024 squished (no aspect ratio preservation)
Strategy B: 1024x1024 aspect ratio preserved + zero padding
Strategy C: 614x1024 aspect ratio preserved, no padding, no distortion

All strategies apply:
  - OTSU background cropping
  - CLAHE (clip=2, tile=8x8)
  - Synchronized horizontal flip for CC+MLO pairs

Label mapping:
  BI-RADS 1, 2 -> 0 (routine)
  BI-RADS 3, 4, 5 -> 1 (needs attention)
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


BIRADS_TO_LABEL = {
    'BI-RADS 1': 0, 'BI-RADS 2': 0,
    'BI-RADS 3': 1, 'BI-RADS 4': 1, 'BI-RADS 5': 1,
}


def crop_breast(img: np.ndarray) -> np.ndarray:
    """OTSU threshold to crop black background."""
    try:
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        if num_labels < 2:
            return img
        largest  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
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


def apply_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def resize_strategy_a(img: np.ndarray) -> np.ndarray:
    """Strategy A: squish to 1024x1024."""
    resized = cv2.resize(img, (1024, 1024),
                         interpolation=cv2.INTER_LINEAR)
    return np.stack([resized]*3, axis=0).astype(np.float32) / 255.0


def resize_strategy_b(img: np.ndarray) -> np.ndarray:
    """Strategy B: aspect ratio preserve + zero pad to 1024x1024."""
    h, w   = img.shape[:2]
    scale  = 1024 / max(h, w)
    new_h  = int(h * scale)
    new_w  = int(w * scale)
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    pad_h  = 1024 - new_h
    pad_w  = 1024 - new_w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return np.stack([padded]*3, axis=0).astype(np.float32) / 255.0


def resize_strategy_c(img: np.ndarray) -> np.ndarray:
    """
    Strategy C: aspect ratio preserve, no padding.
    Fixed output size: 614x1024 (derived from 912x1520 native ratio).
    All images forced to exactly this size to allow batching.
    """
    resized = cv2.resize(img, (614, 1024),
                         interpolation=cv2.INTER_LINEAR)
    return np.stack([resized]*3, axis=0).astype(np.float32) / 255.0


RESIZE_FNS = {
    'a': resize_strategy_a,
    'b': resize_strategy_b,
    'c': resize_strategy_c,
}


def load_and_preprocess(img_path: str,
                         strategy: str = 'b') -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Load grayscale
    2. OTSU crop
    3. CLAHE
    4. Resize (strategy a, b, or c)
    Returns CHW float32 [0,1]
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    img = crop_breast(img)
    img = apply_clahe(img)
    return RESIZE_FNS[strategy](img)


def augment_pair(cc: np.ndarray,
                 mlo: np.ndarray) -> tuple:
    """
    Synchronized augmentation for CC+MLO pairs.
    Horizontal flip is shared — prevents laterality mismatch.
    Rotation and brightness are independent per view.
    """
    if np.random.random() < 0.5:
        cc  = np.ascontiguousarray(cc[:, :, ::-1])
        mlo = np.ascontiguousarray(mlo[:, :, ::-1])

    def _aug(img):
        img = np.transpose(img, (1, 2, 0))
        h, w = img.shape[:2]
        if np.random.random() < 0.5:
            angle = np.random.uniform(-10, 10)
            M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        if np.random.random() < 0.5:
            img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    return _aug(cc), _aug(mlo)


def build_breast_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    df['label'] = df['breast_birads'].map(BIRADS_TO_LABEL)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    records = []
    for (study_id, laterality), group in df.groupby(
            ['study_id', 'laterality']):
        cc_rows  = group[group['view_position'] == 'CC']
        mlo_rows = group[group['view_position'] == 'MLO']
        if len(cc_rows) == 0 or len(mlo_rows) == 0:
            continue
        records.append({
            'study_id':     study_id,
            'laterality':   laterality,
            'cc_image_id':  cc_rows.iloc[0]['image_id'],
            'mlo_image_id': mlo_rows.iloc[0]['image_id'],
            'label':        int(cc_rows.iloc[0]['label']),
            'split':        cc_rows.iloc[0]['split'],
        })

    breast_df = pd.DataFrame(records)
    train_df  = breast_df[
        breast_df['split'] == 'training'].reset_index(drop=True)
    test_df   = breast_df[
        breast_df['split'] == 'test'].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    for name, sdf in [('train', train_df), ('test', test_df)]:
        print(f"  {name}: pos={int((sdf['label']==1).sum())} "
              f"neg={int((sdf['label']==0).sum())}")
    return train_df, test_df


def get_cv_splits(train_df: pd.DataFrame,
                  n_splits: int = 5,
                  random_state: int = 42):
    """Study-level stratified CV to prevent data leakage."""
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
            train_df[train_df['study_id'].isin(tr_s)].reset_index(
                drop=True),
            train_df[train_df['study_id'].isin(val_s)].reset_index(
                drop=True),
        ))
    print(f"Created {n_splits} study-level stratified folds")
    return splits


class VinDrPairedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 strategy: str = 'b', augment: bool = False):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.strategy = strategy
        self.do_aug   = augment
        n_pos = int((df['label']==1).sum())
        n_neg = int((df['label']==0).sum())
        aug_s = 'aug' if augment else 'no aug'
        print(f"VinDrPairedDataset [{strategy}] ({aug_s}): "
              f"{len(df)} breasts | pos={n_pos} neg={n_neg}")

    def __len__(self):
        return len(self.df)

    def _load(self, study_id, image_id):
        path = os.path.join(
            self.img_dir, study_id, f"{image_id}.png")
        return load_and_preprocess(path, self.strategy)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cc  = self._load(row['study_id'], row['cc_image_id'])
        mlo = self._load(row['study_id'], row['mlo_image_id'])
        if self.do_aug:
            cc, mlo = augment_pair(cc, mlo)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return (torch.from_numpy(np.ascontiguousarray(cc)),
                torch.from_numpy(np.ascontiguousarray(mlo)),
                label)
