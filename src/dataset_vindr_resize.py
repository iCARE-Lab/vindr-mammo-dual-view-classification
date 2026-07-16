"""
dataset_vindr_resize.py
-----------------------
Dataset loader for paired CC+MLO mammogram classification on VinDr-Mammo.

Each breast is represented as a pair of images:
  - CC  (craniocaudal)      — top-down view
  - MLO (mediolateral oblique) — angled side view

Label mapping (BI-RADS binary classification):
  BI-RADS 1, 2  ->  0  (routine, no follow-up required)
  BI-RADS 3, 4, 5  ->  1  (needs attention, follow-up indicated)

Three resize strategies are supported:
  Strategy A: Direct resize to 1024x1024 (distorts aspect ratio)
  Strategy B: Resize preserving aspect ratio, zero-pad to 1024x1024
  Strategy C: Resize preserving aspect ratio to 614x1024, no padding

All strategies apply the same preprocessing pipeline:
  1. Otsu thresholding to remove the black background
  2. CLAHE for local contrast enhancement
  3. Resize (strategy-specific)
  4. Convert to 3-channel float32 in [0, 1]
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


# Binary label mapping for BI-RADS scores
BIRADS_TO_LABEL = {
    'BI-RADS 1': 0, 'BI-RADS 2': 0,
    'BI-RADS 3': 1, 'BI-RADS 4': 1, 'BI-RADS 5': 1,
}


def crop_breast(img: np.ndarray) -> np.ndarray:
    """
    Remove the black background from a mammogram image.

    Uses Otsu thresholding to binarize the image, then finds the
    largest connected component (the breast tissue) using connected
    component analysis. Crops to the bounding box of that component
    with a 2% margin on each side.

    If cropping fails or reduces the image to less than 10% of its
    original size, the original image is returned unchanged.

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
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast across the image by operating on small
    tile regions (8x8 pixels) rather than the whole image. The clip
    limit of 2.0 prevents over-amplification of noise.

    Args:
        img: Grayscale image as a NumPy array.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def resize_strategy_a(img: np.ndarray) -> np.ndarray:
    """
    Strategy A: Resize image directly to 1024x1024.

    This does not preserve the original aspect ratio and will
    distort the breast geometry. Included as a baseline to
    quantify the effect of aspect ratio preservation.

    Args:
        img: Grayscale image after background removal and CLAHE.

    Returns:
        3-channel float32 array of shape (3, 1024, 1024) in [0, 1].
    """
    resized = cv2.resize(img, (1024, 1024),
                         interpolation=cv2.INTER_LINEAR)
    return np.stack([resized]*3, axis=0).astype(np.float32) / 255.0


def resize_strategy_b(img: np.ndarray) -> np.ndarray:
    """
    Strategy B: Resize preserving aspect ratio, zero-pad to 1024x1024.

    Scales the image so its longest dimension equals 1024, then pads
    the shorter dimension symmetrically with zeros to reach 1024x1024.
    This avoids geometric distortion while producing a fixed square
    output that is compatible with any backbone architecture.

    Args:
        img: Grayscale image after background removal and CLAHE.

    Returns:
        3-channel float32 array of shape (3, 1024, 1024) in [0, 1].
    """
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
    Strategy C: Resize to 614x1024 preserving aspect ratio, no padding.

    The target size of 614x1024 matches the native VinDr-Mammo aspect
    ratio (912x1520). All images are forced to exactly this size to
    allow batching. No zero-padding is added, so every pixel contains
    breast tissue information. Note that this rectangular output is
    architecture-dependent and may not be compatible with all backbones.

    Args:
        img: Grayscale image after background removal and CLAHE.

    Returns:
        3-channel float32 array of shape (3, 1024, 614) in [0, 1].
    """
    resized = cv2.resize(img, (614, 1024),
                         interpolation=cv2.INTER_LINEAR)
    return np.stack([resized]*3, axis=0).astype(np.float32) / 255.0


# Map strategy identifier to the corresponding resize function
RESIZE_FNS = {
    'a': resize_strategy_a,
    'b': resize_strategy_b,
    'c': resize_strategy_c,
}


def load_and_preprocess(img_path: str,
                         strategy: str = 'b') -> np.ndarray:
    """
    Load a mammogram image and apply the full preprocessing pipeline.

    Steps:
        1. Load image in grayscale.
        2. Remove black background using Otsu thresholding.
        3. Enhance local contrast with CLAHE.
        4. Resize using the specified strategy (a, b, or c).

    Args:
        img_path: Path to the PNG image file.
        strategy: Resize strategy identifier ('a', 'b', or 'c').

    Returns:
        Preprocessed image as a float32 array of shape (3, H, W) in [0, 1].

    Raises:
        FileNotFoundError: If the image cannot be read from img_path.
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
    Apply data augmentation to a paired CC and MLO image.

    Horizontal and vertical flips are applied with the same random
    decision for both views. This is necessary because flipping changes
    the apparent breast laterality — applying opposite flips to CC and
    MLO would break their anatomical correspondence.

    Random rotation up to +/-10 degrees is applied independently to
    each view, since small rotations do not affect laterality.

    Args:
        cc:  CC view as a float32 array of shape (3, H, W).
        mlo: MLO view as a float32 array of shape (3, H, W).

    Returns:
        Tuple of augmented (cc, mlo) arrays.
    """
    # Synchronized horizontal flip
    if np.random.random() < 0.5:
        cc  = np.ascontiguousarray(cc[:, :, ::-1])
        mlo = np.ascontiguousarray(mlo[:, :, ::-1])

    # Synchronized vertical flip
    if np.random.random() < 0.5:
        cc  = np.ascontiguousarray(cc[:, ::-1, :])
        mlo = np.ascontiguousarray(mlo[:, ::-1, :])

    def _rotate(img):
        """Apply a random rotation within +/-10 degrees."""
        img = np.transpose(img, (1, 2, 0))
        h, w = img.shape[:2]
        if np.random.random() < 0.5:
            angle = np.random.uniform(-10, 10)
            M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    return _rotate(cc), _rotate(mlo)


def build_breast_dataframe(csv_path: str):
    """
    Build a breast-level paired dataframe from the VinDr-Mammo CSV.

    Groups images by (study_id, laterality) to form CC+MLO pairs.
    If a breast has more than one CC or MLO image, the first is used.
    Uses the official VinDr-Mammo train/test split column.

    Args:
        csv_path: Path to breast-level_annotations.csv.

    Returns:
        Tuple of (train_df, test_df) DataFrames, each with columns:
        study_id, laterality, cc_image_id, mlo_image_id, label, split.
    """
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
    """
    Create stratified cross-validation splits at the study level.

    Splitting at the study level ensures that both breasts of the
    same patient always appear in the same fold, preventing any
    form of patient-level data leakage between training and validation.

    Args:
        train_df:     Breast-level training DataFrame.
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
            train_df[train_df['study_id'].isin(tr_s)].reset_index(
                drop=True),
            train_df[train_df['study_id'].isin(val_s)].reset_index(
                drop=True),
        ))
    print(f"Created {n_splits} study-level stratified folds")
    return splits


class VinDrPairedDataset(Dataset):
    """
    PyTorch Dataset for paired CC+MLO mammogram classification.

    Each item in the dataset is one breast, represented by its CC
    and MLO views. Both views are preprocessed and optionally augmented
    before being returned as tensors.

    Args:
        df:       DataFrame with columns: study_id, cc_image_id,
                  mlo_image_id, label.
        img_dir:  Root directory containing images at
                  {img_dir}/{study_id}/{image_id}.png.
        strategy: Resize strategy ('a', 'b', or 'c').
        augment:  Whether to apply data augmentation.
    """

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
        """Load and preprocess a single image."""
        path = os.path.join(
            self.img_dir, study_id, f"{image_id}.png")
        return load_and_preprocess(path, self.strategy)

    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (cc_tensor, mlo_tensor, label) where tensors are
            float32 of shape (3, H, W) and label is a scalar float32.
        """
        row = self.df.iloc[idx]
        cc  = self._load(row['study_id'], row['cc_image_id'])
        mlo = self._load(row['study_id'], row['mlo_image_id'])
        if self.do_aug:
            cc, mlo = augment_pair(cc, mlo)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return (torch.from_numpy(np.ascontiguousarray(cc)),
                torch.from_numpy(np.ascontiguousarray(mlo)),
                label)
