#!/usr/bin/env python3
"""
Script: Evaluate CNN Residual Augmentation vs Baseline/YOLO

Purpose:
This script evaluates how much additional signal the CNN residual predictor adds
on top of baseline logistic regression probabilities (with and without YOLO features).
It computes correlation between true residual and CNN prediction, fits an augmentation
logistic regression on the validation set, and evaluates metric deltas (AUC, pseudo-R²)
and threshold-based metrics (accuracy, precision, recall, F1, ROC AUC) on the test set.

Functionality:
- Loads existing artifacts when available:
  - CNN test predictions: data/processed/cnn_test_predictions.csv
  - CNN dataset index:   data/processed/cnn_samples_residual.csv (to identify Val/Test)
  - Baseline and YOLO pipelines: models/baseline_logistic_regression.pkl, models/logistic_regression_with_yolo.pkl
  - Enriched split CSVs for features/labels: data/processed/clusters_{train,val,test}_with_residuals.csv
- If needed for Val CNN predictions, loads the saved CNN weights and runs fast inference on Val images.
- Correlation: Pearson r(true_residual, yhat_CNN) on Test with scatter figure.
- Augmentation regression on Val: Crash ~ p_baseline + yhat_CNN (and optional YOLO variant).
- Test evaluation:
  - Compute AUC and McFadden pseudo-R² deltas for baseline vs baseline+CNN (and YOLO variants).
  - Optimize thresholds on Val for F1 and Youden J; also evaluate fixed 0.5.
  - Report threshold metrics on Test for each model and threshold.
- Saves CSV summaries and figures to reports/.

How to run:
    conda activate berlin-road-crash
    python src/modeling/evaluate_cnn_augmentation.py \
        --use_saved_preds  # use saved predictions if present; recompute if missing

Outputs:
- reports/CNN/cnn_augmentation_regression.csv
- reports/CNN/cnn_vs_baseline_threshold_metrics.csv
- reports/Comparison_Baseline_vs_YOLO/cnn_residual_vs_true_scatter.png
- reports/Comparison_Baseline_vs_YOLO/augmentation_regression_coefficients.png
- reports/Comparison_Baseline_vs_YOLO/delta_auc_bars.png
- reports/Comparison_Baseline_vs_YOLO/delta_pseudo_r2_bars.png
- reports/Comparison_Baseline_vs_YOLO/threshold_metrics_bars.png
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

# Statsmodels for augmentation regression (p-values, pseudo-R²)
import statsmodels.api as sm

# Optional torch imports for CNN inference on Val
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import hashlib
import requests
from io import BytesIO


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_CNN_DIR = PROJECT_ROOT / 'reports' / 'CNN'
REPORTS_COMP_DIR = PROJECT_ROOT / 'reports' / 'Comparison_Baseline_vs_YOLO'

CNN_INDEX_PATH = DATA_DIR / 'cnn_samples_residual.csv'
CNN_TEST_PRED_PATH = DATA_DIR / 'cnn_test_predictions.csv'
CNN_BEST_WEIGHTS = MODELS_DIR / 'cnn_residual_best.pth'

BASELINE_PIPELINE_PATH = MODELS_DIR / 'baseline_logistic_regression.pkl'
YOLO_PIPELINE_PATH = MODELS_DIR / 'logistic_regression_with_yolo.pkl'

CLUSTERS_TRAIN = DATA_DIR / 'clusters_train_with_residuals.csv'
CLUSTERS_VAL = DATA_DIR / 'clusters_val_with_residuals.csv'
CLUSTERS_TEST = DATA_DIR / 'clusters_test_with_residuals.csv'


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dirs():
    """Create output directories if missing."""
    REPORTS_CNN_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_COMP_DIR.mkdir(parents=True, exist_ok=True)


def compute_pseudo_r2(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute McFadden pseudo-R² using statsmodels-compatible formula.

    Parameters:
        y_true: Binary labels (0/1)
        proba: Predicted probabilities (0..1)
    Returns:
        McFadden pseudo-R²
    """
    eps = 1e-12
    proba = np.clip(proba, eps, 1 - eps)
    ll_model = np.sum(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))
    p_bar = np.mean(y_true)
    ll_null = np.sum(y_true * np.log(p_bar + eps) + (1 - y_true) * np.log(1 - p_bar + eps))
    return 1.0 - (ll_model / ll_null)


def find_optimal_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Find thresholds optimizing F1 and Youden J on validation set.

    Returns dict with keys: 'fixed_0_5', 'f1_opt', 'youden_opt'.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_f1 = -1.0
    best_thr_f1 = 0.5
    best_j = -1.0
    best_thr_j = 0.5
    y_true_bin = (y_true > 0.5).astype(int)
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true_bin, y_pred)
        tp = ((y_pred == 1) & (y_true_bin == 1)).sum()
        fn = ((y_pred == 0) & (y_true_bin == 1)).sum()
        fp = ((y_pred == 1) & (y_true_bin == 0)).sum()
        tn = ((y_pred == 0) & (y_true_bin == 0)).sum()
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        youden = tpr + tnr - 1.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr_f1 = thr
        if youden > best_j:
            best_j = youden
            best_thr_j = thr
    return {'fixed_0_5': 0.5, 'f1_opt': best_thr_f1, 'youden_opt': best_thr_j}


def compute_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1/ROC AUC for a given threshold."""
    y_bin = (y_true > 0.5).astype(int)
    y_pred = (proba >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_bin, y_pred),
        'precision': precision_score(y_bin, y_pred, zero_division=0),
        'recall': recall_score(y_bin, y_pred, zero_division=0),
        'f1_score': f1_score(y_bin, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_bin, proba)
    }


# -----------------------------------------------------------------------------
# CNN Val inference helpers (reuse minimal parts from training script)
# -----------------------------------------------------------------------------
class ResidualIndexDataset(Dataset):
    """Dataset for loading images by URL using the CNN index file; returns only images."""
    def __init__(self, csv_path: Path, split: str, cache_dir: Path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transforms.Compose([
            transforms.Resize((424, 424)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _cache_path(self, url: str) -> Path:
        h = hashlib.md5(url.encode()).hexdigest()
        return (PROJECT_ROOT / 'data' / 'cache' / 'images') / f"{h}.jpg"

    def _ensure_cached(self, url: str, path: Path):
        if path.exists():
            return
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
            img = img.resize((424, 424), Image.Resampling.LANCZOS)
            img.save(path, 'JPEG', quality=85, optimize=True)
        except Exception:
            Image.new('RGB', (424, 424), color='black').save(path, 'JPEG', quality=85, optimize=True)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        url = row['image_url']
        path = self._cache_path(url)
        self._ensure_cached(url, path)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, int(row['cluster_id'])


class ResidualPredictor(torch.nn.Module):
    """ResNet18-based regression head to predict residual value."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


def infer_cnn_predictions(split: str, batch_size: int = 32) -> pd.DataFrame:
    """Run CNN inference for the given split ('val' or 'test') and return DataFrame with predictions.

    Returns columns: cluster_id, predicted_residual
    """
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    ds = ResidualIndexDataset(CNN_INDEX_PATH, split=split, cache_dir=PROJECT_ROOT / 'data' / 'cache' / 'images')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count() or 1))
    model = ResidualPredictor().to(device)
    if CNN_BEST_WEIGHTS.exists():
        ckpt = torch.load(CNN_BEST_WEIGHTS, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for imgs, cluster_ids in dl:
            imgs = imgs.to(device)
            y = model(imgs).detach().cpu().numpy()
            preds.append(y)
            ids.append(cluster_ids.numpy())
    preds = np.concatenate(preds, axis=0) if preds else np.array([])
    ids = np.concatenate(ids, axis=0) if ids else np.array([])
    return pd.DataFrame({'cluster_id': ids.astype(int), 'predicted_residual': preds})


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load enriched split CSVs containing features and labels for baseline/YOLO models."""
    return pd.read_csv(CLUSTERS_TRAIN), pd.read_csv(CLUSTERS_VAL), pd.read_csv(CLUSTERS_TEST)


def get_baseline_prob(df: pd.DataFrame, pipeline_path: Path) -> np.ndarray:
    """Load sklearn pipeline and compute probabilities for given split DataFrame.

    Expects df to contain the original feature columns used by the pipeline.
    """
    import joblib
    pipe = joblib.load(pipeline_path)
    X = df.drop(columns=['match_label', 'split'], errors='ignore')
    # Keep only columns known by the pipeline where possible
    proba = pipe.predict_proba(X)[:, 1]
    return proba


def fit_aug_logit_val(y_val: np.ndarray, p_base_val: np.ndarray, yhat_cnn_val: np.ndarray) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    """Fit augmentation logistic regression on Val: Crash ~ p_baseline + yhat_CNN."""
    X = np.column_stack([p_base_val, yhat_cnn_val])
    X = sm.add_constant(X)
    model = sm.Logit(y_val, X)
    return model.fit(disp=False)


def evaluate_models(args):
    ensure_dirs()

    train_df, val_df, test_df = load_splits()
    y_val = val_df['match_label'].astype(int).values
    y_test = test_df['match_label'].astype(int).values

    # Baseline and YOLO probabilities
    p_base_val = get_baseline_prob(val_df, BASELINE_PIPELINE_PATH)
    p_base_test = get_baseline_prob(test_df, BASELINE_PIPELINE_PATH)

    p_yolo_val = None
    p_yolo_test = None
    if YOLO_PIPELINE_PATH.exists():
        p_yolo_val = get_baseline_prob(val_df, YOLO_PIPELINE_PATH)
        p_yolo_test = get_baseline_prob(test_df, YOLO_PIPELINE_PATH)

    # CNN predictions for Val/Test
    # Test: prefer saved predictions; else infer
    if CNN_TEST_PRED_PATH.exists():
        cnn_test = pd.read_csv(CNN_TEST_PRED_PATH)
        # Align to test_df by cluster_id
        test_merge = test_df[['cluster_id']].merge(
            cnn_test[['cluster_id', 'predicted_residual']], on='cluster_id', how='left')
        yhat_cnn_test = test_merge['predicted_residual'].values
        # Correlation with true residual if available
        if 'true_residual' in cnn_test.columns:
            corr, p = pearsonr(cnn_test['true_residual'].values, cnn_test['predicted_residual'].values)
        else:
            # Fall back to correlation using clusters_test_with_residuals residuals
            corr, p = pearsonr(test_df['residual'].values, yhat_cnn_test)
    else:
        test_inf = infer_cnn_predictions('test')
        test_merge = test_df[['cluster_id']].merge(test_inf, on='cluster_id', how='left')
        yhat_cnn_test = test_merge['predicted_residual'].values
        corr, p = pearsonr(test_df['residual'].values, yhat_cnn_test)

    # Val predictions: needed for augmentation regression and threshold search
    if args.use_saved_preds and (REPORTS_CNN_DIR / 'cnn_val_predictions.csv').exists():
        yhat_cnn_val = pd.read_csv(REPORTS_CNN_DIR / 'cnn_val_predictions.csv')['predicted_residual'].values
    else:
        val_inf = infer_cnn_predictions('val')
        # Save for reuse
        val_inf.to_csv(REPORTS_CNN_DIR / 'cnn_val_predictions.csv', index=False)
        yhat_cnn_val = val_inf.merge(val_df[['cluster_id']], on='cluster_id', how='right')['predicted_residual'].values

    # Scatter: CNN residual vs true residual (Test)
    plt.figure(figsize=(6, 5))
    plt.scatter(test_df['residual'].values, yhat_cnn_test, s=10, alpha=0.5)
    z = np.polyfit(test_df['residual'].values, yhat_cnn_test, 1)
    xp = np.linspace(test_df['residual'].min(), test_df['residual'].max(), 100)
    plt.plot(xp, z[0] * xp + z[1], color='red', linewidth=2)
    plt.xlabel('True residual (Test)')
    plt.ylabel('CNN predicted residual')
    plt.title(f'CNN residual vs true (r={corr:.3f}, p={p:.3g})')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(REPORTS_COMP_DIR / 'cnn_residual_vs_true_scatter.png', dpi=200)
    plt.close()

    # Augmentation regression on Val (baseline + CNN)
    aug_res = fit_aug_logit_val(y_val, p_base_val, yhat_cnn_val)
    params = aug_res.params
    bse = aug_res.bse
    pvalues = aug_res.pvalues

    # Evaluate on Test: baseline-only vs baseline+CNN
    # Build augmented probabilities by refitting a simple logistic on Val and applying on Test
    X_val_aug = sm.add_constant(np.column_stack([p_base_val, yhat_cnn_val]))
    X_test_aug = sm.add_constant(np.column_stack([
        p_base_test,
        yhat_cnn_test
    ]))
    # Use fitted coefficients to compute probabilities on Test
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    logit_test = X_test_aug @ params.values
    p_aug_test = sigmoid(logit_test)

    # Metrics and deltas
    auc_base = roc_auc_score(y_test, p_base_test)
    auc_aug = roc_auc_score(y_test, p_aug_test)
    pseudo_base = compute_pseudo_r2(y_test, p_base_test)
    pseudo_aug = compute_pseudo_r2(y_test, p_aug_test)

    # Optional YOLO comparisons
    results_rows = []
    results_rows.append({
        'model': 'baseline', 'AUC': auc_base, 'pseudo_R2': pseudo_base
    })
    results_rows.append({
        'model': 'baseline+cnn', 'AUC': auc_aug, 'pseudo_R2': pseudo_aug
    })

    if p_yolo_val is not None and p_yolo_test is not None:
        auc_yolo = roc_auc_score(y_test, p_yolo_test)
        pseudo_yolo = compute_pseudo_r2(y_test, p_yolo_test)
        results_rows.append({'model': 'yolo', 'AUC': auc_yolo, 'pseudo_R2': pseudo_yolo})

        # YOLO + CNN augmentation
        aug_yolo_res = fit_aug_logit_val(y_val, p_yolo_val, yhat_cnn_val)
        params_y = aug_yolo_res.params
        X_test_aug_y = sm.add_constant(np.column_stack([p_yolo_test, yhat_cnn_test]))
        p_aug_y_test = sigmoid(X_test_aug_y @ params_y.values)
        auc_aug_y = roc_auc_score(y_test, p_aug_y_test)
        pseudo_aug_y = compute_pseudo_r2(y_test, p_aug_y_test)
        results_rows.append({'model': 'yolo+cnn', 'AUC': auc_aug_y, 'pseudo_R2': pseudo_aug_y})

    results_df = pd.DataFrame(results_rows)
    # Compute deltas relative to baseline and YOLO where applicable
    def delta(col, a, b):
        va = results_df.loc[results_df['model'] == a, col].values
        vb = results_df.loc[results_df['model'] == b, col].values
        return (va[0] - vb[0]) if len(va) and len(vb) else np.nan
    delta_auc_base = delta('AUC', 'baseline+cnn', 'baseline')
    delta_pr2_base = delta('pseudo_R2', 'baseline+cnn', 'baseline')
    delta_auc_yolo = delta('AUC', 'yolo+cnn', 'yolo') if 'yolo' in results_df['model'].values else np.nan
    delta_pr2_yolo = delta('pseudo_R2', 'yolo+cnn', 'yolo') if 'yolo' in results_df['model'].values else np.nan

    # Save augmentation regression table (coefficients and p-values)
    aug_table = pd.DataFrame({
        'term': ['Intercept', 'baseline_prob', 'cnn_pred_residual'],
        'coef': params.values,
        'std_err': bse.values,
        'p_value': pvalues.values
    })
    aug_table.to_csv(REPORTS_CNN_DIR / 'cnn_augmentation_regression.csv', index=False)

    # Figures: coefficients bar plot
    plt.figure(figsize=(6, 4))
    terms = ['baseline_prob', 'cnn_pred_residual']
    coefs = [params['x1'], params['x2']]
    errs = [bse['x1'], bse['x2']]
    colors = ['#4C78A8', '#F58518']
    plt.bar(terms, coefs, yerr=errs, color=colors, alpha=0.9)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Augmentation regression coefficients (Val)')
    plt.tight_layout()
    plt.savefig(REPORTS_COMP_DIR / 'augmentation_regression_coefficients.png', dpi=200)
    plt.close()

    # Figures: delta bars
    plt.figure(figsize=(6, 4))
    names = ['ΔAUC (base+cnn - base)']
    vals = [delta_auc_base]
    if not np.isnan(delta_auc_yolo):
        names.append('ΔAUC (yolo+cnn - yolo)')
        vals.append(delta_auc_yolo)
    plt.bar(names, vals, color=['#72B7B2', '#54A24B'][:len(vals)])
    plt.tight_layout()
    plt.savefig(REPORTS_COMP_DIR / 'delta_auc_bars.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    names = ['ΔPseudo-R² (base+cnn - base)']
    vals = [delta_pr2_base]
    if not np.isnan(delta_pr2_yolo):
        names.append('ΔPseudo-R² (yolo+cnn - yolo)')
        vals.append(delta_pr2_yolo)
    plt.bar(names, vals, color=['#E45756', '#F2CF5B'][:len(vals)])
    plt.tight_layout()
    plt.savefig(REPORTS_COMP_DIR / 'delta_pseudo_r2_bars.png', dpi=200)
    plt.close()

    # Threshold metrics: optimize on Val, report on Test for each model variant
    rows = []
    variants = [('baseline', p_base_val, p_base_test)]
    if 'baseline+cnn' in results_df['model'].values:
        variants.append(('baseline+cnn', sigmoid(X_val_aug @ params.values), p_aug_test))
    if p_yolo_val is not None and p_yolo_test is not None:
        variants.append(('yolo', p_yolo_val, p_yolo_test))
        if 'yolo+cnn' in results_df['model'].values:
            # Recompute Val augmented probabilities for YOLO variant
            X_val_aug_y = sm.add_constant(np.column_stack([p_yolo_val, yhat_cnn_val]))
            variants.append(('yolo+cnn', sigmoid(X_val_aug_y @ params_y.values), p_aug_y_test))

    for name, p_val, p_test in variants:
        thr = find_optimal_thresholds(y_val, p_val)
        for label, tval in [('0.5', thr['fixed_0_5']), ('F1', thr['f1_opt']), ('Youden', thr['youden_opt'])]:
            m = compute_threshold_metrics(y_test, p_test, tval)
            rows.append({
                'model': name,
                'threshold': label,
                **m
            })
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(REPORTS_CNN_DIR / 'cnn_vs_baseline_threshold_metrics.csv', index=False)

    # Figure: grouped bar chart for F1 under three thresholds per model
    plt.figure(figsize=(8, 5))
    models_order = list(dict.fromkeys([r['model'] for r in rows]))
    thr_order = ['0.5', 'F1', 'Youden']
    x = np.arange(len(models_order))
    width = 0.25
    for i, thr_name in enumerate(thr_order):
        vals = [thr_df[(thr_df.model == m) & (thr_df.threshold == thr_name)]['f1_score'].values[0] for m in models_order]
        plt.bar(x + i * width - width, vals, width=width, label=thr_name)
    plt.xticks(x, models_order)
    plt.ylabel('F1-score (Test)')
    plt.title('Threshold metrics comparison (Test)')
    plt.legend(title='Threshold')
    plt.tight_layout()
    plt.savefig(REPORTS_COMP_DIR / 'threshold_metrics_bars.png', dpi=200)
    plt.close()

    # Console summary
    print("\n=== Correlation (Test) ===")
    print(f"Pearson r(true residual, yhat_CNN) = {corr:.4f} (p={p:.3g})")
    print("\n=== AUC and pseudo-R² (Test) ===")
    print(results_df)
    print("\nDeltas vs baseline:")
    print(f"  ΔAUC (base+cnn - base): {delta_auc_base:.4f}")
    print(f"  ΔPseudo-R² (base+cnn - base): {delta_pr2_base:.4f}")
    if not np.isnan(delta_auc_yolo):
        print(f"  ΔAUC (yolo+cnn - yolo): {delta_auc_yolo:.4f}")
    if not np.isnan(delta_pr2_yolo):
        print(f"  ΔPseudo-R² (yolo+cnn - yolo): {delta_pr2_yolo:.4f}")


def main():
    """CLI entrypoint to run the augmentation evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate CNN residual augmentation vs baseline/YOLO')
    parser.add_argument('--use_saved_preds', action='store_true', help='Use saved Val/Test CNN predictions when available')
    args = parser.parse_args()
    evaluate_models(args)


if __name__ == '__main__':
    main()


