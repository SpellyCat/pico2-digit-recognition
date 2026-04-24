from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PCA_COMPONENTS = 64
HIDDEN_UNITS = 64
MAX_ITER = 45
ALPHA = 1e-4
LEARNING_RATE_INIT = 0.001
BATCH_SIZE = 256
LOGIT_SCALE_GRID = np.linspace(0.60, 2.50, 39)
OUTPUT_MODEL = Path(__file__).with_name("pico_mnist_model.json")


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        return x_train, y_train, x_test, y_test
    except Exception:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        x = x.reshape(-1, 28, 28)
        return x[:60000], y[:60000], x[60000:], y[60000:]


def fit_pca_and_scaler(x_train_flat: np.ndarray) -> tuple[PCA, StandardScaler]:
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=False, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(x_train_flat)
    scaler = StandardScaler()
    scaler.fit(train_pca)
    return pca, scaler


def transform_features(pca: PCA, scaler: StandardScaler, x_flat: np.ndarray) -> np.ndarray:
    return scaler.transform(pca.transform(x_flat))


def train_mlp(x_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,), activation="relu", solver="adam",
        alpha=ALPHA, batch_size=BATCH_SIZE, learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER, random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=8, tol=1e-4, verbose=False,
    )
    model.fit(x_train, y_train)
    return model


def quantize_matrix(mat: np.ndarray) -> tuple[np.ndarray, float]:
    mat = np.asarray(mat, dtype=np.float64)
    max_abs = float(np.max(np.abs(mat)))
    if max_abs == 0.0:
        return np.zeros_like(mat, dtype=np.int16), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.rint(mat / scale), -127, 127).astype(np.int16)
    return q, scale


def forward_quantized(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, s1: float,
                      w2: np.ndarray, b2: np.ndarray, s2: float, logit_scale: float = 1.0) -> np.ndarray:
    hidden = np.maximum(0.0, b1 + s1 * (x @ w1.T))
    logits = b2 + s2 * (hidden @ w2.T)
    return logits * logit_scale


def predict_quantized(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, s1: float,
                      w2: np.ndarray, b2: np.ndarray, s2: float, logit_scale: float = 1.0) -> np.ndarray:
    return np.argmax(forward_quantized(x, w1, b1, s1, w2, b2, s2, logit_scale), axis=1)


def softmax_nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return log_loss(y_true, probs, labels=list(range(10)))


def tune_logit_scale(x_val: np.ndarray, y_val: np.ndarray, w1: np.ndarray, b1: np.ndarray, s1: float,
                     w2: np.ndarray, b2: np.ndarray, s2: float) -> float:
    best_scale, best_nll = 1.0, float("inf")
    for scale in LOGIT_SCALE_GRID:
        logits = forward_quantized(x_val, w1, b1, s1, w2, b2, s2, float(scale))
        nll = softmax_nll(logits, y_val)
        if nll < best_nll:
            best_nll = nll
            best_scale = float(scale)
    return best_scale


def export_pico_model_json(path: Path, w1_q: np.ndarray, b1: np.ndarray, s1: float,
                           w2_q: np.ndarray, b2: np.ndarray, s2: float, logit_scale: float) -> None:
    model_data = {
        "h_units": HIDDEN_UNITS,
        "log_s": logit_scale,
        "w1_s": s1,
        "w2_s": s2,
        "w1": w1_q.tolist(),
        "b1": b1.tolist(),
        "w2": w2_q.tolist(),
        "b2": b2.tolist()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(model_data, f, separators=(',', ':'))


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()
    x_train_full, x_val, y_train_full, y_val = train_test_split(
        x_train, y_train, test_size=6000, random_state=RANDOM_STATE, stratify=y_train
    )

    x_train_flat = x_train_full.reshape(len(x_train_full), -1)
    x_val_flat = x_val.reshape(len(x_val), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    pca, scaler = fit_pca_and_scaler(x_train_flat)
    x_train_feat = transform_features(pca, scaler, x_train_flat)
    
    model = train_mlp(x_train_feat, y_train_full)
    
    # 核心数学融合：将 PCA 矩阵、Scaler 参数与第一层权重完美融合，跳过 Pico 端的特征处理
    P = pca.components_.T 
    S = np.diag(1.0 / scaler.scale_)
    M1 = P @ S 
    B_pca = - (pca.mean_ @ M1) - (scaler.mean_ / scaler.scale_)

    W1_raw = model.coefs_[0]
    b1_raw = model.intercepts_[0]

    # 获得可直接处理 784 像素的新权重与偏置！
    W_fused = M1 @ W1_raw 
    b_fused = (B_pca @ W1_raw) + b1_raw

    # 对融合后的参数进行量化
    w1_q, s1 = quantize_matrix(W_fused.T)
    w2_q, s2 = quantize_matrix(model.coefs_[1].T)
    b1 = np.asarray(b_fused, dtype=np.float64)
    b2 = np.asarray(model.intercepts_[1], dtype=np.float64)

    # 注意：此时调参和预测都直接使用扁平化且未经过 PCA 的 784 个原始像素验证其准确性
    logit_scale = tune_logit_scale(x_val_flat, y_val, w1_q, b1, s1, w2_q, b2, s2)
    print(f"Selected logit scale: {logit_scale:.4f}")

    q_val_pred = predict_quantized(x_val_flat, w1_q, b1, s1, w2_q, b2, s2, logit_scale)
    q_test_pred = predict_quantized(x_test_flat, w1_q, b1, s1, w2_q, b2, s2, logit_scale)
    print(f"Quantized val accuracy:  {accuracy_score(y_val, q_val_pred):.4f}")
    print(f"Quantized test accuracy: {accuracy_score(y_test, q_test_pred):.4f}")

    export_pico_model_json(OUTPUT_MODEL, w1_q, b1, s1, w2_q, b2, s2, logit_scale)
    print(f"Exported Pico JSON model: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()