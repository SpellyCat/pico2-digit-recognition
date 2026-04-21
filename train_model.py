from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
from scipy.ndimage import affine_transform, shift as ndi_shift
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HIDDEN_UNITS = 40
N_AUG = 5
DECIMALS = 1
ALPHA = 0.0001
LEARNING_RATE = 0.0013
MAX_ITER = 450


def preprocess(img: np.ndarray) -> np.ndarray:
    img = np.array(img, dtype=float)
    img = np.clip(img, 0.0, None)

    total = float(img.sum())
    if total <= 1e-8:
        return img

    ys, xs = np.indices(img.shape)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)

    dx = (img.shape[1] - 1) / 2.0 - cx
    dy = (img.shape[0] - 1) / 2.0 - cy
    img = ndi_shift(img, shift=(dy, dx), order=1, mode="constant", cval=0.0)

    total = float(img.sum())
    if total > 1e-8:
        ys, xs = np.indices(img.shape)
        cx = float((xs * img).sum() / total)
        cy = float((ys * img).sum() / total)
        mu02 = float((((ys - cy) ** 2) * img).sum() / total)
        mu11 = float((((xs - cx) * (ys - cy)) * img).sum() / total)
        if mu02 > 1e-6:
            skew = mu11 / mu02
            mat = np.array([[1.0, 0.0], [-skew, 1.0]], dtype=float)
            offset = np.array([0.0, skew * cy], dtype=float)
            img = affine_transform(img, mat, offset=offset, order=1, mode="constant", cval=0.0)

    mx = float(img.max())
    if mx > 0:
        img = img / mx * 16.0
    return img


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sx = rng.uniform(-1.3, 1.3)
    sy = rng.uniform(-1.3, 1.3)
    ang = rng.uniform(-20, 20)
    shear = rng.uniform(-0.35, 0.35)

    theta = np.deg2rad(ang)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    sh = np.array([[1.0, shear], [0.0, 1.0]], dtype=float)
    mat = rot @ sh
    inv = np.linalg.inv(mat)

    center = np.array(img.shape) / 2.0 - 0.5
    offset = center - inv @ center - np.array([sy, sx], dtype=float)
    out = affine_transform(img, inv, offset=offset, order=1, mode="constant", cval=0.0, output_shape=img.shape)
    return out


def make_augmented(X: np.ndarray, y: np.ndarray, n_aug: int = N_AUG, seed: int = 0):
    rng = np.random.default_rng(seed)
    imgs = []
    labels = []
    for img, label in zip(X, y):
        imgs.append(preprocess(img))
        labels.append(label)
        for _ in range(n_aug):
            imgs.append(preprocess(augment(img, rng)))
            labels.append(label)
    return np.array(imgs), np.array(labels)


def fmt(v: float) -> str:
    return f"{float(v):.{DECIMALS}f}"


def write_vector(f, name, vec):
    f.write(f"{name} = [\n")
    f.write("    " + ", ".join(fmt(v) for v in vec) + "\n")
    f.write("]\n\n")


def write_matrix(f, name, mat):
    f.write(f"{name} = [\n")
    for row in mat:
        f.write("    [" + ", ".join(fmt(v) for v in row) + "],\n")
    f.write("]\n\n")


def export_model(path: Path, scaler: StandardScaler, mlp: MLPClassifier) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated for Pico 2 deployment\n")
        f.write(f"# Architecture: 64 -> {HIDDEN_UNITS} -> 10\n")
        f.write("# Model tuned for center-of-mass shift + deskew preprocessing\n\n")
        write_vector(f, "SCALER_MEAN", np.round(scaler.mean_, DECIMALS))
        write_vector(f, "SCALER_SCALE", np.round(scaler.scale_, DECIMALS))
        write_matrix(f, "WEIGHTS1", np.round(mlp.coefs_[0], DECIMALS).T)
        write_vector(f, "BIASES1", np.round(mlp.intercepts_[0], DECIMALS))
        write_matrix(f, "WEIGHTS2", np.round(mlp.coefs_[1], DECIMALS).T)
        write_vector(f, "BIASES2", np.round(mlp.intercepts_[1], DECIMALS))


def main() -> None:
    digits = load_digits()
    X = digits.images.astype(float)
    y = digits.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_aug, y_aug = make_augmented(X_train, y_train, n_aug=N_AUG, seed=0)
    X_test_p = np.array([preprocess(img) for img in X_test])

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(HIDDEN_UNITS,),
            activation="tanh",
            solver="adam",
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE,
            max_iter=MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_aug.reshape(len(X_aug), -1), y_aug)

    scaler = model.named_steps["standardscaler"]
    mlp = model.named_steps["mlpclassifier"]

    train_acc = accuracy_score(y_aug, model.predict(X_aug.reshape(len(X_aug), -1)))
    test_acc = accuracy_score(y_test, model.predict(X_test_p.reshape(len(X_test_p), -1)))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print(f"Iterations: {mlp.n_iter_}")

    mean = np.round(scaler.mean_, DECIMALS)
    scale = np.round(scaler.scale_, DECIMALS)
    W1 = np.round(mlp.coefs_[0], DECIMALS)
    b1 = np.round(mlp.intercepts_[0], DECIMALS)
    W2 = np.round(mlp.coefs_[1], DECIMALS)
    b2 = np.round(mlp.intercepts_[1], DECIMALS)

    Xn = (X_test_p.reshape(len(X_test_p), -1) - mean) / scale
    hidden = np.tanh(Xn @ W1 + b1)
    logits = hidden @ W2 + b2
    pred = logits.argmax(axis=1)
    rounded_acc = accuracy_score(y_test, pred)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    rounded_correct_conf = float(conf[pred == y_test].mean())

    print(f"Rounded test accuracy: {rounded_acc:.4f}")
    print(f"Rounded correct-confidence: {rounded_correct_conf:.4f}")

    out_path = Path(__file__).with_name("pico_model.py")
    export_model(out_path, scaler, mlp)
    print(f"Exported: {out_path}")
    print(f"Export size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
