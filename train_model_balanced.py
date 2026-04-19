from __future__ import annotations

from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
HIDDEN_UNITS = 32
DECIMALS = 2
MAX_ITER = 1000
LEARNING_RATE = 0.002
ALPHA = 0.001
ACTIVATION = "tanh"


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
        f.write(f"# Architecture: 64 -> {HIDDEN_UNITS} -> 10\n\n")
        write_vector(f, "SCALER_MEAN", scaler.mean_.tolist())
        write_vector(f, "SCALER_SCALE", scaler.scale_.tolist())
        write_matrix(f, "WEIGHTS1", mlp.coefs_[0].T.tolist())
        write_vector(f, "BIASES1", mlp.intercepts_[0].tolist())
        write_matrix(f, "WEIGHTS2", mlp.coefs_[1].T.tolist())
        write_vector(f, "BIASES2", mlp.intercepts_[1].tolist())


def main() -> None:
    digits = load_digits()
    X = digits.data.astype("float64")
    y = digits.target.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation=ACTIVATION,
        solver="adam",
        alpha=ALPHA,
        batch_size=64,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )

    model.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_acc = accuracy_score(y_test, model.predict(X_test_s))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print(f"Iterations: {model.n_iter_}")

    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)

    final_model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation=ACTIVATION,
        solver="adam",
        alpha=ALPHA,
        batch_size=64,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )

    final_model.fit(X_all_s, y)
    full_train_acc = accuracy_score(y, final_model.predict(X_all_s))
    print(f"Full-data train accuracy: {full_train_acc:.4f}")

    out_path = Path(__file__).with_name("pico_model_balanced.py")
    export_model(out_path, final_scaler, final_model)
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
