# src/train.py
#commandes à mettre pour éxécuter :
# python train.py 
#et
#python train.py --class weights

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import load_combined_wine_data, prepare_data, split_and_scale
from neural_network import MLPClassifier


LABEL_NAMES = {0: "Mauvaise", 1: "Moyenne", 2: "Bonne"}
TYPE_NAMES = {0: "Red", 1: "White"}


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long)
    return X_t, y_t


def accuracy_from_logits(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, yb)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    all_preds = []
    all_true = []

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = criterion(logits, yb)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, yb)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    return total_loss / len(loader), total_acc / len(loader), y_true, y_pred


def build_criterion(y_train, device, use_class_weights: bool):
    if not use_class_weights:
        return nn.CrossEntropyLoss(), None

    classes = np.array([0, 1, 2])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(y_train),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("\nPoids de classes (0/1/2) :", class_weights)

    return nn.CrossEntropyLoss(weight=class_weights_t), class_weights


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def plot_training_curves(history_df: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Valeur")
    plt.title("Courbes d'entraînement (loss et accuracy)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, class_names):
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Matrice de confusion (Test)")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def export_dl_metrics_json(out_path: str, y_true, y_pred, model_name: str, selection_metric: str = "accuracy"):
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "best_model": model_name,
        "selection_metric": selection_metric,
        "models": {
            model_name: {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                ),
            }
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_weights", action="store_true", help="Active la pondération des classes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden", type=str, default="128,64", help="ex: '128,64'")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "gelu"])
    parser.add_argument("--batch_norm", action="store_true", help="Active BatchNorm")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()

    # ===== 1) Charger dataset combiné + préparer X/y =====
    df = load_combined_wine_data().reset_index(drop=True)
    X, y = prepare_data(df)  # y = quality_grouped (0/1/2)

    wine_type_all = df["wine_type"].astype(int).copy()

    # ===== 2) Split + scaling (IMPORTANT : on passe bien X ET y) =====
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)

    # Pour exporter un CSV des prédictions test avec wine_type, on refait le split sur indices
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2, random_state=42, stratify=y.iloc[idx_train])

    # ===== 3) DataLoaders =====
    X_train_t, y_train_t = to_tensor(X_train, y_train)
    X_val_t, y_val_t = to_tensor(X_val, y_val)
    X_test_t, y_test_t = to_tensor(X_test, y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False)

    # ===== 4) Modèle =====
    hidden_units = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        n_classes=3,
        hidden_units=hidden_units,
        activation=args.activation,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    ).to(device)

    # ===== 5) Loss + Optim =====
    criterion, _ = build_criterion(y_train, device, use_class_weights=args.class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ===== 6) Entraînement + early stopping =====
    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        )

        print(
            f"[COMBINED] Epoch {epoch:03d} "
            f"| train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"| val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            patience_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print("Early stopping déclenché.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== 7) Test final + métriques =====
    te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print("\n===== TEST FINAL (COMBINED) =====")
    print(f"loss={te_loss:.4f} acc={te_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nMatrice de confusion (0=Mauvaise,1=Moyenne,2=Bonne):\n", cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            zero_division=0,
        )
    )

    # ===== 8) Sauvegardes outputs =====
    suffix = "cw" if args.class_weights else "nocw"

    history_df = pd.DataFrame(history)
    history_csv = f"outputs/dl_history_{suffix}.csv"
    history_df.to_csv(history_csv, index=False)

    training_fig = f"outputs/dl_training_curve_{suffix}.png"
    plot_training_curves(history_df, training_fig)

    cm_fig = f"outputs/dl_confusion_matrix_{suffix}.png"
    plot_confusion_matrix(cm, cm_fig, class_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]])

    # CSV prédictions test (avec wine_type)
    df_test = pd.DataFrame({
        "index": idx_test,
        "wine_type": wine_type_all.iloc[idx_test].to_numpy(),
        "wine_type_name": pd.Series(wine_type_all.iloc[idx_test].to_numpy()).map(TYPE_NAMES).to_numpy(),
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": pd.Series(y_true).map(LABEL_NAMES).to_numpy(),
        "pred_label": pd.Series(y_pred).map(LABEL_NAMES).to_numpy(),
        "correct": (y_true == y_pred).astype(int),
    })
    pred_csv = f"outputs/dl_test_predictions_{suffix}.csv"
    df_test.to_csv(pred_csv, index=False)

    # Modèle
    model_path = f"models/mlp_combined_{suffix}.pt"
    torch.save(model.state_dict(), model_path)

    # JSON metrics POUR compare_models.py (noms simples et stables)
    metrics_json = f"outputs/dl_metrics_{suffix}.json"
    export_dl_metrics_json(
        out_path=metrics_json,
        y_true=y_true,
        y_pred=y_pred,
        model_name=f"MLP_{suffix}",
        selection_metric="accuracy",
    )

    print("\n===== OUTPUTS SAUVEGARDÉS =====")
    print("Historique :", history_csv)
    print("Courbes :", training_fig)
    print("Matrice confusion :", cm_fig)
    print("Prédictions test :", pred_csv)
    print("Modèle :", model_path)
    print("Métriques JSON :", metrics_json)


if __name__ == "__main__":
    main()