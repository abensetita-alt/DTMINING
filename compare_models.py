# Comparaison : Modèles classiques (RandomForest) vs Deep Learning (MLP nocw / cw)

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.environ["MPLBACKEND"] = "Agg"  

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# FICHIERS JSON 

CLASSIC_JSON = os.path.join(OUT_DIR, "classic_metrics.json")

# JSON exportés par train.py (Deep Learning)
DL_JSON_NOCW = os.path.join(OUT_DIR, "dl_metrics_nocw.json")
DL_JSON_CW = os.path.join(OUT_DIR, "dl_metrics_cw.json")



# Si soucis

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Vérification de l'éxécution JSON"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_accuracy(metrics_obj: dict) -> float:
    """
    Récupère l'accuracy depuis un JSON de métriques.
    """
    if "accuracy" in metrics_obj:
        return float(metrics_obj["accuracy"])

    for k in ["acc", "test_accuracy", "test_acc", "test_accu"]:
        if k in metrics_obj:
            return float(metrics_obj[k])

    if "test" in metrics_obj and isinstance(metrics_obj["test"], dict):
        if "accuracy" in metrics_obj["test"]:
            return float(metrics_obj["test"]["accuracy"])
        for k in ["acc", "test_accuracy", "test_acc"]:
            if k in metrics_obj["test"]:
                return float(metrics_obj["test"][k])

    raise KeyError(
        "Impossible de trouver l'accuracy dans ce JSON. "
        "Clés dispo : " + ", ".join(list(metrics_obj.keys()))
    )



# CLASSIQUE : RandomForest

classic = load_json(CLASSIC_JSON)

classic_best_name = classic.get("best_model", "RandomForest")
classic_models = classic.get("models", {})

if classic_best_name not in classic_models:
    # fallback si jamais le JSON ne contient pas best_model correctement
    if "RandomForest" in classic_models:
        classic_best_name = "RandomForest"
    else:
        # dernier fallback : prendre le 1er modèle trouvé
        if len(classic_models) == 0:
            raise KeyError("classic_metrics.json ne contient pas de clé 'models' exploitable.")
        classic_best_name = list(classic_models.keys())[0]

classic_best_metrics = classic_models[classic_best_name]
classic_acc = extract_accuracy(classic_best_metrics)



# DEEP LEARNING : MLP nocw / cw

dl_nocw = load_json(DL_JSON_NOCW)
dl_cw = load_json(DL_JSON_CW)

dl_acc_nocw = extract_accuracy(dl_nocw)
dl_acc_cw = extract_accuracy(dl_cw)



# TABLEAU + EXPORT CSV

rows = [
    {"Approche": "Classique", "Modèle": classic_best_name, "Accuracy": classic_acc},
    {"Approche": "DeepLearning", "Modèle": "MLP (nocw)", "Accuracy": dl_acc_nocw},
    {"Approche": "DeepLearning", "Modèle": "MLP (cw)", "Accuracy": dl_acc_cw},
]

summary_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

csv_path = os.path.join(OUT_DIR, "comparison_summary.csv")
summary_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"[OK] Summary saved to: {os.path.abspath(csv_path)}")
print(summary_df)



# PLOT (bar chart)
plt.figure(figsize=(9, 5))
plt.bar(summary_df["Modèle"], summary_df["Accuracy"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparaison des accuracies (Classique vs Deep Learning)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "comparison_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Plot saved to: {os.path.abspath(plot_path)}")

best_row = summary_df.iloc[0]
print(f"\n[RESULT] Meilleure accuracy : {best_row['Approche']} – {best_row['Modèle']} ({best_row['Accuracy']:.4f})")