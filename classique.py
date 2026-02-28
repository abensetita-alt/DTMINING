
##**Modèles classiques (Arbre de descision, Random Forest et KNN)**

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import os
os.environ["MPLBACKEND"] = "Agg"  # pas de fenêtre GUI (important pour le correcteur)

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Chargement des datasets
red_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    delimiter=";"
)

white_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    delimiter=";"
)

print("Red wine shape:", red_wine.shape)
print("White wine shape:", white_wine.shape)

## Chargement et Préparation des Données

red_wine_df = red_wine.copy()
white_wine_df = white_wine.copy()

# Add 'wine_type' column (0 for red, 1 for white)
red_wine_df['wine_type'] = 0
white_wine_df['wine_type'] = 1

# Combine the datasets
wine_df = pd.concat([red_wine_df, white_wine_df], ignore_index=True)

# Define features (X) and target (y)
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

print("Shape of combined DataFrame (wine_df):", wine_df.shape)
print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)
print("First 5 rows of X:")
print(X.head())
print("First 5 values of y:")
print(y.head())


## Division des Données en Ensembles d'Entraînement et de Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


## Optimisation et Évaluation du Classificateur Arbre de Décision

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the parameter grid for Decision Tree Classifier
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize StratifiedKFold for cross-validation
# quality (target variable) has imbalanced classes, so StratifiedKFold is appropriate.
# Adjusted n_splits to 2 to avoid 'least populated class' warning, as the smallest class has 2 samples.
strat_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search_dtc = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=strat_kfold,
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)

# Fit GridSearchCV to the training data
grid_search_dtc.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters for Decision Tree Classifier:", grid_search_dtc.best_params_)

# Get the best estimator
best_dtc = grid_search_dtc.best_estimator_

# Make predictions on the test set using the best estimator
y_pred_best_dtc = best_dtc.predict(X_test)

# Evaluate the best model
accuracy_best_dtc = accuracy_score(y_test, y_pred_best_dtc)
print(f"\nOptimized Decision Tree Classifier Accuracy: {accuracy_best_dtc:.4f}")

print("\nOptimized Decision Tree Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_dtc, zero_division=0))


## Optimisation et Évaluation du Classificateur Forêt Aléatoire

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Define the parameter grid for Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize StratifiedKFold for cross-validation
# Adjusted n_splits to 2 to avoid 'least populated class' warning, as the smallest class has 2 samples in y_train.
strat_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search_rfc = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=strat_kfold,
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)

# Fit GridSearchCV to the training data
grid_search_rfc.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters for Random Forest Classifier:", grid_search_rfc.best_params_)

# Get the best estimator
best_rfc = grid_search_rfc.best_estimator_

# Make predictions on the test set using the best estimator
y_pred_best_rfc = best_rfc.predict(X_test)

# Evaluer le meilleur modèle
accuracy_best_rfc = accuracy_score(y_test, y_pred_best_rfc)
print(f"\nOptimized Random Forest Classifier Accuracy: {accuracy_best_rfc:.4f}")

print("\nOptimized Random Forest Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_rfc, zero_division=0))


## Optimisation et Évaluation du Classificateur K-Plus Proches Voisins (KNN)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Initialisation du StandardScaler object
scaler = StandardScaler()

# Apprendre le scaler sur les données d'entraînement, puis transformer les données d'entraînement et de test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définition de la grille de recherche des hyperparamètres pour KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2] # 1 for Manhattan distance, 2 for Euclidean distance
}

# Initialisation de StratifiedKFold pour cross-validation
# Adjusted n_splits to 2 to avoid 'least populated class' warning, as the smallest class has 2 samples in y_train.
strat_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialisation de GridSearchCV
grid_search_knn = GridSearchCV(
    estimator=KNeighborsClassifier(n_jobs=-1),
    param_grid=param_grid,
    cv=strat_kfold,
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)

# Ajustement de GridSearchCV sur les données d'entraînement normalisées
grid_search_knn.fit(X_train_scaled, y_train)

# Affichage des meilleurs paramêtre trouvés 
print("Best parameters for K-Nearest Neighbors Classifier:", grid_search_knn.best_params_)

# Choix du meilleur éstimateur 
best_knn = grid_search_knn.best_estimator_

# Prédictions sur le jeu de test normalisé en utilisant le meilleur estimateur
y_pred_best_knn = best_knn.predict(X_test_scaled)

# Evaluation du meilleur modèle
accuracy_best_knn = accuracy_score(y_test, y_pred_best_knn)
print(f"\nOptimized K-Nearest Neighbors Classifier Accuracy: {accuracy_best_knn:.4f}")

print("\nOptimized K-Nearest Neighbors Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_knn, zero_division=0))

#Affichage des performances des 3 modèles 

print("\nOptimized Decision Tree Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_dtc, zero_division=0))

print("\nOptimized Random Forest Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_rfc, zero_division=0))

print("\nOptimized K-Nearest Neighbors Classifier Classification Report:")
print(classification_report(y_test, y_pred_best_knn, zero_division=0))

optimized_models = ['Optimized Decision Tree', 'Optimized Random Forest', 'Optimized K-Nearest Neighbors']
optimized_accuracies = [accuracy_best_dtc, accuracy_best_rfc, accuracy_best_knn]

performance_optimized_df = pd.DataFrame({
    'Model': optimized_models,
    'Accuracy': optimized_accuracies
})

print("Optimized Model Performance Comparison:")
print(performance_optimized_df)

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='Accuracy', data=performance_optimized_df, palette='viridis', hue='Model', legend=False)
plt.title('Comparison of Optimized Classifier Accuracies')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1) # Accuracy is between 0 and 1

import matplotlib.pyplot as plt

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_01.png"), dpi=150, bbox_inches="tight")

out_path = os.path.abspath(os.path.join(OUT_DIR, "figure_01.png"))
print(f"[PLOT] Figure enregistrée : {out_path}")

plt.close()

# Importance des caractéristiques 

importances = best_rfc.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Importance des Caractéristiques par le Classificateur Forêt Aléatoire:")
print(feature_importance_df)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', hue='Feature', legend=False)
plt.title('Importance des Caractéristiques pour le Classificateur Forêt Aléatoire')
plt.xlabel('Importance')
plt.ylabel('Caractéristique')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_02.png"), dpi=150, bbox_inches="tight")

out_path = os.path.abspath(os.path.join(OUT_DIR, "figure_02.png"))
print(f"[PLOT] Figure enregistrée : {out_path}")

plt.close()

###Tableau comparatif des prédictions du modéle Random Forest

import sys, time
import pandas as pd

def categorize_quality(quality_score):
    if quality_score <= 4:
        return 'Mauvaise'
    elif quality_score <= 6:
        return 'Moyenne'
    else:
        return 'Bonne'

t0 = time.perf_counter()
print("[CHK] Début bloc comparaison"); sys.stdout.flush()

# --- 1) Sécuriser index/tailles pour éviter un alignement coûteux
# On force tout sur le même index, celui de y_test
idx = getattr(y_test, "index", None)
if idx is None:
    # y_test est peut-être un ndarray -> on fabrique un RangeIndex cohérent
    idx = pd.RangeIndex(start=0, stop=len(y_test), step=1)

# Convertir les prédictions en Series avec le même index
if not isinstance(y_pred_best_rfc, pd.Series):
    y_pred_ser = pd.Series(y_pred_best_rfc, index=idx, name="Qualité Prédite")
else:
    # Réindexer pour correspondre
    y_pred_ser = y_pred_best_rfc.reindex(idx)

# y_test en Series (si besoin) + réindexage
if not isinstance(y_test, pd.Series):
    y_test_ser = pd.Series(y_test, index=idx, name="Vraie Qualité")
else:
    y_test_ser = y_test.reindex(idx).rename("Vraie Qualité")

# Récupération sûre du type de vin depuis X_test
# - On reindexe sur idx pour être aligné
# - .to_numpy() évite de propager l'index lors de l'assignation
wine_type = X_test.reindex(idx)["wine_type"].to_numpy()

t1 = time.perf_counter()
print(f"[TIMING] Préparation des séries : {t1 - t0:.3f}s"); sys.stdout.flush()

# --- 2) Construction du DataFrame sans alignements implicites
comparison_df_enhanced = pd.DataFrame({
    'Vraie Qualité': y_test_ser,
    'Qualité Prédite': y_pred_ser,
    'Type de Vin': wine_type
}, index=idx)

# --- 3) Catégorisation (vectorisée donc rapide)
comparison_df_enhanced['Catégorie Vraie'] = pd.cut(
    comparison_df_enhanced['Vraie Qualité'],
    bins=[-float("inf"), 4, 6, float("inf")],
    labels=['Mauvaise', 'Moyenne', 'Bonne'],
    right=True
)
comparison_df_enhanced['Catégorie Prédite'] = pd.cut(
    comparison_df_enhanced['Qualité Prédite'],
    bins=[-float("inf"), 4, 6, float("inf")],
    labels=['Mauvaise', 'Moyenne', 'Bonne'],
    right=True
)

comparison_df_enhanced['Catégorie Correcte'] = (
    comparison_df_enhanced['Catégorie Vraie'] == comparison_df_enhanced['Catégorie Prédite']
)

t2 = time.perf_counter()
print(f"[TIMING] Construction + catégorisation : {t2 - t1:.3f}s"); sys.stdout.flush()

# --- 4) Affichage rapide et non bloquant (éviter IPython.display dans un .py)
print("Comparaison des Prédictions du Modèle Random Forest (tableau enrichi) :")
with pd.option_context('display.max_rows', 20, 'display.max_columns', 20, 'display.width', 160):
    print(comparison_df_enhanced.head(20).to_string())
sys.stdout.flush()

# --- 5) Résumés
correct_category_predictions = comparison_df_enhanced['Catégorie Correcte'].sum()
total_predictions = len(comparison_df_enhanced)
category_accuracy = correct_category_predictions / total_predictions
print(f"\nPrécision de la Catégorie (Mauvaise/Moyenne/Bonne) : {category_accuracy:.2f}")
print("\nPrécision de la Catégorie par Type de Vin :")
print(comparison_df_enhanced.groupby('Type de Vin', observed=True)['Catégorie Correcte'].mean())
sys.stdout.flush()

t3 = time.perf_counter()
print(f"[TIMING] Affichages + métriques : {t3 - t2:.3f}s"); sys.stdout.flush()
print(f"[TIMING] Total bloc comparaison : {t3 - t0:.3f}s"); sys.stdout.flush()

