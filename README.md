# Classification de la qualité des vins – Étude comparative (Machine Learning vs Deep Learning)

## Présentation globale du projet

Ce projet vise à prédire la qualité des vins à partir de leurs caractéristiques physico-chimiques, en utilisant le dataset *Wine Quality* issu du repository UCI Machine Learning.

L’objectif principal est de comparer deux approches :

- Des modèles classiques de Machine Learning (Decision Tree, Random Forest, KNN)
- Un modèle de Deep Learning basé sur un réseau de neurones multicouches (MLP)

L’analyse ne se limite pas à l’accuracy globale, mais intègre également la gestion du déséquilibre des classes et l’étude de métriques plus équilibrées.

---

## Objectifs

- Transformer la note de qualité initiale en un problème de classification à 3 classes
- Optimiser les modèles classiques via validation croisée et recherche d’hyperparamètres
- Implémenter un MLP en PyTorch
- Étudier l’impact du déséquilibre des classes
- Comparer les performances selon plusieurs métriques :
  - Accuracy
  - Macro F1-score
  - Balanced Accuracy

---

## Jeu de données

Source : UCI Machine Learning Repository  
Dataset : Wine Quality (vins rouges et blancs)

Les bases « red » et « white » ont été fusionnées et une variable binaire `wine_type` a été ajoutée :

- 0 : vin rouge
- 1 : vin blanc

La variable cible originale (*quality*) a été regroupée en trois catégories :

- 0 → Mauvaise qualité (≤ 4)
- 1 → Qualité moyenne (5–6)
- 2 → Bonne qualité (≥ 7)

---

## Méthodologie

### Modèles de Machine Learning Classiques

- Decision Tree (GridSearchCV, validation croisée stratifiée 5-fold)
- Random Forest (GridSearchCV, validation croisée stratifiée 5-fold)
- K-Nearest Neighbors (GridSearchCV, validation croisée stratifiée 5-fold)

Les données sont séparées en :
- 80 % pour l'entraînement
- 20 % pour le test  
avec stratification des classes.

---

### Approche Deep Learning (MLP – PyTorch)

- Architecture entièrement connectée
- Fonction de perte : CrossEntropyLoss
- Optimiseur : Adam
- Early stopping basé sur la validation
- Comparaison entre :
  - MLP sans pondération des classes
  - MLP avec pondération des classes (gestion du déséquilibre)

Les données sont séparées en :
- 64 % entraînement
- 16 % validation
- 20 % test

Standardisation des variables via `StandardScaler` (ajusté uniquement sur l’entraînement).

---

## Résultats principaux

| Modèle | Accuracy |
|--------|----------|
| Random Forest | 0.8554 |
| KNN | 0.8292 |
| Decision Tree | 0.7977 |
| MLP (sans pondération) | 0.8077 |
| MLP (avec pondération) | 0.6185 |

Le Random Forest obtient la meilleure accuracy globale.

Le MLP avec pondération améliore significativement la Balanced Accuracy (0.6765) et la détection des classes minoritaires, au prix d’une baisse d’accuracy globale.

---

## 🧠 Principales conclusions

- Les méthodes ensemblistes restent extrêmement performantes sur des données tabulaires structurées.
- L’accuracy seule peut masquer un déséquilibre important entre classes.
- La pondération des classes améliore l’équité de la classification.
- Le Deep Learning ne surpasse pas automatiquement les modèles classiques sur des datasets de taille modérée.

---

## 📂 Structure du projet
