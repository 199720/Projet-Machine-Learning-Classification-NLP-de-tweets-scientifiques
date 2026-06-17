# Classification de Tweets Scientifiques

**Module :** Machine Learning I  
**Année :** 2025–2026

---

## Présentation

Ce projet explore la classification automatique de tweets en lien avec la science, à partir du dataset **SciTweets** (`scitweets_export.tsv`, 1 140 tweets annotés).  
Trois tâches de classification sont traitées, de la plus simple à la plus fine :

| Notebook | Tâche | Type |
|---|---|---|
| `premiere_Classification.ipynb` | **SCI vs NON-SCI** | Binaire |
| `deuxieme_Classification.ipynb` | **{CLAIM, REF} vs CONTEXT** | Binaire (parmi les tweets SCI) |
| `troisieme_Classification.ipynb` | **CLAIM vs REF vs CONTEXT** | Multi-classe (parmi les tweets SCI) |

---

## Tâche 1 — SCI vs NON-SCI

Classification binaire de l'ensemble du corpus (1 139 tweets valides, ~67 % NON-SCI / ~33 % SCI).

**Pipeline :**
- Deux stratégies de prétraitement comparées : **P1** (NLTK — lemmatisation WordNet) et **P2** (spaCy — filtrage par POS, expansion des contractions, encodage des entités)
- Vectorisation TF-IDF avec n-grammes (1–3)
- Gestion du déséquilibre : `class_weight`, SMOTE, sur/sous-échantillonnage
- Optimisation des hyperparamètres avec **Optuna** (TPE, 60–80 trials par modèle)
- Validation croisée stratifiée à 5 folds

**Classifieurs testés :** Naive Bayes · Régression Logistique · SVM (RBF) · K-NN · Forêt Aléatoire

**Meilleur modèle :** Régression Logistique (P2, `class_weight='balanced'`, trigrammes)
- F1 (SCI) = **0.782** | Accuracy = **0.851** | AUC-ROC = **0.920** | AP = **0.856**

---

## Tâche 2 — {CLAIM, REF} vs CONTEXT

Classification binaire sur les seuls tweets scientifiques (375 tweets, déséquilibre sévère : 342 vs 33).

**Points clés :**
- Augmentation des données par substitution de synonymes via WordNet (33 → 330 tweets Context)
- 6 méthodes de rééchantillonnage comparées (original, upsampling, downsampling, combiné, up+down, class_weight)
- Optimisation Optuna (20 trials)
- Analyse des features discriminantes par coefficients de Régression Logistique

**Meilleur modèle :** SVM linéaire (`class_weight='balanced'`)
- F1 macro = **0.705** | Accuracy = **0.907**

---

## Tâche 3 — CLAIM vs REF vs CONTEXT

Classification multi-classe à 3 étiquettes sur les tweets scientifiques.

**Points clés :**
- Pipeline avec **SMOTE** pour gérer le déséquilibre multi-classe
- SVM linéaire optimisé par Optuna (C = 0.0112)
- Tableau comparatif progressif de 8 expériences (baseline → augmentation → SMOTE → Optuna)
- Évaluation par F1 macro, F1 par classe (CLAIM / REF / CONTEXT), courbes d'apprentissage

---

## Stack technique

```
Python 3.x · scikit-learn · imbalanced-learn · Optuna
NLTK · spaCy (en_core_web_sm) · contractions · emoji
pandas · numpy · matplotlib · seaborn · wordcloud · joblib
```

---

## Lancer les notebooks

```bash
jupyter notebook
```

Le fichier de données `scitweets_export.tsv` doit être placé dans le même répertoire que les notebooks.
