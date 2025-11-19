# Jigsaw Unintended Bias in Toxicity Classification

Lien vers le notion : https://www.notion.so/Projet-de-MTI850-27f03bede59d80f2be2ad6fbe570e161?source=copy_link 

Lien vers le kaggle : https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data


## 0. Chargement et inspection initiale
- Lecture des fichiers `train.csv`, `test.csv`, `identity_columns`, `target`.  
- Exploration des distributions de labels (`target`, `severe_toxicity`, etc.).  
- Détection de déséquilibres (classe toxique vs non-toxique).  
- Vérification des **colonnes d’identité** (ex. `male`, `female`, `gay`, `black`, `muslim`…).  
- Observation du vocabulaire, longueur moyenne, ponctuation, emojis.

---

## 1. Approche "Classique" Spark ML

### 1.1 Nettoyage et prétraitement
1.1.1 Prétraitement sur échantillon réduit (tests de pipeline rapide).  
1.1.2 Prétraitement complet (regex, minuscules, stopwords, lemmatisation).  
1.1.3 Gestion du déséquilibre des classes (`target`, identités protégées) :  
- oversampling / undersampling,  
- poids de classe dans le modèle.  

### 1.2 Vectorisation
- TF-IDF unigrames/bigrames.  
- CountVectorizer + IDF.  
- Word2Vec Spark (moyenne des embeddings).  
- Option : représentation multi-entrée (texte + colonnes d’identité).

### 1.3 Modélisation
- Régression logistique (baseline).  
- Random Forest / GBTClassifier.  
- Calibration des probabilités (Platt Scaling).  

### 1.4 Optimisation et validation
- GridSearchCV / TrainValidationSplit.  
- Validation stratifiée sur les identités (stratifier sur `target` ∧ identities).  
- Évaluation : ROC-AUC global + moyenne pondérée sur sous-groupes.

---

## 2. Analyse du biais et métriques d’équité
- Définition des sous-groupes (identités > 0.5).  
- Calcul du **Subgroup AUC**, **BPSN AUC**, **BNSP AUC** (comme sur Kaggle).  
- Visualisation des scores par identité.  
- Discussion : sources de biais (échantillonnage, corrélation identité/toxicité).

---

## 3. Interprétabilité et robustesse
- Poids TF-IDF ou coefficients de la régression logistique → mots déclencheurs.  
- LIME / SHAP pour textes représentatifs.  
- Test contre phrases neutres contenant des identités (ex. “I am a Christian”).  

---

## 4. Approche moderne : Transformers (BERT)

### 4.1 Avec John Snow Labs Spark-NLP
- Pipeline : `DocumentAssembler` → `BertEmbeddings` → `ClassifierDLApproach`.  
- Entraînement distribué sur Spark.  
- Évaluation des performances et biais par sous-groupe.  

### 4.2 Fine-tuning BERT (manuellement)
- Tokenisation Hugging Face (`bert-base-uncased`, `roberta-base`).  
- Gestion du déséquilibre : weighted loss / focal loss.  
- Validation multi-identités et early stopping.  
- Export du modèle pour inference Spark (`onnx` ou `TorchDistributor`).  

---

## 5. Comparaison et évaluation des modèles
- Baselines Spark ML vs BERT.  
- Métriques :  
  - ROC-AUC global, Subgroup AUC, Average Bias AUC.  
  - F1, Precision, Recall, Balanced Accuracy.  
- Analyse des erreurs :  
  - faux positifs sur phrases neutres,  
  - biais d’identité persistants.

---

## 6. Amélioration et hybridation
- Stacking Spark ML + BERT (Logistic Stacker).  
- Data augmentation (synonymes, back-translation).  
- Fine-tuning multi-task (toxicité + identité).  
- Distillation BERT → modèle léger pour inférence rapide.

---

## 7. Soumission Kaggle et documentation
- Génération des prédictions sur `test.csv`.  
- Création du fichier `submission.csv`.  
- Documentation technique : pipeline, choix de métriques, gestion du biais.  
- Discussion sur l’éthique de la modération automatique.

---

## Points forts de cette structure
- Mieux alignée avec les enjeux du dataset (Jigsaw = biais et équité).  
- Séparation claire entre pipeline classique et pipeline moderne.  
- Intègre les métriques spécifiques du concours (Subgroup AUC, BPSN, BNSP).  
- Ajoute une dimension interprétabilité et analyse éthique.
