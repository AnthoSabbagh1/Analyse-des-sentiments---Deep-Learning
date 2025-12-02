# DEVA — Multimodal Sentiment Analysis on MELD

> Transformer audio & vision into textual descriptions for unified multimodal sentiment prediction.

---

Lien du dataset : https://www.kaggle.com/datasets/argish/meld-preprocessed/data


> 🎭 **Texte + Audio + Vision** unifiés pour prédire le sentiment humain avec l’architecture DEVANet.

---

## 🚀 Introduction
Ce projet implémente **DEVANet**, un modèle multimodal combinant :
- 📝 **Texte** (BERT + Transformer)
- 🔊 **Audio** (features & Mel-spectrograms)
- 👀 **Vision** (émotions faciales)

Objectif 🎯 : prédire le sentiment **positive / negative / neutral** pour chaque utterance du dataset **MELD**.

L’architecture repose sur :
- un encodeur texte avancé,
- des encodeurs audio & visuels simplifiés,
- une **Cross-Modal Attention**,
- une **Multimodal Fusion Unit (MFU)** apprenant à pondérer les modalités,
- un prédicteur final donnant un score continu ∈ [-1, 1].

---

## 📚 Dataset MELD

### 📁 Structure des fichiers `.pt`
Chaque entrée contient :
- 🗣️ `utterance` — texte  
- 🎭 `emotion` — émotion MELD  
- 🎧 `audio` — vecteur audio brut  
- 🔉 `audio_mel` — Mel-spectrogram  
- 👤 `face` — features faciales  

---

### 🔄 Mapping émotions → sentiment
| Emotion | Sentiment |
|---------|-----------|
| joy, surprise | 😊 positive |
| sadness, anger, fear, disgust | 😡 negative |
| neutral | 😐 neutral |

---

### 📊 Données échantillonnées (stratified)
| Set | Total | 😀 Pos | 😡 Neg | 😐 Neu |
|------|-------|--------|--------|--------|
| Train | 300 | 100 | 100 | 100 |
| Dev | 150 | 50 | 50 | 50 |
| Test | 150 | 50 | 50 | 50 |

Toutes les modalités sont disponibles pour 100% des échantillons.

---

## 🧠 Architecture DEVANet

### 📝 1. Encodeur Texte  
- BERT tokenizer + BERT embeddings  
- Transformer Encoder  
- Token spécial **Em** pour booster la représentation globale  

---

### 🎧 2. AudioDescriptionGenerator  
Transforme les Mel-features en embedding 768D :  
- Projection linéaire  
- AdaptiveAvgPool1d  
- MLP  

---

### 👀 3. VisionDescriptionGenerator  
Encode l’émotion en un embedding dense 768D :  
- `nn.Embedding(num_emotions → 768)`

---

### 🔁 4. Cross-Modal Attention  
Permet au texte d’extraire les informations pertinentes dans :
- 🔊 audio  
- 👀 vision  

Module utilisé : `nn.MultiheadAttention`

---

### 🌀 5. MFU — Multimodal Fusion Unit  
Le cœur de DEVANet.  
Combine texte + audio + vision via :  
- projections linéaires  
- attention croisée  
- **pondération apprenable** :  
  - α = poids audio  
  - β = poids vision  

---

### 🎯 6. Prédicteur final  
Une MLP prédit un **score de sentiment continu** dans [-1, 1].  
Le signe du score donne la classe.

---

## 📈 Entraînement & Évaluation

### ⚙️ Setup
- Optimiseur : **AdamW**  
- Loss : **MSE**  
- Nombre d’époques : 10  
- Sauvegarde : `best_deva_meld.pt`  

### 📏 Métriques utilisées
- **Acc-2** (binaire positive / négative)  
- **Acc-5**  
- **Acc-7**  
- **F1-score**  
- **MAE**  
- **Pearson corr**  

---

## 🧪 Résultats sur Test

| Metric | Value |
|--------|--------|
| Test Loss | ⭐ 0.0054 |
| **Acc-2** | **0.8733** |
| Acc-5 | 0.2000 |
| Acc-7 | 0.1467 |
| **F1** | **0.8768** |
| MAE | 0.0518 |
| **Pearson** | **0.9975** |

Les résultats montrent une excellente cohérence (corrélation ≈ 1) et une très bonne précision binaire.

---

## 🧩 MFU — Analyse des poids α & β

### 🔢 Poids appris
| Poids | Valeur |
|--------|---------|
| α (audio) | 0.9973 |
| β (vision) | 1.0011 |

### 🧮 Contribution normalisée
| Modalité | Contribution |
|-----------|--------------|
| 🔊 Audio | 49.9% |
| 👀 Vision | 50.1% |

📌 **Le modèle privilégie très légèrement la vision (+0.2%).**

---

## 🔍 Exemples de Prédictions

### ✔ Example 1
- **Text** : "Why do all you're coffee mugs have numbers on the bottom?"  
- **True** : positive  
- **Pred** : positive (1.015)  
- **Emotion** : surprise  

### ✔ Example 2
- **Text** : "Where’s number 27?!"  
- **True** : negative  
- **Pred** : negative (-0.842)  
- **Emotion** : anger  

### ✔ Example 3
- **Text** : "Y'know what?"  
- **True** : neutral  
- **Pred** : neutral (-0.015)  

### ✔ Examples 4 & 5  
Plusieurs utterances neutres → toutes prédites correctement.

---

## ▶️ Comment Exécuter le Projet

### 📦 Prérequis
- Python 3.x  
- Données MELD prétraitées disponibles localement  
- Librairies essentielles  

### 🔧 Auteurs 
- Yohan MARCEL
- Thomas MATHIOT
- Nicolas PINIER
- Anthony SABBAGH
