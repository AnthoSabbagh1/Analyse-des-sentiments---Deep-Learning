# DEVA — Multimodal Sentiment Analysis on MELD

> Transformer audio & vision into textual descriptions for unified multimodal sentiment prediction.

---

Lien du dataset : https://www.kaggle.com/datasets/argish/meld-preprocessed/data

## 📘 Introduction

DEVA est un modèle d’analyse de sentiment multimodal appliqué au dataset **MELD**.  
Il prédit le sentiment (**positif**, **négatif**, **neutre**) à partir de trois modalités :

- 📝 Texte  
- 🎧 Audio  
- 🎥 Vision (expressions faciales)

Les modalités non textuelles sont converties en **descriptions textuelles**, puis fusionnées avec le texte via une **Cross-Modal Attention** et une **Minor Fusion Unit (MFU)**.

---

## ✨ Fonctionnalités

- 🔗 Fusion multimodale : texte + audio + vision  
- 🧠 Encodage textuel : BERT + Transformer Encoder  
- 🗣️ Génération de descriptions audio et visuelles  
- 🧩 Minor Fusion Unit (MFU) avec poids apprenables (α / β)  
- 📊 Évaluation complète : Acc-2, Acc-5, Acc-7 F1, MAE, Pearson  
- ⚖️ Sampling stratifié pour un dataset équilibré  

---

## 🏛️ Architecture du Modèle (DEVANet)

### 🔹 TextEncoder
- Tokenisation : `BertTokenizer`  
- Embeddings : `BertModel`  
- Ajout d’un token spécial `Em`  
- Passage dans un `TransformerEncoder`

### 🔹 Générateurs de Descriptions
- **AudioDescriptionGenerator**  
  Ex : “The speaker has high energy, low pitch, and fast speech.”
- **VisionDescriptionGenerator**  
  Ex : “The person displays wide open eyes and raised eyebrows.”

### 🔹 Cross-Modal Attention
- Attention multi-têtes (`MultiheadAttention`)  
- Interaction texte ↔ description audio  
- Interaction texte ↔ description visuelle  

### 🔹 Minor Fusion Unit (MFU)
- Fusion du texte, de l'attention audio et de l'attention vision  
- Poids apprenables :  
  - α (audio)  
  - β (vision)

### 🔹 Prédicteur Final
- Réseau feed-forward  
- Sortie : score de sentiment ∈ [-1.0, 1.0]

---

## 📚 Dataset : MELD

Le dataset **MELD** contient des dialogues annotés de la série *Friends* avec :

- texte  
- émotions et sentiments  
- audio + MFCCs  
- images faciales  

Découpage : **train / validation / test**  
Sampling stratifié sur train et validation.

---

## ⚙️ Installation

### Prérequis

Python 3.x
torch
transformers
numpy
pandas
scikit-learn
scipy
tqdm
matplotlib
seaborn
pathlib
