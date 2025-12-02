🧠 Projet DEVA — Analyse Multimodale de Sentiment sur le Dataset MELD
📌 Introduction

Ce projet implémente un modèle d’analyse de sentiment multimodal, inspiré de l’architecture DEVA, et appliqué au dataset MELD (Multimodal EmotionLines Dataset).
L’objectif est de prédire le sentiment d’une utterance (positif, négatif, neutre) dans une conversation, en combinant trois sources d’information :

Texte

Audio

Vidéo (expressions faciales)

L’approche repose sur une idée clé :

🔄 Transformer les modalités non textuelles en descriptions textuelles, puis les fusionner avec le texte original via attention croisée et une Minor Fusion Unit (MFU).

✨ Fonctionnalités

🔊 Traitement Multimodal : fusion texte + audio + vision

📝 Encodage Textuel Avancé : BERT + Transformer Encoder

🎙️ Génération de Descriptions Audio et Visuelles

🔗 MFU (Minor Fusion Unit) : fusion adaptative des modalités via poids apprenables

📊 Évaluation complète : Acc-2, F1, MAE, Pearson

⚖️ Sampling Stratifié : équilibrage des classes de sentiment

🏗️ Architecture du Modèle (DEVANet)
1. TextEncoder

Tokenisation via BertTokenizer

Embeddings contextuels via BertModel

Ajout d’un token apprenable Em

Passage par un nn.TransformerEncoder

2. Générateurs de Descriptions

AudioDescriptionGenerator
Convertit des features audio en phrases telles que :
"The speaker has high energy, low pitch, and fast speech."

VisionDescriptionGenerator
Génère une description de l’expression faciale :
"The person displays wide open eyes and raised eyebrows."

3. CrossModalAttention

Mécanisme nn.MultiheadAttention permettant au texte :

d’interroger la description audio

d’interroger la description visuelle

4. Minor Fusion Unit (MFU)

Module central du modèle :

Combine les informations provenant du texte, de l’audio et de la vision

Poids apprenables :

α → contribution de l’audio

β → contribution de la vision

5. Prédicteur

Un réseau feed-forward qui transforme l’embedding fusionné en un score de sentiment continu entre -1.0 et 1.0.

📚 Dataset : MELD

Le dataset MELD contient des dialogues de la série Friends annotés avec :

Émotions

Sentiments

Audio (formes d’onde + MFCCs)

Vidéo (visages)

Le dataset est divisé en train/val/test, avec sampling stratifié sur l’entraînement et la validation.

⚙️ Installation et Utilisation
🔧 Prérequis
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

📥 Téléchargement des données

Les données prétraitées MELD peuvent être obtenues via KaggleHub, ou via un chemin local similaire :

C:\Users\tmath\.cache\kagglehub\datasets\argish\meld-preprocessed\versions\1\preprocessed_data


Assurez-vous d’adapter ce chemin selon votre environnement.

▶️ Exécuter le Notebook
git clone <URL_DE_TON_DEPOT>
cd <ton_dossier_projet>


Installer les dépendances :

pip install -r requirements.txt
# ou manuellement
pip install torch transformers numpy pandas scikit-learn scipy tqdm matplotlib seaborn


Ouvrir le .ipynb dans Jupyter Notebook ou Google Colab et exécuter les cellules dans l’ordre.

📈 Résultats

Après 10 époques d'entraînement avec données samplifiées :

Performance (Test Set)
Métrique	Valeur
Test Loss	0.0027
Acc-2	0.7467
F1-Score	0.7518
MAE	0.0390
Pearson	0.9983
🔍 Analyse des Poids du MFU
Poids	Valeur	Contribution
α (Audio)	0.9938	49.8%
β (Vision)	1.0033	50.2%

➡️ Le modèle utilise légèrement plus la vision que l’audio pour prédire le sentiment.

🔮 Exemples de Prédictions

Le notebook inclut une section illustrant :

l’utterance originale

les descriptions générées (audio + vision)

le score prédit

la comparaison avec le label réel
