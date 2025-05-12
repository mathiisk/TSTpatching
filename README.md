# MECHANISTIC INTERPRETABILITY ON TRANSFORMER-BASED TIME SERIES CLASSIFICATION MODEL
### Author: Matiss Kalnare
### Supervisor Niki van Stein

This repository contains a PyTorch-based toolkit created as part of a BSc thesis, focused on **interpretability**  techique adaptions to Transformer-based Time Series (TST) classification model. 
It provides two pipelines to probe and interpret the inner workings of a model.

**1. Activation Patching / Causal Tracing**: causal intervention on specific parts (Heads, MLP layers, etc.) to form / confirm hypothesis about influential components wihtin the Transformer

**2. Sparse Autoencoders**: a neural network trained on the activation values of a specific layer within the Transformer that highlights possible interpretable features / concepts that the Transformer has learned

Included is a modular training script for a TST, some pretrained models on benchmark datasets, and Interactive Jupyter notebooks to follow work through the two mechanistic interpretability methods.

# Components
- **TSTtrainer.py** Core training and evaluation script for Transformer-based TSC models. Defines convolutional patch embeddings, learnable postional encodings, and a Transformer encoder with a classification head

- **TSTpatching.ipynb** Pipeline for causal tracing. Loads a pre-trained model, performs selective patching across various components, and measures downstream influence on predicitons. Forms causal / attribution graphs

- **SAE.ipynb** Pipeline for training and analyzing sparse autoencoders. Visualiuzes learned features / concepts. 

- **Pre-trained models** Ready to use weights for Yoga (univariate), JapaneseVowels (multivariate) datasets to experiment on

# Installation
```
# Clone
git clone https://github.com/mathiisk/TSTpatching.git
cd TSTpatching

# Install packages
pip install -r requirements.txt
```
CUDA-enabled GPU recommended but not necessary

# Usage
1. Training a new model
```
python TSTtrainer.py --dataset DATASETNAME --epochs NUMEPOCHS --batch_size BATCHSIZE
```
Specify Dataset name from https://www.timeseriesclassification.com/dataset.php, e.g, JapaneseVowels

Specify number of epochs for trianing, e.g., 100

Specify batch size, e.g., 4

The trained weights will be saved as TST_<dataset>.pth in the project root

2. Run Activation Patching Experiments
```
jupyter notebook TSTpatching.ipynb
```
Follow the cells to load a pre-trained model, apply patching, and analyze the results

3. Run Sparse Autoencoder Experiments
Follow the cells to train a sparse autoencoder, visualize sparse codes


# BSc Thesis Context
This repository aims to support a BSc thesis investigating whether **mechanistic interpretability** methods from NLP (activation patching, sparse autoencoders) can be adapted to Transformer-based time series classifiers. 
The primary goal is to determine the feasibility and insight potential of these methods in revealing internal causal components within TST models