# Adversarial Attack Experiments

This repository contains various scripts and Jupyter Notebooks for experimenting with adversarial attacks on machine learning models using the Adversarial Robustness Toolbox (ART). It includes attacks on both image classification models (TensorFlow/Keras) and audio classification models (PyTorch).

## Structure

The repository is organized into the following directories:

### 1. `audio_attacks/`
Contains scripts for experimenting with adversarial attacks on **Audio data**. This section is built using **PyTorch** and **Torchaudio**, featuring Digital Signal Processing (DSP) techniques.
*   `audio_attack_pytorch.py`: An end-to-end pipeline demonstrating adversarial attacks (FGSM) on audio waveforms using PyTorch and ART. Includes generation of Mel-spectrograms to visualize the adversarial noise using `librosa`.

### 1. `cifar_attacks/`
Contains Jupyter Notebooks for training models and testing adversarial attacks on the **CIFAR-10** dataset. It includes scripts to test different hyperparameters for attacks (e.g., varying `gamma` and `theta`).

**Key files:**
*   `train_cifar.ipynb`: Training a baseline neural network on CIFAR-10.
*   `test_cifar_*.ipynb`: Various test scripts evaluating the model against adversarial examples generated with different parameters (e.g., `gamma_0.3`, `thetha_0.5`, `optimal`, `std`).

### 2. `mnist_attacks/`
Contains scripts and notebooks for experimenting with adversarial attacks on the **MNIST** dataset. Uses Saliency Map Method (JSMA) for generating adversarial examples.

**Key files:**
*   `train_image.py`: A Python script that trains a Keras model on MNIST and generates adversarial examples using the `SaliencyMapMethod` from the `adversarial-robustness-toolbox`.
*   `train.ipynb` / `test.ipynb`: Jupyter notebooks for training and testing on the MNIST dataset.

### 3. `misc/`
Contains miscellaneous exploratory scripts.
*   `Untitled.ipynb`: Statistical analysis scripts (e.g., chi-square tests for hypothesis testing using pandas and scipy).

## Requirements

To run these scripts, you will need the following libraries:
pip install torch torchaudio librosa matplotlib adversarial-robustness-toolbox
*   PyTorch (`torch`, `torchaudio`)
*   TensorFlow
*   Keras
*   Adversarial Robustness Toolbox (`adversarial-robustness-toolbox`)
*   Librosa
*   NumPy
*   Pandas
*   SciPy
*   Matplotlib

## Usage
Navigate to the respective directories (`cifar_attacks/` or `mnist_attacks/`) and open the Jupyter Notebooks or run the Python scripts to train the models and generate adversarial examples.
