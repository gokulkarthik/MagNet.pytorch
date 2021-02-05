# MagNet - PyTorch Implementation

PyTorch implementation of `MagNet: a Two-Pronged Defense against Adversarial Examples`

Paper: https://arxiv.org/pdf/1705.09064.pdf


## Steps

1. Attack Models: Trained classifier models for the datasets MNIST, Fashion-MNIST & CIFAR-10 are availabe in the `models` directory. If you want to train your own classifier define them in `classifiers.py` and train them using `train_classifier.py`

2. Defensive Models: Trained autoencoder models for the datasets MNIST, Fashion-MNIST & CIFAR-10 are availabe in the `models` directory. If you want to train your own autoencoder define them in `defensive_models.py` and train them using `train_defensive_model.py`

3. Adversarial Examples: `generate_adversarial_examples.py` will generated the adversarial images using common adversarial attacks using Foolbox.

4. Evaluation: Performance of a defensive model against various attacks can be evalauted using `evaluate_defensive_model.py`. Check the summary csv files for each dataset inside the `results'` directory