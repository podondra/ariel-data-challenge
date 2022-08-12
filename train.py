import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import ariel


class Model(nn.Module):
    def __init__(self, dropout_probability, n_hiddens, T):
        super(Model, self).__init__()
        self.T = T
        self.model = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(192, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(n_hiddens, ariel.N_TARGETS))
        self.cuda()

    def forward(self, X):
        return self.model(X)
    
    def loss(self, Y_pred, Y):
        return F.mse_loss(Y_pred, Y)

    @torch.no_grad()
    def sample(self, X):
        # T = 5000 is the maximum for this competition
        Y_pred = torch.zeros(X.shape[0], self.T, ariel.N_TARGETS)
        for i in range(X.shape[0]):
            Y_pred[i] = self.forward(X[i:i + 1].expand(self.T, -1, -1))
        return Y_pred

    def evaluate(self, dataset, Y_train_mean, Y_train_std):
        X, quartiles = dataset
        Y_pred = self.sample(X)
        Y_pred = ariel.unstandardise(Y_pred, Y_train_mean, Y_train_std)
        quartiles_pred = np.quantile(Y_pred, ariel.QUARTILES, axis=1)
        return ariel.light_score(quartiles, quartiles_pred)


def make_model(config):
    return Model(config["dropout_probability"], config["n_hiddens"], config["T"])


if __name__ == "__main__":
    # get data
    X = ariel.read_spectra()
    quartiles = ariel.read_quartiles_table()
    Y = torch.from_numpy(quartiles[1]).float()

    # train and validation set split
    # TODO out-of-distribution split?
    ids = torch.arange(ariel.N)
    ids_train, ids_valid = train_test_split(ids, train_size=0.8, random_state=36)
    idx_train = torch.zeros_like(ids, dtype=torch.bool)
    idx_train[ids_train] = True
    idx_valid = ~idx_train
    X_train, X_valid = X[idx_train], X[idx_valid]
    Y_train = Y[idx_train]
    quartiles_train, quartiles_valid = quartiles[:, idx_train], quartiles[:, idx_valid]

    # data preparation
    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    X_train = ariel.standardise(X_train, X_train_mean, X_train_std)
    X_valid = ariel.standardise(X_valid, X_train_mean, X_train_std)

    Y_train_mean, Y_train_std = Y_train.mean(dim=0), Y_train.mean(dim=0)
    Y_train = ariel.standardise(Y_train, Y_train_mean, Y_train_std)

    X_train, X_valid = X_train.cuda(), X_valid.cuda()
    Y_train = Y_train.cuda()

    hyperparameter_defaults = dict(
            batch_size=256,
            dropout_probability=0.1,
            learning_rate=0.001,
            patience=32,
            n_hiddens=128,
            T=128,
            weight_decay=0)

    ariel.train(
            make_model,
            TensorDataset(X_train, Y_train),
            (X_valid, quartiles_valid),
            Y_train_mean, Y_train_std,
            hyperparameter_defaults)
