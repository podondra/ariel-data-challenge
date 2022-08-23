from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb


N_WAVES = 52
N_TARGETS = 6
QUARTILES = [0.16, 0.5, 0.84]
N = 21987
HYPERPARAMETER_DEFAULTS = dict(
        batch_size=256,
        learning_rate=0.0002,
        patience=2048,
        n_hiddens=7,
        n_neurons=128,
        weight_decay=0)


def read_spectra(path="data/train/spectra.hdf5", n=N):
    with h5py.File(path, "r") as file:
        X = torch.zeros((n, 1, 52), dtype=torch.float64)
        for i in range(n):
            key = "Planet_" + str(i)
            X[i, 0] = torch.from_numpy(file[key]["instrument_spectrum"][:])
    return X


def read_fm_parameter_table(path="data/train/ground_truth/fm_parameter_table.csv", n=N):
    fm_parameter = pd.read_csv(path, index_col="planet_ID", nrows=n)
    return torch.from_numpy(fm_parameter.values)


def read_quartiles_table(path="data/train/ground_truth/quartiles_table.csv", n=N):
    df = pd.read_csv(path, index_col="planet_ID", nrows=n)
    quartiles = ["q1", "q2", "q3"]
    targets = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    X = np.zeros((len(quartiles), df.shape[0], len(targets)))
    for i, quartile in enumerate(quartiles):
        for j, target in enumerate(targets):
            X[i, :, j] = df.loc[:, target + '_' + quartile]
    return X


def standardise(tensor, mean, std):
    return (tensor - mean) / std


def unstandardise(tensor, mean, std):
    return tensor * std + mean


def light_score(quartiles, quartiles_pred):
    return 100 * (10 - np.sqrt(((1 - quartiles_pred / quartiles) ** 2).mean()))


def light_track_format(quartiles, filename="data/light_track.csv"):
    # prepares submission file for the light track
    # assume test data are arranged in assending order of the planet ID
    df = pd.DataFrame()
    df.index.name = "planet_ID"
    for i, target in enumerate(['T', 'log_H2O', 'log_CO2','log_CH4','log_CO','log_NH3']):
        for j, quartile in enumerate(['q1','q2','q3']):
            df[target + "_" + quartile] = quartiles[j, :, i]
    df.to_csv(filename)
    return df


def regular_track_format(traces, weights=None, filename="data/regular_track.hdf5"):
    # convert input into regular track format
    # assume that test data are arranged in assending order of the planet ID
    # weight takes into account the importance of each point in the tracedata
    if weights is None:
        weights = np.full(traces.shape[:2], 1 / traces.shape[1])
    with h5py.File(filename, "w") as tracefile:
        for i, (trace, weight) in enumerate(zip(traces, weights)):
            grp = tracefile.create_group("Planet_" + str(i))
            grp.attrs['ID'] = i
            grp.create_dataset('tracedata', data=trace)
            grp.create_dataset('weights', data=weight)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.input = nn.Linear(N_WAVES, self.n_neurons)
        self.linears = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.add_module("linear" + str(i), self.linears[-1])
        self.output = nn.Linear(self.n_neurons, 2 * N_TARGETS)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, X):
        X = torch.flatten(X, 1)
        X = F.relu(self.input(X))
        for linear in self.linears:
            X = F.relu(linear(X))
        X = self.output(X)
        mean, var = X[:, :N_TARGETS], X[:, N_TARGETS:]
        var = F.softplus(var) + 1e-6
        return mean, var

    def loss(self, Y_pred, Y):
        mean, var = Y_pred[0], Y_pred[1]
        return torch.mean(0.5 * torch.log(var) + ((Y - mean).square()) / (2 * var))

    def evaluate(self, dataset):
        X, quartiles = dataset
        mean, var = self(X)
        mean, var = mean.cpu().numpy(), var.cpu().numpy()
        std = np.sqrt(var)
        quartiles_pred = np.stack([norm.ppf(quartile, loc=mean, scale=std) for quartile in QUARTILES])
        return light_score(quartiles, quartiles_pred)


def train(Model, trainset, validset, config):
    X_train, Y_train, quartiles_train = trainset
    X_valid, Y_valid, quartiles_valid = validset
    with wandb.init(config=config, project="ariel-data-challenge"):
        config = wandb.config
        model = Model(config)
        wandb.watch(model)
        trainloader = DataLoader(
                TensorDataset(X_train, Y_train),
                batch_size=config["batch_size"],
                shuffle=True)
        optimiser = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"])
        score_valid_best = float("-inf")
        i = 0
        while i < config["patience"]:
            model.train()
            for X_batch, Y_batch in trainloader:
                optimiser.zero_grad()
                loss = model.loss(model(X_batch), Y_batch)
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                loss_train = model.loss(model(X_train), Y_train).item()
                loss_valid = model.loss(model(X_valid), Y_valid).item()
                score_train = model.evaluate((X_train, quartiles_train))
                score_valid = model.evaluate((X_valid, quartiles_valid))
            if score_valid > score_valid_best:
                i = 0
                score_valid_best = score_valid
                score_train_at_best = score_train
                loss_train_at_best = loss_train
                loss_valid_at_best = loss_valid
                model_state_best = deepcopy(model.state_dict())
            else:
                i += 1
            wandb.log({
                "loss_train": loss_train,
                "loss_valid": loss_valid,
                "light_score_train": score_train,
                "light_score_valid": score_valid,
                "light_score_valid_best": score_valid_best})
            wandb.run.summary["loss_train"] = loss_train_at_best
            wandb.run.summary["loss_valid"] = loss_valid_at_best
            wandb.run.summary["light_score_train"] = score_train_at_best
            wandb.run.summary["light_score_valid"] = score_valid_best
        model.load_state_dict(model_state_best)
        torch.save(model_state_best, f"models/{wandb.run.name}.pt")


if __name__ == "__main__":
    # get data
    X = read_spectra()
    quartiles = read_quartiles_table()
    Y = torch.from_numpy(quartiles[1])

    # train and validation set split
    # TODO out-of-distribution split?
    ids = torch.arange(N)
    ids_train, ids_valid = train_test_split(ids, train_size=0.8, random_state=36)
    idx_train = torch.zeros_like(ids, dtype=torch.bool)
    idx_train[ids_train] = True
    idx_valid = ~idx_train
    X_train, X_valid = X[idx_train], X[idx_valid]
    Y_train, Y_valid = Y[idx_train], Y[idx_valid]
    quartiles_train, quartiles_valid = quartiles[:, idx_train], quartiles[:, idx_valid]

    # data preparation
    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    X_train = standardise(X_train, X_train_mean, X_train_std)
    X_valid = standardise(X_valid, X_train_mean, X_train_std)

    X_train, X_valid = X_train.float(), X_valid.float()
    Y_train, Y_valid = Y_train.float(), Y_valid.float()

    X_train, X_valid = X_train.cuda(), X_valid.cuda()
    Y_train, Y_valid = Y_train.cuda(), Y_valid.cuda()

    train(
            Model,
            (X_train, Y_train, quartiles_train),
            (X_valid, Y_valid, quartiles_valid),
            HYPERPARAMETER_DEFAULTS)
