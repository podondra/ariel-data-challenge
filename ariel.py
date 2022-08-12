from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import wandb


N_WAVES = 52
N_TARGETS = 6
QUARTILES = [0.16, 0.5, 0.84]
N = 21987


def read_spectra(path="data/train/spectra.hdf5", n=N):
    with h5py.File(path, "r") as file:
        X = torch.zeros((n, 1, 52))
        for i in range(n):
            key = "Planet_" + str(i)
            X[i, 0] = torch.from_numpy(file[key]["instrument_spectrum"][:])
    return X


def read_fm_parameter_table(path="data/train/ground_truth/fm_parameter_table.csv", n=N):
    fm_parameter = pd.read_csv(path, index_col="planet_ID", nrows=n)
    return torch.from_numpy(fm_parameter.values).float()


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


@torch.no_grad()
def sample(model, X, T=128):
    # T = 5000 is the maximum for this competition
    Y_pred = torch.zeros(X.shape[0], T, N_TARGETS)
    for i in range(X.shape[0]):
        Y_pred[i] = model(X[i:i + 1].expand(T, -1, -1))
    return Y_pred


def light_score(quartiles, quartiles_pred):
    return 100 * (10 - np.sqrt(((1 - quartiles_pred / quartiles) ** 2).mean()))


def evaluate(model, dataset, Y_train_mean, Y_train_std):
    X, quartiles = dataset
    Y_pred = sample(model, X)
    Y_pred = unstandardise(Y_pred, Y_train_mean, Y_train_std)
    quartiles_pred = np.quantile(Y_pred, QUARTILES, axis=1)
    return light_score(quartiles, quartiles_pred)


def train(model, trainloader, validset, Y_train_mean, Y_train_std, patience=4):
    config = dict(
            patience=patience,
            modelname=model.__class__.__name__)
    with wandb.init(config=config, project="ariel"):
        wandb.watch(model)
        optimiser = optim.Adam(model.parameters())
        score_valid_best = 0
        i = 0
        while i < patience:
            for X_batch, Y_batch in trainloader:
                optimiser.zero_grad()
                loss = model.loss(model(X_batch), Y_batch)
                loss.backward()
                optimiser.step()
            score_valid = evaluate(model, validset, Y_train_mean, Y_train_std)
            if score_valid > score_valid_best:
                i = 0
                score_valid_best = score_valid
                model_state_best = deepcopy(model.state_dict())
            else:
                i += 1
            wandb.log({"light_score_valid": score_valid})
            wandb.run.summary["light_score_valid"] = score_valid_best
        model.load_state_dict(model_state_best)
        torch.save(model_state_best, f"models/{model.__class__.__name__.lower()}.pt")
