from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch.utils.data import DataLoader
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


def light_score(quartiles, quartiles_pred):
    return 100 * (10 - np.sqrt(((1 - quartiles_pred / quartiles) ** 2).mean()))


def train(Model, trainset, validset, Y_train_mean, Y_train_std, config):
    with wandb.init(config=config, project="ariel"):
        config = wandb.config
        model = Model(config["dropout_probability"], config["T"])
        wandb.watch(model)
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        # TODO optimise learning rate and weight decay
        optimiser = optim.Adam(model.parameters())
        score_valid_best = 0
        i = 0
        while i < config["patience"]:
            for X_batch, Y_batch in trainloader:
                optimiser.zero_grad()
                loss = model.loss(model(X_batch), Y_batch)
                loss.backward()
                optimiser.step()
            score_valid = model.evaluate(validset, Y_train_mean, Y_train_std)
            if score_valid > score_valid_best:
                i = 0
                score_valid_best = score_valid
                model_state_best = deepcopy(model.state_dict())
            else:
                i += 1
            wandb.log({"light_score_valid": score_valid})
            wandb.run.summary["light_score_valid"] = score_valid_best
        model.load_state_dict(model_state_best)
        torch.save(model_state_best, f"models/{wandb.run.name}.pt")
