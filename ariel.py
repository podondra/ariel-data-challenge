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
from torch.utils.data import DataLoader, Dataset
import wandb


N_WAVES = 52
N_AUXILLARY = 9
N_TARGETS = 6
QUARTILES = [0.16, 0.5, 0.84]
N = 21987
HYPERPARAMETER_DEFAULTS = dict(
        batch_size=256,
        learning_rate=0.0001,
        patience=2048,
        n_hiddens=7,
        n_neurons=128,
        weight_decay=0)


def read_spectra(path="data/train/spectra.hdf5", n=N):
    with h5py.File(path, "r") as file:
        spectra = torch.zeros((4, n, 52))
        for i in range(n):
            key = "Planet_" + str(i)
            spectra[0, i] = torch.from_numpy(file[key]["instrument_wlgrid"][:])
            spectra[1, i] = torch.from_numpy(file[key]["instrument_spectrum"][:])
            spectra[2, i] = torch.from_numpy(file[key]["instrument_noise"][:])
            spectra[3, i] = torch.from_numpy(file[key]["instrument_width"][:])
    return spectra


def read_traces(path="data/train/ground_truth/traces.hdf5", n=N):
    with h5py.File(path, "r") as file:
        mean = torch.zeros((n, 6))
        var = torch.zeros((n, 6))
        for i in range(n):
            key = "Planet_" + str(i)
            tracedata = file[key]["tracedata"][:]
            weights = file[key]["weights"][:]
            mean[i] = torch.from_numpy(tracedata.T @ weights)
    return mean


def read_auxillary_table(path="data/train/auxillary_table.csv", n=N):
    auxillary = pd.read_csv(path, index_col="planet_ID", nrows=n)
    return torch.from_numpy(auxillary.values).float()


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


def standardise(tensor, mean, std):
    return (tensor - mean) / std


class SpectraDataset(Dataset):
    def __init__(self, X, auxillary, Y, quartiles, X_mean, X_std):
        self.X = X
        self.auxillary = auxillary
        self.Y = Y
        self.quartiles = quartiles
        self.X_mean = X_mean
        self.X_std = X_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, aux, y = self.X[idx], self.auxillary[idx], self.Y[idx]
        x = (x - self.X_mean) / self.X_std
        return x, aux, y


class NoisySpectraDataset(Dataset):
    def __init__(self, X, noise, auxillary, Y, quartiles, X_mean, X_std):
        self.X = X
        self.noise = noise
        self.auxillary = auxillary
        self.Y = Y
        self.quartiles = quartiles
        self.X_mean = X_mean
        self.X_std = X_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, aux, y = self.X[idx], self.auxillary[idx], self.Y[idx]
        x = x + torch.normal(torch.tensor(0.0, device=torch.device("cuda")), noise_train[idx])
        x = (x - self.X_mean) / self.X_std
        return x, aux, y


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.input = nn.Linear(N_WAVES + N_AUXILLARY, self.n_neurons)
        self.linears = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.add_module("linear" + str(i), self.linears[-1])
        self.output = nn.Linear(self.n_neurons, 2 * N_TARGETS)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, X, aux):
        X = torch.cat((X, aux), dim=1)
        X = F.relu(self.input(X))
        for linear in self.linears:
            X = F.relu(linear(X))
        X = self.output(X)
        mean, var = X[:, :N_TARGETS], X[:, N_TARGETS:]
        var = F.softplus(var) + 1e-6
        return mean, var

    def loss(self, Y_pred, Y):
        mean, var = Y_pred
        return torch.mean(0.5 * torch.log(var) + 0.5 * (Y - mean).square() / var)

    @torch.no_grad()
    def predict(self, dataset, batch_size=2048):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        mean, var = list(zip(*[self(X_batch, aux_batch) for X_batch, aux_batch, _ in dataloader]))
        mean, var = torch.concat(mean), torch.concat(var)
        return mean, var

    def evaluate(self, Y_pred, dataset):
        mean, var = Y_pred
        mean, var = mean.cpu().numpy(), var.cpu().numpy()
        std = np.sqrt(var)
        quartiles_pred = np.stack([norm.ppf(quartile, loc=mean, scale=std) for quartile in QUARTILES])
        return light_score(dataset.quartiles, quartiles_pred)


def train(Model, trainset, validset, config):
    with wandb.init(config=config, project="ariel-data-challenge"):
        config = wandb.config
        model = Model(config)
        wandb.watch(model)
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        optimiser = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"])
        score_valid_best = float("-inf")
        i = 0
        while i < config["patience"]:
            model.train()
            for X_batch, aux_batch, Y_batch in trainloader:
                optimiser.zero_grad()
                loss = model.loss(model(X_batch, aux_batch), Y_batch)
                loss.backward()
                optimiser.step()
            model.eval()
            output_train = model.predict(trainset)
            output_valid = model.predict(validset)
            loss_train = model.loss(output_train, trainset.Y).item()
            loss_valid = model.loss(output_valid, validset.Y).item()
            score_train = model.evaluate(output_train, trainset)
            score_valid = model.evaluate(output_valid, validset)
            if score_valid > score_valid_best:
                i = 0
                score_valid_best = score_valid
                score_train_at_best = score_train
                loss_train_at_best = loss_train
                loss_valid_at_best = loss_valid
                model_state_best = deepcopy(model.state_dict())
                torch.save(model_state_best, f"models/{wandb.run.name}.pt")
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
        return model


if __name__ == "__main__":
    spectra = read_spectra()
    X = spectra[1]
    noise = spectra[2]
    auxillary = read_auxillary_table()
    quartiles = read_quartiles_table()
    Y = read_traces()

    # train and validation set split
    # TODO out-of-distribution split?
    ids = torch.arange(N)
    ids_train, ids_valid = train_test_split(ids, train_size=0.8, random_state=36)
    idx_train = torch.zeros_like(ids, dtype=torch.bool)
    idx_train[ids_train] = True
    idx_valid = ~idx_train

    X_train, X_valid = X[idx_train], X[idx_valid]
    noise_train, noise_valid = noise[idx_train], noise[idx_valid]
    auxillary_train, auxillary_valid = auxillary[idx_train], auxillary[idx_valid]
    Y_train, Y_valid = Y[idx_train], Y[idx_valid]
    quartiles_train, quartiles_valid = quartiles[:, idx_train], quartiles[:, idx_valid]

    # data preparation
    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    auxillary_train_mean = auxillary_train.mean(dim=0)
    auxillary_train_std = auxillary_train.std(dim=0)
    auxillary_train = standardise(auxillary_train, auxillary_train_mean, auxillary_train_std)
    auxillary_valid = standardise(auxillary_valid, auxillary_train_mean, auxillary_train_std)

    X_train, X_valid = X_train.cuda(), X_valid.cuda()
    noise_train, noise_valid = noise_train.cuda(), noise_valid.cuda()
    auxillary_train, auxillary_valid = auxillary_train.cuda(), auxillary_valid.cuda()
    Y_train, Y_valid = Y_train.cuda(), Y_valid.cuda()

    trainset = NoisySpectraDataset(
            X_train, noise_train, auxillary_train,
            Y_train, quartiles_train,
            X_train_mean, X_train_std)
    validset = SpectraDataset(
            X_valid, auxillary_valid,
            Y_valid, quartiles_valid,
            X_train_mean, X_train_std)

    model = train(Model, trainset, validset, HYPERPARAMETER_DEFAULTS)
