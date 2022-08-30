from copy import deepcopy

import h5py
import numpy as np
import ot
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb


DEFAULT_PRIOR_BOUNDS = np.array([
    [0, 3000],    # T range
    [-12, -2],    # 1. gas range
    [-12, -2],    # 2. gas range
    [-12, -2],    # 3. gas range
    [-12, -2],    # 4. gas range
    [-12, -2]])    # 5. gas range
HYPERPARAMETER_DEFAULTS = dict(
        batch_size=256,
        learning_rate=0.00001,
        n_hiddens=5,
        n_neurons=2048,
        patience=2048)
N = 21987
N_AUXILIARY = 9
N_TARGETS = 6
N_WAVES = 52
QUARTILES = [0.16, 0.5, 0.84]


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
        means = torch.zeros((n, 6))
        variances = torch.zeros((n, 6))
        for i in range(n):
            key = "Planet_" + str(i)
            tracedata = file[key]["tracedata"][:]
            weights = file[key]["weights"][:]
            mean = tracedata.T @ weights
            means[i] = torch.from_numpy(mean)
            variances[i] = torch.from_numpy(np.square(tracedata - mean).T @ weights)
    return means, variances


def read_auxiliary_table(path="data/train/auxiliary_table.csv", n=N):
    auxiliary = pd.read_csv(path, index_col="planet_ID", nrows=n)
    return torch.from_numpy(auxiliary.values).float()


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


def normalise(array, bounds_matrix):
    return (array - bounds_matrix[:, 0]) / (bounds_matrix[:, 1] - bounds_matrix[:, 0])


def wasserstein(trace1, trace2, w1=None, w2=None, bounds_matrix=DEFAULT_PRIOR_BOUNDS):
    w1 = ot.unif(len(trace1)) if w1 is None else w1
    w2 = ot.unif(len(trace2)) if w2 is None else w2
    trace1 = normalise(trace1, bounds_matrix)
    trace2 = normalise(trace2, bounds_matrix)
    M = ot.dist(trace1, trace2)
    M /= M.max()
    return ot.emd2(w1, w2, M, numItermax=100000)


def regular_score(tracefile, traces_pred, ids):
    # calculate the score for regular track from a predicted trace matrix (N X M X 6)
    # and a ground truth HDF5 file
    score = 0
    n = ids.shape[0]
    with h5py.File(tracefile, "r") as traces:
        for i, trace_pred in tqdm(zip(ids, traces_pred), total=n):
            key = 'Planet_' + str(i.item())
            tracedata = traces[key]['tracedata'][:]
            weights = traces[key]['weights'][:]
            score += wasserstein(trace_pred, tracedata, w2=weights)
    return 1000 * (1 - score / n)


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
    def __init__(
            self,
            ids, X, X_train_mean, X_train_std,
            auxiliary, auxiliary_train_mean, auxiliary_train_std,
            Y, quartiles):
        self.ids = ids
        self.X = standardise(X, X_train_mean, X_train_std)
        self.X_train_mean, self.X_train_std = X_train_mean, X_train_std
        self.auxiliary = standardise(auxiliary, auxiliary_train_mean, auxiliary_train_std)
        self.auxiliary_train_mean, self.auxiliary_train_std = auxiliary_train_mean, auxiliary_train_std
        self.Y = Y
        self.quartiles = quartiles
        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.auxiliary = self.auxiliary.cuda()
            self.Y = (self.Y[0].cuda(), self.Y[1].cuda())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.auxiliary[idx], (self.Y[0][idx], self.Y[1][idx])


class NoisySpectraDataset(Dataset):
    def __init__(self):
        # TODO scale noise and standardise X
        if torch.cuda.is_available():
            self.zero = torch.tensor(0.0, device=torch.device("cuda"))
        else:
            self.zero = torch.tensor(0.0)
        raise NotImplementedError()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx] + torch.normal(self.zero, self.noise[idx])
        return x, self.auxiliary[idx], (self.Y[0][idx], self.Y[1][idx])


def nll(mean_pred, var_pred, mean, var):
    return torch.mean(0.5 * torch.log(var_pred) + 0.5 * (mean - mean_pred).square() / var_pred)


def kl_divergence(mean_pred, var_pred, mean, var):
    return (torch.log(torch.sqrt(var_pred) / torch.sqrt(var))
            + 0.5 * (var + torch.square(mean_pred - mean)) / var_pred - 0.5)


def crps(mean_pred, var_pred, mean, var):
    std_pred = torch.sqrt(var_pred)
    mean_std = (mean - mean_pred) / std_pred
    pi = torch.tensor(np.pi)
    pdf = (1.0 / torch.sqrt(2.0 * pi)) * torch.exp(-0.5 * torch.square(mean_std))
    cdf = 0.5 + 0.5 * torch.erf(mean_std / torch.sqrt(torch.tensor(2.0)))
    return std_pred * (mean_std * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.input = nn.Linear(N_WAVES + N_AUXILIARY, self.n_neurons)
        self.linears = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.add_module("linear" + str(i), self.linears[-1])
        self.output = nn.Linear(self.n_neurons, 2 * N_TARGETS)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, X, auxiliary):
        X = torch.cat((X, auxiliary), dim=1)
        X = F.relu(self.input(X))
        for linear in self.linears:
            X = F.relu(linear(X))
        X = self.output(X)
        mean, var = X[:, :N_TARGETS], X[:, N_TARGETS:]
        var = F.softplus(var) + 1e-6
        return mean, var

    def loss(self, Y_pred, Y):
        mean_pred, var_pred = Y_pred
        mean, var = Y
        return torch.mean(crps(mean_pred, var_pred, mean, var))

    @torch.no_grad()
    def predict(self, dataset, batch_size=2048):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        output = [self(X_batch, auxiliary_batch) for X_batch, auxiliary_batch, _ in dataloader]
        mean, var = list(zip(*output))
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
        # TODO wandb.watch(model)
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])
        score_valid_best = float("-inf")
        i = 0
        while i < config["patience"]:
            model.train()
            for X_batch, auxiliary_batch, Y_batch in trainloader:
                optimiser.zero_grad()
                loss = model.loss(model(X_batch, auxiliary_batch), Y_batch)
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
                model_state_at_best = deepcopy(model.state_dict())
                torch.save(model_state_at_best, f"models/{wandb.run.name}.pt")
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
        model.load_state_dict(model_state_at_best)
        return model


def get_datasets():
    spectra = read_spectra()
    X = spectra[1]
    noise = spectra[2]
    auxiliary = read_auxiliary_table()
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
    auxiliary_train, auxiliary_valid = auxiliary[idx_train], auxiliary[idx_valid]
    Y_train, Y_valid = (Y[0][idx_train], Y[1][idx_train]), (Y[0][idx_valid], Y[1][idx_valid])
    quartiles_train, quartiles_valid = quartiles[:, idx_train], quartiles[:, idx_valid]

    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    auxiliary_train_mean = auxiliary_train.mean(dim=0)
    auxiliary_train_std = auxiliary_train.std(dim=0)

    trainset = SpectraDataset(
            ids_train,
            X_train, X_train_mean, X_train_std,
            auxiliary_train, auxiliary_train_mean, auxiliary_train_std,
            Y_train, quartiles_train)
    validset = SpectraDataset(
            ids_valid,
            X_valid, X_train_mean, X_train_std,
            auxiliary_valid, auxiliary_train_mean, auxiliary_train_std,
            Y_valid, quartiles_valid)
    return trainset, validset


if __name__ == "__main__":
    trainset, validset = get_datasets()
    model = train(Model, trainset, validset, HYPERPARAMETER_DEFAULTS)
