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


DEFAULT_PRIOR_BOUNDS = np.array([[0, -12, -12, -12, -12, -12], [3000, -2, -2, -2, -2, -2]])
DEFAULT_HYPERPARAMETERS = dict(
        batch_size=256,
        learning_rate=0.0015,
        loss="kl_divergence",
        n_hiddens=4,
        n_neurons=2048,
        patience=2048)
N = 21987
N_AUXILIARY = 9
N_TARGETS = 6
N_WAVES = 52
QUARTILES = [0.16, 0.5, 0.84]


def read_spectra(ids, path="data/train/spectra.hdf5"):
    n = ids.shape[0]
    with h5py.File(path, "r") as file:
        spectra = torch.zeros((4, n, 52))
        for i, identifier in enumerate(ids):
            key = "Planet_" + str(identifier)
            spectra[0, i] = torch.from_numpy(file[key]["instrument_wlgrid"][:])
            spectra[1, i] = torch.from_numpy(file[key]["instrument_spectrum"][:])
            spectra[2, i] = torch.from_numpy(file[key]["instrument_noise"][:])
            spectra[3, i] = torch.from_numpy(file[key]["instrument_width"][:])
    return spectra


def read_traces(ids, path="data/train/ground_truth/traces.hdf5"):
    n = ids.shape[0]
    with h5py.File(path, "r") as file:
        means = torch.zeros((n, 6))
        variances = torch.zeros((n, 6))
        for i, identifier in enumerate(ids):
            key = "Planet_" + str(identifier)
            trace = file[key]["tracedata"][:]
            weights = file[key]["weights"][:]
            mean = trace.T @ weights
            means[i] = torch.from_numpy(mean)
            variances[i] = torch.from_numpy(np.square(trace - mean).T @ weights)
    return means, variances


def read_auxiliary_table(ids, path="data/train/auxiliary_table.csv"):
    auxiliary = pd.read_csv(path, index_col="planet_ID")
    return torch.from_numpy(auxiliary.loc[ids].values).float()


def read_fm_parameter_table(ids, path="data/train/ground_truth/fm_parameter_table.csv"):
    fm_parameter = pd.read_csv(path, index_col="planet_ID")
    return torch.from_numpy(fm_parameter.loc[ids].values)


def read_quartiles_table(ids, path="data/train/ground_truth/quartiles_table.csv"):
    df = pd.read_csv(path, index_col="planet_ID").loc[ids]
    quartiles = ["q1", "q2", "q3"]
    targets = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    X = np.zeros((len(quartiles), df.shape[0], len(targets)))
    for i, quartile in enumerate(quartiles):
        for j, target in enumerate(targets):
            X[i, :, j] = df.loc[:, target + '_' + quartile]
    return X


def light_score(quartiles, quartiles_pred):
    return 100 * (10 - np.sqrt(np.mean(np.square((quartiles - quartiles_pred) / quartiles))))


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


def normalise(matrix, prior_bounds):
    return (matrix - prior_bounds[0]) / (prior_bounds[1] - prior_bounds[0])


def wasserstein(trace1, trace2, w2, prior_bounds=DEFAULT_PRIOR_BOUNDS):
    w1 = ot.unif(trace1.shape[0])
    trace1 = normalise(trace1, prior_bounds)
    trace2 = normalise(trace2, prior_bounds)
    M = ot.dist(trace1, trace2)
    M /= M.max()
    return ot.emd2(w1, w2, M, numItermax=100000)


def regular_score(traces_pred, ids, tracefile="data/train/ground_truth/traces.hdf5"):
    # calculate the score for regular track from a predicted trace matrix (N X M X 6)
    # and a ground truth HDF5 file
    score = 0
    n = ids.shape[0]
    with h5py.File(tracefile, "r") as traces:
        for i, trace_pred in tqdm(zip(ids, traces_pred), total=n):
            key = "Planet_" + str(i)
            trace = traces[key]["tracedata"][:]
            weights = traces[key]["weights"][:]
            score += wasserstein(trace_pred, trace, w2=weights)
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
            grp.attrs["ID"] = i
            grp.create_dataset("tracedata", data=trace)
            grp.create_dataset("weights", data=weight)


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
        self.cnn = nn.Sequential(
                nn.Conv1d(1, 8, 3, padding="same"),
                nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 3, padding="same"),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding="same"),
                nn.Conv1d(32, 32, 3, padding="same"),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding="same"),
                nn.Conv1d(64, 64, 3, padding="same"),
                nn.MaxPool1d(2))
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.input = nn.Linear(201, self.n_neurons)
        self.linears = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.add_module("linear" + str(i), self.linears[-1])
        self.output = nn.Linear(self.n_neurons, 2 * N_TARGETS)
        if torch.cuda.is_available():
            self.cuda()
        losses = {
            "crps": crps,
            "kl_divergence": kl_divergence,
            "nll": nll}
        self.loss_function = losses[config["loss"]]

    def forward(self, X, auxiliary):
        X = torch.unsqueeze(X, 1)
        X = self.cnn(X)
        X = torch.flatten(X, start_dim=1)
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
        return torch.mean(self.loss_function(mean_pred, var_pred, mean, var))

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


def get_dataset(
        ids,
        X_train_mean=None, X_train_std=None,
        auxiliary_train_mean=None, auxiliary_train_std=None):
    spectra = read_spectra(ids)
    X, noise = spectra[1], spectra[2]
    auxiliary = read_auxiliary_table(ids)
    quartiles = read_quartiles_table(ids)
    Y = read_traces(ids)
    return SpectraDataset(
            ids,
            X,
            X.mean() if X_train_mean is None else X_train_mean,
            X.std() if X_train_std is None else X_train_std,
            auxiliary,
            auxiliary.mean(dim=0) if auxiliary_train_mean is None else auxiliary_train_mean,
            auxiliary.std(dim=0) if auxiliary_train_std is None else auxiliary_train_std,
            Y,
            quartiles)


if __name__ == "__main__":
    # train and validation set split
    # TODO out-of-distribution split?
    ids = np.arange(N)
    ids_train, ids_valid = train_test_split(ids, train_size=0.8, random_state=36)
    trainset = get_dataset(ids_train)
    validset = get_dataset(
            ids_valid,
            trainset.X_train_mean,
            trainset.X_train_std,
            trainset.auxiliary_train_mean,
            trainset.auxiliary_train_std)
    model = train(Model, trainset, validset, DEFAULT_HYPERPARAMETERS)
