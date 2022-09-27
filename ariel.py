from copy import deepcopy

import h5py
import numpy as np
import ot
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import distributions
from torch import nn
from torch import optim
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb


DEFAULT_PRIOR_BOUNDS = np.array([[0, -12, -12, -12, -12, -12], [3000, -2, -2, -2, -2, -2]])
DEFAULT_HYPERPARAMETERS = dict(
        batch_size=256,
        dropout_probability=0.2,
        learning_rate=0.0001,
        n_epochs=2048,
        n_epochs_pretrain=150,
        n_hiddens=5,
        n_neurons=1024,
        noisy_spectra=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 91392
N_ANNOTATED = 21988
N_AUXILIARY = 9
N_TARGETS = 6
N_WAVES = 52
QUARTILES = [0.16, 0.5, 0.84]
T = 250


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
        Y = torch.zeros((n, N_TARGETS))
        covariance = torch.zeros((n, N_TARGETS, N_TARGETS))
        for i, identifier in enumerate(ids):
            key = "Planet_" + str(identifier)
            trace = file[key]["tracedata"][:]
            weights = file[key]["weights"][:]
            mean = trace.T @ weights
            Y[i] = torch.from_numpy(mean)
            covariance[i] = torch.from_numpy((weights * (trace - mean).T @ (trace - mean)))
    return Y, covariance


def read_auxiliary_table(ids, path="data/train/auxiliary_table.csv"):
    auxiliary = pd.read_csv(path, index_col="planet_ID")
    return torch.from_numpy(auxiliary.loc[ids].values).float()


def read_fm_parameter_table(ids, path="data/train/ground_truth/fm_parameter_table.csv"):
    fm_parameter = pd.read_csv(path, index_col="planet_ID")
    return torch.from_numpy(fm_parameter.loc[ids].values).float()


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
    return ot.emd2(w1, w2, M)


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


def nll(Y_pred, Y):
    mean_pred, L_pred = Y_pred
    mean, _ = Y
    return (torch.sum(torch.log(L_pred.diagonal(dim1=-2, dim2=-1)), dim=-1)
            + 0.5 * _batch_mahalanobis(L_pred, mean - mean_pred))


def kl_divergence(Y_pred, Y):
    mean_pred, L_pred = Y_pred
    mean, covariance = Y
    distribution_pred = distributions.MultivariateNormal(mean_pred, scale_tril=L_pred)
    distribution = distributions.MultivariateNormal(mean, covariance_matrix=covariance)
    return distributions.kl.kl_divergence(distribution, distribution_pred)


def crps(mean_pred, var_pred, mean, var):
    # TODO can be applied to covariance matrix?
    std_pred = torch.sqrt(var_pred)
    mean_std = (mean - mean_pred) / std_pred
    pi = torch.tensor(np.pi)
    pdf = (1.0 / torch.sqrt(2.0 * pi)) * torch.exp(-0.5 * torch.square(mean_std))
    cdf = 0.5 + 0.5 * torch.erf(mean_std / torch.sqrt(torch.tensor(2.0)))
    return std_pred * (mean_std * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # TODO reduce CNN to VGG Net-A
        self.cnn = nn.Sequential(
                nn.Conv1d(1, 8, 3, padding="same", bias=False), nn.BatchNorm1d(8), nn.ReLU(),
                nn.Conv1d(8, 8, 3, padding="same", bias=False), nn.BatchNorm1d(8), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 3, padding="same", bias=False), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 16, 3, padding="same", bias=False), nn.BatchNorm1d(16), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding="same", bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 3, padding="same", bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding="same", bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding="same", bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2))
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.linear0 = nn.Linear(201, self.n_neurons, bias=False)
        self.dropout0 = nn.Dropout(config["dropout_probability"])
        self.batchnorm0 = nn.BatchNorm1d(self.n_neurons)
        self.linears = []
        self.dropouts = []
        self.batchnorms = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons, bias=False))
            self.add_module("linear" + str(i), self.linears[-1])
            self.batchnorms.append(nn.BatchNorm1d(self.n_neurons))
            self.add_module("batchnorm" + str(i), self.batchnorms[-1])
            self.dropouts.append(nn.Dropout(config["dropout_probability"]))
            self.add_module("dropout" + str(i), self.dropouts[-1])
        self.output = nn.Linear(self.n_neurons, N_TARGETS + N_TARGETS * (N_TARGETS + 1) // 2)
        self.to(DEVICE)
        self.loss_function = kl_divergence
        self.ind_diag = torch.arange(N_TARGETS)
        self.ind_tril = torch.tril_indices(row=N_TARGETS, col=N_TARGETS, offset=-1)

    def pretrain(self, flag):
        self.loss_function = nll if flag else kl_divergence

    def forward(self, X, auxiliary):
        X = torch.unsqueeze(X, 1)
        X = self.cnn(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.cat((X, auxiliary), dim=1)
        X = F.relu(self.batchnorm0(self.linear0(X)))
        X = self.dropout0(X)
        for linear, batchnorm, dropout in zip(self.linears, self.batchnorms, self.dropouts):
            X = F.relu(batchnorm(linear(X)))
            X = dropout(X)
        X = self.output(X)
        mean = X[:, :N_TARGETS]
        variance = X[:, N_TARGETS:2 * N_TARGETS]
        L_vector = X[:, 2 * N_TARGETS:]
        L = torch.zeros((L_vector.shape[0], N_TARGETS, N_TARGETS), device=DEVICE)
        L[:, self.ind_diag, self.ind_diag] = F.softplus(variance) + 1e-6
        L[:, self.ind_tril[0], self.ind_tril[1]] = L_vector
        return mean, L

    def loss(self, Y_pred, Y):
        return torch.mean(self.loss_function(Y_pred, Y))

    @torch.no_grad()
    def predict(self, dataset, batch_size=2048):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        output = [self(X_batch, auxiliary_batch) for X_batch, auxiliary_batch, _ in dataloader]
        mean, L = list(zip(*output))
        mean, L = torch.concat(mean), torch.concat(L)
        return mean, L

    def evaluate(self, Y_pred, dataset):
        mean, L = Y_pred
        mean, L = mean.cpu(), L.cpu()
        sample = distributions.MultivariateNormal(mean, scale_tril=L).sample((T, ))
        quartiles_pred = np.quantile(sample.numpy(), QUARTILES, axis=0)
        return light_score(dataset.quartiles, quartiles_pred)


def standardise(tensor, mean, std):
    return (tensor - mean) / std


def scale(X):
    return (X - X.mean(dim=1, keepdim=True)) / X.std(dim=1, keepdim=True)


class SpectraDataset(Dataset):
    def __init__(
            self, ids, X, noise, auxiliary, Y, covariance=None, quartiles=None,
            auxiliary_mean=None, auxiliary_std=None):
        self.ids = ids
        self.X_orig = X
        self.X = scale(self.X_orig).to(DEVICE)
        self.X_orig = self.X_orig.to(DEVICE)
        self.noise = noise.to(DEVICE)
        self.auxiliary_mean = auxiliary.mean(dim=0) if auxiliary_mean is None else auxiliary_mean
        self.auxiliary_std = auxiliary.std(dim=0) if auxiliary_std is None else auxiliary_std
        self.auxiliary = standardise(auxiliary, self.auxiliary_mean, self.auxiliary_std)
        self.auxiliary = self.auxiliary.to(DEVICE)
        self.Y = Y.to(DEVICE)
        if covariance is None:
            self.covariance = torch.full((self.Y.shape[0], ), torch.nan, device=DEVICE)
        else:
            self.covariance = covariance.to(DEVICE)
        self.quartiles = quartiles

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.auxiliary[idx], (self.Y[idx], self.covariance[idx])


class NoisySpectraDataset(SpectraDataset):
    def sample(self):
        self.X = scale(self.X_orig + torch.normal(mean=0.0, std=self.noise))


def get_data(ids, pretrain=False):
    spectra = read_spectra(ids)
    data = {"ids": ids, "X": spectra[1], "noise": spectra[2], "auxiliary": read_auxiliary_table(ids)}
    if pretrain:
        data["Y"] = read_fm_parameter_table(ids)
    else:
        data["Y"], data["covariance"] = read_traces(ids)
        data["quartiles"] = read_quartiles_table(ids)
    return data


def train(model, config, n_epochs, trainset, validset=None):
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])
    for epoch in range(n_epochs):
        model.train()
        if config["noisy_spectra"]:
            trainloader = DataLoader(trainset.sample(), batch_size=config["batch_size"], shuffle=True)
        for X_batch, auxiliary_batch, Y_batch in trainloader:
            optimiser.zero_grad()
            loss = model.loss(model(X_batch, auxiliary_batch), Y_batch)
            loss.backward()
            optimiser.step()
        model.eval()
        output_train = model.predict(trainset)
        log = dict()
        log["loss_train"] = model.loss(output_train, (trainset.Y, trainset.covariance)).item()
        if trainset.quartiles is not None:
            log["light_score_train"] = model.evaluate(output_train, trainset)
        if validset is not None:
            output_valid = model.predict(validset)
            log["loss_valid"] = model.loss(output_valid, (validset.Y, validset.covariance)).item()
            log["light_score_valid"] = model.evaluate(output_valid, validset)
        wandb.log(log)
    return model


if __name__ == "__main__":
    ids_pretrain = np.arange(N_ANNOTATED, N)
    data_pretrain = get_data(ids_pretrain, pretrain=True)
    pretrainset = NoisySpectraDataset(**data_pretrain)
    ids_train = np.arange(N_ANNOTATED)
    ids_train, ids_valid = train_test_split(ids_train, train_size=0.8, random_state=36)
    data_train = get_data(ids_train)
    data_valid = get_data(ids_valid)
    trainset = NoisySpectraDataset(
            **data_train,
            auxiliary_mean=pretrainset.auxiliary_mean, auxiliary_std=pretrainset.auxiliary_std)
    validset = SpectraDataset(
            **data_valid,
            auxiliary_mean=pretrainset.auxiliary_mean, auxiliary_std=pretrainset.auxiliary_std)
    config = DEFAULT_HYPERPARAMETERS
    with wandb.init(config=config, project="ariel-data-challenge"):
        config = wandb.config
        model = Model(config)
        model.pretrain(True)
        # TODO pre-train with early stopping on trainset?
        model = train(model, config, config["n_epochs_pretrain"], pretrainset, validset)
        model.pretrain(False)
        model = train(model, config, config["n_epochs"], trainset, validset)
        torch.save(model.state_dict(), f"models/{wandb.run.name}.pt")
