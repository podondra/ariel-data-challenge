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
import wandb


DEFAULT_PRIOR_BOUNDS = np.array([[0, -12, -12, -12, -12, -12], [7000, -1, -1, -1, -1, -1]])
DEFAULT_HYPERPARAMETERS = dict(
        batch_size=256,
        learning_rate=0.0001,
        loss="kl_divergence",
        n_epochs=2048,
        n_hiddens=5,
        n_neurons=1024,
        patience=2048)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ANNOTATED = 21988
N_AUXILIARY = 9
N_TARGETS = 6
N_TEST = 800
N_WAVES = 52
QUARTILES = [0.16, 0.5, 0.84]


def read_spectra(ids, path="data/train/spectra.hdf5"):
    n = ids.shape[0]
    with h5py.File(path, "r") as file:
        spectra = torch.zeros((4, n, N_WAVES))
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
        variance = torch.zeros((n, N_TARGETS))
        for i, identifier in enumerate(ids):
            key = "Planet_" + str(identifier)
            trace = file[key]["tracedata"][:]
            weights = file[key]["weights"][:]
            mean = trace.T @ weights
            Y[i] = torch.from_numpy(mean)
            variance[i] = torch.from_numpy(np.square(trace - mean).T @ weights)
    return Y, variance


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
    matrix = (matrix - prior_bounds[0]) / (prior_bounds[1] - prior_bounds[0])
    matrix[matrix < 0] = 0
    matrix[matrix > 1] = 1
    return matrix


def earth_movers_distance(trace1, trace2, w1, w2, prior_bounds=DEFAULT_PRIOR_BOUNDS):
    trace1 = normalise(trace1, prior_bounds)
    trace2 = normalise(trace2, prior_bounds)
    M = ot.dist(trace1, trace2)
    M /= M.max()
    return ot.emd2(w1, w2, M)


def regular_score(traces_pred, ids, tracefile="data/train/ground_truth/traces.hdf5"):
    # calculate the score for regular track from a predicted trace matrix (N X T X 6)
    weights_pred = ot.unif(traces_pred.shape[1])
    score = 0
    n = ids.shape[0]
    with h5py.File(tracefile, "r") as traces:
        for i, trace_pred in zip(ids, traces_pred):
            key = "Planet_" + str(i)
            trace = traces[key]["tracedata"][:]
            weights = traces[key]["weights"][:]
            score += earth_movers_distance(trace_pred, trace, weights_pred, weights)
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


def nll(mean_pred, var_pred, mean, var):
    return torch.log(var_pred) + torch.square(mean - mean_pred) / var_pred


def crps(mean_pred, var_pred, mean, var):
    std_pred = torch.sqrt(var_pred)
    mean_std = (mean - mean_pred) / std_pred
    pi = torch.tensor(np.pi)
    pdf = (1.0 / torch.sqrt(2.0 * pi)) * torch.exp(-0.5 * torch.square(mean_std))
    cdf = 0.5 + 0.5 * torch.erf(mean_std / torch.sqrt(torch.tensor(2.0)))
    return std_pred * (mean_std * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))


def kl_divergence(mean_pred, var_pred, mean, var):
    return torch.log(var_pred / var) + (var + torch.square(mean_pred - mean)) / var_pred


def wasserstein(mean_pred, var_pred, mean, var):
    return torch.square(mean_pred - mean) + torch.square(torch.sqrt(var_pred) - torch.sqrt(var))


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv1d(1, 8, 3, padding="same"), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 3, padding="same"), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding="same"), nn.ReLU(),
                nn.Conv1d(32, 32, 3, padding="same"), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding="same"), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding="same"), nn.ReLU(),
                nn.MaxPool1d(2))
        self.n_neurons = config["n_neurons"]
        self.n_hiddens = config["n_hiddens"]
        self.linear0 = nn.Linear(201, self.n_neurons)
        self.linears = []
        for i in range(1, self.n_hiddens + 1):
            self.linears.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.add_module("linear" + str(i), self.linears[-1])
        self.output = nn.Linear(self.n_neurons, 2 * N_TARGETS)
        self.to(DEVICE)
        losses = {
            "crps": crps,
            "kl_divergence": kl_divergence,
            "nll": nll,
            "wasserstein": wasserstein}
        self.loss_function = losses[config["loss"]]

    def forward(self, X, auxiliary):
        X = torch.unsqueeze(X, 1)
        X = self.cnn(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.cat((X, auxiliary), dim=1)
        X = F.relu(self.linear0(X))
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
        #sample = np.random.normal(loc=mean, scale=std, size=(T, *mean.shape)).swapaxes(0, 1)
        #quartiles = np.quantile(sample, QUARTILES, axis=1)
        quartiles_pred = np.stack([norm.ppf(quartile, loc=mean, scale=std) for quartile in QUARTILES])
        return light_score(dataset.quartiles, quartiles_pred)


def standardise(tensor, mean, std):
    return (tensor - mean) / std


def scale(tensor):
    return (tensor - tensor.mean(dim=1, keepdim=True)) / tensor.std(dim=1, keepdim=True)


class SpectraDataset(Dataset):
    def __init__(self, ids, X, auxiliary, auxiliary_mean, auxiliary_std, Y, variance, quartiles):
        self.ids = ids
        self.X = scale(X).to(DEVICE)
        self.auxiliary = standardise(auxiliary, auxiliary_mean, auxiliary_std).to(DEVICE)
        self.auxiliary_mean, self.auxiliary_std = auxiliary_mean, auxiliary_std
        self.Y = Y.to(DEVICE)
        self.variance = variance.to(DEVICE)
        self.quartiles = quartiles

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.auxiliary[idx], (self.Y[idx], self.variance[idx])


def get_dataset(ids, auxiliary_mean=None, auxiliary_std=None):
    spectra = read_spectra(ids)
    X = spectra[1]
    auxiliary = read_auxiliary_table(ids)
    Y, variance = read_traces(ids)
    quartiles = read_quartiles_table(ids)
    return SpectraDataset(
            ids, X, auxiliary,
            auxiliary.mean(dim=0) if auxiliary_mean is None else auxiliary_mean,
            auxiliary.std(dim=0) if auxiliary_std is None else auxiliary_std,
            Y, variance, quartiles)


def train_early_stopping(model, config, trainset, validset):
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])
        log = dict()
        log["light_score_valid_best"] = float("-inf")
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
            log["loss_train"] = model.loss(output_train, (trainset.Y, trainset.variance)).item()
            log["loss_valid"] = model.loss(output_valid, (validset.Y, validset.variance)).item()
            log["light_score_train"] = model.evaluate(output_train, trainset)
            log["light_score_valid"] = model.evaluate(output_valid, validset)
            if log["light_score_valid"] > log["light_score_valid_best"]:
                i = 0
                loss_train_at_best = log["loss_train"]
                loss_valid_at_best = log["loss_valid"]
                score_train_at_best = log["light_score_train"]
                log["light_score_valid_best"] = log["light_score_valid"]
                model_state_at_best = deepcopy(model.state_dict())
            else:
                i += 1
            wandb.log(log)
            wandb.run.summary["loss_train"] = loss_train_at_best
            wandb.run.summary["loss_valid"] = loss_valid_at_best
            wandb.run.summary["light_score_train"] = score_train_at_best
            wandb.run.summary["light_score_valid"] = log["light_score_valid_best"]
        model.load_state_dict(model_state_at_best)
        return model


def train_epochs(model, config, trainset, validset):
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])
    log = dict()
    for epoch in range(config["n_epochs"]):
        model.train()
        for X_batch, auxiliary_batch, Y_batch in trainloader:
            optimiser.zero_grad()
            loss = model.loss(model(X_batch, auxiliary_batch), Y_batch)
            loss.backward()
            optimiser.step()
        model.eval()
        output_train = model.predict(trainset)
        log["loss_train"] = model.loss(output_train, (trainset.Y, trainset.variance)).item()
        log["light_score_train"] = model.evaluate(output_train, trainset)
        if validset is not None:
            output_valid = model.predict(validset)
            log["loss_valid"] = model.loss(output_valid, (validset.Y, validset.variance)).item()
            log["light_score_valid"] = model.evaluate(output_valid, validset)
        wandb.log(log)
    return model


if __name__ == "__main__":
    # train and validation set split
    ids_train = np.arange(N_ANNOTATED)
    validset = None
    #ids_train, ids_valid = train_test_split(ids_train, train_size=0.8, random_state=36)
    trainset = get_dataset(ids_train)
    #validset = get_dataset(ids_valid, trainset.auxiliary_mean, trainset.auxiliary_std)
    config = DEFAULT_HYPERPARAMETERS
    with wandb.init(config=config, project="ariel-data-challenge"):
        config = wandb.config
        model = Model(config)
        model = train_epochs(model, config, trainset, validset)
        #model = train_early_stopping(model, config, trainset, validset)
        torch.save(model.state_dict(), f"models/{wandb.run.name}.pt")
