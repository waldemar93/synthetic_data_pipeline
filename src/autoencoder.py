import copy
import json
import os
import optuna
import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from torch import nn
import pickle

from src.base import PROJECT_DIR
from src.helpers import load_experiment_data, load_columns_json_for_experiment


def preprocess_data(data: pd.DataFrame) -> Tuple[
    pd.DataFrame, StandardScaler,
    StandardScaler, OneHotEncoder, OneHotEncoder]:
    """
    Preprocess the input DataFrame by standardizing continuous columns and one-hot encoding binary columns with missing values.

    :param data: A DataFrame containing the input data with columns "CR1", "OSTM", "OSSTAT", "EFSTM", "EFSSTAT", "RFSSTAT"
    :return: A tuple containing the preprocessed DataFrame, the StandardScaler instances for OSTM and EFSTM columns, and the OneHotEncoder instances for EFSSTAT and RFSSTAT columns
    """

    data = data.copy()  # Create a copy of the input DataFrame
    # One-Hot-Encoding for binary columns with missing values
    one_hot_encoder_efsstat = OneHotEncoder(sparse_output=False, drop='first', categories='auto')
    efsstat_encoded = one_hot_encoder_efsstat.fit_transform(data['EFSSTAT'].values.reshape(-1, 1))

    one_hot_encoder_rfsstat = OneHotEncoder(sparse_output=False, drop='first', categories='auto')
    rfsstat_encoded = one_hot_encoder_rfsstat.fit_transform(data['RFSSTAT'].values.reshape(-1, 1))

    # log transform continuous column
    data.loc[:, 'OSTM'] = np.log1p(data['OSTM'].values.reshape(-1, 1))
    data.loc[:, 'EFSTM'] = np.log1p(data['EFSTM'].values.reshape(-1, 1))

    # Standardize continuous columns
    scaler_ostm = StandardScaler()#MinMaxScaler(feature_range=(-1, 1))
    data.loc[:, 'OSTM'] = scaler_ostm.fit_transform(data['OSTM'].values.reshape(-1, 1))

    scaler_efstm = StandardScaler()#MinMaxScaler(feature_range=(-1, 1))
    data.loc[:, 'EFSTM'] = scaler_efstm.fit_transform(data['EFSTM'].values.reshape(-1, 1))

    # Replace original binary columns with encoded versions and remove the original columns
    data = data.drop(columns=['EFSSTAT', 'RFSSTAT'])
    data['EFSSTAT_0'] = efsstat_encoded[:, 0]
    # data['EFSSTAT_1'] = efsstat_encoded[:, 1]
    data['RFSSTAT_0'] = rfsstat_encoded[:, 0]
    data['RFSSTAT_1'] = rfsstat_encoded[:, 1]

    return data, scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat


def preprocess_data_with_scalers(df: DataFrame, scaler_ostm: StandardScaler, scaler_efstm: StandardScaler,
                                 one_hot_encoder_efsstat: OneHotEncoder, one_hot_encoder_rfsstat: OneHotEncoder) -> DataFrame:
    efsstat_encoded = one_hot_encoder_efsstat.transform(df['EFSSTAT'].values.reshape(-1, 1))
    rfsstat_encoded = one_hot_encoder_rfsstat.transform(df['RFSSTAT'].values.reshape(-1, 1))

    # log transform continuous column
    df.loc[:, 'OSTM'] = np.log1p(df['OSTM'].values.reshape(-1, 1))
    df.loc[:, 'EFSTM'] = np.log1p(df['EFSTM'].values.reshape(-1, 1))

    # Standardize continuous columns
    df.loc[:, 'OSTM'] = scaler_ostm.transform(df['OSTM'].values.reshape(-1, 1))
    df.loc[:, 'EFSTM'] = scaler_efstm.transform(df['EFSTM'].values.reshape(-1, 1))

    # Replace original binary columns with encoded versions and remove the original columns
    df = df.drop(columns=['EFSSTAT', 'RFSSTAT'])
    df['EFSSTAT_0'] = efsstat_encoded[:, 0]
    # data['EFSSTAT_1'] = efsstat_encoded[:, 1]
    df['RFSSTAT_0'] = rfsstat_encoded[:, 0]
    df['RFSSTAT_1'] = rfsstat_encoded[:, 1]

    return df


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            #nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def create_autoencoder(input_dim: int, latent_dim: int) -> Autoencoder:
    """
    Create an Autoencoder model with the specified input and latent dimensions.

    :param input_dim: The input dimension of the Autoencoder
    :param latent_dim: The dimension of the latent representation in the Autoencoder
    :return: An instance of the Autoencoder model
    """

    return Autoencoder(input_dim, latent_dim)


def train_autoencoder(
        model: Autoencoder,
        data: pd.DataFrame,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        device: torch.device) -> None:
    """
    Train the Autoencoder model on the provided data.

    :param model: The Autoencoder model to train
    :param data: A DataFrame containing the preprocessed data
    :param batch_size: The batch size for the DataLoader
    :param epochs: The number of training epochs
    :param learning_rate: The learning rate for the optimizer
    :param device: The device (CPU or GPU) to train the model on
    """

    data_np = data.values
    tensor_data = torch.tensor(data_np, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0].to(device)
            _, decoded = model(x)
            loss = criterion(decoded, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


def encode_data(model: Autoencoder, data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the input data using the trained Autoencoder model.

    :param model: The trained Autoencoder model
    :param data: A DataFrame containing the preprocessed data
    :return: A DataFrame containing the encoded data (latent representation)
    """

    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    encoded_data = model.encoder(tensor_data).detach().numpy()
    encoded_df = pd.DataFrame(encoded_data, columns=['latent'])
    return encoded_df


def decode_data(
        model: Autoencoder,
        encoded_data: pd.DataFrame,
        original_data_columns: pd.Index,
        scaler_ostm: StandardScaler,
        scaler_efstm: StandardScaler,
) -> pd.DataFrame:
    """
    Decode the encoded data using the trained Autoencoder model.

    :param model: The trained Autoencoder model
    :param encoded_data: A DataFrame containing the encoded (latent) data
    :param original_data_columns: The column names of the original preprocessed data
    :param scaler_ostm: The StandardScaler instance for the OSTM column
    :param scaler_efstm: The StandardScaler instance for the EFSTM column
    :return: A DataFrame containing the decoded data
    """

    tensor_encoded_data = torch.tensor(encoded_data.values, dtype=torch.float32)
    decoded_data = model.decoder(tensor_encoded_data).detach().numpy()

    # Invert the scalers
    decoded_data[:, original_data_columns.get_loc('OSTM')] = scaler_ostm.inverse_transform(
        decoded_data[:, original_data_columns.get_loc('OSTM')].reshape(-1, 1)).flatten()
    decoded_data[:, original_data_columns.get_loc('EFSTM')] = scaler_efstm.inverse_transform(
        decoded_data[:, original_data_columns.get_loc('EFSTM')].reshape(-1, 1)).flatten()

    # inverse the log transform
    decoded_data[:, original_data_columns.get_loc('OSTM')] = np.expm1(decoded_data[:, original_data_columns.get_loc('OSTM')].reshape(-1, 1)).flatten()
    decoded_data[:, original_data_columns.get_loc('EFSTM')] = np.expm1(decoded_data[:, original_data_columns.get_loc('EFSTM')].reshape(-1, 1)).flatten()

    decoded_df = pd.DataFrame(decoded_data, columns=original_data_columns)

    # Round the binary variables
    decoded_df["CR1"] = (decoded_df["CR1"] > 0.5).astype(int)
    decoded_df["OSSTAT"] = (decoded_df["OSSTAT"] > 0.5).round().astype(int)

    # Convert one-hot-encoded variables back to the original format
    decoded_df["EFSSTAT"] = (decoded_df["EFSSTAT_0"] > 0.5).astype(int)
    decoded_df["RFSSTAT"] = decoded_df.apply(lambda row: 1 if row['RFSSTAT_0'] < 0.5 and row['RFSSTAT_1'] > 0.5 else 0 if row['RFSSTAT_0'] > 0.5 and row['RFSSTAT_1'] < 0.5 else -1, axis=1)

    # Remove the one-hot-encoded columns
    decoded_df.drop(["EFSSTAT_0", "RFSSTAT_0", "RFSSTAT_1"], axis=1, inplace=True)

    return decoded_df


def save_models_scalers(
        model: Autoencoder,
        scaler_ostm: StandardScaler,
        scaler_efstm: StandardScaler,
        one_hot_encoder_efsstat: OneHotEncoder,
        one_hot_encoder_rfsstat: OneHotEncoder,
        model_path: str,
        scalers_path: str,
) -> None:
    """
    Save the Autoencoder model and StandardScaler instances to disk.

    :param model: The Autoencoder model to save
    :param scaler_ostm: The StandardScaler instance for the OSTM column
    :param scaler_efstm: The StandardScaler instance for the EFSTM column
    :param one_hot_encoder_efsstat: The OneHotEncoder instance for the EFSSTAT column
    :param one_hot_encoder_rfsstat: The OneHotEncoder instance for RFSSTAT column
    :param model_path: The file path to save the Autoencoder model
    :param scalers_path: The file path to save the StandardScaler instances
    """

    torch.save(model.state_dict(), model_path)

    with open(scalers_path, "wb") as f:
        pickle.dump((scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat), f)


def load_models_scalers(
        model_path: str, scalers_path: str
) -> Tuple[Autoencoder, StandardScaler, StandardScaler, OneHotEncoder, OneHotEncoder]:
    """
    Load the Autoencoder model and StandardScaler instances from disk.

    :param model_path: The file path to load the Autoencoder model
    :param scalers_path: The file path to load the StandardScaler instances
    :return: A tuple containing the loaded Autoencoder model, the StandardScaler and OneHotEncoder instances
    """

    state_dict = torch.load(model_path)

    # Determine input_dim and latent_dim from the state_dict
    input_dim = state_dict["encoder.0.weight"].shape[1]
    latent_dim = state_dict["encoder.2.weight"].shape[0]

    model = Autoencoder(input_dim, latent_dim)
    model.load_state_dict(state_dict)

    with open(scalers_path, "rb") as f:
        scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat = pickle.load(f)

    return model, scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat


def optimize_autoencoder(trials):
    def suggest_params(trial: optuna.trial.Trial):
        lr = trial.suggest_float('lr', 0.000001, 0.001, log=True)
        epochs = trial.suggest_categorical('epochs', [500, 1000, 2500, 5000, 10000])
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        }
        trial.set_user_attr("params", params)
        return params

    def objective_func(trial: optuna.trial.Trial):
        df, _, _ = load_experiment_data('synthetic_aml')
        column_json = load_columns_json_for_experiment('synthetic_aml')
        df = df[column_json['outcome']]

        params = suggest_params(trial)
        preprocessed_data, scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat = preprocess_data(df)

        input_dim = preprocessed_data.shape[1]
        latent_dim = 1

        model = create_autoencoder(input_dim, latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_autoencoder(model, preprocessed_data, params['batch_size'], params['epochs'], params['lr'], device)

        model.to('cpu')  # Move the model to the appropriate device before calling the functions

        encoded_df = encode_data(model, preprocessed_data)
        decoded_df = decode_data(model, encoded_df, preprocessed_data.columns, scaler_ostm, scaler_efstm)

        sum_e = 0
        for col in df.columns:
            sum_e += mean_squared_error(df[col], decoded_df[col])
        return sum_e / len(df.columns)

    sampler = optuna.samplers.TPESampler(seed=1)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective_func, n_trials=trials, show_progress_bar=True, n_jobs=6)

    # save best_params
    best_params = study.best_trial.user_attrs['params']
    with open('autoencoder_config.json', 'w') as fh:
        json.dump(best_params, fh, indent=2)


def process_data(df: DataFrame, outcome_cols=None) -> DataFrame:
    if outcome_cols is None:
        outcome_cols = ['CR1', 'OSTM', 'OSSTAT', 'EFSTM', 'EFSSTAT', 'RFSSTAT']

    df = copy.deepcopy(df)
    df_outcome = df[outcome_cols]

    model_path = "autoencoder.pth"
    scalers_path = "scalers.pkl"
    model, scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat = load_models_scalers(model_path, scalers_path)
    processed_data = preprocess_data_with_scalers(df_outcome, scaler_ostm, scaler_efstm, one_hot_encoder_efsstat, one_hot_encoder_rfsstat)
    encoded = encode_data(model, processed_data)
    df['outcome'] = encoded
    df = df.drop(outcome_cols, axis=1)
    return df


def decode_synthetic_data(experiment_name: str, df_synth: DataFrame):
    df_synth = copy.deepcopy(df_synth)
    cols = ['CR1', 'OSTM', 'OSSTAT', 'EFSTM', 'EFSSTAT_0', 'RFSSTAT_0', 'RFSSTAT_1']

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                   'data')
    model_path = os.path.join(exp_folder_path, "autoencoder.pth")
    scalers_path = os.path.join(exp_folder_path, "scalers.pkl")

    model, scaler_ostm, scaler_efstm, _, _ = load_models_scalers(model_path, scalers_path)
    df_outcome = decode_data(model, df_synth[['outcome']], DataFrame(columns=cols).columns, scaler_ostm=scaler_ostm, scaler_efstm=scaler_efstm)
    for col in df_outcome.columns:
        df_synth[col] = df_outcome[col]

    df_synth.drop(['outcome'], axis=1, inplace=True)

    return df_synth


if __name__ == '__main__':
    optimize_autoencoder(100)
