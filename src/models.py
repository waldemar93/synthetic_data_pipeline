# CTGAN, TVAE, CTAB_GAN_Plus, tab-ddpm, be-great (transformers)
import json
import os.path
import random
import time
from abc import abstractmethod
from collections import defaultdict
from typing import List, Literal, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import zero
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from src.base import PROJECT_DIR
from src.helpers import get_categorical_columns_from_df, get_integer_columns_from_df
from src_others.CTAB_GAN_Plus.pipeline.data_preparation import DataPrep
from src_others.CTAB_GAN_Plus.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from src_others.CTGAN.ctgan import CTGAN, TVAE
import pickle
from scipy.spatial import distance
from heapq import nsmallest
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader, GenericDataLoader
from synthcity.utils.serialization import load_from_file, save_to_file
from pathlib import Path
import os
from src_others.tab_ddpm import GaussianMultinomialDiffusion, lib
from src_others.tab_ddpm.sample import simple_sample
from src_others.tab_ddpm.train import Trainer
from src_others.tab_ddpm.utils_train import get_model
from synthcity.plugins.survival_analysis.plugin_survae import SurVAEPlugin
from synthcity.plugins.survival_analysis.plugin_survival_gan import SurvivalGANPlugin
from synthcity.plugins.survival_analysis.plugin_survival_ctgan import SurvivalCTGANPlugin
from synthcity.plugins.survival_analysis.plugin_survival_nflow import SurvivalNFlowPlugin
from synthcity.plugins.generic.plugin_rtvae import RTVAEPlugin
from synthcity.plugins.generic.plugin_nflow import NormalizingFlowsPlugin
from synthcity.plugins.generic.plugin_bayesian_network import BayesianNetworkPlugin
from synthcity.plugins.generic.plugin_tvae import TVAEPlugin
from synthcity.plugins.generic.plugin_ctgan import CTGANPlugin
from synthcity.plugins.privacy.plugin_adsgan import AdsGANPlugin


available_models = ['nflow', 'tvae', 'rtvae', 'bayesian_network', 'survae', 'survival_ctgan', 'ctgan', 'survival_nflow',
                    'survival_gan', 'ctab-gan+', 'tab_ddpm']
synthcity_models = ['nflow', 'rtvae', 'bayesian_network', 'survae', 'survival_ctgan', 'survival_nflow', 'survival_gan']
available_models_type = Literal['nflow', 'tvae', 'rtvae', 'bayesian_network', 'survae', 'survival_ctgan', 'ctgan',
                                'survival_nflow', 'survival_gan', 'ctab-gan+', 'tab_ddpm']
available_synthcity_models_type = Literal['nflow', 'rtvae', 'bayesian_network', 'survae', 'survival_ctgan',
                                          'survival_nflow', 'survival_gan']

class GenerativeModel:
    @staticmethod
    @abstractmethod
    def model_dir() -> str:
        """Abstract method to return the model directory name."""
        pass

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize the Generative Model.

        :param kwargs: Keyword arguments for initialization.
        """
        pass

    @abstractmethod
    def train(self, df: DataFrame, random_seed: int = 42) -> None:
        """Abstract method to train the generative model.

        :param df: Input data frame for training.
        :param random_seed: Random seed for reproducibility.
        """
        pass

    @abstractmethod
    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """Abstract method to generate synthetic samples.

        :param sample_size: Number of samples to generate.
        :param random_seed: Random seed for reproducibility.
        """
        pass

    @abstractmethod
    def save_model(self, experiment_name: str, filename: str) -> None:
        """Abstract method to save the trained model.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file to save the model.
        """
        pass

    @abstractmethod
    def save_synthetic_data(self, df_synthetic: DataFrame, filename: str) -> None:
        """Abstract method to save the generated synthetic data.

        :param df_synthetic: Data frame containing synthetic data.
        :param filename: Name of the file to save the data.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_model(experiment_name: str, filename: str):
        """Abstract method to load the trained model.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file containing the model.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_synth_data(filename: str) -> DataFrame:
        """Abstract method to load synthetic data.

        :param filename: Name of the file containing synthetic data.
        """
        pass


class Ctgan(GenerativeModel):
    @staticmethod
    def model_dir():
        """Returns the directory name for the CTGAN model."""
        return 'CTGAN'

    @staticmethod
    def load_model(experiment_name: str, filename: str) -> GenerativeModel:
        """
        Loads a pre-trained CTGAN model.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file from which to load the model.
        :return: An instance of the loaded Ctgan.
        """

        if not filename.endswith('.pkl'):
            filename += '.pkl'
        loaded_ctgan = Ctgan()
        loaded_ctgan.model = CTGAN.load(
            os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                         Ctgan.model_dir(), 'models', filename))
        return loaded_ctgan

    def __init__(self, **kwargs):
        """
        Initializes the CTGAN instance.

        :param kwargs: Arguments to pass to the CTGAN instance.
        """

        self.training_seed = None
        self.sampling_seed = None
        self.model = CTGAN(**kwargs)
        self.params = kwargs

    def train(self, df: DataFrame, categorical_cols: List[str] = None, random_seed: int = 42) -> None:
        """
        Train the CTGAN model on the given data.

        :param df: DataFrame on which the model is trained.
        :param categorical_cols: List of columns in df which are categorical. Defaults to None.
        :param random_seed: Seed for reproducibility. Defaults to 42.
        """

        self.training_seed = random_seed
        _set_random_seed(random_seed)
        if categorical_cols is None:
            categorical_cols = get_categorical_columns_from_df(df)
        self.model.fit(df, discrete_columns=categorical_cols)

    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """
        Generate synthetic samples using the trained CTGAN model.

        :param sample_size: Number of samples to generate.
        :param random_seed: Seed for reproducibility. Defaults to 42.
        :return: A DataFrame containing the generated samples.
        """

        self.sampling_seed = random_seed
        _set_random_seed(random_seed)
        return self.model.sample(n=sample_size)

    def save_model(self, experiment_name: str, filename: str) -> None:
        """
        Save the trained CTGAN model to a file.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file to which to save the model.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        self.model.save(
            os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                         self.model_dir(), 'models', filename))


class CtabGanPlus(GenerativeModel):
    @staticmethod
    def load_columns_info(experiment_name: str):
        """
        Load columns information for the specified experiment.

        :param experiment_name: Name of the experiment from which to load column info.
        :return: JSON information about the columns.
        """

        path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                            CtabGanPlus.model_dir(), 'columns_ctabgan.json')
        with open(path, 'r') as fh:
            info_json = json.load(fh)
            return info_json

    @staticmethod
    def model_dir() -> str:
        """Returns the directory name for the CTAB-GAN+ model."""
        return 'CTAB-GAN+'

    @staticmethod
    def load_model(experiment_name: str, filename: str) -> GenerativeModel:
        """
        Loads a saved CTAB-GAN+ model for the specified experiment and filename.

        :param experiment_name: Name of the experiment from which to load the model.
        :param filename: Filename of the saved model.
        :return: Loaded CtabGanPlus model.
        """

        if not filename.endswith('.pkl'):
            filename += '.pkl'
        filehandler = open(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                        CtabGanPlus.model_dir(), 'models', filename), 'rb')
        return pickle.load(filehandler)

    def __init__(self, **kwargs) -> None:
        """
        Initialize a new instance of CtabGanPlus.

        :param kwargs: Additional keyword arguments to pass to the CTABGANSynthesizer model.
        """

        self.training_seed = None
        self.sampling_seed = None
        self.data_prep = None

        self.model = CTABGANSynthesizer(**kwargs)
        self.params = kwargs

    def train(self, df: DataFrame, problem_type: dict[str: str] = None, cat_columns: list[str] = None,
              int_columns: list[str] = None,
              log_columns: list[str] = None, general_columns: list[str] = None,
              mixed_columns: dict[str: float] = None,
              non_categorical_columns: list[str] = None, random_seed: int = 42) -> None:
        """
        Train the model using the provided dataframe.

        :param df: Input dataframe for training.
        :param problem_type: Type of problem, either regression or classification with the column to predict.
        :param cat_columns: List of categorical columns.
        :param int_columns: List of integer columns.
        :param log_columns: List of logarithmically transformed columns.
        :param general_columns: List of general columns.
        :param mixed_columns: Dictionary of mixed columns with their respective mixing rates.
        :param non_categorical_columns: List of non-categorical columns.
        :param random_seed: Random seed for reproducibility.
        """

        self.training_seed = random_seed
        _set_random_seed(self.training_seed)

        if cat_columns is None:
            cat_columns = []
        if int_columns is None:
            int_columns = []
        if log_columns is None:
            log_columns = []
        else:
            int_columns = [col for col in int_columns if col not in log_columns]
        if mixed_columns is None:
            mixed_columns = {}
        if general_columns is None:
            general_columns = []
        if non_categorical_columns is None:
            non_categorical_columns = []

        if problem_type is None or len(problem_type) == 0:
            problem_type = {None: None}

        self.data_prep = DataPrep(df, cat_columns, log_columns, mixed_columns, general_columns,
                                  non_categorical_columns, int_columns, problem_type)

        self.model.fit(train_data=self.data_prep.df, categorical=self.data_prep.column_types["categorical"],
                       mixed=self.data_prep.column_types["mixed"],
                       general=self.data_prep.column_types["general"],
                       non_categorical=self.data_prep.column_types["non_categorical"], type=problem_type)

    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """
        Generate synthetic samples using the trained model.

        :param sample_size: Number of samples to generate.
        :param random_seed: Random seed for reproducibility.
        :return: Generated samples as a dataframe.
        """

        self.sampling_seed = random_seed
        synth = self.model.sample(sample_size)
        return self.data_prep.inverse_prep(synth)

    def save_model(self, experiment_name: str, filename: str) -> None:
        """
        Save the trained model to a specified location.

        :param experiment_name: Name of the experiment under which the model is saved.
        :param filename: Desired filename for the saved model.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        self.model.generator = self.model.generator.to('cpu')
        self.model.device = 'cpu'
        file = open(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                 self.model_dir(), 'models', filename), 'wb')
        pickle.dump(self, file)


class Tvae(GenerativeModel):
    @staticmethod
    def model_dir():
        """Returns the directory name for the TVAE model."""
        return 'TVAE'

    @staticmethod
    def load_model(experiment_name: str, filename: str) -> GenerativeModel:
        """
        Loads a saved TVAE model for the specified experiment and filename.

        :param experiment_name: Name of the experiment from which to load the model.
        :param filename: Filename of the saved model.
        :return: Loaded TVAE model.
        """

        if not filename.endswith('.pkl'):
            filename += '.pkl'
        loaded_tvae = Tvae()
        loaded_tvae.model = TVAE.load(
            os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                         Tvae.model_dir(), 'models', filename))
        return loaded_tvae

    def __init__(self, **kwargs):
        """
        Initialize a new instance of TVAE.

        :param kwargs: Additional keyword arguments to pass to the TVAE model.
        """

        self.training_seed = None
        self.sampling_seed = None
        self.model = TVAE(**kwargs)
        self.params = kwargs

    def train(self, df: DataFrame, categorical_cols: List[str] = None, random_seed: int = 42) -> None:
        """
        Train the model using the provided dataframe.

        :param df: Input dataframe for training.
        :param categorical_cols: List of categorical columns.
        :param random_seed: Random seed for reproducibility.
        """

        self.training_seed = random_seed
        _set_random_seed(random_seed)
        if categorical_cols is None:
            categorical_cols = get_categorical_columns_from_df(df)
        self.model.fit(df, discrete_columns=categorical_cols)

    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """
        Generate synthetic samples using the trained model.

        :param sample_size: Number of samples to generate.
        :param random_seed: Random seed for reproducibility.
        :return: Generated samples as a dataframe.
        """

        self.sampling_seed = random_seed
        _set_random_seed(random_seed)
        return self.model.sample(samples=sample_size)

    def save_model(self, experiment_name: str, filename: str) -> None:
        """
        Save the trained model to a specified location.

        :param experiment_name: Name of the experiment under which the model is saved.
        :param filename: Desired filename for the saved model.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        self.model.save(
            os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                         self.model_dir(), 'models', filename))


class SynthCityModel(GenerativeModel):
    class CustomUnpickler(pickle.Unpickler):
        """ Custom unpickler to handle special classes during deserialization. """
        def find_class(self, module, name):
            """
            Finds the class for deserialization based on module and name.

            :param module: Module name.
            :param name: Class name.
            """

            if module == 'pathlib' and name == 'WindowsPath':
                return Path
            return super().find_class(module, name)

    @staticmethod
    def __load_from_file(file):
        """
        Load an object from a file using custom unpickler.

        :param file: Path to the file to load from.
        """

        with open(file, 'rb') as f:
            return SynthCityModel.CustomUnpickler(f).load()

    @staticmethod
    def model_dir() -> str:
        pass

    @staticmethod
    def load_model(experiment_name: str, filename: str, model_type: available_synthcity_models_type) -> GenerativeModel:
        """
        Load a saved generative model based on the experiment, filename, and model type.

        :param experiment_name: Name of the experiment to load from.
        :param filename: Filename of the saved model.
        :param model_type: Type of the SynthCity model.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                            model_type.upper(), 'models', filename)
        # model = load_from_file(path)
        model = SynthCityModel.__load_from_file(path)

        gen_model = SynthCityModel(model_type)
        gen_model.model = model
        return gen_model

    def __init__(self, model_name: str, **kwargs):
        """
       Initialize a new instance of the SynthCityModel.

       :param model_name: Name of the model to initialize.
       :param kwargs: Additional keyword arguments to later pass to a synthcity model.
       """

        self.model_name = model_name
        self.model_args = kwargs
        self.training_seed = None
        self.sampling_seed = None
        self.model = None

    @staticmethod
    def load_dataloader_info(experiment_name: str, model_type: str):
        """
        Load dataloader information based on the experiment and model type.

        :param experiment_name: Name of the experiment to load from.
        :param model_type: Type of the SynthCity model.
        """

        path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                            'synthcity_dl.json')
        with open(path, 'r') as fh:
            info_json = json.load(fh)

        surv_plugins = Plugins(categories=["survival_analysis"]).list()

        if model_type.lower() in surv_plugins and 'survival_target_col' in info_json and 'survival_time_to_event_col' in info_json:
            return {'survival_target_col': info_json['survival_target_col'],
                    'survival_time_to_event_col': info_json['survival_time_to_event_col']}
        else:
            return {'target_col': info_json['target_col']}

    def train(self, df: DataFrame, random_seed: int = 42, survival_target_col=None, survival_time_to_event_col=None,
              target_col=None) -> None:
        """
        Train the generative model using the provided data.

        :param df: Input dataframe for training.
        :param random_seed: Seed for reproducibility.
        :param survival_target_col: Target column for survival analysis data.
        :param survival_time_to_event_col: Time to event column for survival analysis data.
        :param target_col: Target column for generic data.
        """
        
        self.training_seed = random_seed
        _set_random_seed(random_seed)

        self.model = get_synthcity_plugin(self.model_name, random_seed=random_seed, **self.model_args)
        # self.model = Plugins().get(self.model_name, random_state=random_seed, **self.model_args)

        if survival_target_col is not None and survival_time_to_event_col is not None:
            loader = SurvivalAnalysisDataLoader(df, target_column=survival_target_col,
                                                time_to_event_column=survival_time_to_event_col,
                                                random_state=random_seed,
                                                train_size=1.0)
        else:
            loader = GenericDataLoader(df, target_column=target_col, random_state=random_seed, train_size=1.0)

        self.model.fit(loader)

    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """
        Generate samples using the trained generative model.

        :param sample_size: Number of samples to generate.
        :param random_seed: Seed for reproducibility.
        :return: Generated samples as a dataframe.
        """

        self.sampling_seed = sample_size
        return self.model.generate(count=sample_size, random_state=random_seed).dataframe()

    def save_model(self, experiment_name: str, filename: str) -> None:
        """
        Save the trained generative model to the specified location.

        :param experiment_name: Name of the experiment to save under.
        :param filename: Desired filename for saving the model.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        save_to_file(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                  self.model_name.upper(), 'models', filename), self.model)


class Tab_ddpm(GenerativeModel):
    @staticmethod
    def model_dir() -> str:
        """Returns the directory name for the TAB_DDPM model."""
        return 'TAB_DDPM'

    @staticmethod
    def load_model(experiment_name: str, filename: str) -> GenerativeModel:
        """
        Load the model from a file.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file where the model is saved.
        """

        if not filename.endswith('.pkl'):
            filename += '.pkl'
        filehandler = open(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                        Tab_ddpm.model_dir(), 'models', filename), 'rb')
        return pickle.load(filehandler)

    def __init__(self, model_type: Literal['resnet', 'mlp'], model_params: dict,
                 gaussian_loss_type: Literal['kl', 'mse'], df: DataFrame, T: dict, info_json: dict, pred_column: str,
                 num_timesteps: int, batch_size: int, lr: float, weight_decay: float, epochs: int) -> None:
        """
        Initialize the Tab_ddpm model.

        :param model_type: Type of the underlying model - 'resnet' or 'mlp'.
        :param model_params: Parameters for the underlying model.
        :param gaussian_loss_type: Type of gaussian loss - 'kl' or 'mse'.
        :param df: Dataframe used for training.
        :param T: Transformations dictionary.
        :param info_json: Information json.
        :param pred_column: Prediction column name.
        :param num_timesteps: Number of timesteps.
        :param batch_size: Batch size for training.
        :param lr: Learning rate.
        :param weight_decay: Weight decay value.
        :param epochs: Number of epochs.
        """

        self.training_seed = None
        self.sampling_seed = None

        self.train_batch_size = batch_size
        self.train_lr = lr
        self.train_weight_decay = weight_decay
        self.train_epochs = epochs

        self.train_dataset, self.columns_order_list, self.columns_cat_ind, self.columns_num_ind = \
            _transform_df_to_dataset_for_tab_ddpm(df, T, info_json, pred_column)

        num_classes = np.array(self.train_dataset.get_category_sizes('train'))
        if len(num_classes) == 0:
            num_classes = np.array([0])
        num_numerical_features = self.train_dataset.X_num['train'].shape[1] if self.train_dataset.X_num is not None \
            else 0
        d_in = np.sum(num_classes) + num_numerical_features
        model_params['d_in'] = d_in
        if pred_column in info_json['boolean'] or pred_column in info_json['categorical']:
            model_params['num_classes'] = len(df[pred_column].unique())
        else:
            # regression
            model_params['num_classes'] = 0

        self.model = get_model(model_type, model_params, None, None)
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_params = model_params

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler='cosine',
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    def train(self, df: DataFrame, random_seed: int = 42) -> None:
        """
        Train the model on given data.

        :param df: Dataframe from training, not used.
        :param random_seed: Seed for random processes.
        """

        # df not used, just for compatibility reasons
        self.training_seed = random_seed
        _set_random_seed(self.training_seed)
        zero.improve_reproducibility(random_seed)

        self.diffusion.train()
        train_loader = lib.prepare_fast_dataloader(self.train_dataset, split='train', batch_size=self.train_batch_size)

        trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=self.train_lr,
            weight_decay=self.train_weight_decay,
            steps=self.train_epochs,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            print_every=5000
        )
        trainer.run_loop()

    def sample(self, sample_size: int, random_seed: int = 42) -> DataFrame:
        """
        Generate synthetic data samples.

        :param sample_size: Number of samples to generate.
        :param random_seed: Seed for random processes.
        :return: Generated samples as a dataframe.
        """

        self.sampling_seed = random_seed
        _set_random_seed(random_seed)

        X_num, X_cat, y_gen = simple_sample(self.diffusion, self.train_dataset, num_samples=sample_size,
                                            model_params=self.model_params,
                                            T_dict={'cat_encoding': None, 'cat_min_frequency': None,
                                                    'cat_nan_policy': None,
                                                    'normalization': 'quantile', 'num_nan_policy': None, 'seed': 0,
                                                    'y_policy': 'default'},
                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                            seed=random_seed)

        # in case encountered NANs, errors, just for hyperparam tuning, returning
        if X_num is None and X_cat is None and y_gen is None:
            return pd.DataFrame()

        # reconstruct the data back to original shape
        cat_column_names = [self.columns_order_list[i] for i in self.columns_cat_ind]
        num_column_names = [self.columns_order_list[i] for i in self.columns_num_ind]
        y_name = [col for col in self.columns_order_list if col not in cat_column_names and col not in num_column_names]

        df_synth_cat = pd.DataFrame(X_cat, columns=cat_column_names)
        df_synth_num = pd.DataFrame(X_num, columns=num_column_names)
        df_synth_y = pd.DataFrame(y_gen, columns=y_name)

        df_synth = pd.concat([df_synth_cat, df_synth_num, df_synth_y], axis=1)
        df_synth = df_synth.reindex(columns=self.columns_order_list)
        return df_synth

    def save_model(self, experiment_name: str, filename: str) -> None:
        """
        Save the model to a file.

        :param experiment_name: Name of the experiment.
        :param filename: Name of the file where the model will be saved.
        """

        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        file = open(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                                 self.model_dir(), 'models', filename), 'wb')
        pickle.dump(self, file)


def _transform_df_to_dataset_for_tab_ddpm(df_train: DataFrame, T: dict, info_json: dict, y: str) \
        -> Tuple[lib.Dataset, List[str], List[int], List[int]]:
    """
    Transform a dataframe into a dataset format compatible with Tab_ddpm.

    :param df_train: Training dataframe.
    :param T: Transformations dictionary.
    :param info_json: Information json.
    :param y: Target column name.
    """

    default_T = {'cat_encoding': None, 'cat_min_frequency': None, 'cat_nan_policy': None, 'normalization': 'quantile',
                 'num_nan_policy': None, 'seed': 0, 'y_policy': 'default'}
    T.update({key: value for key, value in default_T.items() if key not in T})

    n_classes = None
    if y in info_json['categorical']:
        task_type = lib.TaskType.MULTICLASS
        n_classes = len(df_train[y].unique())
    elif y in info_json['boolean']:
        task_type = lib.TaskType.BINCLASS
    else:
        task_type = lib.TaskType.REGRESSION

    param_order_list = list(df_train.columns)
    cat_ind = []
    num_ind = []

    for i, col in enumerate(param_order_list):
        if col == y:
            continue
        if col in info_json['categorical'] or col in info_json['boolean']:
            cat_ind.append(i)
        else:
            num_ind.append(i)

    # convert to np.array
    x_num = df_train.iloc[:, num_ind].to_numpy()
    x_cat = df_train.iloc[:, cat_ind].to_numpy()
    y = df_train[[y]].to_numpy()

    dataset = lib.Dataset(X_num={'train': x_num}, X_cat={'train': x_cat}, y={'train': y}, y_info={},
                          task_type=task_type,
                          n_classes=n_classes)

    T = lib.Transformations(**T)
    dataset = lib.transform_dataset(dataset, T, cache_dir=None)
    # c = dataset.get_category_sizes('train')
    return dataset, param_order_list, cat_ind, num_ind


def _set_random_seed(seed: int) -> None:
    """
    Set seeds for random processes.

    :param seed: Seed value.
    """

    # somehow, getting a bit different results for CTGAN...
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_gen_model(experiment_name: str, model_type: available_models_type, filename: str) -> \
        GenerativeModel:
    """
   Load a generative model based on its type.

   :param experiment_name: Name of the experiment.
   :param model_type: Type of the generative model.
   :param filename: Name of the file where the model is saved.
   """

    if model_type.lower() == 'tvae':
        model = Tvae.load_model(experiment_name, filename)
    elif model_type.lower() == 'ctgan':
        model = Ctgan.load_model(experiment_name, filename)
    elif model_type.lower() == 'ctab-gan+':
        model = CtabGanPlus.load_model(experiment_name, filename)
    elif model_type.lower() == 'tab_ddpm':
        model = Tab_ddpm.load_model(experiment_name, filename)
    else:
        model = SynthCityModel.load_model(experiment_name, filename, model_type.lower())

    return model


def get_synthcity_plugin(plugin_name: str, random_seed: int, **model_args):
    """
    Retrieve an instance of the specified SynthCity plugin.

    :param plugin_name: The name of the desired plugin.
    :param random_seed: Seed for randomness to ensure reproducibility.
    :param model_args: Additional keyword arguments for initializing the plugin.
    :return: An instance of the specified SynthCity plugin.
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if plugin_name.lower() == 'survae':
        return SurVAEPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'survival_gan':
        return SurvivalGANPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'survival_ctgan':
        return SurvivalCTGANPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'survival_nflow':
        return SurvivalNFlowPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'nflow':
        return NormalizingFlowsPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'rtvae':
        return RTVAEPlugin(random_state=random_seed, device=device, **model_args)
    elif plugin_name.lower() == 'bayesian_network':
        return BayesianNetworkPlugin(random_state=random_seed, device=device, **model_args)
    else:
        raise AttributeError(f'Unknown plugin name "{plugin_name}" provided.')


def get_synthcity_plugin_hyperparams(plugin_name: str):
    """
    Retrieve hyperparameter space for the specified SynthCity plugin.

    :param plugin_name: The name of the desired plugin.
    :return: Hyperparameter space of the specified SynthCity plugin.
    """

    if plugin_name.lower() == 'survae':
        return TVAEPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'survival_gan':
        return AdsGANPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'survival_ctgan':
        return CTGANPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'survival_nflow':
        return NormalizingFlowsPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'nflow':
        return NormalizingFlowsPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'rtvae':
        return RTVAEPlugin.hyperparameter_space()
    elif plugin_name.lower() == 'bayesian_network':
        return BayesianNetworkPlugin.hyperparameter_space()
    else:
        raise AttributeError(f'Unknown plugin name "{plugin_name}" provided.')


