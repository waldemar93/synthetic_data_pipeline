import json
import os
from dataclasses import dataclass, MISSING
from typing import Literal, List
import hydra
from omegaconf import OmegaConf
from src.base import PROJECT_DIR
from src.helpers import load_experiment_data, load_columns_json_for_experiment
from src.models import Ctgan, Tvae, CtabGanPlus, Tab_ddpm, available_models, available_models_type, SynthCityModel


@dataclass
class GenTrainingConfig:
    """ Configuration for generative model training. """
    experiment_name: str = MISSING
    train_models: List[available_models_type]
    train_configs: List[str]
    train_seeds: List[int]
    filenames: List[str]


@dataclass
class MainGenTrainingConfig:
    """Main configuration for generative model training class containing a GenTrainingConfig instance."""
    gen_training: GenTrainingConfig = MISSING


def _validate_gen_training_config(cfg: GenTrainingConfig):
    """
    Validate the generative training configuration.

    :param cfg: The configuration object to be validated.
    """
    if len(cfg.train_models) != len(cfg.filenames):
        raise AttributeError(f'The number of filenames needs to match the number of models to train, but '
                             f'{len(cfg.train_models)} != {len(cfg.filenames)}')

    if len(cfg.train_models) != len(cfg.train_configs):
        raise AttributeError(f'The number of train_configs needs to match the number of models to train, but '
                             f'{len(cfg.train_models)} != {len(cfg.train_configs)}')

    for model in cfg.train_models:
        if model.lower() not in available_models:
            raise AttributeError(f'Models in optimize_models need to be one of the following: '
                                 f'{", ".join(available_models)}, but {model} was provided')


def _train_model(experiment_name: str, model_name: str, config_name: str, train_seed: int, save_filename: str):
    config_name = config_name if config_name.endswith('.json') else config_name + '.json'
    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name)
    _, train_df, _ = load_experiment_data(experiment_name)
    """
    Train a specific generative model using the given configuration.

    :param experiment_name: Name of the experiment.
    :param model_name: Name of the generative model to train.
    :param config_name: Configuration name (or path) for the model training. The value "default" will train the model 
    with default hyperparameters.
    :param train_seed: Random seed value for training.
    :param save_filename: Filename to save the trained model.
    """

    if config_name != 'default.json':
        with open(os.path.join(exp_folder_path, model_name.upper(), 'config', config_name), 'r') as fh:
            config = json.load(fh)
    else:
        config = {}

    if model_name.upper() == 'CTGAN':
        if len(config) > 0:
            categorical_cols = config['categorical_cols']
            del config['categorical_cols']
        else:
            columns_json = load_columns_json_for_experiment(experiment_name)
            categorical_cols = columns_json['boolean'] + columns_json['categorical']

        model = Ctgan(**config)
        model.train(train_df, categorical_cols=categorical_cols, random_seed=train_seed)
        model.save_model(experiment_name, save_filename)

    elif model_name.upper() == 'TVAE':
        if len(config) > 0:
            categorical_cols = config['categorical_cols']
            del config['categorical_cols']
        else:
            columns_json = load_columns_json_for_experiment(experiment_name)
            categorical_cols = columns_json['boolean'] + columns_json['categorical']

        model = Tvae(**config)
        model.train(train_df, categorical_cols=categorical_cols, random_seed=train_seed)
        model.save_model(experiment_name, save_filename)

    elif model_name.upper() == 'CTAB-GAN+':
        if len(config) > 0:
            cat_columns = config['columns_info']['cat_columns']
            int_columns = config['columns_info']['int_columns']
            general_columns = config['columns_info']['general_columns']
            log_columns = config['columns_info']['log_columns']
            mixed_columns = config['columns_info']['mixed_columns']
            non_categorical_columns = config['columns_info']['non_categorical_columns']
            problem_type = config['columns_info']['problem_type']

            del config['columns_info']
        else:
            columns_info = CtabGanPlus.load_columns_info(experiment_name)
            cat_columns = columns_info['cat_columns']
            int_columns = columns_info['int_columns']
            general_columns = columns_info['general_columns']
            log_columns = columns_info['log_columns']
            mixed_columns = columns_info['mixed_columns']
            non_categorical_columns = columns_info['non_categorical_columns']
            problem_type = columns_info['problem_type']

        model = CtabGanPlus(**config)
        model.train(train_df, cat_columns=cat_columns, int_columns=int_columns, general_columns=general_columns,
                    log_columns=log_columns, mixed_columns=mixed_columns,
                    non_categorical_columns=non_categorical_columns, problem_type=problem_type, random_seed=train_seed)
        model.save_model(experiment_name, save_filename)

    elif model_name == 'TAB_DDPM':
        model = Tab_ddpm(**config, df=train_df)
        model.train(train_df, random_seed=train_seed)
        model.save_model(experiment_name, save_filename)

    else:
        # synthcity models
        model = SynthCityModel(model_name.lower(), **config)
        dl_config = SynthCityModel.load_dataloader_info(experiment_name, model_name.lower())
        model.train(train_df, random_seed=train_seed, **dl_config)
        model.save_model(experiment_name, save_filename)


@hydra.main(config_path="../config", config_name="4_gen_training", version_base="1.1")
def train_synthesizer(cfg: MainGenTrainingConfig):
    """
    Main function to train generative models using the given configuration.

    :param cfg: The main configuration object for generative training.
    """

    cfg: GenTrainingConfig = cfg.gen_training
    print(OmegaConf.to_yaml(cfg))

    # validate config
    _validate_gen_training_config(cfg)

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', cfg.experiment_name)

    for i, model in enumerate(cfg.train_models):
        # check if output file exists -> continue with next
        filename = cfg.filenames[i] if not cfg.filenames[i].endswith('.pkl') else \
            cfg.filenames[i].rsplit('.pkl', maxsplit=1)[0]
        model_config = cfg.train_configs[i]

        for seed in cfg.train_seeds:
            if os.path.exists(os.path.join(exp_folder_path, model, 'models', f'{filename}_{seed}.pkl')):
                print(f'File with name: {filename} already exists, skip training')
                continue

            _train_model(experiment_name=cfg.experiment_name, model_name=model, config_name=model_config,
                         train_seed=seed, save_filename=f'{filename}_{seed}')


if __name__ == '__main__':
    train_synthesizer()
