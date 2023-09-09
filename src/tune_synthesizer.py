import json
import os
import time
from dataclasses import dataclass, MISSING
from datetime import timedelta
from typing import List, Dict, Optional, Literal
import hydra
import optuna
from catboost import _catboost
from omegaconf import OmegaConf
from pandas import DataFrame
from sklearn.model_selection import KFold
from synthcity.plugins import Plugins
from synthcity.utils.optuna_sample import suggest_all
from src.base import PROJECT_DIR
from src.helpers import load_experiment_data, load_columns_json_for_experiment, load_catboost_params
from src.models import Ctgan, Tvae, CtabGanPlus, available_models, SynthCityModel, get_synthcity_plugin, Tab_ddpm, \
    get_synthcity_plugin_hyperparams, available_models_type
from src.evaluation_functions import basic_statistical_measure_num, discriminator_measure_rf, k_means_score, \
    log_transformed_correlation_score, regularized_support_coverage, ml_efficiency_cat, survival, survival_sightedness, \
    survival_auc_opt, survival_auc_abs_opt


@dataclass
class GenOptimizationConfig:
    """ Configuration for generative model optimization. """
    experiment_name: str = MISSING
    optimize_models: List[Literal['CTGAN', 'CTAB-GAN+', 'TVAE', 'TAB_DDPM']] = MISSING
    optimization_sampler: Literal['RandomSampler', 'GridSampler', 'TPESampler', 'CmaEsSampler', 'NSGAIISampler',
                                  'QMCSampler'] = MISSING
    optimization_trials: int = MISSING
    optimization_seed: int = MISSING
    optimization_metrics: List[Literal["basic_statistical_measure_num", "log_transformed_correlation_score",
                                       "regularized_support_coverage", "discriminator_measure_rf", "k_means_score",
                                       "ml_efficiency_cat", "survival_auc_opt", "survival_auc_abs_opt",
                                       "survival_sightedness", "survival"]] = MISSING
    optimization_metrics_weights: List[float] = MISSING
    predict_column: str = MISSING
    # maximum of ten metrics allowed, for each metrics params can be given here, necessary for ml_efficiency
    # params need to be the same as the functions
    optimization_metric1_params: Optional[Dict] = MISSING
    optimization_metric2_params: Optional[Dict] = MISSING
    optimization_metric3_params: Optional[Dict] = MISSING
    optimization_metric4_params: Optional[Dict] = MISSING
    optimization_metric5_params: Optional[Dict] = MISSING
    optimization_metric6_params: Optional[Dict] = MISSING
    optimization_metric7_params: Optional[Dict] = MISSING
    optimization_metric8_params: Optional[Dict] = MISSING
    optimization_metric9_params: Optional[Dict] = MISSING
    optimization_metric10_params: Optional[Dict] = MISSING
    filenames: List[str] = MISSING


@dataclass
class MainGenOptimizationConfig:
    """Main configuration for generative model optimization class containing a GenOptimizationConfig instance."""
    gen_optimization: GenOptimizationConfig = MISSING


def _validate_gen_optimization_config(cfg: GenOptimizationConfig):
    """
    Validate the generative model optimization configuration.

    :param cfg: Instance of GenOptimizationConfig to validate.
    :raises AttributeError: When configuration is invalid.
    """
    pos_metrics = ["basic_statistical_measure_num", "log_transformed_correlation_score", "regularized_support_coverage",
                   "discriminator_measure_rf", "k_means_score", "ml_efficiency_cat", "survival_auc_opt",
                   "survival_auc_abs_opt", "survival_sightedness", "survival"]

    if len(cfg.optimization_metrics) > 10:
        raise AttributeError(f'The final score for the optimization can consist of maximum 10 metrics, '
                             f'but {len(cfg.optimization_metrics)} were provided')

    if len(cfg.optimization_metrics) != len(cfg.optimization_metrics_weights):
        raise AttributeError(f'The number of optimization_metrics needs to match the number of '
                             f'optimization_metrics_weights, but {len(cfg.optimization_metrics)} != '
                             f'{len(cfg.optimization_metrics_weights)}')

    if len(cfg.optimize_models) != len(cfg.filenames):
        raise AttributeError(f'The number of filenames needs to match the number of models to optimize, but '
                             f'{len(cfg.optimize_models)} != {len(cfg.filenames)}')

    for model in cfg.optimize_models:
        if model.lower() not in available_models:
            raise AttributeError(f'Models in optimize_models need to be one of the following: '
                                 f'{", ".join(available_models)}, but {model} was provided')
        if model.lower() == 'tab_ddpm' and 'predict_column' not in cfg:
            raise AttributeError(f'By using TAB_DDPM the column that the model needs to predict while training needs to '
                                 f'be provided (similar to CTAB-GAN+) with the parameter "predict_column".')

    for i, metric in enumerate(cfg.optimization_metrics):
        if metric not in pos_metrics:
            raise AttributeError(f'Metrics in optimization_metrics needs to be one of the following: '
                                 f'{", ".join(pos_metrics)}, but {metric} was provided')

        if metric == 'ml_efficiency_cat':
            if f'optimization_metric{i + 1}_params' not in cfg:
                raise AttributeError(f'For the metric ml_efficiency the following parameters need to be provided: '
                                     f'outcome_col, metric, filename_params_json, relative, '
                                     f'Optional (create_column_lambda) (optimization_metric{i + 1}_params)')
            params = cfg[f'optimization_metric{i + 1}_params']
            if 'predict_col' not in params or 'metric' not in params or 'filename_params_json' not in params \
                    or 'relative' not in params:
                raise AttributeError(f'For the metric ml_efficiency the following parameters need to be provided: '
                                     f'outcome_col, metric, filename_params_json, relative, '
                                     f'Optional (create_column_lambda) (optimization_metric{i + 1}_params)')

        if metric.startswith('survival'):
            if f'optimization_metric{i + 1}_params' not in cfg:
                raise AttributeError(f'For the metric {metric} the following parameters need to be provided: '
                                     f'survival_target_col, survival_time_to_event_col '
                                     f'(optimization_metric{i + 1}_params)')
            params = cfg[f'optimization_metric{i + 1}_params']
            if 'survival_target_col' not in params or 'survival_time_to_event_col' not in params:
                raise AttributeError(f'For the metric {metric} the following parameters need to be provided: '
                                     f'survival_target_col, survival_time_to_event_col '
                                     f'(optimization_metric{i + 1}_params)')


def _call_eval_func(experiment_name: str, eval_func_name: str, params: Dict, orig_df_train: DataFrame,
                    synth_df_train: DataFrame, orig_df_test: Optional[DataFrame]) -> float:
    """
    Call the evaluation function.

    :param experiment_name: Name of the experiment.
    :param eval_func_name: Name of the evaluation function.
    :param params: Parameters required by the evaluation function.
    :param orig_df_train: Original training dataframe.
    :param synth_df_train: Synthetic dataframe with the same dataset size as the training dataframe.
    :param orig_df_test: Optional: original testing dataframe.
    :return: Evaluation score as a float.
    """
    if eval_func_name == 'basic_statistical_measure_num':
        if 'num_columns' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name)
            params['num_columns'] = columns_json['integer'] + columns_json['float']
        return basic_statistical_measure_num(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'log_transformed_correlation_score':
        if 'categorical_cols' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name)
            params['categorical_cols'] = columns_json['boolean'] + columns_json['categorical']
        return log_transformed_correlation_score(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'regularized_support_coverage':
        if 'categorical_cols' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name)
            params['categorical_cols'] = columns_json['boolean'] + columns_json['categorical']
        return regularized_support_coverage(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'ml_efficiency_cat':
        params['training_params'] = load_catboost_params(experiment_name, params['filename_params_json'])
        params.pop('filename_params_json', None)
        columns_json = load_columns_json_for_experiment(experiment_name)
        params['outcome_cols'] = columns_json['outcome']
        try:
            return ml_efficiency_cat(orig_df_train, orig_df_test, synth_df_train, **params)
        except _catboost.CatBoostError as e:
            if "All features are either constant or ignored" in str(e):
                return -1
            else:
                raise  # re-throw the exception if it's not the one we want to ignore

    if eval_func_name.startswith('survival'):
        time_col = params['survival_time_to_event_col']
        target_col = params['survival_target_col']
        count = len(orig_df_train)
        func = eval(eval_func_name)
        return func([orig_df_train[time_col], orig_df_train[target_col]],
                    [synth_df_train[time_col], synth_df_train[target_col]],
                    count)

    func = eval(eval_func_name)
    score = func(orig_df_train, synth_df_train, **params)
    return score


def suggest_ctab_gan_plus_params(trial: optuna.trial.Trial):
    """
    Suggest parameters for the CTAB-GAN+ model.

    :param trial: Optuna trial instance.
    :return: Dictionary of suggested parameters.
    """
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    lr = trial.suggest_float('lr', 0.00002, 0.002, log=True)

    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 6, 8
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    ####

    epochs = trial.suggest_categorical('steps', [100, 300, 500, 1000, 5000])
    batch_size = 2 ** trial.suggest_int('batch_size', 7, 10)
    random_dim = 2 ** trial.suggest_int('random_dim', 4, 7)
    num_channels = 2 ** trial.suggest_int('num_channels', 4, 6)

    params = {
        "lr": lr,
        "epochs": epochs,
        "class_dim": d_layers,
        "batch_size": batch_size,
        "random_dim": random_dim,
        "num_channels": num_channels
    }
    return params


def suggest_tvae_params(trial: optuna.trial.Trial):
    """
    Suggest parameters for the TVAE model.

    :param trial: Optuna trial instance.
    :return: Dictionary of suggested parameters.
    """
    def suggest_dim_int(name, min, max):
        t = trial.suggest_int(name, min, max)
        return t  # 2 ** t

    lr = trial.suggest_float('lr', 0.00002, 0.002, log=True)
    # construct model -> changed to hourglass shape
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 6, 9
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)

    layers_dim = []
    for i in range(n_layers):
        prev_layer_dim = None
        if len(layers_dim) > 0:
            prev_layer_dim = layers_dim[-1]
        if prev_layer_dim is None:
            layers_dim.append(suggest_dim_int('d_layer_1', d_min, d_max))
        else:
            layers_dim.append(suggest_dim_int(f'd_layer{i + 1}', d_min, prev_layer_dim))

    d_layers = [2 ** dim for dim in layers_dim]
    d_layers_rev = d_layers[::-1]
    ####

    epochs = trial.suggest_categorical('epochs', [300, 500, 1000, 5000, 10000])
    batch_size = trial.suggest_categorical('batch_size', [20, 50, 100, 200, 500, 1000])

    # num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -2, 3)))
    embedding_dim = 2 ** trial.suggest_int('embedding_dim', 4, 8)
    loss_factor = trial.suggest_float('loss_factor', 0.001, 10, log=True)

    params = {
        "epochs": epochs,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "loss_factor": loss_factor,
        "compress_dims": d_layers,
        "decompress_dims": d_layers_rev,
        "lr": lr
    }

    trial.set_user_attr("params", params)
    return params


def suggest_ctgan_params(trial: optuna.trial.Trial):
    """
    Suggest parameters for the CTGAN model.

    :param trial: Optuna trial instance.
    :return: Dictionary of suggested parameters.
    """
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    lr = trial.suggest_float('lr', 0.00002, 0.002, log=True)
    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 6, 9
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    ####

    epochs = trial.suggest_categorical('epochs', [100, 300, 500, 1000, 5000])
    batch_size = trial.suggest_categorical('batch_size', [20, 50, 100, 200, 500, 1000])

    embedding_dim = 2 ** trial.suggest_int('embedding_dim', 4, 8)
    log_freq = trial.suggest_categorical('log_frequency', [True, False])

    params = {
        "epochs": epochs,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "generator_dim": d_layers,
        "discriminator_dim": d_layers,
        "log_frequency": log_freq,
        "generator_lr": lr,
        "discriminator_lr": lr
    }

    trial.set_user_attr("params", params)
    return params


def suggest_tab_ddpm_params(trial: optuna.trial.Trial):
    """
    Suggest parameters for the TAB_DDPM model.

    :param trial: Optuna trial instance.
    :return: Dictionary of suggested parameters.
    """
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )

    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    lr = trial.suggest_float('lr', 0.00001, 0.003, log=True)

    weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    steps = trial.suggest_categorical('steps', [100, 500, 1000, 5000, 20000, 30000])

    gaussian_loss_type = 'mse'
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 500, 1000])
    dim_t = trial.suggest_categorical('dim_t', [32, 64, 128])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.3, 0.5])

    params = {
        "model_type": "mlp",
        "gaussian_loss_type": gaussian_loss_type,
        "num_timesteps": num_timesteps,
        "epochs": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,

        'model_params': {'rtdl_params': {'d_layers': d_layers, 'dropout': dropout}, 'dim_t': dim_t, 'is_y_cond': True},
    }

    trial.set_user_attr("params", params)
    return params


def create_gen_objective_func(cfg: GenOptimizationConfig, model_name: available_models_type):
    """
    Create the objective function for generator model optimization based on the provided configuration.

    :param cfg: Configuration object for generator optimization.
    :param model_name: Name of the generative model.
    :return: The objective function to be optimized by Optuna.
    """
    def objective_function(trial: optuna.trial.Trial):
        # after every iteration (fold) the result needs to be not worse than 0.9 of the best result to continue the
        # training
        at_least_allowed = 0.90

        df, _, _ = load_experiment_data(cfg.experiment_name)

        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        kf = kfold.split(df)
        folds_idxs = [k for k in kf]

        training_times = []
        evaluation_times = []
        scores = []
        for i, idx in enumerate(folds_idxs):
            train_idx = idx[0]
            test_idx = idx[1]

            df_train_fold = df.loc[train_idx]
            df_test_fold = df.loc[test_idx]

            columns_json = load_columns_json_for_experiment(cfg.experiment_name)
            start_time = time.time()

            if model_name.upper() == 'CTGAN':
                params = suggest_ctgan_params(trial)
                print('params', params)
                gen_model = Ctgan(**params)
                cat_cols = columns_json['boolean'] + columns_json['categorical']
                gen_model.train(df_train_fold, random_seed=1, categorical_cols=cat_cols)
                params['categorical_cols'] = cat_cols
                trial.set_user_attr("params", params)

            elif model_name.upper() == 'TVAE':
                params = suggest_tvae_params(trial)
                print('params', params)
                gen_model = Tvae(**params)
                cat_cols = columns_json['boolean'] + columns_json['categorical']
                gen_model.train(df_train_fold, random_seed=1, categorical_cols=cat_cols)
                params['categorical_cols'] = cat_cols
                trial.set_user_attr("params", params)

            elif model_name.upper() == 'CTAB-GAN+':
                params = suggest_ctab_gan_plus_params(trial)
                print('params', params)
                gen_model = CtabGanPlus(**params)
                columns_info = CtabGanPlus.load_columns_info(cfg.experiment_name)
                params['columns_info'] = columns_info
                gen_model.train(df_train_fold, random_seed=1, **columns_info)
                trial.set_user_attr("params", params)
            elif model_name.upper() == 'TAB_DDPM':
                params = suggest_tab_ddpm_params(trial)
                print('params', params)
                params['T'] = {}
                params['info_json'] = columns_json
                params['pred_column'] = cfg['predict_column']
                gen_model = Tab_ddpm(**params, df=df_train_fold)
                gen_model.train(df_train_fold, random_seed=1)
                trial.set_user_attr("params", params)
            else:
                # Snythcity models
                hyperparams_space = get_synthcity_plugin_hyperparams(model_name)
                params = suggest_all(trial, hyperparams_space)
                print('params', params)
                gen_model = SynthCityModel(model_name.lower(), **params)
                dl_arguments = SynthCityModel.load_dataloader_info(cfg.experiment_name, model_name.lower())
                try:
                    gen_model.train(df_train_fold, random_seed=1, **dl_arguments)
                except ValueError as e:
                    # for nflow error
                    if 'Input contains NaN' in str(e):
                        return 0
                    else:
                        raise  # If it's a different ValueError, we re-raise it.
                trial.set_user_attr('params', params)

            elapsed_time = time.time() - start_time
            training_times.append(elapsed_time)

            df_synth = gen_model.sample(len(df_train_fold))
            # just for tab_ddpm in case it went wrong, give 0 back
            if len(df_synth) == 0:
                return 0

            # correct if datatypes if necessary
            for col in df_train_fold:
                if df_train_fold[col].dtype != df_synth[col].dtype:
                    df_synth[col] = df_synth[col].astype(df_train_fold[col].dtype)

            start_time = time.time()
            for j, eval_method in enumerate(cfg.optimization_metrics):
                if f'optimization_metric{j + 1}_params' not in cfg:
                    params = {}
                else:
                    params = dict(cfg[f'optimization_metric{j + 1}_params'])

                score = _call_eval_func(cfg.experiment_name, eval_func_name=eval_method, params=params,
                                        orig_df_train=df_train_fold,
                                        synth_df_train=df_synth, orig_df_test=df_test_fold)
                print(f'{i}_fold {eval_method}: {score}')

                # scores do not yet exist
                if i == 0:
                    scores.append(score)
                # sum the scores
                else:
                    scores[j] += score
            elapsed_time = time.time() - start_time
            evaluation_times.append(elapsed_time)

            # premature stop in case the performance difference is too big
            if len(trial.study.best_trials) > 0 and i < 4:
                best_val = trial.study.best_trial.value
                tmp_avg_scores = [score / (i + 1) for score in scores]
                tmp_sum_score = 0

                for score_val, score_weight in zip(tmp_avg_scores, cfg.optimization_metrics_weights):
                    tmp_sum_score += score_val * score_weight

                tmp_final_score = tmp_sum_score / sum(cfg.optimization_metrics_weights)
                if tmp_final_score < best_val * at_least_allowed:
                    print(f'Interrupted training after {i + 1} run, because {tmp_final_score} is too far away from '
                          f'{best_val}')
                    return tmp_final_score

        # printing training and evaluation times (total)
        print(f'The training of 5 {model_name} models took in total {str(timedelta(seconds=sum(training_times)))}')
        print(f'The evaluation of 5 {model_name} models took in total {str(timedelta(seconds=sum(evaluation_times)))}')

        # average the scores
        scores = [score / 5 for score in scores]

        sum_score = 0
        # weigh the scores according to the defined weights
        for score_val, score_weight in zip(scores, cfg.optimization_metrics_weights):
            sum_score += score_val * score_weight

        final_score = sum_score / sum(cfg.optimization_metrics_weights)
        return final_score

    return objective_function


@hydra.main(config_path="../config", config_name="3_gen_optimization", version_base="1.1")
def optimize_synthesizer(cfg: MainGenOptimizationConfig):
    """
    Main function to optimize the synthesizer based on provided configurations.

    :param cfg: The main configuration object for generator optimization.
    """
    cfg: GenOptimizationConfig = cfg.gen_optimization
    print(OmegaConf.to_yaml(cfg))

    # validate config
    _validate_gen_optimization_config(cfg)

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', cfg.experiment_name)

    for i, model in enumerate(cfg.optimize_models):
        # check if output file exists -> continue with next
        filename = cfg.filenames[i] if cfg.filenames[i].endswith('.json') else cfg.filenames[i] + '.json'
        if os.path.exists(os.path.join(exp_folder_path, model, 'config', filename)):
            continue

        # create sampler
        if cfg.optimization_sampler == 'RandomSampler':
            sampler = optuna.samplers.RandomSampler(seed=cfg.optimization_seed)
        elif cfg.optimization_sampler == 'TPESampler':
            sampler = optuna.samplers.TPESampler(seed=cfg.optimization_seed)
        elif cfg.optimization_sampler == 'CmaEsSampler':
            sampler = optuna.samplers.CmaEsSampler(seed=cfg.optimization_seed)
        elif cfg.optimization_sampler == 'NSGAIISampler':
            sampler = optuna.samplers.NSGAIISampler(seed=cfg.optimization_seed)
        elif cfg.optimization_sampler == 'QMCSampler':
            sampler = optuna.samplers.QMCSampler(seed=cfg.optimization_seed)
        else:
            raise AttributeError(f'cfg.optimization_sampler {cfg.optimization_sampler} is not supported.')

        objective_function = create_gen_objective_func(cfg, model)

        # create study
        path = os.path.abspath(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments',
                                            cfg.experiment_name, 'studies'))

        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Specify the SQLite database file
        db_file = os.path.join(path, f'{cfg.experiment_name}_{model}_{cfg.optimization_seed}.db')
        # Create the study with the SQLite database storage
        storage = f'sqlite:///{db_file}'

        study = optuna.create_study(direction='maximize', sampler=sampler, storage=storage, load_if_exists=True,
                                    study_name=f'{cfg.experiment_name}_{model}_{cfg.optimization_seed}')

        print(f'Number of completed trials before optimization: {len(study.trials)}')
        # start optimization
        study.optimize(objective_function, n_trials=cfg.optimization_trials, show_progress_bar=True, n_jobs=6)

        # save best_params
        best_params = study.best_trial.user_attrs['params']
        with open(os.path.join(exp_folder_path, model, 'config', filename), 'w') as fh:
            json.dump(best_params, fh, indent=2)


if __name__ == '__main__':
    optimize_synthesizer()



