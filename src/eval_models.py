import copy
import json
import os
from collections import defaultdict
from dataclasses import MISSING, dataclass
from typing import List, Literal, Optional, Dict
import hydra
import numpy as np
from omegaconf import OmegaConf
from pandas import DataFrame
from tabulate import tabulate
from src.autoencoder import decode_synthetic_data
from src.base import PROJECT_DIR
from src.evaluation_tableevaluator import plot_pca, plot_cumsums, plot_correlation_difference
from src.helpers import load_experiment_data, load_columns_json_for_experiment, load_catboost_params
from src.models import load_gen_model, available_models, available_models_type
from src.evaluation_functions import (synthetic_eval_check, neg_values_check, basic_statistical_measure_num,
                                      discriminator_measure_rf, k_means_score, log_transformed_correlation_score,
                                      regularized_support_coverage, ml_efficiency_cat, survival_sightedness,
                                      survival_auc_opt, survival_auc_abs_opt, survival)


@dataclass
class EvalConfig:
    """ Configuration for the evaluation of generative models. """
    experiment_name: str = MISSING
    models: List[available_models_type] = MISSING
    sampling_seeds: List[int] = MISSING
    output: Literal['txt', 'avg_json'] = MISSING
    metrics: List[Literal["basic_statistical_measure_num", "log_transformed_correlation_score",
                          "regularized_support_coverage", "discriminator_measure_rf", "k_means_score",
                          "ml_efficiency_cat", "synthetic_eval_check", "neg_values_check", "plot_pca", "plot_cumsums",
                          "plot_correlation_difference", "survival_auc_abs_opt", "survival_auc_opt",
                          "survival_sightedness", "survival"]] = MISSING
    # maximum of twenty metrics allowed, for each metrics params can be given here, necessary for ml_efficiency
    # params need to be the same as the functions
    metric1_params: Optional[Dict] = MISSING
    metric2_params: Optional[Dict] = MISSING
    metric3_params: Optional[Dict] = MISSING
    metric4_params: Optional[Dict] = MISSING
    metric5_params: Optional[Dict] = MISSING
    metric6_params: Optional[Dict] = MISSING
    metric7_params: Optional[Dict] = MISSING
    metric8_params: Optional[Dict] = MISSING
    metric9_params: Optional[Dict] = MISSING
    metric10_params: Optional[Dict] = MISSING
    metric11_params: Optional[Dict] = MISSING
    metric12_params: Optional[Dict] = MISSING
    metric13_params: Optional[Dict] = MISSING
    metric14_params: Optional[Dict] = MISSING
    metric15_params: Optional[Dict] = MISSING
    metric16_params: Optional[Dict] = MISSING
    metric17_params: Optional[Dict] = MISSING
    metric18_params: Optional[Dict] = MISSING
    metric19_params: Optional[Dict] = MISSING
    metric20_params: Optional[Dict] = MISSING

    # optional parameters
    synth_preprocess_lambda: Optional[List[str]] = MISSING
    synth_raw_size: Optional[int] = MISSING
    synth_remove_lambda: Optional[List[str]] = MISSING
    synth_final_size: Optional[int] = MISSING


@dataclass
class MainEvalConfig:
    """Main configuration for the evaluation of generative models, containing a EvalConfig instance."""
    eval_gen: EvalConfig = MISSING


def _validate_eval_config(cfg: EvalConfig):
    """
    Validate the evaluation configuration.

    :param cfg: Configuration instance to validate.
    """

    pos_metrics = ["basic_statistical_measure_num", "log_transformed_correlation_score", "regularized_support_coverage",
                   "discriminator_measure_rf", "k_means_score", "ml_efficiency_cat", "synthetic_eval_check",
                   "neg_values_check", "plot_pca", "plot_cumsums", "plot_correlation_difference",
                   "survival_auc_abs_opt", "survival_auc_opt", "survival_sightedness",  "survival"]

    for model in cfg.models:
        if model.lower() not in available_models:
            raise AttributeError(f'Models in models need to be one of the following: '
                                 f'{", ".join(available_models)}, but {model} was provided')

    for i, metric in enumerate(cfg.metrics):
        if metric not in pos_metrics:
            raise AttributeError(f'Metrics in metrics needs to be one of the following: '
                                 f'{", ".join(pos_metrics)}, but {metric} was provided')

        if metric == 'ml_efficiency_cat':
            if f'metric{i + 1}_params' not in cfg:
                raise AttributeError(f'For the metric ml_efficiency the following parameters need to be provided: '
                                     f'outcome_col, metric, filename_params_json, relative, '
                                     f'Optional (create_column_lambda) (optimization_metric{i + 1}_params)')
            params = getattr(cfg, f'metric{i + 1}_params')
            if 'predict_col' not in params or 'metric' not in params or 'filename_params_json' not in params \
                    or 'relative' not in params:
                raise AttributeError(f'For the metric ml_efficiency the following parameters need to be provided: '
                                     f'outcome_col, metric, filename_params_json, relative, '
                                     f'Optional (create_column_lambda) (optimization_metric{i + 1}_params)')

            elif metric == 'synthetic_eval_check':
                if f'metric{i + 1}_params' not in cfg:
                    raise AttributeError(f'For the metric synthetic_eval_check the following parameter need to be '
                                         f'provided: eval_str, name (metric{i + 1}_params)')
                params = getattr(cfg, f'metric{i + 1}_params')
                if 'eval_str' and 'name' not in params:
                    raise AttributeError(f'For the metric synthetic_eval_check the following parameters need to be '
                                         f'provided: eval_str, name (metric{i + 1}_params)')
        if metric.startswith('survival'):
            if f'metric{i + 1}_params' not in cfg:
                raise AttributeError(f'For the metric {metric} the following parameters need to be provided: '
                                     f'survival_target_col, survival_time_to_event_col '
                                     f'(optimization_metric{i + 1}_params)')
            params = cfg[f'metric{i + 1}_params']
            if 'survival_target_col' not in params or 'survival_time_to_event_col' not in params:
                raise AttributeError(f'For the metric {metric} the following parameters need to be provided: '
                                     f'survival_target_col, survival_time_to_event_col '
                                     f'(optimization_metric{i + 1}_params)')

    if 'synth_raw_size' in cfg and 'synth_remove_lambda' not in cfg:
        raise AttributeError(f'synth_raw_size can only be provided, if synth_remove_lambda is also provided. '
                             f'If you do not want to exclude any synthetic data, use synth_final_size instead')

    if 'synth_raw_size' not in cfg and 'synth_remove_lambda' in cfg:
        raise AttributeError(f'synth_remove_lambda can only be provided, if synth_raw_size is also provided.')

    if 'synth_raw_size' in cfg and not isinstance(cfg.synth_raw_size, int):
        raise AttributeError(f'The parameter synth_raw_size needs to be an Integer, '
                             f'but {cfg.synth_raw_size} was provided.')

    if 'synth_final_size' in cfg and not isinstance(cfg.synth_final_size, int):
        raise AttributeError(f'The parameter synth_final_size needs to be an Integer, '
                             f'but {cfg.synth_final_size} was provided.')

    if 'synth_raw_size' in cfg and 'synth_final_size' in cfg and cfg.synth_final_size >= cfg.synth_raw_size:
        raise AttributeError(f'The parameter synth_raw_size need to be always bigger than synth_final_size.')

    if not isinstance(OmegaConf.to_object(cfg.synth_preprocess_lambda), list):
        raise AttributeError(f'synth_preprocess_lambda needs to be a list.')

    if not isinstance(OmegaConf.to_object(cfg.synth_remove_lambda), list):
        raise AttributeError(f'synth_remove_lambda needs to be a list.')


def call_eval_func(experiment_name: str, eval_func_name: str, params: Dict, orig_df_train: DataFrame,
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

    params = copy.deepcopy(params)
    if eval_func_name == 'basic_statistical_measure_num':
        if 'num_columns' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name, for_training=False)
            params['num_columns'] = columns_json['integer'] + columns_json['float']
        return basic_statistical_measure_num(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'log_transformed_correlation_score':
        if 'categorical_cols' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name, for_training=False)
            params['categorical_cols'] = columns_json['boolean'] + columns_json['categorical']
        return log_transformed_correlation_score(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'regularized_support_coverage':
        if 'categorical_cols' not in params:
            columns_json = load_columns_json_for_experiment(experiment_name, for_training=False)
            params['categorical_cols'] = columns_json['boolean'] + columns_json['categorical']
        return regularized_support_coverage(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'ml_efficiency_cat':
        params['training_params'] = load_catboost_params(experiment_name, params['filename_params_json'])
        params.pop('filename_params_json', None)
        columns_json = load_columns_json_for_experiment(experiment_name, for_training=False)
        params['outcome_cols'] = columns_json['outcome']
        return ml_efficiency_cat(orig_df_train, orig_df_test, synth_df_train, **params)

    if eval_func_name == 'discriminator_measure_rf':
        return discriminator_measure_rf(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'k_means_score':
        return k_means_score(orig_df_train, synth_df_train, **params)

    if eval_func_name == 'synthetic_eval_check':
        return synthetic_eval_check(synth_df_train, **params)

    if eval_func_name == 'neg_values_check':
        return neg_values_check(orig_df_train, synth_df_train)

    if eval_func_name.startswith('survival'):
        time_col = params['survival_time_to_event_col']
        target_col = params['survival_target_col']
        count = len(orig_df_train)
        func = eval(eval_func_name)
        return func([orig_df_train[time_col], orig_df_train[target_col]],
                    [synth_df_train[time_col], synth_df_train[target_col]],
                    count)

    raise NotImplementedError(f'There is no implementation yet for the following evaluation function: {eval_func_name}')


def call_visual_eval_func(experiment_name: str, eval_func_name: str, params: Dict, df_orig: DataFrame,
                          df_synth: DataFrame, model_type: str, model_name: str, train_seed: str, sampling_seed: str):
    """
    Call a specific visual evaluation function based on its name.

    :param experiment_name: Name of the experiment.
    :param eval_func_name: Name of the evaluation function.
    :param params: Dictionary of parameters for the evaluation function.
    :param df_orig: Original dataframe.
    :param df_synth: Synthetic dataframe.
    :param model_type: Type of the model.
    :param model_name: Name of the model.
    :param train_seed: Training seed value.
    :param sampling_seed: Sampling seed value.
    """

    columns_json = load_columns_json_for_experiment(experiment_name)
    cat_cols = columns_json['boolean'] + columns_json['categorical']

    eval_folder = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                               model_type, 'eval')
    filename = f'{model_name}_t{train_seed}_s{sampling_seed}.png'

    if eval_func_name == 'plot_pca':
        filename = 'pca_' + filename
        plot_pca(df_orig, df_synth, categorical_columns=cat_cols,
                 fname=os.path.join(eval_folder, f'{model_name}_{train_seed}', filename))

    elif eval_func_name == 'plot_cumsums':
        filename = 'cumsum_' + filename
        if 'columns' in params:
            plot_cumsums(df_orig[params['columns']], df_synth[params['columns']],
                         fname=os.path.join(eval_folder, f'{model_name}_{train_seed}', filename))
        else:
            plot_cumsums(df_orig, df_synth,
                         fname=os.path.join(eval_folder, f'{model_name}_{train_seed}', filename))

    elif eval_func_name == 'plot_correlation_difference':
        filename = 'corr_' + filename
        if 'columns' in params:
            plot_correlation_difference(df_orig[params['columns']], df_synth[params['columns']],
                                    fname=os.path.join(eval_folder, f'{model_name}_{train_seed}', filename),
                                    cat_cols=[c for c in cat_cols if c in params['columns']])
        else:
            plot_correlation_difference(df_orig, df_synth,
                                        fname=os.path.join(eval_folder, f'{model_name}_{train_seed}', filename),
                                        cat_cols=cat_cols)
    else:
        raise NotImplementedError(f'There is no implementation yet for the following evaluation function: '
                                  f'{eval_func_name}')


@hydra.main(config_path="../config", config_name="5_eval_gen", version_base="1.1")
def evaluate_synthesizer(cfg: MainEvalConfig):
    """
    Evaluates the generative models based on the provided configuration.

    :param cfg: Configuration object which contains evaluation settings and parameters.
    """

    cfg: EvalConfig = cfg.eval_gen
    print(OmegaConf.to_yaml(cfg))

    # list of metrics that are just figures
    metrics_fig = {"plot_pca", "plot_cumsums", "plot_correlation_difference"}

    # validate config
    _validate_eval_config(cfg)

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', cfg.experiment_name)

    _, df_train, df_test = load_experiment_data(cfg.experiment_name, for_training=False)

    for model_type in cfg.models:
        for filename in os.listdir(os.path.join(exp_folder_path, model_type, 'models')):
            print(f'eval for {model_type} {filename} started')
            model_name_short = filename.rsplit('_', maxsplit=1)[0]
            train_seed = filename.rsplit('_', maxsplit=1)[1].split('.', maxsplit=1)[0]

            os.makedirs(os.path.join(exp_folder_path, model_type, 'eval', f'{model_name_short}_{train_seed}'),
                        exist_ok=True)

            # there should be several models trained with different seeds
            model = load_gen_model(cfg.experiment_name, model_type, filename)

            for sampling_seed in cfg.sampling_seeds:
                run_results = {}

                if ('synth_final_size' not in cfg or cfg.synth_final_size is None
                        or cfg.synth_final_size == 'None' or cfg.synth_final_size == ''):
                    df_synth = model.sample(len(df_train), random_seed=sampling_seed)
                else:
                    if ('synth_raw_size' not in cfg or cfg.synth_raw_size is None
                            or cfg.synth_raw_size == 'None' or cfg.synth_raw_size == ''):
                        df_synth = model.sample(cfg.synth_final_size, random_seed=sampling_seed)
                    else:
                        df_synth = model.sample(cfg.synth_raw_size, random_seed=sampling_seed)

                # TODO hardcoded
                if os.path.exists(os.path.join(exp_folder_path, 'data', 'autoencoder.pth')):
                    df_synth = decode_synthetic_data(cfg.experiment_name, df_synth)

                # correct datatypes if necessary
                for col in df_train:
                    if df_train[col].dtype != df_synth[col].dtype:
                        df_synth[col] = df_synth[col].astype(df_train[col].dtype)

                if ('synth_preprocess_lambda' in cfg and cfg.synth_preprocess_lambda is not None
                        and cfg.synth_preprocess_lambda != 'None' and cfg.synth_preprocess_lambda != ''):
                    for lambda_expr in cfg.synth_preprocess_lambda:
                        df_synth = df_synth.apply(lambda row: (exec(lambda_expr, globals(), {'row': row}), row)[1], axis=1)

                if ('synth_remove_lambda' in cfg and cfg.synth_remove_lambda is not None
                        and cfg.synth_remove_lambda != 'None' and cfg.synth_remove_lambda != ''):
                    for remove_exp in cfg.synth_remove_lambda:
                        df_synth = df_synth[eval(remove_exp)]

                if ('synth_final_size' in cfg and cfg.synth_final_size is not None
                        and cfg.synth_final_size != 'None' and cfg.synth_final_size != ''):
                    df_synth = df_synth.sample(n=cfg.synth_final_size, random_state=1)

                # save csv
                df_synth.to_csv(os.path.join(exp_folder_path, model_type, 'synthetic_data',
                                             f'{model_name_short}_t{train_seed}_s{sampling_seed}.csv'), index=False)

                for i, metric in enumerate(cfg.metrics):
                    # get params for the metric if exists
                    if f'metric{i + 1}_params' not in cfg:
                        params = {}
                        metric_name = metric
                    else:
                        params = dict(getattr(cfg, f'metric{i + 1}_params'))
                        metric_name = params.pop('name', metric)

                    # returns a float
                    if metric not in metrics_fig:
                        run_results[metric_name] = call_eval_func(cfg.experiment_name, metric, params, df_train,
                                                                  df_synth, df_test)

                    # figures
                    else:
                        call_visual_eval_func(cfg.experiment_name, metric, params=params, df_orig=df_train,
                                              df_synth=df_synth, model_type=model_type, model_name=model_name_short,
                                              train_seed=str(train_seed), sampling_seed=str(sampling_seed))

                # save results
                with open(os.path.join(exp_folder_path, model_type, 'eval', f'{model_name_short}_{train_seed}',
                                       f'{model_name_short}_t{train_seed}_s{sampling_seed}.json'), 'w') as fh:
                    json.dump(run_results, fh, indent=2)

    # average all results
    average(cfg.experiment_name, cfg.models, 'avg_json')


def average(experiment_name: str, model_types: List[str], output: Literal['txt', 'avg_json'] = 'txt'):
    """
    Averages the evaluation results of generative models and writes the averaged results.

    :param experiment_name: Name of the experiment for which the results should be averaged.
    :param model_types: List of types of models for which the results should be averaged.
    :param output: Format in which to write the averaged results. Either 'txt' or 'avg_json'.
    """

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name)
    final_avg_results = {}

    for model_type in model_types:
        # files = os.listdir(os.path.join(exp_folder_path, model_type, 'eval'))
        # json_files = [file for file in files if file.endswith('.json')]
        eval_folder_path = os.path.join(exp_folder_path, model_type, 'eval')
        json_paths = [os.path.join(root, file) for root, dirs, files in os.walk(eval_folder_path) for file in files if
                      file.endswith('.json')]

        # (shortname, train_seed, sample_seed) as key, results dict as value
        results_dict = {}
        short_name2train_seeds = defaultdict(set)
        short_name2sampling_seeds = defaultdict(set)

        # keeping track of how many samples were averaged
        short_name_count = defaultdict(int)

        for path in json_paths:
            filename = os.path.basename(path)
            if not filename.__contains__('_t') or not filename.__contains__('_s'):
                continue

            short_name, other_str = filename.rsplit('_t', maxsplit=1)
            train_seed, other_str = other_str.split('_s', maxsplit=1)
            sample_seed = other_str.rsplit('.json', maxsplit=1)[0]

            short_name = model_type + '_' + short_name

            short_name_count[short_name] += 1

            with open(path, 'r') as fh:
                json_file = json.load(fh)

            short_name2train_seeds[short_name].add(train_seed)
            short_name2sampling_seeds[short_name].add(sample_seed)

            results_dict[(short_name, train_seed, sample_seed)] = json_file

        # averaged results
        t_seed_results_dict = defaultdict(lambda: defaultdict(list))
        s_seed_results_dict = defaultdict(lambda: defaultdict(list))
        all_results_dict = defaultdict(lambda: defaultdict(list))

        for name in short_name2train_seeds:
            # metrics
            metrics = []
            if len(short_name2train_seeds[name]) > 0 and len(short_name2sampling_seeds[name]) > 0:
                metrics = list(results_dict[(name, list(short_name2train_seeds[name])[0],
                                             list(short_name2sampling_seeds[name])[0])].keys())

            # append all results that should be averaged
            for t_seed in short_name2train_seeds[name]:
                for s_seed in short_name2sampling_seeds[name]:
                    results = results_dict[(name, t_seed, s_seed)]

                    if len(metrics) == 0:
                        metrics = list(results.keys())

                    for key, val in results.items():
                        t_seed_results_dict[(name, t_seed)][key].append(val)
                        s_seed_results_dict[(name, s_seed)][key].append(val)
                        all_results_dict[name][key].append(val)

            train_seed_data = []
            train_seed_data_head = ['metric', 'training_seed', 'avg', 'min', 'max', 'std']

            for metric in metrics:
                # average training seed
                for t_seed in sorted([int(s) for s in short_name2train_seeds[name]]):
                    t_seed = str(t_seed)
                    val = t_seed_results_dict[(name, t_seed)][metric]
                    train_seed_data.append([metric, t_seed, np.average(val), min(val), max(val), np.std(val)])

            sampling_seed_data = []
            sampling_seed_data_head = ['metric', 'sampling_seed', 'avg', 'min', 'max', 'std']

            for metric in metrics:
                # average sampling seed
                for s_seed in sorted([int(s) for s in short_name2sampling_seeds[name]]):
                    s_seed = str(s_seed)
                    val = s_seed_results_dict[(name, s_seed)][metric]
                    sampling_seed_data.append([metric, s_seed, np.average(val), min(val), max(val), np.std(val)])

            all_data = []
            all_data_head = ['metric', 'avg', 'min', 'max', 'std']

            for metric in metrics:
                val = all_results_dict[name][metric]
                all_data.append([metric, np.average(val), min(val), max(val), np.std(val)])

            if output == 'txt':
                with open(os.path.join(exp_folder_path, model_type, 'eval', f'{name}.txt'), 'w') as fh:
                    fh.write(f'Evaluation of {name} ({model_type})\n\n')
                    fh.write(f'Results according to {len(short_name2train_seeds[name])} training seeds '
                             f'{sorted(list(short_name2train_seeds[name]))}:\n')
                    fh.write(tabulate(train_seed_data, headers=train_seed_data_head))
                    fh.write('\n\n')
                    fh.write(f'Results according to {len(short_name2sampling_seeds[name])} sampling seeds '
                             f'{sorted(list(short_name2sampling_seeds[name]))}:\n')
                    fh.write(tabulate(sampling_seed_data, headers=sampling_seed_data_head))
                    fh.write('\n\n')
                    fh.write(f'All results ({short_name_count[name]}):\n')
                    fh.write(tabulate(all_data, headers=all_data_head))

            else:
                results = {}
                for metric in metrics:
                    results[metric] = np.average(all_results_dict[name][metric])
                final_avg_results[name] = results

                with open(os.path.join(exp_folder_path, model_type, 'eval', f'{name}.json'), 'w') as fh:
                    json.dump(results, fh, indent=2)

        with open(os.path.join(exp_folder_path, 'eval.json'), 'w') as fh:
            json.dump(final_avg_results, fh, indent=2)


if __name__ == '__main__':
    evaluate_synthesizer()
