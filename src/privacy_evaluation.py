import copy
import heapq
import json
import os
from dataclasses import dataclass, MISSING
from typing import Literal, Optional, List, Dict
import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.base import PROJECT_DIR
from src.helpers import load_experiment_data, load_columns_json_for_experiment


@dataclass
class PrivacyEvalConfig:
    """ Configuration for the privacy evaluation of synthetic data. """
    experiment_name: str = MISSING
    hamming_dist_num_bins: int = MISSING
    synthetic_data: Dict[str, List[str]]


@dataclass
class MainPrivacyEvalConfig:
    """Main configuration for the privacy evaluation synthetic data, containing a PrivacyEvalConfig instance."""
    privacy_eval_gen: PrivacyEvalConfig = MISSING


def _validate_config(cfg: PrivacyEvalConfig) -> None:
    """
    Validate the provided configuration for the privacy evaluation.

    :param cfg: Configuration to validate.
    """

    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', cfg.experiment_name)
    if not os.path.exists(exp_folder_path):
        raise AttributeError(f'The provided experiment: "{cfg.experiment_name}" does not exist.')

    for model in cfg.synthetic_data:
        model_folder_path = os.path.join(exp_folder_path, model.upper())
        if not os.path.exists(model_folder_path):
            raise AttributeError(f'The provided model folder "{model.upper()}" does not exist for the experiment '
                                 f'"{cfg.experiment_name}".')
        if not isinstance(OmegaConf.to_object(cfg.synthetic_data[model]), list):
            raise AttributeError(f'A list of filenames needs to be provided for each generative model. '
                                 f'For the model "{model.upper()}" no list was provided.')
        for path in cfg.synthetic_data[model]:
            path = path if path.endswith('.csv') else path + '.csv'
            file_path = os.path.join(model_folder_path, 'synthetic_data', path)
            if not os.path.exists(file_path):
                raise AttributeError(f'The provided filename "{path}" for the model "{model.upper()}" does not exist '
                                     f'for the experiment "{cfg.experiment_name}".')

    if not isinstance(cfg.hamming_dist_num_bins, int):
        raise AttributeError(f'The parameter "hamming_dist_num_bins" needs to be an Integer, '
                             f'but {cfg.hamming_dist_num_bins} was provided.')


def hamming_distance(df_from, df_to, categorical_cols, num_bins=10):
    """
    Calculate the Hamming distance for each datapoint in df_from to each datapoint to df_to.

    :param df_from: Source dataframe.
    :param df_to: Target dataframe.
    :param categorical_cols: List of columns that are considered categorical.
    :param num_bins: Number of bins for discretizing numerical columns.

    :return: A tuple containing: Average minimum hamming distance, Average minimum hamming distance difference for the
    closest two datapoints, List of percentile for the minimum hamming distances, Number of exact matches.
    """

    df_from = copy.deepcopy(df_from)
    df_to = copy.deepcopy(df_to)

    # all columns that are not categorical are considered numerical
    num_cols = [col for col in df_to.columns if col not in categorical_cols]

    for col in num_cols:
        # get the cutoff values for num_bins for each numerical columns
        cut_offs = pd.qcut(df_to[col], q=num_bins, retbins=True, duplicates='drop')[1]

        # Set the first bin to negative infinity and the last to positive infinity
        cut_offs[0] = -np.inf
        cut_offs[-1] = np.inf

        df_to[col] = pd.cut(df_to[col], bins=cut_offs, labels=list(range(len(cut_offs) - 1)))
        df_from[col] = pd.cut(df_from[col], bins=cut_offs, labels=list(range(len(cut_offs) - 1)))

    # calculate dist for every datapoint in synth_df to every point in orig_df
    min_dists = []
    min_dist_two_dif = []
    for i in range(len(df_from)):
        # calculate the Hamming distance to every point in orig_df
        dists = (df_to != df_from.iloc[i]).sum(axis=1)
        smallest_two = heapq.nsmallest(2, dists)
        min_dists.append(smallest_two[0])
        min_dist_two_dif.append(smallest_two[1] - smallest_two[0])

    # calculate the average of the minimum Hamming distances
    avg_min_dist = np.mean(min_dists)
    avg_min_dist_two_dif = np.mean(min_dist_two_dif)

    # min dist extra
    min_dists.sort()
    num_zeros = min_dists.count(0)
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    # Percentiles
    percentile_values = np.percentile(min_dists, [p * 100 for p in percentiles])

    return avg_min_dist, avg_min_dist_two_dif, percentile_values, num_zeros


def eval_hamming_distances(df_from: pd.DataFrame, df_to: pd.DataFrame,
                           categorical_cols: List[str],
                           mode: Literal['balanced', 'unbalanced'] = 'balanced',
                           balanced_count: Optional[int] = None,
                           df_from_df_to_same=False,
                           hamming_dist_num_bins=10):
    """
    Evaluate Hamming distances between two dataframes.

    :param df_from: Source dataframe.
    :param df_to: Target dataframe.
    :param categorical_cols: List of columns that are considered categorical.
    :param mode: Evaluation mode ('balanced' or 'unbalanced').
    :param balanced_count: Number of samples in each balanced split.
    :param df_from_df_to_same: Indicates if source and target dataframes are the same.
    :param hamming_dist_num_bins: Number of bins for discretizing numerical columns.

    :return: A tuple containing: Average minimum hamming distance, Average minimum hamming distance difference for the
    closest two datapoints, List of percentile for the minimum hamming distances, Number of exact matches.
    """

    if mode == 'balanced' and balanced_count is None:
        raise AttributeError(f'The chosen mode is "balanced", but "balanced_count" was not provided.')

    if mode == 'balanced':
        from_split_count = round(len(df_from) / balanced_count)
        to_split_count = round(len(df_to) / balanced_count)

        # shuffle the data
        df_from = df_from.sample(frac=1, random_state=1).reset_index(drop=True)
        df_to = df_to.sample(frac=1, random_state=1).reset_index(drop=True)

        # separate data into sets
        df_from_sets = np.array_split(df_from, from_split_count)
        df_to_sets = np.array_split(df_to, to_split_count)
    else:
        df_from_sets = [df_from]
        df_to_sets = [df_to]

    # for averaging
    avg_min_dist, avg_min_dist_two_dif, num_zeros = 0., 0., 0.
    avg_percentiles = {0.05: 0., 0.25: 0., 0.5: 0., 0.75: 0., 0.95: 0.}
    skip = 0

    for i in range(len(df_from_sets)):
        for j in range(len(df_to_sets)):
            # same random seeds used for shuffling of the same df -> same sets
            if df_from_df_to_same and j <= i:
                skip += 1
                continue

            df_from_set = df_from_sets[i]
            df_to_set = df_to_sets[j]

            min_dist, min_dist_two_dif, percentiles, zeros = hamming_distance(df_from_set, df_to_set,
                                                                              categorical_cols=categorical_cols,
                                                                              num_bins=hamming_dist_num_bins)

            avg_min_dist += min_dist
            avg_min_dist_two_dif += min_dist_two_dif
            num_zeros += zeros

            avg_percentiles[0.05] += percentiles[0]
            avg_percentiles[0.25] += percentiles[1]
            avg_percentiles[0.5] += percentiles[2]
            avg_percentiles[0.75] += percentiles[3]
            avg_percentiles[0.95] += percentiles[4]

    exp_count = len(df_from_sets) * len(df_to_sets) if not df_from_df_to_same \
        else len(df_from_sets) * len(df_to_sets) - skip

    avg_min_dist /= exp_count
    avg_min_dist_two_dif /= exp_count
    avg_percentiles[0.05] /= exp_count
    avg_percentiles[0.25] /= exp_count
    avg_percentiles[0.5] /= exp_count
    avg_percentiles[0.75] /= exp_count
    avg_percentiles[0.95] /= exp_count

    return avg_min_dist, avg_min_dist_two_dif, avg_percentiles, num_zeros


def evaluate_privacy(experiment_name: str, gen_model_names: Dict[str, List[str]], hamming_dist_num_bins=10):
    """
    Evaluate the privacy of generated synthetic data against original data.

    :param experiment_name: Name of the experiment.
    :param gen_model_names: Mapping of generative model names to a list of synthetic data filenames.
    :param hamming_dist_num_bins: Number of bins for Hamming distance calculation.
    """

    results = {}
    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name)
    _, df_train, df_test = load_experiment_data(experiment_name, for_training=False)
    columns_json = load_columns_json_for_experiment(experiment_name)
    categorical_cols = columns_json['boolean'] + columns_json['categorical']

    results['parameters'] = {'hamming_dist_num_bins': hamming_dist_num_bins,
                             'balanced_count': len(df_test)}

    orig_train_avg_min_dist, orig_train_avg_min_dist_two_dif, orig_train_avg_percentiles, orig_train_num_zeros = eval_hamming_distances(
        df_train, df_train, categorical_cols=categorical_cols, balanced_count=len(df_test), df_from_df_to_same=True,
        hamming_dist_num_bins=hamming_dist_num_bins)

    orig_test_avg_min_dist, orig_test_avg_min_dist_two_dif, orig_test_avg_percentiles, orig_test_num_zeros = eval_hamming_distances(
        df_train, df_test, categorical_cols=categorical_cols, balanced_count=len(df_test),
        hamming_dist_num_bins=hamming_dist_num_bins)

    results['original train'] = {
        'train_avg_min_dist': orig_train_avg_min_dist,
        'test_avg_min_dist': orig_test_avg_min_dist,
        'train_avg_min_dist_two_dif': orig_train_avg_min_dist_two_dif,
        'test_avg_min_dist_two_dif': orig_test_avg_min_dist_two_dif,
        'train_avg_percentile_0.05': orig_train_avg_percentiles[0.05],
        'test_avg_percentile_0.05': orig_test_avg_percentiles[0.05],
        'train_avg_percentile_0.25': orig_train_avg_percentiles[0.25],
        'test_avg_percentile_0.25': orig_test_avg_percentiles[0.25],
        'train_avg_percentile_0.50': orig_train_avg_percentiles[0.5],
        'test_avg_percentile_0.50': orig_test_avg_percentiles[0.5],
        'train_avg_percentile_0.75': orig_train_avg_percentiles[0.75],
        'test_avg_percentile_0.75': orig_test_avg_percentiles[0.75],
        'train_avg_percentile_0.95': orig_train_avg_percentiles[0.95],
        'test_avg_percentile_0.95': orig_test_avg_percentiles[0.95],
        'train_num_zeros_sum': orig_train_num_zeros,
        'test_num_zeros_sum': orig_test_num_zeros,
        'dif_train_test': 1 - (orig_train_avg_min_dist / orig_test_avg_min_dist)
    }

    for gen_model in gen_model_names:
        for filename in gen_model_names[gen_model]:
            filename = filename if filename.endswith('.csv') else filename + '.csv'
            df_synth = pd.read_csv(os.path.join(exp_folder_path, gen_model.upper(), 'synthetic_data', filename))

            for col in df_train:
                if df_train[col].dtype != df_synth[col].dtype:
                    df_synth[col] = df_synth[col].astype(df_train[col].dtype)

            train_avg_min_dist, train_avg_min_dist_two_dif, train_avg_percentiles, train_num_zeros = eval_hamming_distances(df_synth, df_train, categorical_cols=categorical_cols, balanced_count=len(df_test), hamming_dist_num_bins=hamming_dist_num_bins)
            test_avg_min_dist, test_avg_min_dist_two_dif, test_avg_percentiles, test_num_zeros = eval_hamming_distances(df_synth, df_test, categorical_cols=categorical_cols, balanced_count=len(df_test), hamming_dist_num_bins=hamming_dist_num_bins)
            results[gen_model.upper()+'_'+filename] = {
                'train_avg_min_dist': train_avg_min_dist,
                'test_avg_min_dist': test_avg_min_dist,
                'train_avg_min_dist_two_dif': train_avg_min_dist_two_dif,
                'test_avg_min_dist_two_dif': test_avg_min_dist_two_dif,
                'train_avg_percentile_0.05': train_avg_percentiles[0.05],
                'test_avg_percentile_0.05': test_avg_percentiles[0.05],
                'train_avg_percentile_0.25': train_avg_percentiles[0.25],
                'test_avg_percentile_0.25': test_avg_percentiles[0.25],
                'train_avg_percentile_0.50': train_avg_percentiles[0.5],
                'test_avg_percentile_0.50': test_avg_percentiles[0.5],
                'train_avg_percentile_0.75': train_avg_percentiles[0.75],
                'test_avg_percentile_0.75': test_avg_percentiles[0.75],
                'train_avg_percentile_0.95': train_avg_percentiles[0.95],
                'test_avg_percentile_0.95': test_avg_percentiles[0.95],
                'train_num_zeros_sum': train_num_zeros,
                'test_num_zeros_sum': test_num_zeros,
                'dif_train_test': 1 - (train_avg_min_dist / test_avg_min_dist),
                'dif_syn_train_orig_train': (train_avg_min_dist / orig_train_avg_min_dist) - 1,
                'dif_syn_test_orig_test': (test_avg_min_dist / orig_test_avg_min_dist) - 1
            }

    # save results
    with open(os.path.join(exp_folder_path, 'eval_privacy.json'), 'w') as fh:
        json.dump(results, fh, indent=2)


@hydra.main(config_path="../config", config_name="6_privacy_eval_gen", version_base="1.1")
def main_privacy_eval(cfg: MainPrivacyEvalConfig) -> None:
    """
    Main function to execute privacy evaluation.

    :param cfg: Configuration for privacy evaluation.
    """
    cfg: PrivacyEvalConfig = cfg.privacy_eval_gen
    print(OmegaConf.to_yaml(cfg))
    _validate_config(cfg)
    evaluate_privacy(cfg.experiment_name, gen_model_names=cfg.synthetic_data,
                     hamming_dist_num_bins=cfg.hamming_dist_num_bins)


if __name__ == '__main__':
    main_privacy_eval()

