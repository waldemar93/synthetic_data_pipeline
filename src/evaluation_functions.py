import copy
import heapq
import os
import sys
from typing import List, Optional, Dict, Literal, Tuple
from catboost import CatBoostClassifier
from dython.nominal import associations, _comp_assoc
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from src.helpers import load_experiment_data, load_catboost_params, load_columns_json_for_experiment
from synthcity.plugins.core.models.survival_analysis.metrics import nonparametric_distance


def synthetic_eval_check(df: DataFrame, eval_str: str):
    """
    Evaluate a string expression and return the result.

    :param df: The dataframe which needs to be used within the evaluation string.
    :param eval_str: The string containing the expression to be evaluated.
    """

    return eval(eval_str)


def neg_values_check(orig_df: DataFrame, synth_df: DataFrame):
    """
    Checks for negative values in columns of a synthetic dataframe that should only have non-negative values
    based on the original dataframe. It calculates the proportion of rows in the synthetic dataframe that have
    invalid negative values for columns that should only contain non-negative values.

    :param orig_df: The original dataframe used as a reference for acceptable values.
    :param synth_df: The synthetic dataframe that is checked for invalid negative values.

    :return: Proportion of valid rows in the synthetic dataframe (rows that don't contain invalid negative values).
    """

    result = {}
    not_valid_idx = set()
    columns = orig_df.select_dtypes(include=[int, float]).columns
    columns_not_to_check = []
    for col in columns:
        if sum(orig_df[col] < 0) > 0:
            columns_not_to_check.append(col)
    columns_to_check = [col for col in columns if col not in columns_not_to_check]
    for col in columns_to_check:
        if sum(synth_df[col] < 0) > 0:
            idx = set(synth_df.index[synth_df[col] < 0])
            not_valid_idx = not_valid_idx | idx
            result[col] = len(idx) / len(synth_df)
    result['not_valid_rows'] = len(not_valid_idx) / len(synth_df)
    return 1-result['not_valid_rows']


### survival functions taken from synthcity implementation
def survival(real: Tuple[np.ndarray, np.ndarray], syn: Tuple[np.ndarray, np.ndarray], n_points: int = 1000):
    auc_opt, auc_abs_opt, sightedness = nonparametric_distance(real, syn, n_points)
    auc_opt = 1 - abs(auc_opt)
    auc_abs_opt = 1 - abs(auc_abs_opt)
    sightedness = 1 - abs(sightedness)
    return sum([auc_opt, auc_abs_opt, sightedness])/3


def survival_auc_opt(real: Tuple[np.ndarray, np.ndarray], syn: Tuple[np.ndarray, np.ndarray], n_points: int = 1000):
    auc_opt, _, _ = nonparametric_distance(real, syn, n_points)
    auc_opt = 1 - abs(auc_opt)
    return auc_opt


def survival_auc_abs_opt(real: Tuple[np.ndarray, np.ndarray], syn: Tuple[np.ndarray, np.ndarray], n_points: int = 1000):
    _, auc_abs_opt, _ = nonparametric_distance(real, syn, n_points)
    auc_abs_opt = 1 - abs(auc_abs_opt)
    return auc_abs_opt


def survival_sightedness(real: Tuple[np.ndarray, np.ndarray], syn: Tuple[np.ndarray, np.ndarray], n_points: int = 1000):
    _, _, sightedness = nonparametric_distance(real, syn, n_points)
    sightedness = 1 - abs(sightedness)
    return sightedness


def basic_statistical_measure_num(orig_df: DataFrame, synth_df: DataFrame, num_columns: List[str] = None,
                                  clip_val: int = 1):
    """
    Computes the basic statistical measure score (mean, median and std) for a set of numerical columns between two
    pandas DataFrames.

    :param orig_df: The original dataset used for comparison.
    :param synth_df: The synthetic dataset generated for comparison.
    :param num_columns: A list of numerical column names to be used for computation. If None, all numerical columns
                        in `orig_df` will be selected.
    :param clip_val: A value to clip the computed differences in mean, median and standard deviation at. If None,
                     no clipping will be applied.

    :return: The basic statistical measure score between `orig_df` and `synth_df`, ranging from 0 (worst) to 1 (best).
    """

    if num_columns is None:
        num_columns = list(orig_df.select_dtypes(include=[int, float]))

    # compute the relative difference between real and synth dataset for mean, median and std
    mean_dif = abs((orig_df[num_columns].mean() - synth_df[num_columns].mean()) / orig_df[num_columns].mean())
    median_dif = abs((orig_df[num_columns].median() - synth_df[num_columns].median()) / orig_df[num_columns].mean())
    std_dif = abs((orig_df[num_columns].std() - synth_df[num_columns].std()) / orig_df[num_columns].mean())

    if clip_val is not None:
        mean_dif.clip(upper=clip_val, inplace=True)
        median_dif.clip(upper=clip_val, inplace=True)
        std_dif.clip(upper=clip_val, inplace=True)

    # compute the scores for mean, median and std
    mean_score = 1 - (sum(mean_dif) / len(mean_dif))
    median_score = 1 - (sum(median_dif) / len(median_dif))
    std_score = 1 - (sum(std_dif) / len(std_dif))
    # best 1, worst 0, final score
    basic_measure = (mean_score + median_score + std_score) / 3

    return basic_measure


def log_transformed_correlation_score(orig_df: DataFrame, synth_df: DataFrame,
                                      categorical_cols: Optional[List[str]] = None, clip_val: Optional[int] = 1) \
        -> float:
    """
    Computes the log-transformed correlation score between two datasets.

    Given two datasets `orig_df` and `synth_df`, this function first calculates their correlation matrices using
    Pearson's R for continuous-continuous cases, Correlation Ratio for categorical-continuous cases and Theil's U
    for categorical-categorical cases. It then takes the absolute value of each correlation coefficient, applies a
    logarithm transformation to it (with zeros replaced by NaNs), and multiplies the result by the sign of the
    original correlation. The relative error between the log-transformed correlation matrices is then computed, and
    optionally clipped to a maximum value specified by `clip_val`. Finally, the score is calculated as 1 minus the
    mean of the relative errors, where a higher score (highest 1, lowest 0) indicates a better match between the two
    datasets.

    Does not work well for nearly perfect correlations (~1)!

    :param orig_df: The original dataset, as a pandas DataFrame.
    :param synth_df: The synthetic dataset, as a pandas DataFrame.
    :param categorical_cols: The column names that should be treated as categorical.
    :param clip_val: An optional maximum value to clip the relative errors to. If None, no clipping is applied.

    :return: The log-transformed correlation overlap score between the two datasets. A score of 1 indicates a perfect
                 match, while a score of 0 indicates no correlation overlap at all.
    """

    if categorical_cols is None:
        categorical_cols = 'auto'

    orig_corr = _comp_assoc(orig_df, categorical_cols, False, theil_u=True, clustering=False, bias_correction=True,
                            nan_strategy='replace', nan_replace_value=-1)[0]
    synth_corr = _comp_assoc(synth_df, categorical_cols, False, theil_u=True, clustering=False, bias_correction=True,
                             nan_strategy='replace', nan_replace_value=-1)[0]
    for col in orig_corr.columns:
        orig_corr[col] = orig_corr[col].astype(float)
        synth_corr[col] = synth_corr[col].astype(float)

    # Signs for the correlation matrices
    sign_orig_corr = np.sign(orig_corr)
    sign_synth_corr = np.sign(synth_corr)

    # Logn of the correlation matrices
    log_orig_corr = np.log(abs(orig_corr), out=np.zeros_like(orig_corr), where=(orig_corr != 0))
    log_synth_corr = np.log(abs(synth_corr), out=np.zeros_like(synth_corr), where=(synth_corr != 0))

    # Relative errors in correlation matrices
    rel_errors = abs((((sign_orig_corr * log_orig_corr) - (sign_synth_corr * log_synth_corr)) /
                      (sign_orig_corr * log_orig_corr)))

    # Clip the relative errors to a maximum value
    if clip_val is not None:
        rel_errors = rel_errors.clip(upper=clip_val)

    # calculate the mean, 1 highest value, 0 lowest value
    score = 1 - rel_errors.mean().mean()

    return score


def correlation_sum_score(orig_df: DataFrame, synth_df: DataFrame, categorical_cols: Optional[List[str]] = None) \
        -> float:
    """
    Calculate the correlation sum score between two dataframes using correlation matrices.

    :param orig_df: The original dataframe to compare.
    :param synth_df: The synthetic dataframe to compare.
    :param categorical_cols: Optional: A list of column names that are categorical. Defaults to None. If `None`, the
                             function will try to detect the categorical columns automatically.

    :return: The correlation sum score between the two dataframes. This is the mean of the absolute differences
             between the correlation matrices of the two dataframes.
    """

    if categorical_cols is None:
        categorical_cols = 'auto'

    # Calculate the correlation matrices for the two datasets
    orig_corr = associations(orig_df, plot=False, theil_u=True, nominal_columns=categorical_cols)['corr']
    synth_corr = associations(synth_df, plot=False, theil_u=True, nominal_columns=categorical_cols)['corr']

    # Relative errors in correlation matrices
    rel_errors = abs(orig_corr - synth_corr)

    # calculate the mean, 1 highest value, 0 lowest value
    return rel_errors.mean().mean()


def regularized_support_coverage(orig_df: DataFrame, synth_df: DataFrame, categorical_cols: List[str],
                                 clip_ratio: float = 1, clip_col: int = 1, include_num: bool = True,
                                 num_bins: int = 10) -> float:
    """
    Calculates the regularized support coverage metric for synthetic data generation evaluation.

    :param orig_df: Original dataset used as the basis for generating the synthetic data.
    :param synth_df: Synthetic dataset generated by a data synthesizer model.
    :param categorical_cols: List of names of the categorical columns in the original dataset.
    :param clip_ratio: The maximum value to which a ratio can be clipped. Defaults to 1.
    :param clip_col: The maximum value to which the support coverage of a column can be clipped. Defaults to 1.
    :param include_num: Indicates whether to include numerical columns in the calculation. Defaults to True.
    :param num_bins: The number of bins to be used when creating histogram-like buckets for numerical columns. Defaults to 10.

    :return: The final regularized support coverage metric.
    """

    # depending on the size of synthetic data
    scaling_factor = len(orig_df) / len(synth_df)

    # sum of all support coverage metrics
    sum_support_coverage = 0

    for col in categorical_cols:
        # for each categorical column get the count of each value for real and synthetic data
        orig_val_counts = orig_df[col].value_counts(sort=False, dropna=False)
        synth_val_counts = synth_df[col].value_counts(sort=False, dropna=False)

        # the number of unique values in the real data
        n = len(orig_val_counts.keys())
        support_cov = 0

        for key in orig_val_counts.keys():
            # if value not in synthetic data, it counts as 0
            if key not in synth_val_counts.keys():
                continue

            # determine the ratio of synthetic and real data samples coverage, use clip_ratio as maximum
            # because of special int type, nan values are always part of this, even if there are 0 values
            if orig_val_counts[key] == 0:
                n = n - 1
                continue
            ratio = (synth_val_counts[key] / orig_val_counts[key]) * scaling_factor

            if ratio > clip_ratio:
                ratio = clip_ratio

            support_cov += ratio

        # calculate the support coverage for a single variable
        support_cov /= n

        # if the support coverage is higher than clip_col, clip it to this value
        if support_cov > clip_col:
            support_cov = clip_col

        sum_support_coverage += support_cov

    if include_num:
        # all columns that are not categorical are considered numerical
        num_cols = [col for col in orig_df.columns if col not in categorical_cols]

        for col in num_cols:
            # get the cutoff values for num_bins for each numerical columns
            cut_offs = pd.qcut(orig_df[col], q=num_bins, retbins=True, duplicates='drop')[1]
            support_cov = 0

            for i in range(len(cut_offs)-1):
                # get the min value and max value for each cutoff
                min_val = cut_offs[i]
                max_val = cut_offs[i + 1]

                # closed buckets on the left and right
                # in case of the last bucket the maximum value is inclusive, otherwise not
                # count for each bucket how many real and synthetic samples there are
                if (i + 1) == len(cut_offs)-1:
                    orig_bucket_count = len(orig_df[(orig_df[col] >= min_val) & (orig_df[col] <= max_val)])
                    synth_bucket_count = len(synth_df[(synth_df[col] >= min_val) & (synth_df[col] <= max_val)])
                else:
                    orig_bucket_count = len(orig_df[(orig_df[col] >= min_val) & (orig_df[col] < max_val)])
                    synth_bucket_count = len(synth_df[(synth_df[col] >= min_val) & (synth_df[col] < max_val)])

                # calculate the ratio of synthetic datapoints to real datapoints
                ratio = (synth_bucket_count / orig_bucket_count) * scaling_factor

                # clip the ratio if it is higher than clip_ratio
                if ratio > clip_ratio:
                    ratio = clip_ratio

                support_cov += ratio

            # calculate support coverage for a singe numerical variable
            support_cov /= (len(cut_offs)-1)

            # if the support coverage exceeds clip_col, then set it to clip_col
            if support_cov > clip_col:
                support_cov = clip_col

            sum_support_coverage += support_cov

        # divide the sum of all support coverage values with the number of all columns (since all are used) for the
        # final metric
        reg_support_cov = sum_support_coverage / len(orig_df.columns)
    else:
        # divide the sum of all support coverage values with the number of all categorical columns for the final metric
        reg_support_cov = sum_support_coverage / len(categorical_cols)

    return reg_support_cov


def discriminator_measure_rf(orig_df: DataFrame, synth_df: DataFrame, test_ratio: float = 0.2) -> float:
    """
    This function measures the discrimination power of a random forest classifier to distinguish between real and
    synthetic data.

    :param orig_df: The original dataset.
    :param synth_df: The synthetic dataset.
    :param test_ratio: The ratio of test data to split from the combined real and synthetic data. Defaults to 0.2.

    :return: The discrimination score ranging from 0 (worst) to 1 (best).
    """

    # copy the data
    orig_df = copy.deepcopy(orig_df)
    synth_df = copy.deepcopy(synth_df)

    if len(synth_df) > len(orig_df):
        synth_df = synth_df.sample(n=len(orig_df), random_state=42).reset_index(drop=True)
    elif len(orig_df) > len(synth_df):
        orig_df = orig_df.sample(n=len(synth_df), random_state=42).reset_index(drop=True)

    # preprocessing
    # select non-numerical columns (ignoring booleans)
    non_numeric_cols = orig_df.select_dtypes(exclude=[int, float]).columns

    df = pd.concat([orig_df, synth_df], axis=0)
    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder because random forest is used
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(df[col])
        # transform real and synthetic data column-wise
        orig_df[col] = encoder.transform(orig_df[col])
        synth_df[col] = encoder.transform(synth_df[col])

    # determine columns containing NaN values
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        orig_df.fillna(-9999999999, inplace=True)
        synth_df.fillna(-9999999999, inplace=True)

    # add extra column to show which of the datapoints is real and which is not
    orig_df['real_point'] = 1
    synth_df['real_point'] = 0

    # split real and synth data into train and test, size depends on the provided test_ratio
    orig_df_train, orig_df_test, synth_df_train, synth_df_test = train_test_split(orig_df, synth_df, random_state=42,
                                                                                  test_size=test_ratio)

    # concatenate real and synth train dataframes into one dataframe
    df_train = pd.concat([orig_df_train, synth_df_train], ignore_index=True, sort=False)
    # concatenate real and synth test dataframes into one dataframe
    df_test = pd.concat([orig_df_test, synth_df_test], ignore_index=True, sort=False)

    # shuffle train dataset
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # get y for train and test data
    y_train = df_train['real_point']
    y_test = df_test['real_point']

    # remove y from X train and test
    df_train.drop(['real_point'], axis=1, inplace=True)
    df_test.drop(['real_point'], axis=1, inplace=True)

    # initiate a RandomForestClassifier with default parameters
    rf_model = RandomForestClassifier(random_state=42)
    # fit the model on the training data
    rf_model.fit(df_train, y_train)
    # predict on the test data
    y_pred = rf_model.predict(df_test)
    # calculate the accuracy for the test data
    accuracy = accuracy_score(y_test, y_pred)

    # best case 0.5, if it is lower than that, it means that the classifier cannot distinguish between real and synth
    # datapoint, therefore, we set the value to the ideal 0.5
    if accuracy < 0.5:
        accuracy = 0.5

    # 1 best, 0 worst
    score = 1 - (accuracy - 0.5) * 2
    return score


def k_means_score(orig_df: DataFrame, synth_df: DataFrame, k: int = 10, clip_ratio: float = 1):
    """
    Calculate the k-means score, a measure of how well the synthetic data matches the real data, for a given pair of
    original and synthetic dataframes. The k-means algorithm is a commonly used unsupervised machine learning
    technique used to cluster data points into k groups based on their similarity. This function applies the k-means
    algorithm to the original data and calculates the coverage of the synthetic data for each cluster, where
    coverage refers to the ratio of the number of synthetic data points to the number of original data points in the
    cluster.

    :param orig_df: The original dataframe.
    :param synth_df: The synthetic dataframe.
    :param k: The number of clusters to use for the KMeans algorithm. Defaults to 10.
    :param clip_ratio: The maximum ratio of synthetic to real data samples coverage for each cluster. Defaults to 1.

    :return: The k-means score, which ranges from 0 to clip_ratio, with higher values indicating a better match
    between the original and synthetic data.
    """

    # depending on the size of synthetic data
    scaling_factor = len(orig_df) / len(synth_df)

    # copy data
    orig_df = copy.deepcopy(orig_df)
    synth_df = copy.deepcopy(synth_df)

    # get different types of columns
    num_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns if len(orig_df[col].unique()) > 3]
    non_numeric_cols = list(orig_df.select_dtypes(exclude=[int, float]).columns)
    bool_missing_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns
                         if len(orig_df[col].unique()) == 3]
    non_numeric_cols += bool_missing_cols

    # special handling of columns containing NaN
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()
    num_nan_cols = []
    if len(nan_columns) > 0:
        num_nan_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns if col in nan_columns]
        cat_nan_cols = [col for col in nan_columns if col not in num_nan_cols]

        # replace NaN value with a very large numerical value, so it can be perceived as an outlier
        for col in num_nan_cols:
            orig_df[col].fillna(-9999999999, inplace=True)
            synth_df[col].fillna(-9999999999, inplace=True)

        for col in cat_nan_cols:
            orig_df[col].fillna('empty', inplace=True)
            synth_df[col].fillna('empty', inplace=True)

        # boolean values will be handled with OneHotEncoder
        num_nan_cols = [col for col in num_nan_cols if col not in non_numeric_cols]

    if len(non_numeric_cols) > 0:
        # use OneHotEncoder for categorical (and boolean with missing values) columns
        oe_scaler = OneHotEncoder()
        oe_scaler.fit(orig_df[non_numeric_cols])

        train_cat = oe_scaler.transform(orig_df[non_numeric_cols]).toarray()
        orig_df.drop(non_numeric_cols, axis=1, inplace=True)

        synth_cat = oe_scaler.transform(synth_df[non_numeric_cols]).toarray()
        synth_df.drop(non_numeric_cols, axis=1, inplace=True)
    else:
        train_cat = None
        synth_cat = None

    train_num_nan = None
    synth_num_nan = None

    # since NaN values are replaced with large numbers, these columns are encoded with RobustScaler
    if len(num_nan_cols) > 0:
        # remove the columns with NaN values from num_cols
        num_cols = [col for col in num_cols if col not in num_nan_cols]

        rs_scaler = RobustScaler()
        rs_scaler.fit(orig_df[num_nan_cols])

        train_num_nan = rs_scaler.transform(orig_df[num_nan_cols])
        orig_df.drop(num_nan_cols, axis=1, inplace=True)

        synth_num_nan = rs_scaler.transform(synth_df[num_nan_cols])
        synth_df.drop(num_nan_cols, axis=1, inplace=True)

    if len(num_cols) > 0:
        # use StandardScaler for all non NaN numerical columns
        std_scaler = StandardScaler()
        std_scaler.fit(orig_df[num_cols])

        train_num = std_scaler.transform(orig_df[num_cols])
        orig_df.drop(num_cols, axis=1, inplace=True)

        synth_num = std_scaler.transform(synth_df[num_cols])
        synth_df.drop(num_cols, axis=1, inplace=True)
    else:
        train_num = None
        synth_num = None

    # combine all transformed features to a numpy array for real and synthetic data

    arrays_to_concatenate = [arr for arr in [orig_df.to_numpy(), train_cat, train_num, train_num_nan] if arr is not None]
    arrays_to_concatenate_synth = [arr for arr in [synth_df.to_numpy(), synth_cat, synth_num, synth_num_nan] if arr is not None]

    real = np.concatenate(arrays_to_concatenate, axis=1)
    synth = np.concatenate(arrays_to_concatenate_synth, axis=1)

    # initiate KMeans model with k clusters
    model = KMeans(n_clusters=k, n_init=10)
    # fit model to the real data
    model.fit(real)
    # get the labels for each datapoint for the real and synthetic data
    real_clusters = list(model.predict(real))
    synth_clusters = list(model.predict(synth))

    # sum of the support coverage metric for all the clusters
    sum_val = 0
    # for each cluster determine the ratio of synthetic and real data samples coverage, use clip_ratio as maximum
    for i in range(k):
        real_count = real_clusters.count(i)
        synth_count = synth_clusters.count(i)

        ratio = (synth_count / real_count) * scaling_factor

        if ratio > clip_ratio:
            ratio = clip_ratio

        sum_val += ratio

    # total score is the sum of all the ratios (to a maximum of clip_ratio), divided by the number of clusters
    # higher number is better, 0 worst
    score = sum_val / k
    return score


def ml_efficiency_cat(orig_df_train: DataFrame, orig_df_test: DataFrame, synth_df_train: DataFrame, predict_col: str,
                      metric: Literal['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc'], relative: bool,
                      outcome_cols: List[str], training_params: Dict, create_column_lambda: Optional[str] = None) \
        -> float:
    """
    Evaluate the efficiency of a machine learning model trained on synthetic data as compared to real data using
    CatBoostClassifier.

    :param orig_df_train: Training set with original data.
    :param orig_df_test: Test set with original data.
    :param synth_df_train: Synthetic data with the same size as the original training dataset.
    :param predict_col: The column name in the dataframes that contains the target variable to predict.
    :param metric: The evaluation metric to use. Options are 'f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', and 'mcc'.
    :param relative: If True, returns the relative efficiency (score on synthetic / score on original). If score on synthetic
                     is better, returns 1.
    :param outcome_cols: List of columns in the dataframe that indicate outcome variables.
    :param training_params: Dictionary of training parameters to pass to CatBoostClassifier.
    :param create_column_lambda: An optional lambda expression string to create the predict_col if not present in
                                 the dataframe.

    :return: A score representing the efficiency of the model trained on synthetic data between 0 and 1.
    """

    # copy the data
    orig_df_train = copy.deepcopy(orig_df_train)
    orig_df_test = copy.deepcopy(orig_df_test)
    synth_df_train = copy.deepcopy(synth_df_train)

    if predict_col not in orig_df_train.columns:
        # use create_column_lambda expression for the creation
        orig_df_train[predict_col] = orig_df_train.apply(eval(create_column_lambda), axis=1)
        orig_df_test[predict_col] = orig_df_test.apply(eval(create_column_lambda), axis=1)
        synth_df_train[predict_col] = synth_df_train.apply(eval(create_column_lambda), axis=1)

    # remove outcome cols
    columns_to_remove = [col for col in outcome_cols if col != predict_col]
    orig_df_train.drop(columns_to_remove, axis=1, inplace=True)
    orig_df_test.drop(columns_to_remove, axis=1, inplace=True)
    synth_df_train.drop(columns_to_remove, axis=1, inplace=True)

    if str(orig_df_train[predict_col].dtype).startswith('Int') and len(orig_df_train[predict_col].unique()) <= 3:
        # change type to real int, because scores seem not to handle this int type well
        orig_df_train[predict_col] = orig_df_train[predict_col].astype(int)
        orig_df_test[predict_col] = orig_df_test[predict_col].astype(int)
        synth_df_train[predict_col] = synth_df_train[predict_col].astype(int)

        # in case 0 is less probable change it to 1 for a proper f1 score
        if metric == 'f1' and \
                orig_df_train[predict_col].value_counts()[1] > orig_df_train[predict_col].value_counts()[0]:
            orig_df_train[predict_col] = orig_df_train[predict_col].apply(lambda x: 1 if x == 0 else 0)
            orig_df_test[predict_col] = orig_df_test[predict_col].apply(lambda x: 1 if x == 0 else 0)
            synth_df_train[predict_col] = synth_df_train[predict_col].apply(lambda x: 1 if x == 0 else 0)

    # needed for the fitting of encoders
    orig_df = pd.concat([orig_df_train, orig_df_test], ignore_index=True, sort=False)

    # preprocessing
    # select non-numerical columns (ignoring booleans)
    non_numeric_cols = orig_df.select_dtypes(exclude=[int, float]).columns

    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(orig_df[col])
        # transform real and synthetic data column-wise
        orig_df_train[col] = encoder.transform(orig_df_train[col])
        orig_df_test[col] = encoder.transform(orig_df_test[col])
        synth_df_train[col] = encoder.transform(synth_df_train[col])
        orig_df.drop([col], axis=1, inplace=True)

    # determine columns containing NaN values
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        orig_df_train.fillna(-9999999999, inplace=True)
        orig_df_test.fillna(-9999999999, inplace=True)
        synth_df_train.fillna(-9999999999, inplace=True)

    # get y for train and test data
    orig_y_train = orig_df_train[predict_col]
    orig_y_test = orig_df_test[predict_col]
    synth_y_train = synth_df_train[predict_col]

    # remove y
    orig_df_train.drop([predict_col], axis=1, inplace=True)
    orig_df_test.drop([predict_col], axis=1, inplace=True)
    synth_df_train.drop([predict_col], axis=1, inplace=True)

    # initiate a CatBoostClassifier with provided parameters for synthetic data
    gb_model_synth = CatBoostClassifier(**training_params, verbose=False, allow_writing_files=False)

    # fit the model on the synthetic data
    if len(synth_y_train.unique()) == 1:
        # to not get an error we change the last y to the missing category
        # get the missing item
        missing_items = set(orig_y_train.unique()) - set(synth_y_train.unique())
        if len(missing_items) == 1:
            synth_y_train = synth_y_train.copy()
            synth_y_train.iloc[-1] = list(missing_items)[0]

    gb_model_synth.fit(synth_df_train, synth_y_train)

    # predict on the test data
    y_pred_synth = gb_model_synth.predict(orig_df_test)

    # calculate the chosen metric for the test data
    if metric == 'f1':
        score_synth = f1_score(orig_y_test, y_pred_synth)
    elif metric == 'f1_micro':
        score_synth = f1_score(orig_y_test, y_pred_synth, average='micro')
    elif metric == 'f1_macro':
        score_synth = f1_score(orig_y_test, y_pred_synth, average='macro')
    elif metric == 'roc_auc':
        score_synth = roc_auc_score(orig_y_test, y_pred_synth)
    elif metric == 'accuracy':
        score_synth = accuracy_score(orig_y_test, y_pred_synth)
    elif metric == 'mcc':
        score_synth = matthews_corrcoef(orig_y_test, y_pred_synth)
    else:
        raise AttributeError(f'The evaluation metric must be one of the following: '
                             f'{", ".join(["f1", "accuracy", "roc_auc", "f1_macro", "f1_micro"])}, '
                             f'but {metric} was provided')

    if relative:
        gb_model_orig = CatBoostClassifier(**training_params, verbose=False, allow_writing_files=False)
        gb_model_orig.fit(orig_df_train, orig_y_train)
        y_pred_orig = gb_model_orig.predict(orig_df_test)

        if metric == 'f1':
            score_orig = f1_score(orig_y_test, y_pred_orig)
        elif metric == 'f1_micro':
            score_orig = f1_score(orig_y_test, y_pred_orig, average='micro')
        elif metric == 'f1_macro':
            score_orig = f1_score(orig_y_test, y_pred_orig, average='macro')
        elif metric == 'roc_auc':
            score_orig = roc_auc_score(orig_y_test, y_pred_orig)
        elif metric == 'accuracy':
            score_orig = accuracy_score(orig_y_test, y_pred_orig)
        elif metric == 'mcc':
            score_synth = matthews_corrcoef(orig_y_test, y_pred_synth)

        # relative score
        score = score_synth / score_orig
        # in case synthetic score is better than the original one
        if score > 1:
            score = 1

        return score

    # otherwise return the actual score
    return score_synth
