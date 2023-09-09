import json
import os
import pandas as pd
from pandas import DataFrame
from typing import Optional, Tuple, Literal
from src.base import PROJECT_DIR


def load_columns_json_for_experiment(experiment_name: str, for_training=True):
    """
    Load column information from a JSON file based on the experiment name.

    :param experiment_name: The name of the experiment.
    :param for_training: Whether the columns info is for training or evaluation.
    :return: Dictionary containing column information.
    """

    folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name, 'data')
    if not for_training and os.path.exists(os.path.join(folder_path, 'columns_orig.json')):
        filename = 'columns_orig.json'
    else:
        filename = 'columns.json'
    with open(os.path.join(folder_path, filename), 'r') as fh:
        columns_json = json.load(fh)
        return columns_json


def load_catboost_params(experiment_name: str, filename: str):
    """
    Load parameters for CatBoost from a JSON file based on the experiment name.

    :param experiment_name: The name of the experiment.
    :param filename: The name of the file containing the parameters.
    :return: Dictionary containing CatBoost parameters.
    """

    filename = filename if filename.endswith('.json') else filename + '.json'
    folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name,
                               'eval_optim')
    with open(os.path.join(folder_path, filename), 'r') as fh:
        params = json.load(fh)
        return params


def load_experiment_data(experiment_name: str, for_training=True) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load data associated with an experiment.

    :param experiment_name: The name of the experiment.
    :param for_training: Whether to load the data for training or evaluation purposes.
    :return: Tuple containing DataFrames for full data, training data, and test data.
    """

    def get_dtype_for_bool(bool_type: Literal['int', 'bool', 'str']):
        if bool_type == 'bool':
            return bool
        elif bool_type == 'int':
            # handles NaN values
            return pd.Int64Dtype()
        elif bool_type == 'float':
            return float
        else:
            return str

    folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name, 'data')
    # specific for autoencoder or similar experiments
    if not for_training and os.path.exists(os.path.join(folder_path, 'data_orig.csv')):
        column_json_filename = 'columns_orig.json'
        filenames = ['data_orig.csv', 'train_orig.csv', 'test_orig.csv']
    else:
        column_json_filename = 'columns.json'
        filenames = ['data.csv', 'train.csv', 'test.csv']

    columns_dtypes = None

    if os.path.exists(os.path.join(folder_path, column_json_filename)):
        with open(os.path.join(folder_path, column_json_filename), 'r') as fh:
            columns_json = json.load(fh)
            columns_dtypes_bool = {col: get_dtype_for_bool(columns_json['boolean_dtype'])
                                   for col in columns_json['boolean']}
            columns_dtypes_int = {col: pd.Int64Dtype() for col in columns_json['integer']}
            columns_dtypes_float = {col: float for col in columns_json['float']}
            columns_dtypes_cat = {col: str for col in columns_json['categorical']}
            columns_dtypes = {**columns_dtypes_cat, **columns_dtypes_bool, **columns_dtypes_int, **columns_dtypes_float}

    df = pd.read_csv(os.path.join(folder_path, filenames[0]), dtype=columns_dtypes)
    df_train = pd.read_csv(os.path.join(folder_path, filenames[1]), dtype=columns_dtypes)
    df_test = pd.read_csv(os.path.join(folder_path, filenames[2]), dtype=columns_dtypes)

    # correct integer columns that do not have NaN values
    if columns_json['boolean_dtype'] == 'int':
        for col in columns_json['boolean']:
            if not df[col].isnull().values.any():
                df[col] = df[col].astype(int)
                df_train[col] = df_train[col].astype(int)
                df_test[col] = df_test[col].astype(int)
    for col in columns_json['integer']:
        if not df[col].isnull().values.any():
            df[col] = df[col].astype(int)
            df_train[col] = df_train[col].astype(int)
            df_test[col] = df_test[col].astype(int)

    if not for_training and os.path.exists(os.path.join(folder_path, 'eval_preprocess_attr.txt')):
        with open(os.path.join(folder_path, 'eval_preprocess_attr.txt'), mode='r') as fh:
            for lambda_expr in fh.readlines():
                lambda_expr = lambda_expr.strip()
                if lambda_expr == '':
                    continue
                df = df.apply(lambda row: (exec(lambda_expr, globals(), {'row': row}), row)[1], axis=1)
                df_train = df_train.apply(lambda row: (exec(lambda_expr, globals(), {'row': row}), row)[1], axis=1)
                df_test = df_test.apply(lambda row: (exec(lambda_expr, globals(), {'row': row}), row)[1], axis=1)

    return df, df_train, df_test


def get_categorical_columns_from_df(df: DataFrame) -> list[str]:
    """
    Extract categorical columns (including boolean) from a DataFrame.

    :param df: Input DataFrame.
    :return: List of categorical (including boolean) column names.
    """

    categories = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categories.append(col)
        # boolean values (int), 3 categories in cases of nan values
        elif len(df[col].value_counts()) <= 3:
            categories.append(col)
    return categories


def get_integer_columns_from_df(df: DataFrame) -> list[str]:
    """
    Extract integer columns from a DataFrame.

    :param df: Input DataFrame.
    :return: List of integer column names.
    """

    columns = []
    for col in df.columns:
        if df[col].dtype == 'int64':
            columns.append(col)
    return columns


def get_boolean_columns_from_df(df: DataFrame) -> list[str]:
    """
    Extract boolean columns (without NaN values) from a DataFrame.

    :param df: Input DataFrame.
    :return: List of boolean column names (without NaN values).
    """

    columns = []
    for col in df.columns:
        if len(df[col].value_counts()) == 2:
            columns.append(col)
    return columns


def get_possible_categorical_values_from_df(df: DataFrame) -> list[[str]]:
    """
    Obtain the possible categorical values (excluding booleans) sorted by frequency.

    :param df: Input DataFrame.
    :return: List of categorical values (excluding boolean) sorted by their frequency.
    """

    cols_cat = get_categorical_columns_from_df(df)
    cols_bool = get_boolean_columns_from_df(df)
    cols_cat = [c for c in cols_cat if c not in cols_bool]

    results = [sorted(list(df[col].value_counts().keys())) for col in cols_cat]
    return results


def remove_other_outcome_cols(df: DataFrame, target_col: Optional[str] = None,
                              outcome_cols: Optional[list[str]] = None) -> DataFrame:
    """
    Removes outcome variables from the DataFrame, preserving the target column.

    :param df: Input DataFrame.
    :param target_col: The target column to be predicted.
    :param outcome_cols: List of outcome columns to be removed.
    :return: DataFrame without specified outcome columns except the target column.
    """

    if outcome_cols is None:
        outcome_cols = ['CR1', 'OSTM', 'OSSTAT', 'EFSTM', 'EFSSTAT', 'RFSTM', 'RFSSTAT']
    cols = [col for col in df.columns if col not in outcome_cols or col == target_col]
    return df[cols]


def add_ending_to_filename(filename: str, extension: str) -> str:
    """
    Appends an extension to the filename if it does not already have it.

    :param filename: Input filename.
    :param extension: The extension to be added.
    :return: Modified filename with the given extension.
    """
    if not filename.endswith(extension):
        filename += extension
    return filename
