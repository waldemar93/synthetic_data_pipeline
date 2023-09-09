import json
import pickle
from dataclasses import dataclass, MISSING
import hydra
import numpy as np
import omegaconf
import pandas as pd
from typing import Literal, Optional, Tuple, Union, Any, List, Dict
import os
from omegaconf import OmegaConf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.base import PROJECT_DIR


@dataclass
class ProcessConfig:
    """Configuration class for data preprocessing."""
    raw_filename: str = MISSING
    remove_columns: List[str] = MISSING
    remove_rows_for_nan_column: List[str] = MISSING
    map_columns: Dict[str, Dict[str, Union[int, float, str]]] = MISSING
    cat_columns: List[str] = MISSING
    bool_columns: List[str] = MISSING
    bool_column_datatype: Literal["int", "bool", "str", "float"] = MISSING
    int_columns: List[str] = MISSING
    float_columns: List[str] = MISSING
    outcome_columns: List[str] = MISSING
    nan_cat_value: Optional[str] = MISSING
    nan_bool_value: Optional[Union[int, str]] = MISSING
    nan_int_value: Optional[int] = MISSING
    nan_cont_value: Optional[float] = MISSING
    scaler_for_num_col: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = MISSING
    experiment_name: str = MISSING
    overwrite_data_folder_if_exists: bool = MISSING
    create_columns_json: bool = MISSING
    create_folder_for_generative_models: List[str] = MISSING
    split_cols: Union[List[str], str] = MISSING
    split_seed: int = MISSING
    test_ratio: float = MISSING
    remove_rows_for_lambda_conditions: Optional[List[str]] = MISSING
    preprocess_lambda: Optional[List[str]] = MISSING
    preprocess_lambda_for_eval: Optional[List[str]] = MISSING


@dataclass
class MainProcessConfig:
    """Main preprocess configuration class containing a ProcessConfig instance."""
    process: ProcessConfig = MISSING


def preprocess_data(df: DataFrame, remove_columns: List[str], remove_rows_for_nan_column: List[str],
                    map_columns: Dict[str, Dict[str, Union[int, float, str]]], cat_columns: List[str],
                    bool_columns: List[str],
                    bool_column_datatype: Literal["int", "bool", "str", "float"], int_columns: List[str],
                    float_columns: List[str], nan_cat_value: Optional[str], nan_bool_value: Optional[Union[int, str]],
                    nan_int_value: Optional[int], nan_cont_value: Optional[float],
                    scaler_for_num_col: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]],
                    remove_rows_for_lambda_conditions: Optional[List[str]] = None,
                    preprocess_lambda: Optional[List[str]] = None) \
        -> Tuple[DataFrame, Any]:
    """
    Preprocess the provided dataframe based on given conditions.

    :param df: Input dataframe to be preprocessed.
    :param remove_columns: List of columns to be removed.
    :param remove_rows_for_nan_column: Rows to be removed based on NaN values in specified columns.
    :param map_columns: Dictionary specifying mappings for columns.
    :param cat_columns: List of categorical columns.
    :param bool_columns: List of boolean columns.
    :param bool_column_datatype: Datatype to use for boolean columns.
    :param int_columns: List of integer columns.
    :param float_columns: List of float columns.
    :param nan_cat_value: Optional: Replacement value for NaN values in categorical columns.
    :param nan_bool_value: Optional: Replacement value for NaN values in boolean columns.
    :param nan_int_value: Optional: Replacement value for NaN values in integer columns.
    :param nan_cont_value: Optional: Replacement value for NaN values in continuous columns.
    :param scaler_for_num_col: Optional: Scaler to use for numeric columns
    :param remove_rows_for_lambda_conditions: Optional: List of conditions to remove rows using lambda functions.
    :param preprocess_lambda: Optional: List of lambda functions for preprocessing
    (will be executed before the data will be loaded).
    :return: A tuple of the transformed DataFrame and the scaler used (if any), otherwise None.
    """

    def bool_transform(x):
        if bool_column_datatype == 'bool':
            return bool(x)
        elif bool_column_datatype == 'int':
            return int(x)
        elif bool_column_datatype == 'float':
            return float(x)
        else:
            return str(x)

    df.drop(remove_columns, axis=1, inplace=True)

    # test if all columns are mentioned
    assert (len(set(df.columns) - set(bool_columns + cat_columns + int_columns + float_columns)) == 0)
    assert (len(set((bool_columns + cat_columns + int_columns + float_columns)) - set(df.columns)) == 0)

    for col in map_columns:
        df[col] = df[col].map(map_columns[col])
    # remove all nan rows
    df.dropna(axis=0, how='all', inplace=True)
    # for remaining continuous columns: remove empty rows
    for col in remove_rows_for_nan_column:
        df = df[df[col].notna()]

    # remove rows depending on condition
    if remove_rows_for_lambda_conditions is not None and remove_rows_for_lambda_conditions != '' \
            and remove_rows_for_lambda_conditions != 'None':
        for cond in remove_rows_for_lambda_conditions:
            df = df.loc[df.apply(lambda row: not (eval(cond)), axis=1)]

    if nan_bool_value is None or nan_bool_value == 'None':
        nan_bool_value = np.NaN
    if nan_cat_value is None or nan_cat_value == 'None':
        nan_cat_value = np.NaN
    if nan_int_value is None or nan_int_value == 'None':
        nan_int_value = np.NaN
    if nan_cont_value is None or nan_cont_value == 'None':
        nan_cont_value = np.NaN

    for col in bool_columns:
        df[col] = df[col].apply(lambda x: nan_bool_value if pd.isna(x) else bool_transform(x))
        # workaround for ints with NaN, otherwise the variable ends up being a float
        if bool_column_datatype == 'int' and df[col].isnull().values.any():
            df[col] = df[col].astype(pd.Int64Dtype())

    # convert float variables to int (real int)
    for col in int_columns:
        df[col] = df[col].apply(lambda x: nan_int_value if pd.isna(x) else int(x))
        # workaround for ints with NaN, otherwise the variable ends up being a float
        if df[col].isnull().values.any():
            df[col] = df[col].astype(pd.Int64Dtype())

    for col in float_columns:
        df[col] = df[col].apply(lambda x: nan_cont_value if pd.isna(x) else float(x))

    for col in cat_columns:
        df[col] = df[col].apply(lambda x: nan_cat_value if pd.isna(x) else str(x))

    # preprocessing lambda expressions
    if preprocess_lambda is not None and preprocess_lambda != 'None' and preprocess_lambda != '':
        for lambda_expr in preprocess_lambda:
            df = df.apply(lambda row: (exec(lambda_expr, globals(), {'row': row}), row)[1], axis=1)

    numerical_columns = float_columns + int_columns
    numerical_idx = [df.columns.get_loc(c) for c in numerical_columns]
    scaler = None
    if scaler_for_num_col == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_for_num_col == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_for_num_col == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_for_num_col is not None and scaler_for_num_col != 'None':
        raise NotImplementedError(f'Scaler: {scaler_for_num_col} is not yet implemented')

    if scaler is not None:
        transformed = scaler.fit_transform(df.iloc[:, numerical_idx])
        df.iloc[:, numerical_idx] = transformed
    return df, scaler


def create_experiment_structure(experiment_name: str, overwrite_data_folder_if_exists: bool, create_columns_json: bool,
                                generative_models: List[str], df: DataFrame, num_scaler: Any, df_train: DataFrame,
                                df_test: DataFrame, cols_bool: Optional[List[str]] = None,
                                cols_int: Optional[List[str]] = None, cols_float: Optional[List[str]] = None,
                                cols_cat: Optional[List[str]] = None, cols_outcome: Optional[List[str]] = None,
                                bool_datatype: Optional[str] = None,
                                preprocess_lambda_for_eval: Optional[List[str]] = None) -> None:
    """
       Creates a structured experiment directory with data and configurations.

       :param experiment_name: Name of the experiment.
       :param overwrite_data_folder_if_exists: Flag to overwrite data folder if it exists.
       :param create_columns_json: Flag to create a columns JSON file. Should be usually always True.
       :param generative_models: List of generative model names.
       :param df: Dataframe containing all data after preprocessing.
       :param num_scaler: Optional: Numeric scaler used during preprocessing.
       :param df_train: Training data after preprocessing.
       :param df_test: Test data after preprocessing.
       :param cols_bool: List of boolean columns.
       :param cols_int: List of integer columns.
       :param cols_float: List of float columns.
       :param cols_cat: List of categorical columns.
       :param cols_outcome: List of outcome columns.
       :param bool_datatype: Datatype used for boolean columns.
       :param preprocess_lambda_for_eval: Optional: List of lambda functions for evaluation.
       :return: None
       """
    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', experiment_name)
    # create experiment name folder if it doesn't exist
    os.makedirs(exp_folder_path, exist_ok=True)
    # create data folder if it doesn't exist
    os.makedirs(os.path.join(exp_folder_path, 'data'), exist_ok=True)

    # in case data.csv does already exist and overwrite_data_folder_if_exists is False
    if os.path.exists(os.path.join(exp_folder_path, 'data', 'data.csv')) and not overwrite_data_folder_if_exists:
        raise AttributeError(f'Data for the experiment {experiment_name} already exists, but '
                             f'overwrite_data_folder_if_exists is {overwrite_data_folder_if_exists}.')

    # save data
    df.to_csv(os.path.join(exp_folder_path, 'data', 'data.csv'), index=False)
    df_train.to_csv(os.path.join(exp_folder_path, 'data', 'train.csv'), index=False)
    df_test.to_csv(os.path.join(exp_folder_path, 'data', 'test.csv'), index=False)

    # if eval lambda strings exist
    if preprocess_lambda_for_eval is not None and preprocess_lambda_for_eval != '' \
            and preprocess_lambda_for_eval != 'None':
        with open(os.path.join(exp_folder_path, 'data', 'eval_preprocess_attr.txt'), 'w') as fh:
            fh.writelines(preprocess_lambda_for_eval)

    if num_scaler is not None:
        with open(os.path.join(exp_folder_path, 'data', 'num_scaler.pkl'), 'wb') as fh:
            pickle.dump(num_scaler, fh)

    # create columns.json
    if create_columns_json:
        json_obj = {'boolean': list(cols_bool), 'categorical': list(cols_cat), 'integer': list(cols_int),
                    'float': list(cols_float), 'outcome': list(cols_outcome), 'boolean_dtype': str(bool_datatype)}

        with open(os.path.join(exp_folder_path, 'data', 'columns.json'), 'w') as fh:
            json.dump(json_obj, fh, indent=2)

    # create a folder for each generative model
    for model_name in generative_models:
        os.makedirs(os.path.join(exp_folder_path, model_name), exist_ok=True)

        # create a folder for each of the following strings
        for dir_name in ['config', 'synthetic_data', 'models', 'eval']:
            os.makedirs(os.path.join(exp_folder_path, model_name, dir_name), exist_ok=True)

        os.makedirs(os.path.join(exp_folder_path, 'eval_optim'), exist_ok=True)


def split_data_in_train_test(df: DataFrame, target_cols: Union[List[str], omegaconf.listconfig.ListConfig, str],
                             test_ratio: float, random_seed: int) -> Tuple[DataFrame, DataFrame]:
    """
    Splits the data into training and test sets according to either a specific target column or the combination of
    all boolean outcome variables.

    :param df: DataFrame to split.
    :param target_cols: Column or columns based on which the stratified split will be performed.
    :param test_ratio: The fraction size for the test set.
    :param random_seed: A specified random seed used for the split.
    :return: Tuple of training and test sets as DataFrame
    """

    data_rows_to_add = []

    # just one column
    if (type(target_cols) is list or type(target_cols) is omegaconf.listconfig.ListConfig) and len(target_cols) == 1:
        target_col = target_cols[0]

    elif type(target_cols) is list or type(target_cols) is omegaconf.listconfig.ListConfig:
        df['split'] = df.apply(lambda row: ''.join([str(row[col]) for col in target_cols]), axis=1)

        # if there is just one value for a given combination, add it to train set
        for key, value in zip(list(df['split'].value_counts().keys()), list(df['split'].value_counts())):
            if value == 1:
                i = df[df['split'] == key].index
                datapoint = df[df['split'] == key].copy()
                datapoint.drop(['split'], axis=1, inplace=True)
                data_rows_to_add.append(datapoint)
                df.drop(index=i, inplace=True)

        target_col = 'split'

    # just one column
    else:
        target_col = target_cols

    df_train, df_test = train_test_split(df, test_size=test_ratio, stratify=df[target_col], random_state=random_seed)

    if target_col == 'split':
        # remove the extra column
        df.drop(['split'], axis=1, inplace=True)
        df_train.drop(['split'], axis=1, inplace=True)
        df_test.drop(['split'], axis=1, inplace=True)

        # add removed datapoints to training
        if len(data_rows_to_add) > 0:
            df_train = pd.concat([df_train] + data_rows_to_add, ignore_index=True)
            df_train.reset_index()

    return df_train, df_test


def _load_df(path: str, **kwargs):
    """
    Load a DataFrame from a specified file path.

    :param path: The path to the file containing the data to be loaded.
    :param **kwargs: Additional keyword arguments to pass to the respective pandas data loading function.
    :return: A pandas DataFrame containing the loaded data.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path, **kwargs)
    elif path.endswith('xlsx'):
        return pd.read_excel(path, **kwargs)


@hydra.main(config_path="../config", config_name="1_process", version_base="1.1")
def process(cfg: MainProcessConfig) -> None:
    """
    Main processing function to preprocess data, split it into train and test sets,
    and create an experiment structure.

    This function uses the Hydra library for configuration management and assumes a certain
    directory structure based on the provided configuration.

    :param cfg: A MainProcessConfig instance containing all necessary preprocessing and experiment
                configuration details.
    :return: None
    """
    cfg: ProcessConfig = cfg.process
    print(OmegaConf.to_yaml(cfg))
    df = _load_df(os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'data', 'raw', cfg.raw_filename))

    df_processed, scaler = preprocess_data(df, remove_columns=cfg.remove_columns if 'remove_columns' in cfg else [],
                                           remove_rows_for_nan_column=cfg.remove_rows_for_nan_column if 'remove_rows_for_nan_column' in cfg else [],
                                           map_columns=cfg.map_columns if 'map_columns' in cfg else {},
                                           cat_columns=cfg.cat_columns if 'cat_columns' in cfg else [],
                                           bool_columns=cfg.bool_columns if 'bool_columns' in cfg else [],
                                           bool_column_datatype=cfg.bool_column_datatype if 'bool_column_datatype' in cfg else 'int',
                                           int_columns=cfg.int_columns if 'int_columns' in cfg else [],
                                           float_columns=cfg.float_columns if 'float_columns' in cfg else [],
                                           nan_cat_value=cfg.nan_cat_value if 'nan_cat_value' in cfg else 'empty_val',
                                           nan_bool_value=cfg.nan_bool_value if 'nan_bool_value' in cfg else -1,
                                           nan_int_value=cfg.nan_int_value if 'nan_int_value' in cfg else None,
                                           nan_cont_value=cfg.nan_cont_value if 'nan_cont_value' in cfg else None,
                                           scaler_for_num_col=cfg.scaler_for_num_col if 'scaler_for_num_col' in cfg else None,
                                           remove_rows_for_lambda_conditions=cfg.remove_rows_for_lambda_conditions if 'remove_rows_for_lambda_conditions' in cfg else None,
                                           preprocess_lambda=cfg.preprocess_lambda if 'preprocess_lambda' in cfg else None)

    df_train, df_test = split_data_in_train_test(df=df_processed, target_cols=cfg.split_cols,
                                                 test_ratio=cfg.test_ratio, random_seed=cfg.split_seed)

    create_experiment_structure(experiment_name=cfg.experiment_name,
                                create_columns_json=cfg.create_columns_json if 'create_columns_json' in cfg else True,
                                overwrite_data_folder_if_exists=cfg.overwrite_data_folder_if_exists if 'overwrite_data_folder_if_exists' in cfg else False,
                                df=df_processed, df_train=df_train, df_test=df_test, num_scaler=scaler,
                                generative_models=cfg.create_folder_for_generative_models,
                                cols_bool=cfg.bool_columns if 'bool_columns' in cfg else [],
                                cols_cat=cfg.cat_columns if 'cat_columns' in cfg else [],
                                cols_int=cfg.int_columns if 'int_columns' in cfg else [],
                                cols_float=cfg.float_columns if 'float_columns' in cfg else [],
                                cols_outcome=cfg.outcome_columns,
                                bool_datatype=cfg.bool_column_datatype if 'bool_column_datatype' in cfg else 'int',
                                preprocess_lambda_for_eval=cfg.preprocess_lambda_for_eval if 'preprocess_lambda_for_eval' in cfg else None)


if __name__ == '__main__':
    process()
