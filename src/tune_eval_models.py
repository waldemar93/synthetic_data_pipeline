import json
import os
from dataclasses import dataclass, MISSING
from typing import Optional, Literal
import hydra
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import LabelEncoder
from src.base import PROJECT_DIR
from src.helpers import load_experiment_data, load_columns_json_for_experiment


@dataclass
class EvalOptimizationConfig:
    """ Configuration for evaluation optimization. """
    experiment_name: str = MISSING
    predict_column: str = MISSING
    create_column_lambda: str = MISSING
    optimization_metric: Literal['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc', 'mean_absolute_error',
                                 'root_mean_squared_error', 'mean_squared_error', 'median_absolute_error',
                                 'mean_absolute_percentage_error'] = MISSING
    optimization_direction: Literal['minimize', 'maximize'] = MISSING
    optimization_sampler: Literal['RandomSampler', 'GridSampler', 'TPESampler', 'CmaEsSampler', 'NSGAIISampler',
                                  'QMCSampler'] = MISSING
    optimization_trials: int = MISSING
    optimization_seed: int = MISSING
    filename: str = MISSING


@dataclass
class MainEvalOptimizationConfig:
    """Main configuration for evaluation optimization class containing a EvalOptimizationConfig instance."""
    eval_optimization: EvalOptimizationConfig = MISSING


def _get_cat_feature_names(experiment_name: str):
    """
    Get names of categorical features for given experiment.

    :param experiment_name: Name of the experiment.
    :return: List of categorical feature names.
    """
    columns_json = load_columns_json_for_experiment(experiment_name)

    cat_cols = columns_json['boolean'] + columns_json['categorical']
    cat_cols = [col for col in cat_cols if col not in columns_json['outcome']]

    return cat_cols


def cross_validation_test_cat(model: any, experiment_name: str, outcome_col: str,
                              create_column_lambda: Optional[str] = None,
                              metric: Literal['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc'] = 'f1') \
        -> float:
    """
    Conduct cross-validation test on categorical data using CatBoost model.

    :param model: CatBoost model.
    :param experiment_name: Name of the experiment.
    :param outcome_col: Outcome column for prediction.
    :param create_column_lambda: Optional: Lambda function to create a new column if it doesn't exist.
    :param metric: Evaluation metric. Default is 'f1'.
    :return: Average cross-validation score.
    """
    # load data
    df, _, _ = load_experiment_data(experiment_name)
    columns_json = load_columns_json_for_experiment(experiment_name)

    # check if column exists
    if outcome_col not in df.columns:
        # use create_column_lambda expression for the creation
        df[outcome_col] = df.apply(eval(create_column_lambda), axis=1)

    # remove other outcome cols
    columns_to_remove = [col for col in columns_json['outcome'] if col != outcome_col]
    df.drop(columns_to_remove, axis=1, inplace=True)

    if outcome_col in columns_json['boolean'] and columns_json['boolean_dtype'] == 'int':
        # change type to real int, because scores seem not to handle this int type well
        df[outcome_col] = df[outcome_col].astype(int)
        # in case 0 is less probable change it to 1 for a proper f1 score
        if metric == 'f1':
            if df[outcome_col].value_counts()[1] > df[outcome_col].value_counts()[0]:
                df[outcome_col] = df[outcome_col].apply(lambda x: 1 if x == 0 else 0)

    # preprocessing
    # select non-numerical columns (ignoring booleans if these are ints)
    non_numeric_cols = df.select_dtypes(exclude=[int, float]).columns

    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder because a forest based model is used
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(df[col])
        # transform data column-wise
        df[col] = encoder.transform(df[col])

    # determine columns containing NaN values
    nan_columns = df.columns[df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        df.fillna(-9999999999, inplace=True)

    y = df[outcome_col]
    df_x = df.drop([outcome_col], axis=1)

    # use stratified
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = kfold.split(df_x, y)
    if metric == 'mcc':
        cv_results = cross_validate(model, X=df_x, y=y, cv=kf, scoring=make_scorer(matthews_corrcoef))
        average_score = sum(cv_results[f'test_score']) / 5
    else:
        scoring = {metric}
        cv_results = cross_validate(model, X=df_x, y=y, cv=kf, scoring=scoring)
        average_score = sum(cv_results[f'test_{metric}']) / 5

    return average_score


def cross_validation_test_num(model: any, experiment_name: str, outcome_col: str,
                              create_column_lambda: Optional[str] = None,
                              metric: Literal['mean_absolute_error', 'root_mean_squared_error', 'mean_squared_error',
                                              'median_absolute_error', 'mean_absolute_percentage_error'] =
                              'mean_absolute_error') -> float:
    """
    Conduct cross-validation test on numerical data using CatBoost model.

    :param model: CatBoost model.
    :param experiment_name: Name of the experiment.
    :param outcome_col: Outcome column for prediction.
    :param create_column_lambda: Optional: Lambda function to create a new column if it doesn't exist.
    :param metric: Evaluation metric. Default is 'mean_absolute_error'.
    :return: Average cross-validation score.
    """
    # load data
    df, _, _ = load_experiment_data(experiment_name)
    columns_json = load_columns_json_for_experiment(experiment_name)

    # check if column exists
    if outcome_col not in df.columns:
        # use create_column_lambda expression for the creation
        df[outcome_col] = df.apply(eval(create_column_lambda), axis=1)

    # remove other outcome cols
    columns_to_remove = [col for col in columns_json['outcome'] if col != outcome_col]
    df.drop(columns_to_remove, axis=1, inplace=True)

    # preprocessing
    # select non-numerical columns (ignoring booleans if these are ints)
    non_numeric_cols = df.select_dtypes(exclude=[int, float]).columns

    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder because random forest is used
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(df[col])
        # transform real and synthetic data column-wise
        df[col] = encoder.transform(df[col])

    # determine columns containing NaN values
    nan_columns = df.columns[df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        df.fillna(-9999999999, inplace=True)

    y = df[outcome_col]
    df_x = df.drop([outcome_col], axis=1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = kfold.split(df_x, y)

    metric = f'neg_{metric}'
    scoring = {metric}
    cv_results = cross_validate(model, X=df_x, y=y, cv=kf, scoring=scoring)
    average_score = abs(sum(cv_results[f'test_{metric}'])) / 5
    return average_score


def suggest_catboost_cat_params(trial: optuna.trial.Trial):
    """
    Suggest hyperparameters for CatBoost classifier.

    :param trial: Optuna trial instance.
    :return: Dictionary containing suggested hyperparameters.
    """
    params: dict = {}
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 0.1, 10.0)
    params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)
    params["objective"] = trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"])
    params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.01, 0.1)
    params["bootstrap_type"] = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU",  # "GPU", may affect performance
        "thread_count": 4,
        "random_seed": 42,
        # "devices": "0", # for GPU
    }

    return params


def suggest_catboost_reg_params(trial: optuna.trial.Trial):
    """
    Suggest hyperparameters for CatBoost regressor.

    :param trial: Optuna trial instance.
    :return: Dictionary containing suggested hyperparameters.
    """
    params: dict = {}
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 0.1, 10.0)
    params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)
    params["objective"] = trial.suggest_categorical("objective", ["RMSE", "MAE", "Quantile", "MAPE"])
    params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.01, 0.1)
    params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU",  # "GPU", may affect performance
        "thread_count": 4,
        "random_seed": 42,
        # "devices": "0", # for GPU
    }

    return params


def get_objective_function(experiment_name: str, outcome_var: str,
                           optim_metric: Literal['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc',
                                                 'mean_absolute_error', 'root_mean_squared_error', 'mean_squared_error',
                                                 'median_absolute_error', 'mean_absolute_percentage_error'],
                           create_column_lambda: Optional[str] = None):
    """
    Get objective function for optimization given an optimization metric.

    :param experiment_name: Name of the experiment.
    :param outcome_var: Variable to predict.
    :param optim_metric: Metric for optimization.
    :param create_column_lambda: Optional: Lambda function to create a new column if it doesn't exist.
    :return: Objective function to be used in Optuna study.
    """
    def objective_function(trial: optuna.trial.Trial):
        # depending on the metric check if it is a categorical prediction or a regression
        categorical_task = optim_metric.lower() in ['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc']
        if categorical_task:
            params = suggest_catboost_cat_params(trial)
            params['cat_features'] = _get_cat_feature_names(experiment_name=experiment_name)
            trial.set_user_attr("params", params)
            gbm = CatBoostClassifier(**params, verbose=False, allow_writing_files=False)
            score = cross_validation_test_cat(gbm, experiment_name=experiment_name, outcome_col=outcome_var,
                                              metric=optim_metric.lower(), create_column_lambda=create_column_lambda)
            return score
        else:
            # regression
            params = suggest_catboost_reg_params(trial)
            params['cat_features'] = _get_cat_feature_names(experiment_name=experiment_name)
            trial.set_user_attr("params", params)
            gbm = CatBoostRegressor(**params, verbose=False, allow_writing_files=False)
            score = cross_validation_test_num(gbm, experiment_name=experiment_name, outcome_col=outcome_var,
                                              metric=optim_metric.lower(), create_column_lambda=create_column_lambda)
            return score

    return objective_function


@hydra.main(config_path="../config", config_name="2_eval_optimization", version_base="1.1")
def optimize_eval_hyperparams(cfg: MainEvalOptimizationConfig):
    """
    Optimize hyperparameters for evaluation.

    :param cfg: Configuration instance containing optimization details.
    """
    cfg: EvalOptimizationConfig = cfg.eval_optimization
    print(OmegaConf.to_yaml(cfg))

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

    # create study
    study = optuna.create_study(direction=cfg.optimization_direction, sampler=sampler)

    # create objective function
    objective_func = get_objective_function(experiment_name=cfg.experiment_name, outcome_var=cfg.predict_column,
                                            optim_metric=cfg.optimization_metric,
                                            create_column_lambda=cfg.create_column_lambda if 'create_column_lambda' in cfg else None)

    # start optimization
    study.optimize(objective_func, n_trials=cfg.optimization_trials, show_progress_bar=True)

    # save best params
    best_params = study.best_trial.user_attrs['params']

    filename = cfg.filename if cfg.filename.endswith('.json') else cfg.filename + '.json'
    exp_folder_path = os.path.join(os.getcwd().split(PROJECT_DIR)[0] + PROJECT_DIR, 'experiments', cfg.experiment_name)
    with open(os.path.join(exp_folder_path, 'eval_optim', filename), 'w') as fh:
        json.dump(best_params, fh, indent=2)


if __name__ == '__main__':
    optimize_eval_hyperparams()
