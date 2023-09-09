# synthetic_aml



## Getting started

Clone this repository from GitHub and create a virtual environment and activate it. 
Install the requirements with `pip install -r requirements.txt`.
The installation was tested for **python 10**.

Add **your data** to `data/raw/`. The file needs to be in a **.csv** or a 
**.xlsx** format.

## 1. Preprocessing

Create a **.yaml** file in the folder `config/process/` with the same structure as `process_aml.yaml`. 
All variables of the dataset need to be listed in one of the following attributes: `bool_columns`, `cat_columns`, 
`int_columns` and `float_columns`. The attributes `map_columns`, `remove_rows_for_lambda_conditions`, 
`preprocess_lambda` and `preprocess_lambda_for_eval` are optional.

Execute the preprocessing pipeline with `python -m src.preprocessing process=your_filename` 
(without the **.yaml** ending). Alternatively, you can edit `config/1_process.yaml` and change the attribute `process` 
to your filename (without the **.yaml** ending) and execute the preprocessing pipeline with 
`python -m src.preprocessing`.

The script will create a new folder with the `experiment_name` in the `experiments` folder.
The folder `data` contains `data.csv` (data before the split), `train.csv`, `test.csv` and `columns.json`. 
The `columns.json` should always be created, because it is required for different scripts afterward.
The current version uses 5-fold cross-validation for validation and therefore, there is no separate validation file.
The folder `eval_optim` will contain the configurations of the finetuned catboost hyperparameters. 
There are separate folders for each generative model (`create_folder_for_generative_models`) with the following folder 
structure:
- `config` (config after hyperparameter optimization will be saved here)
- `models` (trained models will be saved here)
- `eval` (everything regarding the evaluation will be saved here)
- `synthetic_data` (synthetic data will be saved here)

## 2. Hyperparameter optimization of ML evaluation models
To tune the hyperparameters of catboost models for the prediction of a variable requires the creation of an individual 
config file in `config/eval_optimization/`. The config file needs to look similar to `synthetic_aml_cr1_1.yaml`.
Note, all outcome columns (`outcome`) that are listed in `data/columns.json` will not be used as features. 
The optional attribute `create_column_lambda` is used to create a temporal column that is used for the prediction.

To start the hyperparameter tuning process is started with 
`python -m src.tune_eval_models eval_optimization=your_filename` (without **.yaml** ending).
Alternatively, you can edit `config/2_eval_optimization.yaml` and change the attribute `eval_optimization` to your 
filename (without the **.yaml** ending) and execute the optimization pipeline with 
`python -m src.tune_eval_models`.

The config with the best hyperparameters is stored at `experiments/your_experiment_name/eval_optim/`.

## 3. Hyperparameter optimization of generative models
To tune the hyperparameters of generative models, a config (or multiple configs) need to be created in 
the folder `config/gen_optimization/` similar to `optim_aml_ctab.yaml`. 
The combined score that will be optimized consists of metrics listed in `optimization_metrics`.
The following metrics can be used: `basic_statistical_measure_num`, `log_transformed_correlation_score`, `regularized_support_coverage`,
                   `discriminator_measure_rf`, `k_means_score`, `ml_efficiency_cat`, `survival_auc_opt`,
                   `survival_auc_abs_opt`, `survival_sightedness`, `survival`. 
It is also possible to use the same metric multiple times (with different parameters).
The weights for the individual metrics are listed in `optimization_metrics_weights`. 

For each metric listed in `optimization_metrics` the parameters can be set through an attribute with the name 
`optimization_metric_{number}_params`. 
Here, **{number}** starts from 1 and depends on the position in the `optimization_metrics` list. 

`ml_efficiency_cat` requires at least the following parameters:
- `predict_col` (column to predict) 
- `metric` (`f1`, `accuracy`, `roc_auc`, `f1_macro`, `f1_micro`, `mcc`)
- `relative` (relative to the original? If yes, it will be cut off at 1)
- `filename_params_json` (filename of the config in `experiments/your_experiment_name/eval_optim`)

`survival`-metrics (`survival_auc_opt`, `survival_auc_abs_opt`, `survival_sightedness` and `survival`) require at least 
the following parameters:
- `survival_target_col` (binary column)
- `survival_time_to_event_col` (duration column)

Other arguments can be changed through the parameters for every metric. The parameter name needs to be the same 
as in the respective function. 

It is advised to create **separate configs for each generative model**, so that they can be started in parallel.

> **Warning**
> 
> **CTAB-GAN+** requires an extra **config** at `experiments/your_experiment_name/CTAB-GAN+/columns_ctabgan.json` 
> with the following parameters: 
> - `cat_columns`
> - `int_columns`
> - `general_columns`
> - `log_columns`
> - `mixed_columns`
> - `non_categorical_columns`
> - `problem_type` (Classification or Regression)
> 
> See `experiments/synthetic_aml/CTAB-GAN+/columns_ctabgan.json` for more details.

> **Warning**
> 
> **Synthcity** models (**nflow, rtvae, bayesian_network, survae, survival_ctgan, survival_nflow, survival_gan**) 
> **require one config file** at `experiments/your_experiment_name/synthcity_dl.json` with the following attributes:
> - `target_col` (for non survival generative models, for supervised learning)
> - `survival_target_col` (for survival generative models, binary column)
> - `survival_time_to_event_col` (for survival generative models, duration column)

To run the hyperparameter optimization pipeline, execute the following script: 
`python -m src.tune_synthesizer gen_optimization=your_optimization_config_name`. 
Alternatively, you can edit `config/3_gen_optimization.yaml` and change the attribute `gen_optimization` to your 
filename (without the **.yaml** ending) and execute the optimization pipeline with 
`python -m src.tune_synthesizer`. 

> **Information**
> 
>In case the optimization was cancelled due to an internal (or external) error, start the script again. It will 
continue with the next trial.

The resulting config file is stored at `experiments/your_experiment_name/generative_model_name/config/`.

## 4. Training of generative models
To train the generative models, a config needs to be created in 
the folder `config/gen_training` similar to `synthetic_aml_best.yaml`. 

The attribute `train_configs` is a list of the configs (filenames) of the individual models stored in 
`experiments/your_experiment_name/generative_models/config/`. The attribute `train_seeds` is a list of different random 
seeds that will be used for the training of each model listed in `train_models`.

> **Information**
> 
> It is also possible to train models with their default hyperparameters. Use **default** for the attribute 
> `train_configs`.

To start the training, execute: `python -m src.train_synthesizer gen_training=your_training_config_name`. 
Alternatively, you can edit `config/4_gen_training.yaml` and change the attribute `gen_training` to your 
filename (without the **.yaml** ending) and start the training with `python -m src.train_synthesizer`. 

The trained models are stored in `experiments/your_experiment_name/generative_model/models/`.

## 5. Evaluation of the generative models
To evaluate the generative models, a config file needs to be created in the folder `config/eval_gen/` similar to 
`synthetic_aml_evaluate.yaml`.

The evaluation pipeline begins by loading a generative model. For each seed listed in `sampling_seeds`, a 
set of synthetic datapoints is sampled from the model.  By default, the number of these synthetic datapoints matches 
the training data size. This default value can be overridden using the `synth_final_size` parameter.

If you wish to enforce specific constraints on the synthetic data using the `synth_remove_lambda` parameter, 
you will need to specify the `synth_raw_size` parameter:
1. The pipeline first samples `synth_raw_size` datapoints from the generative model.
2. Datapoints that don't meet the criteria specified in `synth_remove_lambda` are then removed.
3. From the filtered synthetic data, `synth_final_size` datapoints are subsequently sampled.

You can also apply preprocessing steps to the synthetic data using the `synth_preprocess_lambda` parameter. 
Notably, this preprocessing occurs after the first round of synthetic data sampling from the generative model, 
and before any subsequent evaluations.

The metrics that can be used for the evaluation are similar to the metrics for the optimization of the generative models. 
However, the following visual evaluation methods can be used as well: `plot_pca`, `plot_cumsums`,
          `plot_correlation_difference`. The parameter `columns` should be set for the methods: `plot_cumsums` and 
`plot_correlation_difference` to reduce the number of shown variables. 
Choose the variables that are important for your task!

> **Warning**
> 
> The visualization methods can **only** be used if the synthetic data has the same size as the training set of the 
> original data. 

For each metric listed in `metrics` the parameters can be set through an attribute with the name 
`metric_{number}_params`. **{number}** depends on the position in the `metrics` list (starting with 1).
The requirements for the metrics `ml_efficiency_cat` and `survival-metrics` (`survival_auc_opt`, `survival_auc_abs_opt`, 
`survival_sightedness` and `survival`) are the same as for optimization of the generative models.
Additionally, for each metric the attribute `name` can be provided that will be shown in the evaluation files. 

> **Warning**
> 
> For metrics that are used multiple times in `metrics`, the attribute `name` must be provided.

Execute the evaluation pipeline with: `python -m src.eval_models eval_gen=your_eval_config_name`. 
Alternatively, you can edit `config/5_eval_gen.yaml` and change the attribute `eval_gen` to your 
filename (without the **.yaml** ending) and start the training with `python -m src.eval_models`. 

The evaluation files for each generative model listed in `models` can be found in the folder 
`experiments/generative_model/eval/`. A separate folder for each training seed is created there and contains the 
evaluation of each individual sampling seed. Additionally, an evaluation file for all the models listed in `models` 
is created at `experiments/your_experiment_name/eval.json`. 
This file contains just the averages across all training and sampling seeds for each model. 
The same generative model can be listed multiple times, if it has a different name (e.g., "best" and "default"). 

## 6. Privacy evaluation
Privacy evaluation is intended to be used just for specific synthetic datasets. To evaluate the privacy of 
synthetic datasets, a config file needs to be created in the folder `config/privacy_eval_gen/` similar to 
`synthetic_aml_best.yaml`. The privacy evaluation is based on the hamming distance. 

To run the evaluation, execute: `python -m src.privacy_evaluation privacy_eval_gen=your_privacy_eval_config_name`. 
Alternatively, you can edit `config/6_privacy_eval_gen.yaml` and change the attribute `privacy_eval_gen` to your 
filename (without the **.yaml** ending) and start the training with `python -m src.privacy_eval_models`.

The results will be stored in a file named `eval_privacy.json` in the folder `experiments/your_experiment_name`.
