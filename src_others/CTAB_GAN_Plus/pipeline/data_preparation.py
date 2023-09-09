import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection


class DataPrep(object):

    def __init__(self, raw_df: pd.DataFrame, categorical: list, log: list, mixed: dict, general: list,
                 non_categorical: list, integer: list, type: dict):

        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []

        target_col = list(type.values())[0]
        if target_col is not None:
            y_real = raw_df[target_col]
            X_real = raw_df.drop(columns=[target_col])
            X_real[target_col] = y_real

            self.df = X_real
        else:
            self.df = raw_df

        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for i in relevant_missing_columns:
            if i in self.log_columns:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x == "empty" else x)
                    self.mixed_columns[i] = [-9999999]
            elif i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x == "empty" else x)
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x == "empty" else x)
                    self.mixed_columns[i] = [-9999999]

        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = []
                for idx, val in enumerate(self.df[log_column].values):
                    if val != -9999999:
                        valid_indices.append(idx)
                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower
                if lower > 0:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x) if x != -9999999 else -9999999)
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999)
                else:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999)

        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)

                if column in self.general_columns:
                    self.column_types["general"].append(column_index)

                if column in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)

            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

            elif column in self.general_columns:
                self.column_types["general"].append(column_index)

        super().__init__()

    def inverse_prep(self, data, eps=1):

        df_sample = pd.DataFrame(data, columns=self.df.columns)

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = df_sample[self.label_encoder_list[i]["column"]].astype(
                int)
            df_sample[self.label_encoder_list[i]["column"]] = le.inverse_transform(
                df_sample[self.label_encoder_list[i]["column"]])

        if self.log_columns:
            for i in df_sample:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound > 0:
                        df_sample[i].apply(lambda x: np.exp(x))
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: np.ceil(np.exp(x) - eps) if (np.exp(x) - eps) < 0 else (np.exp(x) - eps))
                    else:
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x) - eps + lower_bound)

        if self.integer_columns:
            for column in self.integer_columns:
                if str(df_sample[column].dtype).startswith('float'):
                    df_sample[column] = (np.round(df_sample[column].values))
                    df_sample[column] = df_sample[column].astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace('empty', np.nan, inplace=True)

        return df_sample
