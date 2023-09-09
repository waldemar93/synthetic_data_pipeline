"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from pipeline.data_preparation import DataPrep
from synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")


class CTABGAN:

    def __init__(self,
                 raw_csv_path="Real_Datasets/Adult.csv",
                 test_ratio=0.20,
                 categorical_columns=None,
                 log_columns=None,
                 mixed_columns=None,
                 general_columns=None,
                 non_categorical_columns=None,
                 integer_columns=None,
                 problem_type=None):
        if non_categorical_columns is None:
            non_categorical_columns = []
        if problem_type is None:
            problem_type = {"Classification": "income"}
        if integer_columns is None:
            integer_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        if general_columns is None:
            general_columns = ["age"]
        if mixed_columns is None:
            mixed_columns = {'capital-loss': [0.0], 'capital-gain': [0.0]}
        if log_columns is None:
            log_columns = []
        if categorical_columns is None:
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                   'gender', 'native-country', 'income']
        self.__name__ = 'CTABGAN'

        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df, self.categorical_columns, self.log_columns, self.mixed_columns,
                                  self.general_columns, self.non_categorical_columns, self.integer_columns,
                                  self.problem_type, self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical=self.data_prep.column_types["categorical"],
                             mixed=self.data_prep.column_types["mixed"],
                             general=self.data_prep.column_types["general"],
                             non_categorical=self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in', end_time - start_time, " seconds.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.raw_df))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
