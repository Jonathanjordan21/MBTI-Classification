import os,sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataTransformationConfig:
    def __init__(self, test_size=0.05, shuffle=False, sampling='undersample', ingestion_config=None):
        self.test_size, self.shuffle, self.sampling = test_size, shuffle, sampling
        self.ingestion_config = ingestion_config

class DataTransformation:
    def __init__(
        self, test_size=0.05, shuffle=False, sampling='undersample', 
        ingestion_config=None, config=None
    ):
        if config!=None:
            self.config = config
        else :
            self.config = DataTransformationConfig(
                test_size, shuffle, sampling, ingestion_config
            )
    
    def transform(self):
        try:
            logging.info("Start Data Transformation...")
            df = pd.read_csv(self.ingestion_config.cleaned_dir)
            if 'under' in self.config.sampling:
                df = self.undersample(df)
            elif 'over' in self.config.sampling:
                df = self.oversample(df)
            train_set, test_set = train_test_split(
                df[['type','post']], test_size=self.config.test_size, shuffle=self.config.shuffle
            )
            logging.info("Train and Test data has been created")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Transformation Complete")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as err:
            CustomException(err, sys)


    def undersample(self, df):
        logging.info("Start undersampling dataset...")
        unique_class = df['type'].unique()

        target_size = min([len(df['post'].loc[df['type']==label]) for label in unique_class])

        df_temp = pd.DataFrame(columns=['type','post'])

        for label in unique_class:
            class_sample = df['post'].loc[df['type']==label]
            undersampled = np.random.choice(class_sample, size=target_size, replace=False)
            for x in undersampled:
                df_temp.loc[len(df_temp)] = [label, x]

        logging.info("Undersampling dataset has finished")

        return df_temp
    

    def oversample(self, df):
        logging.info("Start oversampling dataset...")
        unique_class = df['type'].unique()
        target_size = max([len(df['post'].loc[df['type']==label]) for label in unique_class])

        df_temp = pd.DataFrame(columns=['type','post'])

        for label in unique_class:
            class_sample = df['post'].loc[df['type']==label]
            if len(class_sample) == target_size:
                continue
            oversampled = np.random.choice(class_sample, size=target_size, replace=True)
            for x in oversampled:
                df_temp.loc[len(df_temp)] = [label, x]
        logging.info("Oversampling dataset has finished")

        return df_temp




