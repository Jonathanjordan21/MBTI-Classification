import os, sys, urllib
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')
    cleaned_data_path:str = os.path.join('artifacts', 'cleaned.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def ingestion(self):
        logging.info("Processing Data Ingestion...")
        try :
            self.download_dataset(self.ingestion_config.raw_data_path)
            df=pd.read_csv(self.ingestion_config.raw_data_path)
            df=self.clean_raw_data(df)
            df.to_csv(self.ingestion_config.cleaned_data_path, index=False, header=True)
            logging.info("Data Ingestion Complete")
            return self.ingestion_config
            
        except Exception as err:
            raise CustomException(err, sys)
    
    def download_dataset(self):
        if not os.path.exists(self.ingestion_config.raw_data_path):
            logging.info("Downloading the Dataset...")
            url = 'https://raw.githubusercontent.com/dashascience/-MBTI-Myers-Briggs-Personality-Type-Dataset/master/mbti_1.csv'
            os.mkdir('artifacts')
            urllib.request.urlretrieve(url, self.ingestion_config.raw_data_path)
            logging.info("Dataset has been successfully downloaded")
        else :
            logging.info("Dataset exists!")
    

    def clean_raw_data(self, df):
        import re
        def string_cond(s):
            return (
                'http' not in s and re.search('[a-zA-Z]', s) and 
                len(s) > 15 and s.isascii()
            )

        try :
            logging.info("Cleaning Raw Data...")

            data = []
            for x in df.iterrows():
                for s in x[1]['posts'].split('|||'):
                    if string_cond(s):
                        data.append((x[1]['type'], s))
            df_new = pd.DataFrame(data, columns=['type', 'post'])
            df_new['post'] = df_new['post'].astype(str)
            df_new['type'] = df_new['type'].astype(str)
            filter_chars = '[!"#$%&()*+,-./:;<=\'>?@[\\]^_`{|}~\t\n]'
            df_new['post'] = df_new['post'].map(lambda x : re.sub(filter_chars, "", x))

            logging.info("Raw Data has been successfully cleaned")

        except Exception as err:
            raise CustomException(err, sys)