import os
import sys
from src.logger import logging
from src.exception import customexception
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Initialize the data Ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('initiate_data_ingestion: Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebook/data', 'diamonds.csv'))
            logging.info('Pandas dataframe is ready')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            df = df.drop(columns=['Unnamed: 0'], axis=1)

            logging.info('Starting Train Test Split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Train Test Split completed')
            logging.info('Data Ingestion completed')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)


        except Exception as e:
            logging.error(f'Error occurred in Data Ingestion: {str(e)}')
            return None