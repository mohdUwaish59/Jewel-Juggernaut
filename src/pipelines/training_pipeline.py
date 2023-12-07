import os
import sys
from src.logger import logging
from src.exception import customexception
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transformation_obj  = DataTransformation()
    train_arr, test_arr = data_transformation_obj.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr=train_arr, test_arr=test_arr)