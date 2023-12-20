import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.exception import customexception
from src.logger import logging

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils import save_object_pkl
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, cv_arr):
        logging.info('Starting Model Training')
        try:
            logging.info('SPlitting Dependent and Independent features from train and test data')
            X_Train, y_train, X_Test, y_test, X_cv, y_cv = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
                cv_arr[:,:-1],
                cv_arr[:,-1]
            )

            models = {
                #'Lasso': Lasso(),
                'LinearRegression': LinearRegression(),
                #'Ridge': Ridge(),
                #'ElasticNet': ElasticNet(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                #'KNN': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostingRegressor': CatBoostRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict= evaluate_model(X_Train, y_train, X_Test, y_test, models, param=params)
            print(model_report)
            print("\n================================================================")
            logging.info(f'Model report : {model_report}')

            #Get best model

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model is : {best_model_name}, R2 Score : {best_model_score}')
            print("\n================================================================")
            logging.info(f'Best Model is : {best_model_name}, R2 Score : {best_model_score}')

            save_object_pkl(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            


        except Exception as e:
            logging.error(f'Error occurred in Data Ingestion: {str(e)}')
            return None