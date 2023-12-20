import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import r2_score

import sys, os
from dataclasses import dataclass

from sklearn.tree import DecisionTreeRegressor


from src.logger import logging
from src.exception import customexception

import pickle
from sklearn.model_selection import cross_val_score, GridSearchCV



def save_object_pkl(file_path, obj):
    try:
           dir_path = os.path.dirname(file_path)
           os.makedirs(dir_path, exist_ok=True) 

           with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error(f'Error occurred in utils: {str(e)}')
        return None

'''def evaluate_model(X_Train, y_train, X_Test, y_test, X_cv, y_cv,models):
    try:
        report = {}
        for i in range(len(models)):
             model = list(models.values())[i]
             model.fit(X_Train, y_train)

             y_test_pred = model.predict(X_Test)

             test_model_score = r2_score(y_test,y_test_pred)
             report[list(models.keys())[i]] = test_model_score

             return report


    except Exception as e:
        logging.error(f'Error occurred in utils: {str(e)}')
        return None'''
    
def evaluate_model(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.error(f'Error occurred in utils: {str(e)}')
        return None
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.error(f'Error occurred in utils: {str(e)}')
        return None
