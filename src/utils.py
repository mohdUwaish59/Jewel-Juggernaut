import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import r2_score

import sys, os
from dataclasses import dataclass

from src.logger import logging
from src.exception import customexception

import pickle


def save_object_pkl(file_path, obj):
    try:
           dir_path = os.path.dirname(file_path)
           os.makedirs(dir_path, exist_ok=True) 

           with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error(f'Error occurred in utils: {str(e)}')
        return None

def evaluate_model(X_Train, y_train, X_Test, y_test,models):
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
        return None
