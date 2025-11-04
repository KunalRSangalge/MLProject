import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,confusion_matrix
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor 
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

from dataclasses import dataclass


@dataclass
class Model_Trainer_config():
    Trained_model_file_path = os.path.join('artifact','model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_config()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("spliting train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "LinearRegression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "ElasticNet" : ElasticNet(),    
                "KNN" : KNeighborsRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "XGBoost" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                "Adaboost" : AdaBoostRegressor()
            }

            model_report : dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("best model found on both train and test data")

            save_object(
                file_path=self.model_trainer_config.Trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)

            return r2 
        
        except Exception as e:
            raise CustomException(e,sys)

