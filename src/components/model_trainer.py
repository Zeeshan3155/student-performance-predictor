import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_obj, tune_models

import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_path = ModelTrainerConfig()

    def initiate_trainer(self,train_arr,test_arr):

        logging.info("Initiating model trainer")

        try:
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            models = {
                "Linear Regressor": LinearRegression(),
                "CatBoost": CatBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor()
            }

            params = {
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regressor":{
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report = evaluate_models(X_train,y_train,X_test,y_test,models)

            top_2_scores = sorted(model_report.values(),reverse=True)[:2]

            top2_modelname = [model_name for model_name, model_score in model_report.items() if model_score in top_2_scores]

            top2_model_dict = {model_name:models[model_name] for model_name in top2_modelname}

            logging.info(f"Top 2 model dict{top2_model_dict}")

            tuned_model_report = tune_models(X_train,y_train,X_test,y_test,top2_model_dict,params)

            logging.info(f"Tuned model report {tuned_model_report}")

            best_r2_score = max(list(tuned_model_report.values()),key=lambda x:x[1])[1]

            logging.info(f"best r2 {best_r2_score}")

            best_model_name = list(filter(lambda x : tuned_model_report[x][1] == best_r2_score,tuned_model_report))[0]

            if best_r2_score < 0.6:
                raise CustomException("No best model")
            
            best_model = models[best_model_name]

            logging.info(f"Best model is {best_model_name}: {best_r2_score}")
            
            save_obj(self.model_path.trained_model_path,best_model)
            
            return best_r2_score
    
        except Exception as e:
            raise CustomException(e,sys)
