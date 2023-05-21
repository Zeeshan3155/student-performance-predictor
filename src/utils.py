import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_obj(obj_path,obj):
    try:
        dir_path = os.path.dirname(obj_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(obj_path,"wb") as fp:
            pickle.dump(obj,fp)
        
        logging.info("Object saved")

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):

    try:
        logging.info("Evaluating models")

        report = {}

        for model_name, model in models.items():
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            r2_score_test = r2_score(y_test,y_pred)

            report[model_name] = r2_score_test

        logging.info("Evaluation completed")

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def tune_models(X_train,y_train,X_test,y_test,models,params):
    try:
        logging.info("Initiating model tuning")

        tune_report = {}
        for model_name, model in models.items():
            param = params[model_name]
            rsCV = RandomizedSearchCV(model,param,cv=3)
            rsCV.fit(X_train,y_train)

            y_pred = rsCV.predict(X_test)

            r2score = r2_score(y_test,y_pred)

            tune_report[model_name] = (rsCV.best_params_,r2score)

        logging.info("Tuning complete")
        return tune_report
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object(object_path):
    try:
        with open(object_path,"rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        raise CustomException(e,sys)