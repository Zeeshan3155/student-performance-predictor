import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def predict(self, dataframe):
        try:
            model_path=os.path.join("artifacts","model_trainer.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(object_path=model_path)
            preprocessor=load_object(object_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(dataframe)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                gender,             
                race, 
                parent_education_lvl,
                lunch_type ,
                test_preparation_course,
                reading_score, 
                writing_score,):
        self.gender = gender
        self.race = race
        self.parent_education_lvl = parent_education_lvl
        self.lunch_type = lunch_type
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def make_dataframe(self):
        try:
            data_dict = {
                "gender":[self.gender],            
                "race_ethnicity": [self.race], 
                "parental_level_of_education": [self.parent_education_lvl],
                "lunch": [self.lunch_type],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score], 
                "writing_score": [self.writing_score]
            }

            df = pd.DataFrame(data_dict)

            return df
        except Exception as e:
            raise CustomException(e,sys)
