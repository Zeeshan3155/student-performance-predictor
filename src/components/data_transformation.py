import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import sys
from src.utils import save_obj
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransforamtionConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransforamtionConfig()

    def get_data_transformer(self):
        try:
            cat_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_columns = ['reading_score','writing_score']

            logging.info("Creating data transformation object")

            cat_pipeline = Pipeline(
                [
                    ('cat_imputer',SimpleImputer(strategy='most_frequent')),
                    ('cat_enconding',OneHotEncoder()),
                    ('cat_scaling',StandardScaler(with_mean=False))
                ]
            )

            num_pipeline = Pipeline(
                [
                    ('num_imputer', SimpleImputer(strategy='median')),
                    ('num_scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('cat_pipeline',cat_pipeline,cat_columns),
                    ('num_pipeline',num_pipeline,num_columns)
                ]
            )

            logging.info(f"categorical columns: {cat_columns}")
            logging.info(f"numerical columns: {num_columns}")
            logging.info("Data transformation object created")

            return preprocessor

        except Exception as e:
            CustomException(e,sys)

    def initiate_transformer(self,train_path,test_path):
        
        try:
            logging.info("Initiating data transformer")
            logging.info("Reading data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Sucessfully read data")

            target_column = ['math_score']

            logging.info("Starting data transformation")

            train_independent_features = train_df.drop(columns=target_column)
            train_dependent_features = train_df[target_column]

            test_independent_features = test_df.drop(columns=target_column)
            test_dependent_features = test_df[target_column]

            preprocessor_obj = self.get_data_transformer()

            processed_input_train_arr = preprocessor_obj.fit_transform(train_independent_features)
            processed_input_test_arr = preprocessor_obj.transform(test_independent_features)

            train_arr = np.c_[processed_input_train_arr,np.array(train_dependent_features)]
            test_arr = np.c_[processed_input_test_arr,np.array(test_dependent_features)]

            logging.info("Data transformation completed")

            save_obj(self.data_transformation_config.preprocessor_path,preprocessor_obj)

            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)