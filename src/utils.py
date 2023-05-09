import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle

def save_obj(obj_path,obj):
    dir_path = os.path.dirname(obj_path)

    os.makedirs(dir_path,exist_ok=True)

    with open(obj_path,"wb") as fp:
        pickle.dump(obj,fp)
