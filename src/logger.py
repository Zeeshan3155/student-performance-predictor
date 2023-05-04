import logging
import os
from datetime import datetime


file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

folder_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

logs_path = os.path.join(os.getcwd(),"logs",folder_name)

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,file_name)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

