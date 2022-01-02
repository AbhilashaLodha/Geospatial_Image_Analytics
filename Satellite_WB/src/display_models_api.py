import numpy as np
import os
import sys
from os import listdir
import pandas as pd
import logging
logger = logging.getLogger('displaymodels')
logger.setLevel(logging.DEBUG)

class Display_Models :
    def display_models(self,project_id):
        data_config_var = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id) +"/config_dir"
        sys.path.insert(0, data_config_var)
        from data_config1 import dict1

        model_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/model_dir/model"
        models = listdir(model_dir)
        logger.info("models are : %s", models)
        return models

