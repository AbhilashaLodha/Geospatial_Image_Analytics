import tifffile as tiff
import numpy as np
import os
import sys
from osgeo import gdal
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import logging
logger = logging.getLogger('satelliteprocess')
logger.setLevel(logging.DEBUG)
from Satelllite_WB.src.satellite_config import *
import ntpath

def path_leaf(path):
    # to get the name of the weights
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class Display:

    def display_result(self, thresh,project_id,weights_path):
        logger.info("Reached Line 20, Display result api")
        model_name = path_leaf(weights_path).replace('.hdf5','')
        test_run = "/home/ubuntu/cv_workspace/data/Satelllite_WB/"+ str(project_id)+ "/data_dir/thumbnails/rasters/"
        rgb_mband_dir = "/home/ubuntu/cv_workspace/data/Satelllite_WB/"+ str(project_id)+ "/data_dir/thumbnails/jpegs"
        output_jpegs = "/home/ubuntu/cv_workspace/data/Satelllite_WB/"+ str(project_id)+ "/data_dir/thumbnails/output_jpegs"
        logger.info("Test run dir--%s" % test_run)
        logger.info("Rgb mband--- %s" % rgb_mband_dir)

        import shutil
        results = os.listdir(test_run)
        results.sort()
        logger.info(results)
        results1 = [x for x in results if x.startswith('raster')]
        for x1 in results1:
            test_id = x1.replace('raster','')
            shutil.copy(rgb_mband_dir + '/' + test_id, output_jpegs+'/'+ test_id )
        
        # rgb_mband_dir
        rgb_files = listdir(rgb_mband_dir)
        logger.info("rgb_files: %s", rgb_files)
        logger.info("raster_tiffs: %s", results1)

        both_list=[rgb_files,results1]
        res = test_run
        res_list = [rgb_mband_dir,res]
        res_json= {}
        res_json["rgb_mband_dir"]=output_jpegs 
        res_json["raster_files_dir"]= "/home/ubuntu/cv_workspace/data/Satelllite_WB/"+ str(project_id)+ "/data_dir/thumbnails/rasters"

        return res_json



