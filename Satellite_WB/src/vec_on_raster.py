# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:08:35 2019

@author: suraj.khaitan
"""

# Import required libraries
import numpy as np
from osgeo import gdal, gdal_array
import tifffile as tiff
import os
import ntpath
import logging
logger = logging.getLogger('Vec on raster')
logger.setLevel(logging.DEBUG)

class Overlay:
        
    def vec_on_raster(self,raster, vector_dir, band, output_raster,test_id):
        # Open background raster image
        read_band = tiff.imread(raster)

        # Select band of image to be displayed
        raster_band = read_band

        raster_band = raster_band/raster_band.max()
        raster_band = raster_band*255
        
        # Reshape image for ease in handling channels
        raster_band = np.reshape(raster_band,(-1, raster_band.shape[0], raster_band.shape[1]))

        # Check if raster is 3 channel as mask to be overlayed is RGB
        if raster_band.shape[0]<3:
            # Create dummy channels of same shape
            ch = np.zeros((3, raster_band.shape[1], raster_band.shape[2]))
            ch[0] = raster_band
            ch[1] = raster_band
            ch[2] = raster_band
        else:
            pass
        
        # Read masks in vector directory
        masks = os.listdir(vector_dir)
        logger.info("masks---%s"%masks)
        for mask in masks:
            if mask.startswith(test_id):
                vector = tiff.imread('{}/{}'.format(vector_dir, mask))
                vector = np.array(vector) 
                logger.info("mask--%s"%mask)
                #print("loop will run ---",vector.shape[0])
                # Check polygon pixel values in mask bands
                min_px = []
                # Check polygon pixel values in mask bands
                for i in range(vector.shape[0]):
                    print('Value of polygon in band {}: {}'.format(i+1, vector[i].min()))
                    logger.info("min-----------%s"%(str(vector[i].min() )    )        ) 
                    min_px.append(vector[i].min())
                
                if len(set(min_px))==1 and min_px[0]==255:
                    print('Found blank mask...')
                    continue
                else:
                    #Insert polygons in background raster
                    for i in range(ch.shape[0]):
                        ch[i] = np.where(vector[i] == vector[i].min(), vector[i].min(), ch[i])
                
                
        # Create final raster image with class polygons(Class colors included) 
        gdal_array.SaveArray(ch.astype("float32"), '{}'.format(output_raster), "GTIFF")
        return raster_band, ch, min_px

def path_leaf(path):
    # to get the name of the weights
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
def result_raster(project_id,weights_path,thres):
    test_images = '/home/ubuntu/cv-asset/Satelllite_WB/'+str(project_id)+'/data_dir/mband/test'
    images = os.listdir(test_images)
    logger.info("i---%s"%images)
    logger.info("project_id")
    #raster = '/home/ubuntu/cv-asset/Satelllite_WB/6/data_dir/mband/test/02.tif'
    for raster in images:
        raster = test_images + "/"+ raster
        #print("loop running---------",raster)
        id= path_leaf(raster)
        test_id = id.replace('.tif','')
        path_clipped= '/home/ubuntu/cv-asset/Satelllite_WB/'+str(project_id)+'/data_dir/clippedbands/'+test_id
        clipped_images = os.listdir(path_clipped)
        for i in clipped_images:
            if "B4" in i:
                raster=path_clipped +"/"+i
        model_name = path_leaf(weights_path).replace('.hdf5','')
        logger.info("raster final----------%s "%raster)
        vector_dir = '/home/ubuntu/cv-asset/Satelllite_WB/'+str(project_id)+'/data_dir/pred_masks/'+str(model_name)+'/'+str(thres)
        band = 4
        logger.info("line no 73")
        output_raster = '/home/ubuntu/cv-asset/Satelllite_WB/'+str(project_id)+'/data_dir/pred_masks/'+str(model_name)+'/'+str(thres)+ '/raster{}.tif'.format(test_id)
        o1 = Overlay()
        logger.info("raster---%s"%raster)
        logger.info("output raster----%s"%output_raster)
        x = o1.vec_on_raster(raster, vector_dir, band, output_raster,test_id)
        #logger.info("res---------%s"%x)
    return "success"
