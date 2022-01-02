from osgeo import gdal
import os
import logging
import sys

logger = logging.getLogger('Traiining unet........')
logger.setLevel(logging.DEBUG)
def do_pansharpen(project_id,directory_input):

    logger.info("in pan sharpen file")

    directory_out = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/pan_mband"
    if not os.path.exists(directory_out):
        os.makedirs(directory_out)

    folders = os.listdir(directory_input)
    for idx in range(len(folders)):
        x =  os.listdir(directory_input)[idx]
        logger.info("hi----- %s"%x)
        folders_dir = directory_input  + '/'+ x
        logger.info("folders_dir : %s" % folders_dir)
        if idx < 9:
            directory_pan = directory_out + '/' + '0'+ str(idx+1)
        else:
            directory_pan = directory_out + '/' + str(idx+1)
        #directory_pan = directory_out+'/0'+idx
        if not os.path.exists(directory_pan):
            os.makedirs(directory_pan)
        file_list = os.listdir(folders_dir)
        for file in file_list:
            #open raster file
            logger.info(folders_dir+'/'+file)
            clipped = gdal.Open(folders_dir+'/'+file)
            x = clipped.RasterXSize
            logger.info(x)
            y = clipped.RasterYSize
            logger.info(y)
            os.system("gdal_translate -of GTiff -outsize {} {} -r cubic {} {}".format(x*2,y*2, folders_dir+'/'+file, directory_pan+'/'+file)) 
        #directory_input = jsonObj["directory_input"] 
    
    return directory_out