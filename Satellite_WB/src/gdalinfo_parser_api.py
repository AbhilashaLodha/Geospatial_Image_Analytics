from osgeo import gdal,ogr,osr
import numpy as np
import subprocess
import json
import os 
import sys
from flask import Flask,request
from Satelllite_WB.src.satellite_config import *
import logging
logger = logging.getLogger('In gdalinfo_parser_api file')
logger.setLevel(logging.DEBUG)

class parsing_gdalinfo():

    def GetCornerCoordinates(FileName):
        GdalInfo = subprocess.check_output('gdalinfo {} -json'.format(FileName), shell=True)
        data = json.loads(GdalInfo)
        return data

    def gdalparser(input_type, dir_path):
        if(input_type == "multipage"):
            raster = gdal.Open(dir_path )
            type(raster)
            metadata = raster.GetMetadata()
            prj = raster.GetProjection()
            srs = osr.SpatialReference(wkt=prj)
            data = parsing_gdalinfo.GetCornerCoordinates(dir_path)
        elif(input_type == "raw"):
            raster = gdal.Open(dir_path + "/" + os.listdir(dir_path)[1])
            transform = raster.GetGeoTransform()
            xOrigin = transform[0]
            yOrigin = transform[3]
            pixelWidth = transform[1]
            pixelHeight = -transform[5]
            
            # Check type of the variable 'raster'
            type(raster)
            # Dimensions
            # Metadata for the raster dataset
            metadata = raster.GetMetadata()

            # Projection
            prj = raster.GetProjection()
            srs = osr.SpatialReference(wkt=prj)
            logger.info(os.listdir(dir_path))
            data = parsing_gdalinfo.GetCornerCoordinates(dir_path + "/" + os.listdir(dir_path)[1])
            #logger.info(data['cornerCoordinates'])

        if (str(srs) == ''):
            logger.info("multiband .tif image")
            if (metadata):

                final = {
                    "data_format": data['driverLongName'],
                    "image_size": data['size'],
                    "bands_count": raster.RasterCount,
                    "meta_data": data['metadata'],
                    "bands": data['bands'],
                    "IMAGE_STRUCTURE": data['metadata']['IMAGE_STRUCTURE'],
                    "corner_coordinates": data['cornerCoordinates'],

                }
            else:
                logger.info("stacked .tif image")
                final = {
                    "data_format": data['driverLongName'],
                    "image_size": data['size'],
                    "bands_count": 11,#raster.RasterCount,
                    "bands": data['bands'],
                    "IMAGE_STRUCTURE": data['metadata']['IMAGE_STRUCTURE'],
                    "corner_coordinates": data['cornerCoordinates'],

                }
        else:
            logger.info("single band .tif image")
            final = {
                "data_format": data['driverLongName'],
                "image_size": data['size'],
                "bands_count": 11,#raster.RasterCount,
                "PROJCS": srs.GetAttrValue('projcs'),
                "GEOGCS": srs.GetAttrValue('geogcs'),
                "SPHEROID": srs.GetAttrValue('spheroid'),
                "PROJECTION": srs.GetAttrValue('projection'),
                "PRIMEM": srs.GetAttrValue('primem'),
                "AUTHORITY": srs.GetAttrValue('authority'),
                "UNIT": srs.GetAttrValue('unit'),
                #"origin_pixelsize": data['geoTransform'],
                "origin": (transform[0],transform[3]),
                "pixelsize" : (transform[1],transform[-5]),
                "latitude_of_origin": srs.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
                "central_meridian": srs.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
                "scale_factor": srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR),
                "meta_data": data['metadata'],
                "bands": data['bands'],
                "IMAGE_STRUCTURE": data['metadata']['IMAGE_STRUCTURE'],
                "corner_coordinates": data['cornerCoordinates'],
                "corner_cord_degrees": data['wgs84Extent']

            }
        
        return final
        

    def gdalinfo(self,project_id):
        input_type =  "raw"
        
        logger.info("input_type--%s" % input_type)
        output_gdal = {}
        if(input_type == "multipage"): # get folder name from config/app layer 
            filepath = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
            logger.info("filepath--- %s" % filepath)
            path, dirs, files = next(os.walk(filepath))
            file_count = len(files)
            #logger.info("file count ",file_count)
            Ids = [str(i).zfill(2) for i in range(1, file_count + 1)]
            logger.info("ids",Ids)
            for i in Ids:
                output_gdal[i] = 0
            for i in Ids:
                dir_path= filepath + "/" + i + ".tif"
                logger.info("dir_path--- %s" % dir_path)
                out = parsing_gdalinfo.gdalparser(input_type, dir_path)
                output_gdal[i] = out

        elif(input_type == "raw"):
            filepath = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
            logger.info("filepath--- %s" % filepath)
            folder_count = len(next(os.walk(filepath))[1])
            Ids = [str(i).zfill(2) for i in range(1, folder_count + 1)]
            logger.info(Ids)
            for i in Ids:
                output_gdal[i] = 0
            for img_id in Ids:
                dir_path= filepath + "/" + img_id
                logger.info("dir_path--- %s" % dir_path)
                logger.info(dir_path)
                out = parsing_gdalinfo.gdalparser(input_type, dir_path)
                output_gdal[img_id] = out
        return output_gdal
