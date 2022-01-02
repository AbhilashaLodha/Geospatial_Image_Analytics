import utm
import glob
import os
import sys
import json
import shutil
import subprocess
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import tarfile
from osgeo import gdal
import numpy as np
from osgeo import gdal,osr
import logging
logger = logging.getLogger('satelliteprocess')
logger.setLevel(logging.DEBUG)


class Satellite_Process:

    def search_scene(self,dataset,latitude,longitude,start_date,end_date,max_cloud_cover,tarfile_landsat8,idx):
        api = landsatxplore.api.API('iasp_satellite', 'iaspsatellite2020')
        #search landsat 8 scenes
        scenes = api.search(
            dataset = dataset, #'LANDSAT_8_C1',
            latitude = latitude, #lat,
            longitude = longitude, #lon,
            start_date = start_date, #'2019-01-01',
            end_date = end_date, #'2019-02-28',
            max_cloud_cover = max_cloud_cover) #10)
        logger.info('{} scenes found.'.format(len(scenes)))
        
        for scene in scenes:
            logger.info(scene['acquisitionDate'])
            logger.info(scene['displayId'])
        api.logout()
        #download landsat scenes
        ee = EarthExplorer('chava_sindhu', 'sindhuchava123')
        directory = tarfile_landsat8 
        logger.info("out folder is %s"%directory)
        ee.download(scene_id=scene['displayId'], output_dir=directory)
        ee.logout()


    def extract(self,tarfile_landsat8,rawbands_landsat8):
        logger.info("in extract function")
        tarfolders = os.listdir(tarfile_landsat8) # ['02', '01', '03']
        logger.info("folders inside tarfile")
        logger.info(tarfolders)

        try :
            #untar directory to Landsat_8
            logger.info(" current tarfolder is ")
            logger.info(tarfile_landsat8)
            tarList = tarfile_landsat8 +"/"+ os.listdir(tarfile_landsat8)[0]
            logger.info("tarList is")
            logger.info(tarList)
            tar = tarfile.open(tarList)
            logger.info("files get extracted in")
            logger.info(rawbands_landsat8)
            tar.extractall(rawbands_landsat8)
            tar.close()

        except Exception as e:
            logger.error("Error: " + str(e))


    def remove_tarfiles(self, tarfile_landsat8):
        logger.info("in remove tarfile function")
        
        folder = tarfile_landsat8
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.info('Failed to delete %s. Reason: %s' % (file_path, e))

    def extract_zone(self,rawbands_landsat8):
        zone = {}
        result = {}
        rawfolders = os.listdir(rawbands_landsat8) # ['02', '01', '03']
        logger.info("folders in rawbands before sorting")
        logger.info(rawfolders)      
        rawfolders.sort()
        logger.info("folders in rawbands after sorting")
        logger.info(rawfolders)
        for idx in range(len(rawfolders)):
            if idx in result:
                pass
            else:
                result[idx] = []
            dir_path = rawbands_landsat8+"/"+ rawfolders[idx]
            logger.info("rawfolder path")
            logger.info(dir_path) 
            images = os.listdir(dir_path)
            logger.info("images : ")
            logger.info(images)
            for image in images:
                logger.info("Image Path: ")
                logger.info(image)
                if image.endswith(".txt"):
                    pass
                else:            
                    ds = gdal.Open(dir_path+"/"+image)
                    logger.info(ds)
                    prj=ds.GetProjection()
                    logger.info("prj : ") 
                    logger.info(prj)

                    srs=osr.SpatialReference(wkt=prj)
                    if srs.IsProjected:
                        zone[image] = srs.GetAttrValue('projcs')  
                    zone_letter = zone[image][-1]
                    zone_number = zone[image][-3:-1]
                    logger.info(zone_letter)
                    logger.info(zone_number)
                    lst =[zone_letter,zone_number]
                    logger.info("lst is : ")
                    logger.info(lst)
                break
            result[idx].append(lst)
        logger.info("Zone Info Dict : ") 
        logger.info(result)
        return result


    def clipping(self,latitude,longitude,zones,rawbands_landsat8,clip_dir_landsat8):

        logger.info("In Clip Function")
        folders = os.listdir(rawbands_landsat8)
        logger.info("folders in rawbands before sorting")
        logger.info(folders)  #['02', '01', '03']  
        folders.sort()
        logger.info("folders in rawbands after sorting")
        logger.info(folders)  #['01', '02', '03']

        logger.info(len(zones))
        for i in zones:
            logger.info(zones[i])
            logger.info(zones[i][0][0])
            logger.info(int(zones[i][0][1]))
        logger.info(len(folders))

        if isinstance(latitude, list) & isinstance(longitude, list) :
            logger.info("Lat, long, start_date & end_date are lists")

            for idx in range(len(folders)):
                logger.info("folders are : ")
                logger.info(folders)
                utm_co = utm.from_latlon(latitude[idx], longitude[idx], int(zones[idx][0][1]), str(zones[idx][0][0])) #, #int(zones[idx][1]), zones[idx][0])
                logger.info("utm_co : ")
                logger.info(utm_co)

                xmin = utm_co[0]-9000 #792555.000
                xmax = utm_co[0]+9000 #812205.000
                ymin = utm_co[1]-9000 #-3399015.000
                ymax = utm_co[1]+9000 #-3379155.000
                
                logger.info("idx")
                logger.info(idx)
                logger.info("folders[idx]")
                logger.info(folders[idx])
                folders_dir = rawbands_landsat8 + "/" + folders[idx]
                logger.info("raw folders dir")
                logger.info(folders_dir)

                FileList = os.listdir(folders_dir)

                directory = clip_dir_landsat8
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                for fp in FileList:
                    if fp.endswith(".txt"):
                        pass
                    else:
                        if idx < 9:    
                            local_dir = directory + '/' + '0'+ str(idx+1)
                        else:
                            local_dir = directory + '/' + str(idx+1)
                        
                        if not os.path.exists(local_dir):
                            os.makedirs(local_dir)
                        logger.info("clip band sub dir")
                        logger.info(local_dir)
                        outputfile = local_dir+"/"+ "clip_"+fp[6:-4] + ".TIF"
                        ds = gdal.Open(folders_dir+"/"+fp)
                        ds = gdal.Translate(outputfile, ds, projWin = [xmin, ymax, xmax, ymin])
                        x = ds.ReadAsArray()

        else :
            logger.info("Lat, long, start_date & end_date are single values")
            logger.info(latitude)
            logger.info(longitude)

            for idx in range(len(folders)):
                utm_co = utm.from_latlon(latitude, longitude, int(zones[idx][0][1]), str(zones[idx][0][0])) #, #int(zones[idx][1]), zones[idx][0])
                logger.info("utm_co : ")
                logger.info(utm_co)

                xmin = utm_co[0]-9000 #792555.000
                xmax = utm_co[0]+9000 #812205.000
                ymin = utm_co[1]-9000 #-3399015.000
                ymax = utm_co[1]+9000 #-3379155.000

                folders_dir = rawbands_landsat8 + "/" + os.listdir(rawbands_landsat8)[idx]
                logger.info("folders_dir")

                FileList = os.listdir(folders_dir)

                directory = clip_dir_landsat8
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                for fp in FileList:
                    if fp.endswith(".txt"):
                        pass
                    else:
                        if idx < 9:    
                            local_dir = directory + '/' + '0'+ str(idx+1)
                        else:
                            local_dir = directory + '/' + str(idx+1)
                        
                        if not os.path.exists(local_dir):
                            os.makedirs(local_dir)

                        outputfile = local_dir+"/"+ "clip_"+fp[6:-4] + ".TIF"
                        ds = gdal.Open(folders_dir+"/"+fp)
                        ds = gdal.Translate(outputfile, ds, projWin = [xmin, ymax, xmax, ymin])
                        x = ds.ReadAsArray()

        return directory

    def allFunctionCalls(self,dataset,latitude,longitude,start_date,end_date,max_cloud_cover,project_id):
        try:           
            logger.info("Attempting satellite processing")
            logger.info(latitude)
            tarfile_landsat8_1 = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/tarfile"
            rawbands_landsat8_1 = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/rawbands"
            clip_dir_landsat8 = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
            logger.info("len----%s"%len(latitude))
            for i in range(1,len(latitude)+1):
                if i <= 9:    
                    tarfile_landsat8 = tarfile_landsat8_1 + "/0{}".format(i)
                    rawbands_landsat8 = rawbands_landsat8_1 +"/0{}".format(i)

                else:
                    tarfile_landsat8 = tarfile_landsat8_1 + "/{}".format(i)
                    rawbands_landsat8 = rawbands_landsat8_1 +"/{}".format(i)

                if not os.path.exists(tarfile_landsat8):
                    os.makedirs(tarfile_landsat8)
                if not os.path.exists(rawbands_landsat8):
                    os.makedirs(rawbands_landsat8)
                self.search_scene(dataset, latitude[i-1], longitude[i-1], start_date[i-1], end_date[i-1], max_cloud_cover,tarfile_landsat8,0)
            
                logger.info("Starting to extract tars")
                self.extract(tarfile_landsat8,rawbands_landsat8)
                self.remove_tarfiles(tarfile_landsat8)
                
                
            zones = self.extract_zone(rawbands_landsat8_1)
            logger.info("zones----")
            logger.info(zones)
            directory = self.clipping(latitude, longitude, zones,rawbands_landsat8_1,clip_dir_landsat8)
            logger.info("clipping done")
            
        except Exception as e:
            logger.error("Error: " + str(e))
        return directory
    