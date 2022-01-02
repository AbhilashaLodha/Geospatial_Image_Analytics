from osgeo import gdal, gdal_array
import os
import math
import numpy as np
import pandas as pd
import os
import math
import logging
import sys
logger = logging.getLogger('In indices api file')
logger.setLevel(logging.DEBUG)


class calc_indices:

    def get_bands(raw_dir,clipped_dir):

        logger.info("inside get_bands")
        bands = []
        path_out = raw_dir
        path_main = clipped_dir
        print("clip dir",clipped_dir)
        keys = [1,2,3,4,5,6,7,8,9,10,11]
        bands_dic = {}

        for i in keys:
            bands_dic[i] = None


        for filename in os.listdir(path_main): 
            if(filename[-5].isdigit()):
                if(int(filename[-6].isdigit() and filename[-6])==1):
                    band_number = 10+int(filename[-5])
                else:
                    band_number = int(filename[-5])
                bands_dic[band_number] = filename
            

        logger.info(" bands_dic %s"%bands_dic)

        # Create blank array of bands
        band_array={}

        # Get bands
        for i, band in bands_dic.items():
            if(bands_dic[i]==None):
                band_array[i] = None
            else:
                band = path_main+str(band)
                ds = gdal.Open(band)
                b = ds.GetRasterBand(1)
                arr = b.ReadAsArray()
                band_array[i] = arr

        output_file = path_out+'/'
        return band_array, output_file, ds, bands_dic

    def calcNDVI(band_array, output_file, ds):
        logger.info("inside ndvi function")
        #logger.info("band array---------- %s "% band_array)
        ndvi = (band_array[5].astype(np.float32) - band_array[4].astype(np.float32))/(band_array[5].astype(np.float32) + band_array[4].astype(np.float32))
        #print("ndvi",ndvi)
        gdal_array.SaveArray(ndvi.astype("float32"), output_file+'ndvi.tif', "GTIFF", ds)
        return np.nanmin(ndvi), np.nanmax(ndvi)

    def calcNDWI(band_array, output_file, ds):
        ndwi = (band_array[3].astype(np.float32) - band_array[5].astype(np.float32))/(band_array[3].astype(np.float32) + band_array[5].astype(np.float32))
        gdal_array.SaveArray(ndwi.astype("float32"), output_file+'ndwi.tif', "GTIFF", ds)
        return np.nanmin(ndwi), np.nanmax(ndwi)

    def calcBI(band_array, output_file, ds):
        BI = (band_array[6].astype(np.float32) + band_array[4].astype(np.float32) - (band_array[5].astype(np.float32)+band_array[2].astype(np.float32)))/(band_array[6].astype(np.float32) + band_array[4].astype(np.float32)+ (band_array[5].astype(np.float32)+band_array[2].astype(np.float32)))
        gdal_array.SaveArray(BI.astype("float32"), output_file+'bi.tif', "GTIFF", ds)
        return np.nanmin(BI), np.nanmin(BI)

    def calcSI(band_array, output_file, ds):    
        SI = (np.cbrt((1-(band_array[2]*0.0001)*(1-(band_array[3]*0.0001))*(1-(band_array[4]*0.0001)))))
        gdal_array.SaveArray(SI.astype("float32"), output_file+'si.tif', "GTIFF", ds)
        return np.nanmin(SI), np.nanmin(SI)

    def calcAVI(band_array, output_file, ds):
        avi = np.cbrt((((band_array[5]*0.0001)*(1-(band_array[4]*0.0001))*((band_array[5]*0.0001)-(band_array[4]*0.0001)))))
        gdal_array.SaveArray(avi.astype("float32"), output_file+'avi.tif', "GTIFF", ds)
        return np.nanmin(avi), np.nanmin(avi)

    def calcSAVI(band_array, output_file, ds):
        L=0.5
        savi = ((band_array[5].astype(np.float32) - band_array[4].astype(np.float32))/(band_array[5].astype(np.float32) + band_array[4].astype(np.float32)+L)) *  (1+L)
        gdal_array.SaveArray(savi.astype("float32"), output_file+'savi.tif', "GTIFF", ds)
        return np.nanmax(savi), np.nanmax(savi)

    def calcEVI(band_array, output_file, ds):
        L = 1
        C1 = 6
        C2 = 7.5
        evi = 2.5*( (band_array[5] - band_array[4])/((band_array[5] + C1*band_array[4]-C2*band_array[2])+L))
        evi = evi * 0.0001        
        gdal_array.SaveArray(evi.astype("float32"), output_file+'evi.tif', "GTIFF", ds)
        return np.nanmin(evi.astype("float16")), np.nanmax(evi.astype("float16"))

    def calcMSAVI(band_array, output_file, ds):
        con = (2*band_array[5].astype(np.float32)+1)
        gdal_array.SaveArray(((con - np.sqrt((con*con)-8*(band_array[5].astype(np.float32)-band_array[4].astype(np.float32))))/2).astype("float32"), output_file+'msavi.tif', "GTIFF", ds)
        msavi_min = np.nanmin(((con - np.sqrt((con*con)-8*(band_array[5].astype(np.float32)-band_array[4].astype(np.float32))))/2).astype("float32"))
        msavi_max = np.nanmax(((con - np.sqrt((con*con)-8*(band_array[5].astype(np.float32)-band_array[4].astype(np.float32))))/2).astype("float32"))
        return msavi_min, msavi_max

    def calcNDSI(band_array, output_file, ds):
        ndsi = (band_array[3].astype(np.float32) - band_array[6].astype(np.float32))/(band_array[3].astype(np.float32) + band_array[6].astype(np.float32))
        gdal_array.SaveArray(ndsi.astype("float32"), output_file+'ndsi.tif', "GTIFF", ds)
        return np.nanmin(ndsi), np.nanmax(ndsi)

    def calcNDMI(band_array, output_file, ds):
        ndmi = (band_array[5].astype(np.float32) - band_array[6].astype(np.float32))/(band_array[5].astype(np.float32) + band_array[6].astype(np.float32))
        gdal_array.SaveArray(ndmi.astype("float32"), output_file+'ndmi.tif', "GTIFF", ds)
        return np.nanmin(ndmi),np.nanmax(ndmi)

    def calcNDGI(band_array, output_file, ds):
        ndgi = (band_array[3].astype(np.float32) - band_array[4].astype(np.float32))/(band_array[3].astype(np.float32) + band_array[4].astype(np.float32))
        gdal_array.SaveArray(ndgi.astype("float32"), output_file+'ndgi.tif', "GTIFF", ds)
        # print("min NDGI",np.nanmin(ndgi)) 
        # print("max NDGI",np.nanmax(ndgi))
        return np.nanmin(ndgi), np.nanmax(ndgi)

    def calcNBR(band_array, output_file, ds):
        nbr = (band_array[5].astype(np.float32) - band_array[7].astype(np.float32))/(band_array[5].astype(np.float32) + band_array[7].astype(np.float32))
        gdal_array.SaveArray(nbr.astype("float32"), output_file+'nbr.tif', "GTIFF", ds)
        nbr_min = np.nanmin(nbr)
        nbr_max = np.nanmax(nbr)
        return nbr_min, nbr_max

    def indices_all(self,project_id):
        clip_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands/"
        indices_output_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/indices_raw/"

        dir_count = 0
        for _, dirs, _ in os.walk(clip_dir):
            dir_count += len(dirs)
            break
        print("dir count ", dir_count)
        print("dir_name",dirs)
        
        for i in dirs:
            band_array, output_file, ds, bands_dic =  calc_indices.get_bands(indices_output_dir, clip_dir +i+"/")
            output_file_dir = output_file +i+"/"
            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            logger.info("get_bands done ")
            empty_bands = []
            for k,v in bands_dic.items(): 
                if(bands_dic[k]==None):     
                    empty_bands.append(k)
                    
            print("empty bands",empty_bands)
            if(4 not in empty_bands and 5 not in empty_bands):
                logger.info("inside ndvi if")
                ndvi_min, ndvi_max =  calc_indices.calcNDVI(band_array, output_file_dir, ds)
                logger.info("ndvi")
            if(3 not in empty_bands and 5 not in empty_bands):
                ndwi_min, ndwi_max =  calc_indices.calcNDWI(band_array, output_file_dir, ds)
                logger.info("ndwi")
            if(2 not in empty_bands and 4 not in empty_bands and 5 not in empty_bands and 6 not in empty_bands):
                bi_min, bi_max =  calc_indices.calcBI(band_array, output_file_dir, ds)
                logger.info("bi")
            if(2 not in empty_bands and 3 not in empty_bands and 4 not in empty_bands):
                si_min, si_max =  calc_indices.calcSI(band_array, output_file_dir, ds)
                logger.info("si")
            if(4 not in empty_bands and 5 not in empty_bands):
                avi_min, avi_max =  calc_indices.calcAVI(band_array, output_file_dir, ds)
                logger.info("avi")
            if(4 not in empty_bands and 5 not in empty_bands):
                savi_min, savi_max =  calc_indices.calcSAVI(band_array, output_file_dir, ds)
                logger.info("savi")
            if(2 not in empty_bands and 4 not in empty_bands and 5 not in empty_bands):
                evi_min, evi_max =  calc_indices.calcEVI(band_array, output_file_dir, ds) 
                logger.info("evi")
            if(4 not in empty_bands and 5 not in empty_bands):
                msavi_min, msavi_max =  calc_indices.calcMSAVI(band_array, output_file_dir, ds)
                logger.info("msavi")
            if(3 not in empty_bands and 6 not in empty_bands):
                ndsi_min, ndsi_max =  calc_indices.calcNDSI(band_array, output_file_dir, ds)
                logger.info("ndsi")
            if(6 not in empty_bands and 5 not in empty_bands):
                ndmi_min, ndmi_max =  calc_indices.calcNDMI(band_array, output_file_dir, ds)
                logger.info("ndmi")
            if(4 not in empty_bands and 3 not in empty_bands):
                ndgi_min, ndgi_max =  calc_indices.calcNDGI(band_array, output_file_dir, ds)
                logger.info("ndgi")
            if(5 not in empty_bands and 7 not in empty_bands):           
                nbr_min, nbr_max =  calc_indices.calcNBR(band_array, output_file_dir, ds)
                logger.info("nbr")
        return indices_output_dir




