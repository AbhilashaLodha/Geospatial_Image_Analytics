import numpy as np
from PIL import Image
from osgeo import gdal
import os
import sys
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('colourcomposition........')
logger.setLevel(logging.DEBUG)

class Colour_Composition:

    def colourcomposition(self,project_id,indices_raw_dir):

        indices_color_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/indices_color/"

        if not os.path.exists(indices_color_dir):
            os.makedirs(indices_color_dir) 

        folders = os.listdir(indices_raw_dir)
        logger.info(folders)
        folders.sort()
        logger.info(folders)

        # logger.info("The index is : %s" % self.index)
        
        for ele in folders:
            logger.info("we are presently in %s", ele)
            files = indices_raw_dir + '/' + ele
            filelist = os.listdir(files)
            logger.info("filelist : %s", filelist)

            for m in range(len(filelist)):
                logger.info("index : ")
                logger.info(filelist[m])
    
                local_dir = indices_color_dir + '/' + ele
                
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                vmin = 0
                ds = gdal.Open(files + "/" + filelist[m])
                red = ds.GetRasterBand(1)
                red = red.ReadAsArray()
                plt.figure(figsize=(8,6))

                if filelist[m] == "ndvi.tif" or filelist[m] == "savi.tif" or filelist[m] == "msavi.tif"  :
                    cmap = 'YlGn'
                    im = plt.imshow(red, vmin=vmin, cmap=cmap)
                    plt.colorbar(im, fraction=0.015)
                    plt.savefig(local_dir+"/%s.png" % filelist[m][:-4],dpi=600, bbox_inches='tight')
                    plt.close()

                elif filelist[m] == "ndwi.tif" or filelist[m] == "ndmi.tif":
                    cmap = 'Blues'
                    im = plt.imshow(red, vmin=vmin, cmap=cmap)
                    plt.colorbar(im, fraction=0.015)
                    plt.savefig(local_dir+"/%s.png" % filelist[m][:-4],dpi=600, bbox_inches='tight')
                    plt.close()

                elif filelist[m] == "ndsi.tif" or filelist[m] == "ndgi.tif":
                    cmap = 'Spectral'
                    im = plt.imshow(red, vmin=vmin, cmap=cmap)
                    plt.colorbar(im, fraction=0.015)
                    plt.savefig(local_dir+"/%s.png" % filelist[m][:-4],dpi=600, bbox_inches='tight')
                    plt.close()

                elif filelist[m] == "bi.tif":
                    cmap = 'YlOrBr'
                    im = plt.imshow(red, vmin=vmin, cmap=cmap)
                    plt.colorbar(im, fraction=0.015)
                    plt.savefig(local_dir+"/%s.png" % filelist[m][:-4],dpi=600, bbox_inches='tight')
                    plt.close()

        logger.info("Coloured-composed imgs can be found in %s" % indices_color_dir)
        return "done"

