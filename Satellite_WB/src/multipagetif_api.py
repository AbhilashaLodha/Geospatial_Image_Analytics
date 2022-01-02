import os
import glob
import subprocess
from Satelllite_WB.src.satellite_config import *
import sys
import logging
logger = logging.getLogger('In multipage api file')
logger.setLevel(logging.DEBUG)

#On Ubuntu
class multipage():

    def to_multipage(self,project_id,classes_raw_bands):
        import os
        import shutil
        import glob
        mask_dir_classes ="/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
        mask_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/mband/"
        logger.info("input directory, to be merged --%s" % mask_dir_classes)
        logger.info("multipage tif directory--- %s" % mask_dir)
        p1= "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/indices_raw"
        rgb_tif = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/stacked"
        src_files = os.listdir(p1)
        dest = mask_dir_classes
        classes = classes_raw_bands
        for file_name in src_files:
            files = os.listdir(p1+"/"+file_name)
            if("RGB" in classes):
                shutil.copy(rgb_tif + '/' + file_name + ".tif", dest+'/'+ file_name + '/' )
            for file_move in files:
                file_test = file_move[:-4]
                if file_test in classes:
                    full_file_name = os.path.join(p1+"/"+file_name, file_move)
                    print(full_file_name)
                    if os.path.isfile(full_file_name):
                        print( dest+'/'+ file_name + '/' + file_move)
                        shutil.copy(full_file_name, dest+'/'+ file_name + '/' + file_move)
                        

        dir_count = 0
        for _, dirs, _ in os.walk(mask_dir_classes):
            dir_count += len(dirs)
            break
        trainIds = [str(i).zfill(2) for i in range(1, dir_count+1)]
        order_classes = []

        z = 0
        for i in trainIds:  
            lis=glob.glob(mask_dir_classes+"/"+i+"/"+"*")
            if(len(lis)==0):
                logger.info("No images in the provided folder")
            lis = sorted(lis)
            for k in range(len(classes_raw_bands)):
                for j in lis:
                    if(classes_raw_bands[k] in j):
                        if(classes_raw_bands[k]=="B1" and ("B11" in j or "B10" in j)):
                            pass
                        else:
                            order_classes.append(j)
            if("RGB" in classes_raw_bands):
                path_rgb = mask_dir_classes+"/"+i+"/"+ i + ".tif"
                order_classes.append(path_rgb)
            logger.info(order_classes)      
            output = i + ".tif"
            
            multiple = ""
            for p in range(len(classes_raw_bands)):
                single = "{} ".format(order_classes[z])
                logger.info(single)
                z = z+1
                multiple = multiple + single
                if(z==len(order_classes)):
                    z = 0
                    break
            order_classes = []
            command = "tiffcp {}".format(multiple) + mask_dir + "{}".format(output)
            os.system(command)
            mask_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/mband/"
        return {"mask_dir":mask_dir}
