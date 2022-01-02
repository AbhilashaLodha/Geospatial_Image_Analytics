# imports
import glob
import os
import os.path
import subprocess
import sys
from Satelllite_WB.src.core import preprocessing
from Satelllite_WB.src.core import masking_2
import logging
logger = logging.getLogger('In annotation1 api file')
logger.setLevel(logging.DEBUG)

class Annotations1():

    def paths(self,data_dir,input_rgb_dir_name,input_dir_name,stacked_dir_name, jpegs_dir_name):
        rgb_bands_dir = os.path.join(data_dir, input_rgb_dir_name) #rgb_mband
        logger.info("rgb_bands_dir : " + str(rgb_bands_dir))
        src_dir = os.path.join(data_dir, input_dir_name) # Input Dir - pan_mband
        logger.info("src_dir : " + str(src_dir))
        stacked_dir = os.path.join(data_dir, stacked_dir_name) # Dir to contain the stacked tiffs
        rgb_dir = os.path.join(data_dir, jpegs_dir_name) # Dir to contain the JPEGs
        if not os.path.exists(rgb_bands_dir):
            os.makedirs(rgb_bands_dir)
        if not os.path.exists(rgb_dir):
            os.makedirs(rgb_dir)
        if not os.path.exists(stacked_dir):
            os.makedirs(stacked_dir)

        return rgb_bands_dir,src_dir, stacked_dir, rgb_dir
    
    def images(self,src_dir):
        dir_count = 0
        for _, dirs, _ in os.walk(src_dir):
            dir_count += len(dirs)
            break
        trainIds = [str(i).zfill(2) for i in range(1, dir_count+1)]

        return trainIds

    def rgb_multiband(self,rgb_bands_dir, src_dir,trainIds):
        for img_id in trainIds:
            logger.info("img_id")
            logger.info(img_id)
            preprocessing.rgb_multipage(os.path.join(src_dir, '{}'.format(img_id)),
                                        os.path.join(rgb_bands_dir),img_id)
        logger.info("multipaging rgb bands done ")

    def stack_bands(self,rgb_bands_dir,stacked_dir,trainIds):
        # Stack bands
        for img_id in trainIds:
            preprocessing.stack_multipage_bands(os.path.join(rgb_bands_dir, '{}.tif'.format(img_id)),
                                                os.path.join(stacked_dir, '{}.tif'.format(img_id)))

        logger.info("stacking is done")
        # TODO Pan Sharpening
    def to_jpg(self,stacked_dir, rgb_dir,trainIds):
        # TODO Extract JPEG from stacked TIFF - enhance
        for img_id in trainIds:
            preprocessing.tiff_to_jpeg(os.path.join(stacked_dir, '{}.tif'.format(img_id)),
                                                os.path.join(rgb_dir, '{}.jpg'.format(img_id)))

        logger.info("Extract JPEG from stacked TIFF---done")


    def move_jpg(self, project_id, src):
        # folder creation
        parent_dir = "/var/www/html/LabelMeAnnotationTool/"
        images_dir = parent_dir + "Images/" + str(project_id)
        annotations_dir = parent_dir + "Annotations/" + str(project_id)

        if not os.path.exists(images_dir):
            os.system("sudo mkdir %s" %images_dir)
            logger.info("folder created")

        # Moving jpgs from data_dir/jpegs folder to Images folder
        logger.info("in move_jpg")
        logger.info(os.getcwd())

        for jpgfile in glob.iglob(os.path.join(src, "*.jpg")):
            # shutil.copy2(jpgfile, dest)
            os.system('sudo cp "%s" "%s"' % (jpgfile, images_dir))
        logger.info("Files moved")

        # Check if labelme_satellite.txt exists 
        txtInVar = "/var/www/html/LabelMeAnnotationTool/annotationCache/DirLists"
        labelme_satellite = txtInVar + "/labelme_satellite.txt"

        if os.path.exists(labelme_satellite):
            os.system('sudo rm %s' %labelme_satellite)
            logger.info("Labelme_satellite.txt removed")

        # populate code
        os.chdir("/var/www/html/LabelMeAnnotationTool/annotationTools/sh/")
        logger.info(os.getcwd())
        os.system('sudo /var/www/html/LabelMeAnnotationTool/annotationTools/sh/populate_dirlist.sh labelme_satellite.txt %s' %project_id)
        
        
    def lableme(self,data_dir, jpegs_dir_name):
        # Batch image annotation
        completed_status = masking_2.batch_annotate(data_dir, jpegs_dir_name)
        print(completed_status.returncode)
        print(completed_status.stdout)
        print("Batch image annotation---done")


    def to_tifmask(self,trainIds,classes,rgb_dir, mask_dir_classes, stacked_dir):
        class_name_to_id0 = {'__ignore__':-1,'_background_':0,'vegetation':1,'water':0}
        class_name_to_id1 = {'__ignore__':-1,'_background_':0,'vegetation':0,'water':1}
        index = ((img_id,i) for img_id in trainIds for i in range(len(classes)))
        for img_id,i in index:
            print("check if this loop is working")
            print(img_id,i)
            # TODO JSON to PNG mask creation
            x  = "class_name_to_id"+str(i)
            class_name_to_id = eval(x)
            print(class_name_to_id)
            mask_arr = masking_2.create_mask_png(os.path.join(rgb_dir, '{}.json'.format(img_id)),class_name_to_id)
            #os.path.join("class_name_to_id"+str(i))
            #mask_arr = masking_2.create_masks_png(rgb_dir)
            print("jpg to png MASK done5555555555")
            print(mask_arr)
            # TODO PNG mask to tiff mask creation
            masking_2.mask_to_tiff(mask_arr, os.path.join(mask_dir_classes, '{}_'.format(img_id)+classes[i]+".tif"),
                                os.path.join(stacked_dir, '{}.tif'.format(img_id)))
            print("png to tif MASK  done5555555555")

    def to_multipage(self,trainIds):
        for img_id in trainIds:
            masking_2.tif_multipagetif(img_id)
            print("multi paging is done")

    def annotation_allfuns(self,project_id):

        input_type = "raw"
        data_dir= "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir"
        input_rgb_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/rgb_mband"
        if input_type == "raw":
            input_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
        else:
            input_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/mband"

        stacked_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/stacked"
        jpegs_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/jpegs"

        # try:
        rgb_bands_dir, src_dir, stacked_dir, rgb_dir = self.paths(data_dir,input_rgb_dir_name,input_dir_name,stacked_dir_name, jpegs_dir_name)
        trainIds = self.images(src_dir)
        self.rgb_multiband(rgb_bands_dir, src_dir,trainIds)
        self.stack_bands(rgb_bands_dir,stacked_dir,trainIds)
        self.to_jpg(stacked_dir, rgb_dir,trainIds)
        self.move_jpg(project_id,jpegs_dir_name)

        return "success"

    # except Exception as e:
            # logger.error("Error: " + str(e))

