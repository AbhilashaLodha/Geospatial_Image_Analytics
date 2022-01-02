# imports
import glob
import os
import os.path
import subprocess
import sys
import shutil
from Satelllite_WB.src.core import preprocessing
from Satelllite_WB.src.core import masking_2
import logging
logger = logging.getLogger('In annotation2 api file')
logger.setLevel(logging.DEBUG)

class Annotations2():

    def paths(self,data_dir,input_dir_name,mask_dir_single,mask_dir_name,stacked_dir_name, jpegs_dir_name):
        src_dir = os.path.join(data_dir, input_dir_name) # Input Dir - pan_mband
        logger.info("src_dir : "+ str(src_dir))
        mask_dir = os.path.join(data_dir, mask_dir_name)
        logger.info("mask_dir"+str(mask_dir))
        mask_dir_classes = os.path.join(data_dir, mask_dir_single)# Output Dir
        stacked_dir = os.path.join(data_dir, stacked_dir_name) # Dir to contain the stacked tiffs
        rgb_dir = os.path.join(data_dir, jpegs_dir_name) # Dir to contain the JPEGs

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        if not os.path.exists(mask_dir_classes):
            os.makedirs(mask_dir_classes)

        return src_dir, mask_dir, mask_dir_classes, stacked_dir, rgb_dir
    
    def images(self,src_dir):
        dir_count = 0
        for _, dirs, _ in os.walk(src_dir):
            dir_count += len(dirs)
            break
        trainIds = [str(i).zfill(2) for i in range(1, dir_count+1)]
        return trainIds

    def rgb_multiband(self,rgb_bands_dir, src_dir,trainIds):
        for img_id in trainIds:
            preprocessing.rgb_multipage(os.path.join(src_dir, '{}'.format(img_id)),
                                        os.path.join(rgb_bands_dir),img_id)
        print("multipaging rgb bands done ")

    def stack_bands(self,rgb_bands_dir,stacked_dir,trainIds):
        # Stack bands
        for img_id in trainIds:
            preprocessing.stack_multipage_bands(os.path.join(rgb_bands_dir, '{}.tif'.format(img_id)),
                                                os.path.join(stacked_dir, '{}.tif'.format(img_id)))

        print("stacking is done")
        # TODO Pan Sharpening
    def to_jpg(self,stacked_dir, rgb_dir,trainIds):
        # TODO Extract JPEG from stacked TIFF - enhance
        for img_id in trainIds:
            preprocessing.tiff_to_jpeg(os.path.join(stacked_dir, '{}.tif'.format(img_id)),
                                                os.path.join(rgb_dir, '{}.jpg'.format(img_id)))

        print("Extract JPEG from stacked TIFF---done")


    def lableme(self,data_dir, jpegs_dir_name):
        # Batch image annotation
        completed_status = masking_2.batch_annotate(data_dir, jpegs_dir_name)
        print(completed_status.returncode)
        print(completed_status.stdout)
        print("Batch image annotation---done")


    def move_xml(self, project_id, dest):
        src = "/var/www/html/LabelMeAnnotationTool/Annotations/" + str(project_id)
        for xmlfile in glob.iglob(os.path.join(src, "*.xml")):
            # shutil.copy2(jpgfile, dest)
            os.system('sudo cp "%s" "%s"' % (xmlfile, dest))
        logger.info("Files copied")


    def xml2json(self, project_id, src):
        import xmltodict
        import json

        all_files = os.listdir(src)
        logger.info("Spare aux.xml files are also generated, so we need to remove them")
        aux_xmls = [x for x in all_files if x.endswith('aux.xml')]
        logger.info(aux_xmls)

        for files in aux_xmls:
            os.remove(os.path.join(src,files))
        logger.info("aux.xml removed")
        logger.info(os.listdir(src))

        for xmlfile in glob.iglob(os.path.join(src, "*.xml")):
            with open(xmlfile) as in_file:
                xml = in_file.read()
                base = os.path.splitext(xmlfile)[0]
                logger.info("base")
                logger.info(base)
                with open('%s.json' %base, 'w') as out_file:
                    json.dump(xmltodict.parse(xml), out_file)
                    logger.info(out_file)
        logger.info("Files converted from xml to json")
        logger.info(os.listdir(src))


    def to_tifmask(self,trainIds,classes,rgb_dir, mask_dir_classes, stacked_dir):
        import copy
        dic = {'__ignore__':-1,'_background_':0}
        lis = []
        final_dic = {}

        for i in range(0,len(classes)):
            lis.append("class_name_to_id{}".format(i))
            dic[classes[i]] = 0
        print(dic)
        for j in range(0,len(classes)):  
            #print(class_name_to_id0)
            dic1 = copy.deepcopy(dic)
            dic1[classes[j]] = 1
            final_dic[lis[j]] = dic1
            #print("**",dic)
        print("______ ",final_dic)
        
        index = ((img_id,i) for img_id in trainIds for i in range(len(classes)))
        logger.info("index")
        logger.info(index)
        for img_id,i in index:
            logger.info("check if this loop is working")
            logger.info(img_id)
            logger.info(i)
            # TODO JSON to PNG mask creation
            class_name_to_id = "class_name_to_id"+str(i)
            logger.info(class_name_to_id)
            logger.info(os.path.join(rgb_dir, '{}.json'.format(img_id)))
            logger.info(final_dic[class_name_to_id])
            mask_arr = masking_2.create_mask_png_new(os.path.join(rgb_dir, '{}.json'.format(img_id)),final_dic[class_name_to_id])
            logger.info("jpg to png MASK done5555555555")
            logger.info(mask_arr)
            # TODO PNG mask to tiff mask creation
            masking_2.mask_to_tiff(mask_arr, os.path.join(mask_dir_classes, '{}_'.format(img_id)+classes[i]+".tif"),
                                os.path.join(stacked_dir, '{}.tif'.format(img_id)))
            logger.info("png to tif MASK  done5555555555")

    def to_multipage(self,classes, mask_dir, mask_dir_classes, trainIds):
        
        masking_2.tif_multipagetif(classes,mask_dir,mask_dir_classes, trainIds)
        logger.info("multi paging is done")

    def annotation_allfuns(self,project_id,classes):
        #try:
        logger.info("Inside annotations2")

        input_type = "raw"
        logger.info("classes------%s"%classes)
        logger.info("classes sorted------%s"%sorted(classes))
        classes = sorted(classes)
        logger.info("classes after sorting------%s"%classes)
        #classes = ["buildings","vegetation","water"]
        data_dir= "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir"

        if input_type == "raw":
            input_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/clippedbands"
        else:
            input_dir_name ="/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/mband"


        mask_dir_single = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/gt_mband_classes"
        mask_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/gt_mband"
        stacked_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/stacked"
        jpegs_dir_name = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/jpegs"

        src_dir, mask_dir, mask_dir_classes, stacked_dir, rgb_dir = self.paths(data_dir,input_dir_name,mask_dir_single, mask_dir_name,stacked_dir_name, jpegs_dir_name)
        trainIds = self.images(src_dir)        
        self.move_xml(project_id,jpegs_dir_name)
        self.xml2json(project_id,jpegs_dir_name)
        self.to_tifmask(trainIds,classes,rgb_dir, mask_dir_classes, stacked_dir)
        self.to_multipage(classes,mask_dir, mask_dir_classes, trainIds)

        return "success"
        # except Exception as e:
            # logger.error("Error: " + str(e))
