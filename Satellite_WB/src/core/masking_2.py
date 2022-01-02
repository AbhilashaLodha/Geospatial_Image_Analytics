# imports
import subprocess
import glob
import os
import json
import numpy as np
from PIL import Image
import labelme
import cv2
import gdal
import logging

logger = logging.getLogger('In masking api file')
logger.setLevel(logging.DEBUG)

'''
class_name_to_id = {}
class_name_to_id['__ignore__'] = -1
class_name_to_id['_background_'] = 0
class_name_to_id['buildings'] = 0
class_name_to_id['trees'] = 1
class_name_to_id['roads'] = 0
class_name_to_id['crops'] = 0
class_name_to_id['water'] = 0
'''

def create_masks_png(json_dir):
    for label_file in sorted(glob.glob(json_dir)):
    #for label_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        with open(label_file) as f:
            base = os.path.splitext(os.path.basename(label_file))[0]
            name = os.path.splitext(os.path.basename(label_file))[1]
            print(base, name, label_file)
            data = json.load(f)
            print(data['imagePath'])
            print(data['shapes'][0]['label'])
            img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
            img = np.asarray(Image.open(img_file))
            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            instance1 = np.copy(lbl)
            pos_2 = np.where(lbl == 2)
            instance1[pos_2] = 0
            instance1 = instance1 * 255
            cv2.imwrite(os.path.join(json_dir, base + '_mask' + '.png'), instance1)
            return instance1

def create_mask_png(label_file,x):
    with open(label_file) as f:
        #base = os.path.splitext(os.path.basename(label_file))[0]
        #name = os.path.splitext(os.path.basename(label_file))[1]
        #print(base, name, label_file)
        data = json.load(f)
        print(data['imagePath'])
        img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])

        # print("imagepath",data['annotation']['filename'])
        # img_file = os.path.join(os.path.dirname(label_file), data['annotation']['filename'])

        img = np.asarray(Image.open(img_file))
        '''
        for i in range(0,len(data['shapes'])):
            classname = str(data['shapes'][i]['label'])
            print("class name ",str(classname))
            id = class_name_to_id[classname]
        '''
        #class_name_to_id = {'__ignore__':-1,'_background_':0,'buildings':1,'trees':0,'roads':0,'crops':0,'water':0}
        class_name_to_id = x
        print("class name to id",class_name_to_id)

        
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id
        )
        

        # lbl = labelme.utils.shapes_to_label(
        #     img_shape=img.shape,
        #     shapes=data['annotation']['object'],
        #     label_name_to_value=class_name_to_id
        # )
        
        instance1 = np.copy(lbl)
        pos_2 = np.where(lbl == 2)
        instance1[pos_2] = 0
        instance1 = instance1 * 255
        #cv2.imwrite(os.path.join(json_dir, base + '_mask' + '.png'), instance1)
        return instance1


def create_mask_png_new(label_file,x):
    with open(label_file) as f:
        #base = os.path.splitext(os.path.basename(label_file))[0]
        #name = os.path.splitext(os.path.basename(label_file))[1]
        #print(base, name, label_file)
        data = json.load(f)
        #print(data['imagePath'])
        # img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])

        logger.info("imagepath")
        logger.info(data['annotation']['filename'])
        img_file = os.path.join(os.path.dirname(label_file), data['annotation']['filename'])

        img = np.asarray(Image.open(img_file))
        '''
        for i in range(0,len(data['shapes'])):
            classname = str(data['shapes'][i]['label'])
            print("class name ",str(classname))
            id = class_name_to_id[classname]
        '''
        #class_name_to_id = {'__ignore__':-1,'_background_':0,'buildings':1,'trees':0,'roads':0,'crops':0,'water':0}
        class_name_to_id = x
        print("class name to id",class_name_to_id)
        '''
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id
        )
        '''
        logger.info("before lbl")
        # dict_abc = []
        # dict_abc = data['annotation']['object']

        input = []
        x = data['annotation']['object']
        y = {}
        if(type(x)==type(y)):
            logger.info("checking>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
            input.append(x)
        else:
            input = x

        logger.info("input-- %s" %input)
        logger.info("type")
        logger.info(type(input))
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            # shapes=data['annotation']['object'],
            shapes = input,
            label_name_to_value=class_name_to_id
            )
        logger.info("after lbl")
        instance1 = np.copy(lbl)
        pos_2 = np.where(lbl == 2)
        instance1[pos_2] = 0
        instance1 = instance1 * 255
        #cv2.imwrite(os.path.join(json_dir, base + '_mask' + '.png'), instance1)
        return instance1




def mask_to_tiff(mask_arr, mask_tiff_file, ori_tiff_file):
    print(mask_arr.shape)

    # open dataset with update permission
    tif_image = gdal.Open(ori_tiff_file)

    # get the geotransform as a tuple of 6
    gt = tif_image.GetGeoTransform()
    print("geo transform", gt)
    pj = tif_image.GetProjection()
    print("projection",pj)

    band = tif_image.GetRasterBand(1)
    print(band.DataType)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape
    print("coloumns and rows", cols, rows)


    # create the new dataset
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(mask_tiff_file, rows, cols, 1)  # , gdal.GDT_UInt8
    outdata.SetGeoTransform(gt)  ##sets same geotransform as input
    outdata.SetProjection(pj)  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(mask_arr)
    outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
    print("done...", outdata.GetRasterBand(1).DataType)

    outdata.FlushCache()  ##saves to disk!!
    outdata = None

def batch_annotate(root_dir, batch_dir):
    command = 'activate common3 && cd ' + root_dir + ' && labelme ' + batch_dir
    print(command)
    print("in batch annotation")
    status = subprocess.run(command, shell=True)
    return status
    

def single_annotate(image_dir, image_path):
    command = 'activate common3 && cd ' + image_dir + ' && labelme ' + image_path
    print(command)
    subprocess.run(command, shell=True)


def tif_multipagetif(classes,mask_dir, mask_dir_classes,trainIds):
    lis=glob.glob(mask_dir_classes+"/"+"*.tif")
    print(len(lis))
    if(len(lis)==0):
        lis=glob.glob(mask_dir_classes+"/"+"*.TIF")
    lis = sorted(lis)
    logger.info("lis---%s"%lis)#all gtmband classes file
    #a = mask_dir+"/{}".format(output)
    
    z = 0
    logger.info("trainIds---%s"%trainIds)#[01,02,03]
    for img_id in trainIds:
        multiple = ""
        l1 = len(classes)
        logger.info("l1-----%s"%l1)#3
        print("len of classes",len(classes))
        for p in range(0,len(classes)+1):
            print("z",z)
            single = "{} ".format(lis[z])
            print(z,lis[z])
            z = z+1
            multiple = multiple + single
            #print("b>>>",multiple)
            if(z%(l1)==0):
                break
        output = img_id + ".tif"
        #command = "tiffcp {} ".format(multiple)+mask_dir+"{}".format(multiple,output)
        #command = "tiffcp {} /home/ubuntu/01_Satelllite_WB/Landsat_8Bands/data_dir/tes_gt/gt_mband/{}".format(multiple,output)
        command = "tiffcp {}".format(multiple) + mask_dir +"/" + "{}".format(output)
        os.system(command)
        print("___",command)
    print("converted to multipage tif")
    return "succesful"
    '''
    singleTif_dir = "data_add\gt_mband_classes\\"
    n  = "data_add\gt_mband_classes\\{}_natural_gas.tif".format(img_id)
    s  = "data_add\gt_mband_classes\\{}_solar.tif".format(img_id)
    h  = "data_add\gt_mband_classes\\{}_hydro.tif".format(img_id)
    outfile = "data_add\gt_mband\{}.tif".format(img_id)
    print("----",outfile)
    #command = 'C:\\"Program Files"\\IrfanView\\i_view64.exe' + ' /multitif=(data_coalpile\gt_mband\{}.tif,'.format(img_id) + n + "," + s +")/killmesoftly"
    command = 'C:\\Users\\chava.sindhu\\Downloads\\iview453_x64\\i_view64.exe' + ' /multitif=(data_add\gt_mband\{}.tif,'.format(img_id) + n + "," + s +","+ h +")/killmesoftly"
    print(command)
    status = subprocess.run(command, shell=True)
    print("converted to multipage tif")
    return status
    '''


    '''
    def tif_multipagetif():
        trainIds = [str(i).zfill(2) for i in range(1, 3)]  # all availiable ids: from "01" to "xx"
        for img_id in trainIds:
            b = "data\gt_mband_building\01.tif"
            t  = "data\gt_mband_trees\01.tif"
            command = "C:\\Program Files\\IrfanView\\i_view64.exe" + " /multitif=(data\gt_mband\{},".format(img_id), b,t+")/killmesoftly"
            #command = 'C:\\"Program Files"\\IrfanView\\i_view32.exe' + ' /multitif=(data\gt_mband\{},'.format(img_id) + file_list[index] + ")/killmesoftly"
            print(command)
            status = subprocess.run(command, shell=True)
            return status
    '''
    '''
    def tif_multipagetif(img_id):
        singleTif_dir = "data\gt_mband_classes\\"
        #b = "data\gt_mband_classes\\{}_buildings.tif".format(img_id)          
        #t  = "data\gt_mband_classes\\{}_trees.tif".format(img_id)
        #r  = "data\gt_mband_classes\\{}_roads.tif".format(img_id)
        c  = "data\gt_mband_classes\\{}_crops.tif".format(img_id)
        w  = "data\gt_mband_classes\\{}_water.tif".format(img_id)
        outfile = "data\gt_mband\{}.tif".format(img_id)
        print(outfile)
        #command = "C:\\'Program Files'\\IrfanView\\i_view64.exe" + " /multitif=(data\gt_mband\{},".format(img_id) +")/killmesoftly"
        #command = os.path.join("C:\Program Files\IrfanView\i_view64.exe" ,multitif=('data\gt_mband\{}.tif'.format(img_id),b,t), "killmesoftly")
        #command = 'C:\\"Program Files"\\IrfanView\\i_view64.exe' + ' /multitif=(data\gt_mband\{}.tif,'.format(img_id) + b + "," + t+"," + r+"," + c+"," + w+ ")/killmesoftly"
        command = 'C:\\"Program Files"\\IrfanView\\i_view64.exe' + ' /multitif=(data\gt_mband\{}.tif,'.format(img_id) + c + "," + w+ ")/killmesoftly"
        print(command)
        status = subprocess.run(command, shell=True)
        print("converted to multipage tif")
        return status
    '''