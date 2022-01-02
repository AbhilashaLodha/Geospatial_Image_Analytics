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


def create_masks_png(json_dir):
    for label_file in sorted(glob.glob(json_dir)):
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
        data = json.load(f)
        print(data['imagePath'])
        img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])

        img = np.asarray(Image.open(img_file))
        class_name_to_id = x
        print("class name to id",class_name_to_id)
        
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id
        )
        
        instance1 = np.copy(lbl)
        pos_2 = np.where(lbl == 2)
        instance1[pos_2] = 0
        instance1 = instance1 * 255
        return instance1


def create_mask_png_new(label_file,x):
    with open(label_file) as f:
        data = json.load(f)

        logger.info("imagepath")
        logger.info(data['annotation']['filename'])
        img_file = os.path.join(os.path.dirname(label_file), data['annotation']['filename'])

        img = np.asarray(Image.open(img_file))
        #class_name_to_id = {'__ignore__':-1,'_background_':0,'buildings':1,'trees':0,'roads':0,'crops':0,'water':0}
        class_name_to_id = x
        print("class name to id",class_name_to_id)
        logger.info("before lbl")

        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['annotation']['object'],
            label_name_to_value=class_name_to_id
            )
        logger.info("after lbl")
        instance1 = np.copy(lbl)
        pos_2 = np.where(lbl == 2)
        instance1[pos_2] = 0
        instance1 = instance1 * 255
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
        command = "tiffcp {}".format(multiple) + mask_dir +"/" + "{}".format(output)
        os.system(command)
        print("___",command)
    print("converted to multipage tif")
    return "succesful"
