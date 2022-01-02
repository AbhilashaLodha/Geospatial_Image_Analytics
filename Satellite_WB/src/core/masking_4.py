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

class_name_to_id = {}
class_name_to_id['__ignore__'] = -1
class_name_to_id['_background_'] = 0
class_name_to_id['buildings'] = 0
class_name_to_id['trees'] = 0
class_name_to_id['roads'] = 0
class_name_to_id['crops'] = 1
class_name_to_id['water'] = 0

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

def create_mask_png(label_file):
    with open(label_file) as f:
        base = os.path.splitext(os.path.basename(label_file))[0]
        name = os.path.splitext(os.path.basename(label_file))[1]
        print(base, name, label_file)
        data = json.load(f)
        print(data['imagePath'])
        img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
        img = np.asarray(Image.open(img_file))
        '''
        for i in range(0,len(data['shapes'])):
            classname = str(data['shapes'][i]['label'])
            print("class name ",str(classname))
            class_name_to_id1 = class_name_to_id[classname]
        '''

        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        
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

def tif_multipagetif(img_id):
    b = "data\gt_mband_building\\{}.tif".format(img_id)          
    t  = "data\gt_mband_trees\\{}.tif".format(img_id)
    outfile = "data\gt_mband\{}.tif".format(img_id)
    print(outfile)
    #command = "C:\\'Program Files'\\IrfanView\\i_view64.exe" + " /multitif=(data\gt_mband\{},".format(img_id) +")/killmesoftly"
    #command = os.path.join("C:\Program Files\IrfanView\i_view64.exe" ,multitif=('data\gt_mband\{}.tif'.format(img_id),b,t), "killmesoftly")
    command = 'C:\\"Program Files"\\IrfanView\\i_view64.exe' + ' /multitif=(data\gt_mband\{}.tif,'.format(img_id) + b + "," + t+ ")/killmesoftly"
    print(command)
    status = subprocess.run(command, shell=True)
    print("converted to multipage tif")
    return status