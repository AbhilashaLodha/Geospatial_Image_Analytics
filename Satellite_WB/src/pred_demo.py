import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import matplotlib.patches as mpatches

from Satelllite_WB.src.unet_model_deeper import *
from Satelllite_WB.src.gen_patches import *
import os.path
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import sys
from Satelllite_WB.src.satellite_config import UPCONV,thres,color_values,order_values,keys
import ntpath
#import numpy as np
from osgeo import gdal, gdal_array
import json
from PIL import Image
import logging
from keras import backend as K
logger = logging.getLogger('Predicting unet........')
logger.setLevel(logging.DEBUG)
#import sys


def return_prediction(weights_path,project_id,img_path,N_CLASSES,dict_model,dict_classes):

    K.clear_session()
    weights_path = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id) +"/model_dir/model/"+weights_path
    temp = dict_classes["class"]
    temp = sorted(temp)
    logger.info("Json------%s"%temp)
    #logger.info("path-----------%s",pt)N_EPOCHS,PATCH_SZ,BATCH_SIZE
    class_dict = {}
    for i in range(len(temp)):
        class_dict[i]= temp[i]
    #class_dict={0:"vegetation",1:"water"}
    print("cl---dict",class_dict)
    #class_dict
    N_EPOCHS = dict_model["modelnames"][0]["epochs"]#70
    logger.info("No of epochs is %s"%N_EPOCHS)
    PATCH_SZ =dict_model["modelnames"][0]["Patch_size"]#128
    logger.info("Patch size is %s"%PATCH_SZ)
    BATCH_SIZE = dict_model["modelnames"][0]["batch_Size"]#50
    CLASS_WEIGHTS = []
    for i in range(0,N_CLASSES):
        CLASS_WEIGHTS.append(float(1/N_CLASSES))
    logger.info("Proj id--%s" % project_id)
    #CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]


    def normalize(img):
        #Normalize the image
        min = img.min()
        logger.info(" min pixel value" + str(min))
        max = img.max()
        logger.info(" max pixel value"+ str(max))
        x = 2.0 * (img - min) / (max - min) - 1.0
        return x


    def get_model():
        #unet model
        return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)



    def predict(x, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES):
        img_height = x.shape[0]
        img_width = x.shape[1]
        n_channels = x.shape[2]
        # make extended img so that it contains integer number of patches
        npatches_vertical = math.ceil(img_height / patch_sz)
        npatches_horizontal = math.ceil(img_width / patch_sz)
        extended_height = patch_sz * npatches_vertical
        extended_width = patch_sz * npatches_horizontal
        ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
        # fill extended image with mirrors:
        ext_x[:img_height, :img_width, :] = x
        for i in range(img_height, extended_height):
            ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
        for j in range(img_width, extended_width):
            ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

        # now we assemble all patches in one array
        patches_list = []
        for i in range(0, npatches_vertical):
            for j in range(0, npatches_horizontal):
                x0, x1 = i * patch_sz, (i + 1) * patch_sz
                y0, y1 = j * patch_sz, (j + 1) * patch_sz
                patches_list.append(ext_x[x0:x1, y0:y1, :])
        # model.predict() needs numpy array rather than a list
        patches_array = np.asarray(patches_list)
        # predictions:
        patches_predict = model.predict(patches_array, batch_size=4)
        prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
        for k in range(patches_predict.shape[0]):
            i = k // npatches_horizontal
            j = k % npatches_vertical
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
        return prediction[:img_height, :img_width, :]


    def picture_from_mask(mask,class_dict,temp,threshold=0):
        # map the mask file according to different colors for different classes
        colors =  {}
        keys = range(N_CLASSES)
        for k in keys:
            colors[k] = color_values[k]
        #logger.info("colors---",colors)
        z_order={}
        order_keys = range(1, N_CLASSES+1)
        for m in order_keys:
            z_order[m] = order_values[m-1]
        #logger.info("order--",z_order)
        
        existing_color = {
            "water":[135, 204, 250],
            "vegetation":[144,238,144],
            "buildings":[240,230,140]

        }
        #class_dict={0:"vegetation",1:"water"}
        pict = []
        pict.append(255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8))
        j= 1
        response_dict = {}
        logger.info("temp--%s"%temp)
        for w in temp:
            response_dict[w] = w
        #print("dict1------",response_dict)

        for i in range(1,N_CLASSES+1):
            pict.append(255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8))
            cl = z_order[i]
            #ist part temp = dict_classes["class"]
            l = class_dict[cl]
            
            #print("l---",l)
            if (l in existing_color):
                #print("-------",temp[cl])
                response_dict[str(temp[cl])]=existing_color[l]
                #print("dict2------",response_dict)
                for ch in range(3):
                    if j == i:
                        pict[j][ch,:,:][mask[cl,:,:] > threshold] = existing_color[l][ch]
                j = j +1
            # use different color
            else:
                # continue
                #print("-------",temp[cl])
                response_dict[temp[cl]]=colors[cl]
                #print("dict3------",response_dict)
                for ch in range(3):
                    if j == i:
                        pict[j][ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
                j = j +1
        logger.info("dict4 final------%s"%str(response_dict))	
        return pict,response_dict
                
    def path_leaf(path):
        # to get the name of the weights
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    def read_tiff(path):
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)
    def check(thres):
        # to check the path of the output folder asked by the user
        result_path = result_dir +"/{}".format(thres) 
        return result_path


    logger.info("Reached line no 187")
    logger.info(str(N_CLASSES))
    
    logger.info(str(PATCH_SZ))
    logger.info(str(CLASS_WEIGHTS))
    logger.info("ip---%s"%img_path)
    img = Image.open(img_path)
    img.load()
    N_BANDS = img.n_frames
    logger.info(str(N_BANDS))
    id = path_leaf(img_path)
    test_id = id.replace('.tif','')
    model = get_model()
    logger.info("Reached line no 189")
    model.load_weights(weights_path)
    #logger.info("test id",test_id)
    img = normalize(read_tiff(img_path).transpose([1,2,0])) 
    logger.info("In line no 216")  # make channels last
    mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1]) 
    #mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])  # make channels first
    #map = picture_from_mask(mask, 0.5)
    map = [] 
    #logger.info("y--",weights_path)
    result_dir = "/home/ubuntu/cv_workspace/data/Satelllite_WB/" +str(project_id)+"/data_dir/pred_masks"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    #img_path = dict1['parent_dir']+"/data_dir/mband/02.tif"
    logger.info("Result dir--%s" % result_dir)
    e = path_leaf(weights_path)
    b = e.replace('.hdf5', '')
    #logger.info("e-------",e)
    thres = 0.5
    middle_folder =  "{}".format(b) 
    middle_path =os.path.join(result_dir, middle_folder)
    if not os.path.isdir(middle_path):
        os.mkdir(middle_path)

    res = []
    map,response_dict = picture_from_mask(mask,class_dict,temp,thres)
    #logger.info("j---",j)
    thres_folder =  "{}".format(thres) 
    result_path =os.path.join(middle_path, thres_folder) 
    #logger.info("path--",result_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    #tiff.imsave(result_path + '/'+test_id+"result.tif",mask)
    for i in range(1,N_CLASSES+1):
        #logger.info("path--",result_path)
        res.append(result_path+'/'+test_id + '_map_0{}.tif'.format(i))
        logger.info(res[i-1])
        tiff.imsave(res[i-1],map[i])
        #result_path_threshold = check(thres_required)
    
    e = path_leaf(weights_path)
    b = e.replace('.hdf5', '')
    result_path_final = result_dir + "/"+b
    logger.info("reached line 162")
    from colormap import rgb2hex
    dict3 = response_dict
    y = []
    for val in dict3.values(): 
        logger.info(val)
        x = rgb2hex(val[0],val[1],val[2])
        x = str(x)
        logger.info("x---%s"%x)
        y.append(x)
    logger.info("y---%s"%y)
    z = []
    for i in dict3:
        z.append(i)
    logger.info("z---%s"%z)
    out = {}
    keys = range(N_CLASSES)
    for k in keys:
        out[z[k]] = y[k]

    logger.info("out---%s"%out)
        #K.clear_session()
    #sys.path.remove(os.path.dirname(os.path.expanduser(data_config_var)))
    return result_path_final,out

class Overlay:
        
    def vec_on_raster(self,raster, vector_dir, band, output_raster,test_id):
        # Open background raster image
        read_band = tiff.imread(raster)

        # Select band of image to be displayed
        raster_band = read_band[band]

        raster_band = raster_band/raster_band.max()
        raster_band = raster_band*255
        
        # Reshape image for ease in handling channels
        raster_band = np.reshape(raster_band,(-1, raster_band.shape[0], raster_band.shape[1]))

        # Check if raster is 3 channel as mask to be overlayed is RGB
        if raster_band.shape[0]<3:
            # Create dummy channels of same shape
            ch = np.zeros((3, raster_band.shape[1], raster_band.shape[2]))
            ch[0] = raster_band
            ch[1] = raster_band
            ch[2] = raster_band
        else:
            pass
        
        # Read masks in vector directory
        masks = os.listdir(vector_dir)
        logger.info("masks---%s"%masks)
        for mask in masks:
            if mask.startswith(test_id):
                vector = tiff.imread('{}/{}'.format(vector_dir, mask))
                vector = np.array(vector) 
                logger.info("mask--%s"%mask)
                #print("loop will run ---",vector.shape[0])
                # Check polygon pixel values in mask bands
                min_px = []
                # Check polygon pixel values in mask bands
                for i in range(vector.shape[0]):
                    print('Value of polygon in band {}: {}'.format(i+1, vector[i].min()))
                    logger.info("min-----------%s"%(str(vector[i].min() )    )        ) 
                    min_px.append(vector[i].min())
                
                if len(set(min_px))==1 and min_px[0]==255:
                    print('Found blank mask...')
                    continue
                else:
                    #Insert polygons in background raster
                    for i in range(ch.shape[0]):
                        ch[i] = np.where(vector[i] == vector[i].min(), vector[i].min(), ch[i])
                
                
        # Create final raster image with class polygons(Class colors included) 
        gdal_array.SaveArray(ch.astype("float32"), '{}'.format(output_raster), "GTIFF")
        return raster_band, ch, min_px

def path_leaf(path):
    # to get the name of the weights
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
    
def result_raster(project_id,weights_path,thres,img_path):
    raster = img_path
    #logger.info("i---%s"%images)
    logger.info("project_id")
    id= path_leaf(raster)
    test_id = id.replace('.tif','')
    model_name = weights_path.replace('.hdf5','')
    logger.info("Model name----%s"%model_name)
    vector_dir = '/home/ubuntu/cv_workspace/data/Satelllite_WB/'+str(project_id)+'/data_dir/pred_masks/'+str(model_name)+'/'+str(thres)
    band = 4
    logger.info("line no 73")
    output_raster = '/home/ubuntu/cv_workspace/data/Satelllite_WB/'+str(project_id)+'/data_dir/pred_masks/'+str(model_name)+'/'+str(thres)+ '/raster{}.tif'.format(test_id)
    o1 = Overlay()
    logger.info("raster---%s"%raster)
    logger.info("output raster----%s"%output_raster)
    x = o1.vec_on_raster(raster, vector_dir, band, output_raster,test_id)
    #logger.info("res---------%s"%str(x))
    return "success"







    
    
