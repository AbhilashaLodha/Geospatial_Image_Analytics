import json
import logging
import multiprocessing
import os
import re
import string
import threading
import time
import json
import datetime
import ntpath
import cv2
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf
from keras import backend as K
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, send_file
from flask_restful import Resource, Api
from flask_uploads import configure_uploads
from flask_uploads import patch_request_class
from flask import Response,redirect,url_for, flash

from Satellite_WB.src.colour_composition_api_new import Colour_Composition
from Satellite_WB.src.gen_iou_api import Gen_IOU
from Satellite_WB.src.satellite_process_api import Satellite_Process
from Satellite_WB.src import pred_pow
from Satellite_WB.src import pred_demo
from Satellite_WB.src.display_result_api import Display
from Satellite_WB.src.display_models_api import Display_Models
from Satellite_WB.src.copy_jpg_xml_api import Copy_JPG_XML
from Satellite_WB.src.gdalinfo_parser_api import parsing_gdalinfo
from Satellite_WB.src.indices_api import calc_indices
from Satellite_WB.src.multipagetif_api import multipage
from Satellite_WB.src.annotation_pipeline_till_jpegs import Annotations1
from Satellite_WB.src.annotation_pipeline_after_labelme import Annotations2
from Satellite_WB.src.create_folder import folder
from Satellite_WB.src import train_unet
from Satellite_WB.src.unet_model_deeper import *
from Satellite_WB.src.gen_patches import *
from Satellite_WB.src.pansharpening import do_pansharpen as pan
from Satellite_WB.src.gen_patches import *
from Satellite_WB.src.split_train import train_test_split
from Satellite_WB.src.split_test import train_test_split as annotation_split
from Satellite_WB.src import vec_on_raster

@app.route("/generate_iou", methods=["POST"])
def generate_iou():
    logger.info("In Generate IOU")
    jsonObj = get_json_from_request(request)
    print("json object : ", jsonObj)
    # logger.info(" Json type>>>>>>>>>>>>>>>>>>> ", type(jsonObj))
    # logger.info(" Json request>>>>>>>>>>>>>>>>>>> " + str(jsonObj))
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        project_id = jsonObj['usecaseid']
        weights_path = jsonObj['weights_path']
        N_CLASSES = jsonObj['N_CLASSES']
        classes = jsonObj["classes_list"]

        #gt_folder = jsonObj['gt_folder']        
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Please provide the project id predicted mask images")
            errorFlag = True
        if not errorFlag:
            logger.info("going to generate_iou file")
            obj = Gen_IOU()
            output = obj.allFunctionCalls(project_id,weights_path,N_CLASSES,classes)
            base_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)
            data_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir"
            only_plots = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(project_id)+ "/model_dir/plots"
            thumbnail_plots = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/plots"
            if not os.path.exists(base_path1):
                logger.info("Stuck in  Line no 750")
                os.mkdir(base_path1)
            if not os.path.exists(data_path1):
                logger.info("Stuck in  Line no 750")
                os.mkdir(data_path1)
            if not os.path.exists(thumbnail_plots):
                logger.info("Stuck in  Line no 750")
                os.mkdir(thumbnail_plots)
            output2 = create_thumbnail_jpegs(thumbnail_plots,only_plots)
            logger.info("result: %s", output)
            result={}
            result["message"] = "success"
        
    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(result)

@app.route("/satellite_process", methods=["POST"])
def satellite_process():
    logger.info("In Satellite Process")
    jsonObj = get_json_from_request(request)
    logger.info("json object : " + str(jsonObj))
    # logger.info(" Json type>>>>>>>>>>>>>>>>>>> ", type(jsonObj))
    # logger.info(" Json request>>>>>>>>>>>>>>>>>>> " + str(jsonObj))
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        dataset = jsonObj['dataset']
        latitude = jsonObj['latitude']
        longitude = jsonObj['longitude']
        start_date = jsonObj['start_date']
        end_date = jsonObj['end_date']
        max_cloud_cover = jsonObj['max_cloud_cover']
        scene_name = jsonObj['scenename']
        project_id = jsonObj['usecaseid']
        logger.info("id----%s"%str(project_id))
        usecase_params6 = {}
        from Satellite_WB.src import db4
        usecase_params6['status'] = "Download_In_Progress"
        db4.updateUseCase(usecase_params6, project_id)
        errorFlag = False
        if (not dataset or dataset == None):
            logger.info("Please provide dataset for satellite processing")
            errorFlag = True
        if (not latitude or latitude == None):
            logger.info("Please provide latitude value")
            errorFlag = True
        if (not longitude or longitude == None):
            logger.info("Please provide longitude value")
            errorFlag = True
        if (not start_date or start_date == None):
            logger.info("Please provide the start_date")
            errorFlag = True
        if (not end_date or end_date == None):
            logger.info("Please provide the end_date")
            errorFlag = True
        if (not max_cloud_cover or max_cloud_cover == None):
            logger.info("Please provide max_cloud_cover value")
            errorFlag = True
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True        
        if not errorFlag:
            logger.info("going to satellite_process_api file")
            ob = Satellite_Process()
            clip_dir = ob.allFunctionCalls(dataset,latitude,longitude,start_date,end_date,max_cloud_cover,project_id)
            # logger.info("result: %s", result)
            #ob_pan = pan(project_id,clip_dir)
            result = {}
            output = "Success"
            result["message"] = output
            usecase_params6 = {}
            from Satellite_WB.src import db4
            usecase_params6['status'] = "Download_Completed"
            db4.updateUseCase(usecase_params6, project_id)
            #logger.info("result with pan: %s", output)                
    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(jsonObj))
        usecase_params6 = {}
        from Satellite_WB.src import db4
        usecase_params6['status'] = ""
        db4.updateUseCase(usecase_params6, project_id)

    return json.dumps(result)

@app.route("/display_result", methods=["POST"])
def display_result():
    logger.info("In display_result_api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    # errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        threshold = jsonObj['threshold']
        project_id = jsonObj["usecaseid"]
        weights_path = jsonObj["weights_path"]
        errorFlag = False
        if (not threshold or threshold == None):
            logger.info("Please provide threshold value")
            errorFlag = True
        if (threshold<=0 or threshold>1):
            #verify the limits (should be in intervals 0.1,0.2 to 0.9)
            logger.info("invalid parameter, please check") 
            errorFlag = True
        if not errorFlag:
            logger.info("going to display_result_api file")
            res = vec_on_raster.result_raster(project_id,weights_path,threshold)
            logger.info("res---%s"%res)
            #/home/ubuntu/cv-asset/Satellite_WB/6/model_dir/plots
            #output1 = create_thumbnail_jpegs(thumbnail_indices,indices_color)
            base_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)
            data_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir"
            model_name = path_leaf(weights_path).replace('.hdf5','')
            raster_path = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(project_id)+"/data_dir/pred_masks/"+str(model_name)+'/'+str(threshold)
            thumbnail_rasters = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/rasters"
            only_rasters = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/only_rasters/"
            raster_jpeg = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/rasters_jpegs"
            output_jpegs = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/output_jpegs"
            if not os.path.exists(base_path1):
                logger.info("Stuck in  Line no 847")
                os.mkdir(base_path1)
            if not os.path.exists(data_path1):
                logger.info("Stuck in  Line no 847")
                os.mkdir(data_path1)
            if not os.path.exists(thumbnail_rasters):
                logger.info("Stuck in  Line no 847")
                os.mkdir(thumbnail_rasters)
            if not os.path.exists(only_rasters):
                logger.info("Stuck in  Line no 850")
                os.mkdir(only_rasters)
            if not os.path.exists(raster_jpeg):
                logger.info("Stuck in  Line no 853")
                os.mkdir(raster_jpeg)
            if not os.path.exists(output_jpegs):
                logger.info("creating output jpegs folder")
                os.mkdir(output_jpegs)
            results = os.listdir(raster_path)
            logger.info("Results-------%s"%results)
            results1 = [x for x in results if x.startswith('raster')]
            logger.info("Results1-------%s"%results1)
            import shutil
            from shutil import copyfile
            #copyfile(src, dst)
            for img in results1:
                copyfile(raster_path+"/"+img,only_rasters +img)
                #n= tiff.imread(raster_path+"/"+img)
                #mg = path_leaf(img)
                #logger.info("i-------%s"%str(raster_path+"/"+img))
                #logger.info("Stuck in  writing satellite jpegs to folder")
                #cv2.imwrite(only_rasters +img,n)
            output = create_thumbnail_tiff(thumbnail_rasters,only_rasters,raster_jpeg)
            obj = Display()
            # rgb_files, result_tiffs = obj.display_result(threshold)
            if output:
                res_list = obj.display_result(threshold,project_id,weights_path)
            #res_list = str(res_list)
        
    except Exception as e:
        logger.error("Error while doing display result api: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(res_list)  


@app.route("/display_model", methods=["POST"])
def display_model():
    logger.info("In display_models_api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    # errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        project_id = jsonObj["usecaseid"]
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Please provide project_id value")
            errorFlag = True
        if not errorFlag:
            logger.info("going to display_models_api file")
            obj = Display_Models()
            # rgb_files, result_tiffs = obj.display_result(threshold)
            result = obj.display_models(project_id)
            #result = str(result)
            output = {}
            output["model list"] = result
        
    except Exception as e:
        logger.error("Error while doing display result api: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(output)   


@app.route("/copy_jpg_xml", methods=["POST"])
def copy_jpg_xml():
    logger.info("In copy_jpg_xml_api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    # errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        project_id = jsonObj["usecaseid"]
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Please provide project_id value")
            errorFlag = True
        if not errorFlag:
            logger.info("going to copy_jpg_xml_api file")
            obj = Copy_JPG_XML()
            # rgb_files, result_tiffs = obj.display_result(threshold)
            result = obj.allfunctions(project_id)
            #result = str(result)
            output = {}
            output["message"] = result
        
    except Exception as e:
        logger.error("Error while doing display result api: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(output)   


@app.route("/gdalinfo_parser", methods=["POST"])
def gdalinfo_parser():
    logger.info("In gdalinfo_parser api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        project_id = jsonObj["usecaseid"]
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to gdalinfo_parser_api file")
            obj = parsing_gdalinfo()
            output = obj.gdalinfo(project_id)
            logger.info("result: %s", output)
    except Exception as e:
        logger.error("Error while doing parsing gdalinfo api: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(output)

@app.route("/multipagetif", methods=["POST"])
def multipagetif():
    from Satellite_WB.src import satellite_config
    logger.info("In multipagetif api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    # errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        project_id = jsonObj["usecaseid"]
        classes_raw_bands = jsonObj["classes_raw_bands"]
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to multipagetif_api file")
            obj = multipage()
            output = obj.to_multipage(project_id,classes_raw_bands)
            logger.info("result: %s", output)
    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(str(output))


@app.route("/annotations1", methods=["POST"])
def annotations1():
    # from Satellite_WB.src import satellite_config
    # from Satellite_WB.Landsat_8Bands.config_dir import data_config_land

    logger.info("In annotations1 api")
    logger.info(request.data)
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    project_id = jsonObj["usecaseid"]
    errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to annotations1 file")
            '''

            dict_classes = {"class": ["vegetation", "water"]}
            classes = dict_classes["class"]
            data_dir = "/home/ubuntu/cv-asset/Satellite_WB/"+str(project_id)+"/data_dir"
            input_rgb_dir_name = "rgb_mband"
            input_dir_name = "pan_mband"
            mask_dir_single = "gt_mband_classes"
            mask_dir_name = "gt_mband"
            stacked_dir_name = "stacked"
            jpegs_dir_name = "jpegs"
            '''
            obj = Annotations1()           
            output = obj.annotation_allfuns(project_id)
            logger.info("result: %s", output)

            obj_indices = calc_indices()
            indices_raw_dir = obj_indices.indices_all(project_id)
            logger.info("Indices calculation done")

            obj_color = Colour_Composition()
            output_color = obj_color.colourcomposition(project_id,indices_raw_dir)
            logger.info("Color composition done")

            
            result = {}
            result["message"]= output
    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(result)

@app.route("/annotations2", methods=["POST"])
def annotations2():
    # from Satellite_WB.src import satellite_config
    # from Satellite_WB.Landsat_8Bands.config_dir import data_config_land

    logger.info("In annotations2 api")
    jsonObj = get_json_from_request(request)
    logger.info(" Json request " + str(jsonObj))
    project_id = jsonObj["usecaseid"]
    classes = jsonObj["classes_list"]
    errorFlag = False
    # Read  the model parameter dictionary with default parameters from config
    # def_args = inception_config.args
    # args = def_args.copy()
    K.clear_session()
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to annotations2 file")
            '''
            dict_classes = {"class": ["vegetation", "water"]}
            classes = dict_classes["class"]
            data_dir = "/home/ubuntu/cv-asset/Satellite_WB/"+str(project_id)+"/data_dir"
            input_rgb_dir_name = "rgb_mband"
            input_dir_name = "pan_mband"
            mask_dir_single = "gt_mband_classes"
            mask_dir_name = "gt_mband"
            stacked_dir_name = "stacked"
            jpegs_dir_name = "jpegs"
            '''
            obj = Annotations2()
            #output_anno =obj = Annotations1.annotation_pipeline()           
            output_anno = obj.annotation_allfuns(project_id,classes)
            logger.info("result: %s", output_anno)

            
            result = {}
            output = "success"
            result["message"]=output

            

    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(jsonObj))
    return json.dumps(result)

import ntpath
def path_leaf(path):
    # to get the name of the weights
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

@app.route("/demo_satellite", methods=["POST"])
def demo_satellite_complete():
    logger.info("In Demo sat api")
    K.clear_session()
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    weights_path = jsonObj["weights_path"] 
    project_id = jsonObj["usecaseid"]
    img_path = jsonObj["img_path"]
    N_CLASSES = jsonObj["N_CLASSES"]
    dict_model = jsonObj["dict_model"]
    dict_classes = jsonObj["dict_classes"]
    logger.info("Object--%s"%jsonObj)
    #thres_required = jsonObj["thres_required"] 
    #print("wp---",wp)
    errorFlag = False
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to main predict file")
            model_name = path_leaf(weights_path).replace('.hdf5','')
            threshold = 0.5
            demo_path = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+"/data_dir/demo_images"
            image_test = path_leaf(img_path)
            for i in os.listdir(demo_path):
                if not(i == image_test ):
                    os.remove(demo_path+"/"+i)

            raster_path = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+"/data_dir/pred_masks/"+str(model_name)+'/'+str(threshold)
            thumbnail_rasters = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/demo_output"
            only_demo_rasters = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/demo_rasters/"
            demo_final = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/demo_final"
            #output_jpegs = "/home/ubuntu/cv_worksp#ace/data/Satellite_WB/"+ str(project_id)+ "/data_dir/thumbnails/output_jpegs"
            if not os.path.exists(thumbnail_rasters):
                logger.info("Stuck in  Line no 847")
                os.mkdir(thumbnail_rasters)
            if not os.path.exists(only_demo_rasters):
                logger.info("Stuck in  Line no 848")
                os.mkdir(only_demo_rasters)
            if not os.path.exists(demo_final):
                logger.info("Stuck in  Line no 849")
                os.mkdir(demo_final)
            #global updated_weights_path
            #return_prediction
            out,response_dict = pred_demo.return_prediction(weights_path,project_id,img_path,N_CLASSES,dict_model,dict_classes)
            raster_comp = pred_demo.result_raster(project_id,weights_path,0.5,img_path)
            logger.info("Completed making raster file")
            
            
            results = os.listdir(raster_path)
            logger.info("Results-------%s"%results)
            results1 = [x for x in results if x.startswith('raster')]
            logger.info("Results1-------%s"%results1)
            import shutil
            from shutil import copyfile
            #copyfile(src, dst)
            for img in results1:
                copyfile(raster_path+"/"+img,only_demo_rasters +img)
            output = create_thumbnail_tiff(thumbnail_rasters,only_demo_rasters,demo_final)
            if out:
                output = {}
                output["message"]= "success"
                output["Path_of_rasters"]= (os.listdir(thumbnail_rasters)[0])#thumbnail_rasters + "/"+(os.listdir(thumbnail_rasters)[0])
                output["color_values"] = response_dict
            #logger.info("result: %s", output)
            logger.info("Demo api completed")
    
        
    except Exception as e:
        logger.error("Error while Prediction API: " + str(e) )

    return  json.dumps(output)

@app.route("/predict_mask", methods=["POST"])
def predict_mask():
    logger.info("In Pred Mask api")
    K.clear_session()
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    weights_path = jsonObj["weights_path"] 
    project_id = jsonObj["usecaseid"]
    N_CLASSES = jsonObj["N_CLASSES"]
    dict_model = jsonObj["dict_model"]
    dict_classes = jsonObj["dict_classes"]
    logger.info("Object--%s"%jsonObj)
    #img_path = jsonObj["img_path"] 
    #thres_required = jsonObj["thres_required"] 
    #print("wp---",wp)
    errorFlag = False
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("going to main predict file")
            #global updated_weights_path
            out,response_dict = pred_pow.return_prediction(weights_path,project_id,N_CLASSES,dict_model,dict_classes)
            if out:
                output = {}
                output["message"]= "success"
                output["Path of mask files"]= out
                output["color_values"] = response_dict
            #logger.info("result: %s", output)
            logger.info("Prediction api completed")
    
        
    except Exception as e:
        logger.error("Error while Prediction API: " + str(e) )

    return  json.dumps(output)


@app.route("/get_classes", methods=["POST"])
def get_class():
    logger.info("In get classes api for Satellite Processng")
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    class_array= jsonObj["classes"] 
    project_id = jsonObj["usecaseid"]
    #input_type = jsonObj["input_type"] 
    #img_path = jsonObj["img_path"] 
    #thres_required = jsonObj["thres_required"] 
    #print("wp---",wp)
    errorFlag = False
    try:
        if (not class_array or class_array == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if not errorFlag:
            logger.info("In get_classes method")
            parent_dir1 = "/home/ubuntu/cv-asset/Satellite_WB/"+str(project_id)
            config_dir = parent_dir1 +"/config_dir"
            config_file = config_dir + "/data_config1"+".py"
            # write to config file
            logger.info("Line no 1182")
            dict2 = {'class':class_array}
            #write_vars_to_file(sys.stdout, dict1)
            logger.info("Line no 1185")
            file1 = open(config_file, 'a')
            logger.info("Line no 1188")
            file1.write('\ndict_classes = ' + json.dumps(dict2)+"\n")
            file1.close()            
            output = "Success"
            result = {}
            result["message"] = output      
    except Exception as e:
        logger.error("Error in get_classes api: " + str(e) )

    return json.dumps(result)

@app.route("/create_folder", methods=["POST"])
def create():
    logger.info("In create folder api")
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    project_name= jsonObj["proj_name"] 
    project_id =jsonObj["usecaseid"] 
    input_type = jsonObj["input_type"] 
    #img_path = jsonObj["img_path"] 
    #thres_required = jsonObj["thres_required"] 
    #print("wp---",wp)
    errorFlag = False
    try:
        if (not project_name or project_name == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not input_type or input_type == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not(input_type == "raw" or input_type =="multiband")):
            #verify the upper limit
            logger.info("Invalid parameter, please check") 
            errorFlag = True
        if not errorFlag:
            logger.info("going to createfolder.py")
            #global updated_weights_path
            output = folder(input_type,project_name,project_id)
            logger.info("result: %s", output)
            result = {}
            result["message"]=output
            logger.info(" api completed")
    except Exception as e:
        logger.error("Error while creating  folders: " + str(e) )

    return json.dumps(result)



def tojpg(imag,path):
    logger.info("Stuck in Line no 1249")
    logger.info("converting to jpg")
    logger.info("Imag------------%s"%imag)
    logger.info("path------------%s"%path)
    logger.info("name of image--------%s"%glob.glob(imag))
    #from PIL import Image
    #import imageio
    #image = imageio.imread('example.tif')
    import subprocess

    for name in glob.glob(imag):
        logger.info("name----%s"%name)
        image_in = name       
        logger.info("In line no 1271")
        e = path_leaf(name)
        logger.info("e-------%s"%str(e))
        if(e.endswith('tif')):
            logger.info("It ends with small tif")
            b = e.replace('.tif', '')
        else:
            logger.info("It ends with big TIF")
            b = e.replace('.TIF', '')
        #b = e.replace('.TIF', '')
        image_out = path +"/"+ b + '.png'
        subprocess.call(["gdal_translate","-co", "TILED=YES", "-co", "COMPRESS=LZW","-ot", "Byte", "-scale", image_in,str(image_out)  ])
        logger.info("In line no 1268")
    logger.info("Conversion from tif/tiff to jpg completed!")

def create_thumbnail_tiff(thumbnail_clipped,clipped_bands,clipped_jpegs):
    clipped_images = os.listdir(clipped_bands)
    try:
        for i in clipped_images:
            tojpg(clipped_bands + i,clipped_jpegs) 
            logger.info("Jpeg conversion done")     
            #image_path = clipped_jpegs + "/"+ str(i).rstrip(".TIF")+".png"
            #e = path_leaf(name)
            #logger.info("e-------%s"%str(e))
            if(i.endswith('tif')):
                logger.info("It ends with small tif")
                b = i.replace('.tif', '')
            else:
                logger.info("It ends with big TIF")
                b = i.replace('.TIF', '')
            image_path = clipped_jpegs + "/"+ b+".png"
            #logger.info("name-----%s"%str(name))
            img = cv2.imread(image_path)
            logger.info("Image path----%s"%image_path)
            logger.info('Original Dimensions 1 : %s'%img.shape[1])
            logger.info('Original Dimensions 0: %s'%img.shape[0])
            scale_percent = 100 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(thumbnail_clipped + "/"+ b+".jpg",resized)
            #return thumbnail_clipped
    except Exception as e:
        logger.error("Error in create thumbnail tiff" +  str(e))
    return thumbnail_clipped

def create_thumbnail_jpegs(thumbnail_indices,indices_color):
    jpeg_images = os.listdir(indices_color)
    try:
        for i in jpeg_images:
            #tojpg(clipped_bands + i,clipped_jpegs) 
            logger.info("Already in png format")     
            #image_path = clipped_jpegs + "/"+ str(i).rstrip(".TIF")+".png"
            #e = path_leaf(name)
            #logger.info("e-------%s"%str(e))
            #b = i.replace('.png', '')
            image_path = indices_color + "/"+ i
            #logger.info("name-----%s"%str(name))
            img = cv2.imread(image_path)
            logger.info("Image path----%s"%image_path)
            logger.info('Original Dimensions 1 : %s'%img.shape[1])
            logger.info('Original Dimensions 0: %s'%img.shape[0])
            scale_percent = 100 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(thumbnail_indices + "/"+i ,resized)
            #return thumbnail_clipped
    except Exception as e:
        logger.error("Error in create thumbnail jpegs")
    return thumbnail_indices


@app.route("/thumbnail_satellite", methods=["POST"])
def create_tumbnail():
    req_json = get_json_from_request(request) # exected jsonArray
    try:
        useCaseId=req_json['usecaseid']
        logger.info("Line no 1253")
        base_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)
        data_path1 = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir"
        clipped_bands = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(useCaseId)+ "/data_dir/clippedbands/01/"     
        thumbnail_dir = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails"
        clipped_jpegs = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/clipped_jpegs"
        thumbnail_clipped = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/clippedbands"
        thumbnail_indices = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/indices_color"
        indices_color = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(useCaseId)+"/data_dir/indices_color/01"
        jpegs = "/home/ubuntu/cv-asset/Satellite_WB/"+ str(useCaseId)+"/data_dir/jpegs"
        only_jpegs = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/only_jpegs"
        thumbnail_jpegs = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/jpegs"
        rgb_jpegs="/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/thumbnails/rgb_jpegs"
        demo_folder = "/home/ubuntu/cv_workspace/data/Satellite_WB/"+ str(useCaseId)+ "/data_dir/demo_images"

        if not os.path.exists(base_path1):
            logger.info("Stuck in  Line no 1283")
            os.mkdir(base_path1)
        if not os.path.exists(data_path1):
            logger.info("Stuck in  Line no 1283")
            os.mkdir(data_path1)
        if not os.path.exists(thumbnail_dir):
            logger.info("Stuck in  Line no 1283")
            os.mkdir(thumbnail_dir)
        if not os.path.exists(thumbnail_clipped):
            logger.info("Stuck in  Line no 1286")
            os.mkdir(thumbnail_clipped)
        if not os.path.exists(clipped_jpegs):
            logger.info("Stuck in  Line no 1289")
            os.mkdir(clipped_jpegs)
        if not os.path.exists(thumbnail_indices):
            logger.info("Stuck in  Line no 1292")
            os.mkdir(thumbnail_indices)
        if not os.path.exists(only_jpegs):
            logger.info("Stuck in  Line no 1295")
            os.mkdir(only_jpegs)
        if not os.path.exists(thumbnail_jpegs):
            logger.info("Stuck in  Line no 1298")
            os.mkdir(thumbnail_jpegs)
        if not os.path.exists(rgb_jpegs):
            logger.info("Stuck in  Line no 1298")
            os.mkdir(rgb_jpegs)
        if not os.path.exists(demo_folder):
            logger.info("Stuck in  Line no 1301")
            os.mkdir(demo_folder)
        for img in glob.glob(jpegs+"/*.jpg"):
            n= cv2.imread(img)
            img = path_leaf(img)
            logger.info("i-------%s"%img)
            logger.info("Stuck in  writing satellite jpegs to folder")
            cv2.imwrite(only_jpegs +"/"+img,n)




        output = create_thumbnail_tiff(thumbnail_clipped,clipped_bands,clipped_jpegs)
        output1 = create_thumbnail_jpegs(thumbnail_indices,indices_color)
        output2 = create_thumbnail_jpegs(thumbnail_jpegs,only_jpegs)

        dir1 = thumbnail_clipped + '/'
        file_tif = os.listdir(dir1)
        for files in file_tif:
            if files.endswith("B1.jpg"):
                src= dir1 +str(files)
                dst = dir1+"B1.jpg"
                os.rename(src, dst)
            elif files.endswith("B2.jpg"):
                src= dir1 +str(files)
                dst =dir1 + "B2.jpg"
                os.rename(src, dst)
            elif files.endswith("B3.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B3.jpg"
                os.rename(src, dst)
            elif files.endswith("B4.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B4.jpg"
                os.rename(src, dst)
            elif files.endswith("B5.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B5.jpg"
                os.rename(src, dst)
            elif files.endswith("B6.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B6.jpg"
                os.rename(src, dst)
            elif files.endswith("B7.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B7.jpg"
                os.rename(src, dst)
            elif files.endswith("B8.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B8.jpg"
                os.rename(src, dst)
            elif files.endswith("B9.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B9.jpg"
                os.rename(src, dst)
            elif files.endswith("B10.jpg"):
                src=dir1 +str(files)
                dst =dir1 +"B10.jpg"
                os.rename(src, dst)
            elif files.endswith("B11.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"B11.jpg"
                os.rename(src, dst)
            elif files.endswith("BQA.jpg"):
                src= dir1 +str(files)
                dst =dir1 +"BQA.jpg"
                os.rename(src, dst)



        files_rb = os.listdir(thumbnail_jpegs)
        files_rgb = [x for x in files_rb if x.startswith('01')]
        import shutil
        for file_1 in files_rgb:
            logger.info("forming rgb jpegs")
            shutil.copy(thumbnail_jpegs + '/' + file_1, rgb_jpegs+'/'+ "RGB.jpg" )
        
        x_b8 = os.listdir(thumbnail_clipped)
        file_not_reqd = [x for x in x_b8 if x not in ["B1.jpg","B2.jpg","B3.jpg","B4.jpg","B5.jpg","B6.jpg","B7.jpg"]]
        logger.info(file_not_reqd)
        if(file_not_reqd):
            for i in file_not_reqd:
                os.remove(thumbnail_clipped+"/"+i) 


        result = {}
        result['message']= "success"
        result["clipped bands thumbnails"]= output
        result["colored indicesthumbnails"]= output1
        result["jpegs thumbnails"]= rgb_jpegs
    except Exception as e:
        logger.error("Error while Iconizing satellite images: "+str(e)+" input: "+str(req_json))
        #jsonObj['Success']="0"
    return json.dumps(result)


@app.route("/train_unet", methods=["POST"])
def training_unet(): #path/x.hdf5
    logger.info("In Training")
    #logger.info("path in train----%s"% sys.path)
    K.clear_session()
    jsonObj = get_json_from_request(request)
    print("json object : ", jsonObj)
    from Satellite_WB.src import satellite_config
    from Satellite_WB.src.satellite_config import weights_path
    #if not os.path.exists(weights_path):
    #open(weights_path, 'a').close()
    try:
        N_EPOCHS = jsonObj["N_EPOCHS"] 
        PATCH_SZ = jsonObj["PATCH_SZ"]  # should divide by 16 #160
        BATCH_SIZE = jsonObj["BATCH_SIZE"] #threshold = 67
        TRAIN_SZ = jsonObj["TRAIN_SZ"]
        VAL_SZ = jsonObj["VAL_SZ"]
        classes_raw_bands = jsonObj["classes_raw_bands"]
        project_id = jsonObj["usecaseid"]
        #train_size = jsonObj["train_size"]
        test_size = jsonObj["test_size"]
        N_CLASSES = jsonObj["N_CLASSES"]
        VAL_SZ = (TRAIN_SZ * VAL_SZ)/100
        logger.info("Valsize-----------%s"%VAL_SZ)
        
        resumption_flag = 0
        errorFlag = False
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not test_size or test_size == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        if (not N_EPOCHS or N_EPOCHS == None):
            logger.info("we are using  default value beacuse your input in None")
            N_EPOCHS = satellite_config.N_EPOCHS
            #errorFlag = True
        if (N_EPOCHS and not isinstance(int(N_EPOCHS), int)):
            logger.info("invalid N_EPOCHS type please enter numeric")
            errorFlag = True
        if (N_EPOCHS<=0 or N_EPOCHS>200):
            #verify the upper limit
            logger.info("invalid parameter, please check") 
            errorFlag = True
        if (not PATCH_SZ or PATCH_SZ == None):
            logger.info("we are using  default value beacuse your input in None")
            PATCH_SZ = satellite_config.PATCH_SZ
            #errorFlag = True
        if (PATCH_SZ and not isinstance(int(PATCH_SZ), int)):
            logger.info("invalid PATCH_SZ type please enter numeric")
            errorFlag = True
        if (PATCH_SZ<=0 or ((PATCH_SZ%16)!=0)):
            logger.info("invalid parameter, please check, Enter in multiples of 16") 
            errorFlag = True
        if (not BATCH_SIZE or BATCH_SIZE == None):
            logger.info("we are using  default value beacuse your input in None")
            BATCH_SIZE = satellite_config.BATCH_SIZE
            #errorFlag = True
        if (BATCH_SIZE and not isinstance(int(BATCH_SIZE), int)):
            logger.info("invalid BATCH_SIZE type please enter numeric")
            errorFlag = True
        if (BATCH_SIZE<=0):
            logger.info("invalid parameter, please check the value") 
            errorFlag = True
        if (not TRAIN_SZ or TRAIN_SZ == None):
            logger.info("we are using  default value beacuse your input in None")
            TRAIN_SZ = satellite_config.TRAIN_SZ
            #errorFlag = Teeerue
        if (TRAIN_SZ and not isinstance(int(TRAIN_SZ), int)):
            logger.info("invalid TRAIN_SZ type please enter numeric")
            errorFlag = True
        if (TRAIN_SZ<=0):
            logger.info("invalid parameter, please check") 
            errorFlag = True
        if (not VAL_SZ or VAL_SZ == None):
            logger.info("we are using  default value beacuse your input in None")
            VAL_SZ = satellite_config.VAL_SZ
            #errorFlag = True
        if (VAL_SZ and not isinstance(int(VAL_SZ), int)):
            logger.info("invalid VAL_SZ type please enter numeric")
            errorFlag = True
        if (VAL_SZ<=0):
            logger.info("invalid parameter, please check") 
            errorFlag = True
        if not errorFlag:
            logger.info("going to main train file")
            previous_weights_path = None  
            #global updated_weights_path
            if(resumption_flag ==1):
                logger.info(" need to resume training")
                previous_weights_path = None
                logger.info("part A")
                obj = multipage()
                mband_dir = obj.to_multipage(project_id,classes_raw_bands)
                o1 = train_test_split(test_size,project_id)
                o2 = annotation_split(test_size,project_id)
                N_BANDS = len(classes_raw_bands)
                output = train_unet.start_training(N_BANDS,N_CLASSES,N_EPOCHS,PATCH_SZ,BATCH_SIZE,TRAIN_SZ,VAL_SZ,previous_weights_path,project_id,mband_dir)
            else:
                logger.info("part B")
                logger.info("len---%d"%len(classes_raw_bands))
                #if(len(classes_raw_##bands))
                if (len(classes_raw_bands)>=1):
                    obj = multipage()
                    logger.info("multipaging started")
                    mband_dir = obj.to_multipage(project_id,classes_raw_bands)
                    logger.info("splitting train test")

                    o1 = train_test_split(test_size,project_id)
                    o2 = annotation_split(test_size,project_id)
                    logger.info("splitting done")

                    N_BANDS = len(classes_raw_bands)
                    logger.info("bands--%s"%N_BANDS)
                    output = train_unet.start_training(N_BANDS,N_CLASSES,N_EPOCHS,PATCH_SZ,BATCH_SIZE,TRAIN_SZ,VAL_SZ,previous_weights_path,project_id,mband_dir,test_size)
                else :
                    input_type = "raw"
                    logger.info("came in this loop")
                    if(input_type == "raw"):
                        logger.info("came here")
                        
                        classes_raw_bands =  ["B1","B2","B3","B4","B5","B6","B10"]
                        obj = multipage()
                        #mband_dir = obj.to_multipage(project_id,classes_raw_bands)
                        mask_dir = "/home/ubuntu/cv-asset/Satellite_WB/"+str(project_id)+"/data_dir/mband"
                        mband_dir = {"mask_dir":mask_dir}
                        logger.info("part second of B")
                        o1 = train_test_split(test_size,project_id)
                        o2 = annotation_split(test_size,project_id)
                        N_BANDS = len(classes_raw_bands)
                        logger.info("bands------%s"%N_BANDS)
                        output = train_unet.start_training(N_BANDS,N_CLASSES,N_EPOCHS,PATCH_SZ,BATCH_SIZE,TRAIN_SZ,VAL_SZ,previous_weights_path,project_id,mband_dir,test_size)
            logger.info("result: %s", output)
            logger.info("Training api completed")
    except Exception as e:
        from Satellite_WB.src import db2
        usecase_params = {}
        usecase_params['status'] = "training_error"
        usecase_params6 = {}
        from Satellite_WB.src import db4
        usecase_params6['status'] = ""
        db4.updateUseCase(usecase_params6, project_id)
        db2.updateUseCase(usecase_params, project_id)
        logger.error("Error while processing Training api: " + str(e) + " input: " + str(jsonObj))
        #output = str(e)

    return json.dumps(output)
@app.route("/pansharpen", methods=["POST"])
def pansharpening():
    logger.info("In the pan sharpening api")
    jsonObj = get_json_from_request(request)
    print("json object : ", jsonObj)
    from Satellite_WB.src.satellite_config import directory_pan
    #if not os.path.exists(weights_path):
    #open(weights_path, 'a').close()
    #directory_pan = jsonObj["directory_pan"] 
    if not os.path.exists(directory_pan):
        os.makedirs(directory_pan)

    try:
        logger.info("going to main file")
        #global updated_weights_path
        output = pan(jsonObj)
        logger.info("result: %s", output)
        
    except Exception as e:
        logger.error("Error while doing this: " + str(e) + " input: " + str(output))
    return output


@app.route("/folder_creation", methods=["POST"])
def folder_create():
    logger.info("In folder creation api end point of Medical WB")
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    project_id =jsonObj["usecaseid"] 
    # input_type = jsonObj["input_type"] 
    errorFlag = False
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        # if (not input_type or input_type == None):
        #     logger.info("Invalid value for this parameter")
        #     errorFlag = True
        # if (not(input_type == "dicom" or input_type =="nii")):
        #     #verify the upper limit
        #     logger.info("Invalid parameter, please check") 
        #     errorFlag = True
        if not errorFlag:
            logger.info("going to folder_creation api")
            output = folder(project_id)
            logger.info("result: %s", output)
            result = {}
            result["message"]=output
            logger.info("folder_creation api completed")
    except Exception as e:
        logger.error("Error while creating  folders: " + str(e) )

    return json.dumps(result)



@app.route("/analyzation", methods=["POST"])
def analyzation():
    logger.info("In analyze api end point of Medical WB")
    jsonObj = get_json_from_request(request)
    #print("json object : ", jsonObj)
    project_id =jsonObj["usecaseid"] 
    # input_type = jsonObj["input_type"] 
    errorFlag = False
    try:
        if (not project_id or project_id == None):
            logger.info("Invalid value for this parameter")
            errorFlag = True
        # if (not input_type or input_type == None):
        #     logger.info("Invalid value for this parameter")
        #     errorFlag = True
        # if (not(input_type == "dicom" or input_type =="nii")):
        #     #verify the upper limit
        #     logger.info("Invalid parameter, please check") 
        #     errorFlag = True
        if not errorFlag:
            logger.info("going to folder_creation api")
            obj = Analyze()
            output = obj.allfunctioncalls(project_id)
            logger.info("result: %s", output)
            result = {}
            result["message"]=output
            logger.info("analyzation api completed")
    except Exception as e:
        logger.error("Error while creating  folders: " + str(e) )

    return json.dumps(result)

