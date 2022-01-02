import os
import json


def folder(input_type,project_name,project_id):
    parent_dir = "/home/ubuntu/cv-asset/Satelllite_WB/"+str(project_id)
    if not(os.path.exists(parent_dir)):
        os.mkdir(parent_dir)
    data_dir = parent_dir + "/data_dir"
    model_dir = parent_dir +"/model_dir"
    config_dir = parent_dir +"/config_dir"
    if not(os.path.exists(data_dir)):
        os.mkdir(data_dir)
    if not(os.path.exists(model_dir)):
        os.mkdir(model_dir)
    if not(os.path.exists(config_dir)):
        os.mkdir(config_dir)
    #data directory
    tarfile = data_dir + "/tarfile"
    rawbands = data_dir +"/rawbands" 
    clip_dir = data_dir+"/clippedbands"
    result_dir = data_dir +"/pred_masks"
    annotated_img_path = data_dir +"/gt_mband"
    indices_raw_dir = data_dir +"/indices_raw"
    indices_color_dir = data_dir +"/indices_color"
    rgb_mband_dir = data_dir +"/rgb_mband"
    pan_mband = data_dir+"/pan_mband"
    mask_dir = data_dir + "/gt_mband"
    mask_dir_classes = data_dir + "/gt_mband_classes"
    stacked_dir = data_dir + "/stacked"
    rgb_dir = data_dir +"/jpegs"
    mband = data_dir +"/mband"
    merged_bands = data_dir +"/merged_bands"
    thumbnails = data_dir +"/thumbnails"

    data_path_list = [tarfile,rawbands,clip_dir,result_dir,pan_mband,mask_dir,
                    annotated_img_path,indices_raw_dir,indices_color_dir,rgb_mband_dir,
                    mask_dir_classes,stacked_dir,rgb_dir,mband,merged_bands,thumbnails]

    for i in range(len(data_path_list)):
        if not(os.path.exists(data_path_list[i])):
            os.mkdir(data_path_list[i])
    #model directory
    plots_dir = model_dir +"/plots"
    weights_dir = model_dir+"/model"
    log_dir = model_dir+"/log_dir"


    model_path_list = [plots_dir,weights_dir,log_dir]
    for i in range(len(model_path_list)):
        if not(os.path.exists(model_path_list[i])):
            os.mkdir(model_path_list[i])
    return "success"