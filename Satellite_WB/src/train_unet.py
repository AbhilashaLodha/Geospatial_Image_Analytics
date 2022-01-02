from Satelllite_WB.src.unet_model_deeper import unet_model
from Satelllite_WB.src.gen_patches import get_patches
from Satelllite_WB.src import db2
from Satelllite_WB.src import db3
from Satelllite_WB.src import db4
from keras import backend as K

import os.path
import sys
import numpy as np
import tensorflow as tf
import tifffile as tiff
from datetime import datetime
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import glob
import sys
from Satelllite_WB.src.satellite_config import UPCONV
from PIL import Image 
import time
import os
import logging
import json
from keras import backend as K

logger = logging.getLogger('Training unet........')
logger.setLevel(logging.DEBUG)

def start_training(N_BANDS,N_CLASSES,N_EPOCHS,PATCH_SZ,BATCH_SIZE,TRAIN_SZ,VAL_SZ,previous_Weights_path,project_id,mband_dir,test_size):
    with tf.device('/device:GPU:0'):
        logger.info("reached Trainign method")
        K.clear_session()
        test_size = float(test_size)
        usecase_params3 = {}
        usecase_params3['usecaseid'] = str(project_id)
        usecase_params3['status'] = "training_in_progress"
        db3.updateUseCase(usecase_params3, project_id)
        usecase_params4 = {}
        usecase_params4['status'] = "Training_In_Progress"
        db4.updateUseCase(usecase_params4, project_id)
        #PYTHONPATH="/home/ubuntu/cv-asset/Satelllite_WB/Indices_2Classes/config_dir/data_config1.py"

        logger.info("Proj id--%s" % project_id)
        train_img_path = mband_dir['mask_dir'] + "/train"

        '''
        if (input_type=="raw"):
            train_img_path = dict1['parent_dir']+"/data_dir/pan_mband"
        else:
            train_img_path = dict1['parent_dir']+"/data_dir/mband"
        '''
        annotated_img_path = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/gt_mband"+"/train"
        logger.info("annotation path--%s"%annotated_img_path)
        log_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id) +"/model_dir/logdir"
        logs = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/model_dir/logdir/"
        weights_path = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/model_dir/model/unet_weights_"+ str(project_id) +".hdf5"
        logger.info("Weights--%s" % weights_path)
        logger.info("Training path-- %s" % train_img_path)
        def read_tiff(path):
            img = Image.open(path)
            images = []
            for i in range(img.n_frames):
                img.seek(i)
                images.append(np.array(img))
            return np.array(images)
        import os
        import shutil
        from os import path
        from shutil import make_archive
        from zipfile import ZipFile
        import subprocess
        
        def zipfile(file1):
            src = path.realpath(file1)
            #print("src--",src)
            root_dir,tail = path.split(src)
            #print("root_dir--",root_dir)
            #print("tail--",tail)
            zipfile = tail.replace('.hdf5','')+'.zip'
            f_path = root_dir+'/'+ zipfile
            if not(os.path.isfile(f_path)):
                logger.info("Not a zip file")
                logger.info(zipfile)
                cmd = 'zip -j ' + "/home/ubuntu/cv-asset/Satelllite_WB/"+str(project_id)+"/model_dir/model/"+ zipfile + ' ' + "/home/ubuntu/cv-asset/Satelllite_WB/"+str(project_id)+"/model_dir/model/" +str(file1)
                subprocess.getstatusoutput(cmd)
                logger.info("done")
            else:
                logger.info("already zip present")


        def normalize(img):
            #Normalize the image
            min = img.min()
            logger.info(" min pixel value"+ str(min))
            max = img.max()
            logger.info(" max pixel value"+ str(max))
            x = 2.0 * (img - min) / (max - min) - 1.0
            return x



        def get_model():
            #unet model
            return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)



            #os.makedirs(weights_path)
        dir1 = annotated_img_path
        count = len(os.listdir(dir1))#dir is your directory path as string
        logger.info("len---%s"%count)
        #trainIds = [str(i).zfill(2) for i in range(1, count+1)]  # all availiable ids: from "01" to "24"
        trainIds = os.listdir(dir1)
        logger.info("Line no 60")
        X_DICT_TRAIN = dict()
        Y_DICT_TRAIN = dict()
        X_DICT_VALIDATION = dict()
        Y_DICT_VALIDATION = dict()
        logger.info('Reading images')
        logger.info("Classes count --%s"%N_CLASSES)
        CLASS_WEIGHTS = []
        for i in range(0,N_CLASSES):
            CLASS_WEIGHTS.append(float(1/N_CLASSES))
        logger.info("weights--%s"%CLASS_WEIGHTS)
        logger.info("Train image path %s"%train_img_path)
        logger.info("annotated_img_path--%s"%annotated_img_path)
        logger.info("ids----%s"%trainIds)
        for img_id in trainIds:
            img_m = normalize(read_tiff(train_img_path+'/{}'.format(img_id)).transpose([1, 2, 0]))
            logger.info("img_id---%s"%img_id)
            x = read_tiff(annotated_img_path+'/{}'.format(img_id))
            #logger.info("shape---%s"%x.shape)
            mask = x.transpose([1, 2, 0]) / 255
            train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
            X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
            Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
            X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
            Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
            logger.info(img_id + ' read')
        logger.info('Images were read')

        def train_net():
            logger.info("start train net")
            logger.info("not able to get patches due to bad data")
            x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
            logger.info("able to get 1 patches due to bad data")
            x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
            logger.info("able to get both patches due to bad data")
            logger.info(N_CLASSES)
            logger.info(PATCH_SZ)
            logger.info(N_BANDS)
            logger.info(CLASS_WEIGHTS)
            model = unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)
            logger.info("Line no 92")
            #if os.path.isfile(weights_path):
             #   model.load_weights(weights_path)
            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
            # Part of code for early stopping, which can be uncommented if required
            #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
            #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.01)
            logger.info("Line no 103")

            # Part of code for check pointing which can be commented if required
            #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
            csv_logger = CSVLogger(logs+'log_unet.csv', append=True, separator=';')
            #...create a graph...
            # Launch the graph in a session.
            #sess = tf.Session()
            # Create a summary writer, add the 'graph' to the event file.
            #writer = tf.summary.FileWriter('./tensorboard_unet/', sess.graph)
            logger.info("Line no 113")
            #tensorboard  --logdir=tensorboard/ --host ec2-18-235-174-183.compute-1.amazonaws.com --port 6006
            logger.info("Line no 115")
            start = time.time()
            
            tensorboard = TensorBoard(log_dir=log_dir,write_graph=True, write_images=True)
            #with tf.device('/GPU:0'):
            logger.info("Classes count --%s"%N_CLASSES)
            history_callback = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,verbose=1, shuffle=True,callbacks=[model_checkpoint, csv_logger, tensorboard] ,validation_data=(x_val, y_val))
            logger.info("History----%s"%str(history_callback))
            loss = history_callback.history["loss"]
            val_loss = history_callback.history["val_loss"]
            end = time.time()
            total_time = end - start
            logger.info("total time---%s"%total_time)
            logger.info("Loss---%s"%loss)
            logger.info("Val Loss---%s"%val_loss)
            return model,total_time,loss,val_loss

        
        with tf.Session() as sess:
            model,total_time,loss,val_loss = train_net()
            timestr = time.strftime("%Y%m%d_%H%M%S")
            #logger.info(timestr)
            k=str(weights_path)
            result = {}
            b = k.replace('.hdf5', '')
            #logger.info("b--",b)
            global updated_weights_path
            updated_weights_path = b+timestr+'.hdf5'
            logger.info("line no 148")
            os.rename(weights_path,updated_weights_path)
            
            now = datetime.now()
            current_time = now. strftime("%H:%M:%S")
            from datetime import date
            today = date.today()
            # dd/mm/YY
            d1 = today.strftime("%d/%m/%Y")
            #K.clear_session()
            import ntpath
            def path_leaf(path):
                # to get the name of the weights
                head, tail = ntpath.split(path)
                return tail or ntpath.basename(head)
            result["message"] = "success"
            result['modelName'] = path_leaf(updated_weights_path)
            result['testSplit'] = test_size
            result['trainSplit'] = (100- (test_size* 100))/100
            result["architecture"]="UNET"
            if val_loss[0]>=1:
                result["valAccuracy"]= 0.05
            else:
                result["valAccuracy"]= str(abs(round(100*(1 - val_loss[0]), 2)))
            result["trainAccuracy"]= str(round(100*(1 - loss[0]), 2))
            result["total_time_taken"]= round(total_time,2)
            result["batchSize"] = BATCH_SIZE
            result["patchSize"]= PATCH_SZ
            result["epochs"]= N_EPOCHS
            result["trainingDate"]= d1
            result["trainingTime"]= current_time
            dict2 = {'training_result':result}
            usecase_params = {}
            usecase_params['status'] = "training_completed"
            usecase_params['trainingdate'] = str(d1)
            usecase_params['trainingtime'] = str(current_time)
            usecase_params['architecture'] = "UNET"
            usecase_params['trainaccuracy'] = str(round(100*(1 - loss[0]), 2))
            usecase_params['valaccuracy'] = str(round(100*(1 - val_loss[0]), 2))
            usecase_params['traintestsplit'] = str((100- (test_size* 100))/100)
            usecase_params['epochs'] = str(N_EPOCHS)
            usecase_params['batchsize'] = str(BATCH_SIZE)
            usecase_params['patchsize'] = str(PATCH_SZ)
            usecase_params['modelname'] = str(path_leaf(updated_weights_path))
            db2.updateUseCase(usecase_params, project_id)
            usecase_params6 = {}
            usecase_params6['status'] = "Ready_to_test"
            usecase_params6['trainaccuracy'] = str(round(100*(1 - loss[0]), 2))
            db4.updateUseCase(usecase_params6, project_id)
            zipfile(path_leaf(updated_weights_path))
            logger.info("Training completed")
            return result


