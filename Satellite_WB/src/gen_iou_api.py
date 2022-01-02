import tifffile as tiff
import numpy as np
import os
import sys
from osgeo import gdal
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.markers
import pandas as pd
import logging
logger = logging.getLogger('geniou')
logger.setLevel(logging.DEBUG)
import ntpath

def path_leaf(path):
    # to get the name of the weights
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class Gen_IOU: 

    def calculateIOU(self,test_run,gt_folder,N_CLASSES):

        # test_run contains predicted output imgs for various thresholds 
        thresholds= listdir(test_run)
        thresholds.sort()
        # gt_folder contains ground truth imgs
        gt=listdir(gt_folder)
        gt.sort()
        logger.info("gt-------%s"%gt)
        N_TEST_IMG = len(gt)
        logger.info("N_TEST-----%s"%N_TEST_IMG)

        iou_dict = {}
        iou_dictf ={}

        # For each threshold, inside each class, we are calculating the thresholds for each test img
        for threshold in thresholds: # for eg total thresholds - 9
            results = listdir("{}/{}".format(test_run,threshold))
            results.sort()
            logger.info("Line no 45")
            results = [x for x in results if not x.startswith('raster')]
            logger.info("res--%s"%results)
            logger.info("Line no 47")

            for m in range(N_CLASSES):
                logger.info("N_CLASSES--%s"%N_CLASSES) # for eg number of classes - 5
                if m in iou_dict:
                    pass
                else:
                    iou_dict[m] = []

                iou_score_list = []
                logger.info("Line no 58")
                for j in range(N_TEST_IMG):  # for eg number of test images - 5
                    logger.info("in loop after line 63")
                    
                    i=j*N_CLASSES+m
                    logger.info("i is %s" %i)
                    ori = tiff.imread('{}/{}'.format(gt_folder,gt[j]))

                    if N_CLASSES == 1 :
                        logger.info("no of clases = 1")
                        ori = np.expand_dims(ori,axis=0)
                        logger.info("ori shape before is")
                        logger.info(ori.shape)
                        ori = ori[m]   
                        logger.info("ori shape after")
                        logger.info(ori.shape)

                    elif N_CLASSES > 1 :
                        logger.info("no of clases > 1")
                        logger.info("ori shape before is")
                        logger.info(ori.shape)
                        ori = ori[m]   
                        logger.info("ori shape after")
                        logger.info(ori.shape)

                    ds = gdal.Open('{}/{}/{}'.format(test_run,threshold,results[i]))
                    logger.info("ds")
                    b = ds.GetRasterBand(2)
                    pred = b.ReadAsArray()
                    a = pred.min()
                    logger.info("pred shape")
                    logger.info(pred.shape)
                    result = np.zeros(pred.shape) 
                    for x in range(0, pred.shape[0]):
                        for y in range(0, pred.shape[1]):
                            if pred[x, y] == a:
                                result[x, y] = 255 
                    xyz = ori.flatten()
                    logger.info("ori flatten shape")
                    logger.info(xyz.shape)
                    intersection = np.logical_and(ori.flatten(), result.flatten())
                    logger.info("before union")
                    union = np.logical_or(ori.flatten(), result.flatten())
                    
                    # calculating iou score
                    logger.info("Line no 81")
                    iou_score = np.sum(intersection)/np.sum(union)
                    logger.info("iou------%s"%iou_score)
                    iou_score_list.append(iou_score)
            
                iou_dict[m].append(iou_score_list)
                logger.info("Line no 87")

        return iou_dict, N_TEST_IMG, thresholds


    def sortingIOU(self,values,N_CLASSES):
        # iou list index -> sorting
        iou_dictf ={}

        for m in range(N_CLASSES):
            if m in iou_dictf:
                    pass
            else:
                iou_dictf[m]=[]

            for i in range(values[1]):
                file =[]
                for j in range(len(values[0][m])):
                    file.append(values[0][m][j][i])
                iou_dictf[m].append(file)
        return iou_dictf


    def thresholdPlotting(self,iou_dictf,values,N_CLASSES,plots_dir,classes):
        # threshold plotting
        # markers = ['s','*','o','^','v','p','1','2','3','4']
        markers = [(i,j,0) for i in range(2,10) for j in range(1, 3)]
        logger.info("markers : %s" % markers)
        colors = ['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w']
        legend_list = []

        for m in range(N_CLASSES):
            for i in range(len(iou_dictf[m])):
                # plt.plot(values[2],iou_dictf[m][i],'-',color='k', marker=markers[i], markerfacecolor=colors[i] )
                plt.plot(values[2],iou_dictf[m][i],'-',color='k', linestyle='solid', marker=markers[i], markerfacecolor=colors[i])
                ti = 'Test image {}'.format(i+1)
                legend_list.append(ti)
            ax = plt.gca()
            ax.set_facecolor('#add8e6')
            plt.title('IOU V/s Threshold - %s' %classes[m].capitalize(),fontdict={'fontsize':20, 'fontweight':"bold"}, loc='center')
            plt.xlabel('Threshold',fontsize=16,color='darkblue')
            plt.ylabel('IOU',fontsize=16,color='darkblue')
            plt.legend(legend_list, fontsize= 'x-large',loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim((0,1))
            plt.savefig(plots_dir+'/iou-plot-%s.png' %classes[m],dpi=600, bbox_inches='tight')

            plt.show()
            plt.close()

        return 1


    def allFunctionCalls(self,project_id,weights_path,N_CLASSES,classes):
        logger.info("classes")
        logger.info(classes)
        model_name = path_leaf(weights_path).replace('.hdf5','')
        test_run = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/pred_masks"+"/"+str(model_name) +"/"
        gt_folder =  "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/data_dir/gt_mband/test"
        plots_dir = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id)+"/model_dir/plots"
        logger.info("test run--%s"%test_run)
        logger.info("method 1--%s"%gt_folder)
        values = self.calculateIOU(test_run,gt_folder,N_CLASSES)
        logger.info("method 2")
        iou_dictf = self.sortingIOU(values,N_CLASSES)
        logger.info("method 3")
        self.thresholdPlotting(iou_dictf, values,N_CLASSES,plots_dir,classes)
        logger.info("method end")
        return plots_dir
    
