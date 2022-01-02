
#CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]  # data_config
#CLASS_WEIGHTS = [0.35,0.3,0.35]
keys = [1,2,3,4,5,6,7,8,9,10,11]
N_BANDS = 8
N_CLASSES = 5
N_EPOCHS = 1
UPCONV = True
PATCH_SZ = 128 # should divide by 16 #160
BATCH_SIZE = 50 #threshold = 67
TRAIN_SZ = 2000  # train size
VAL_SZ = 500 
thres = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] 
color_values = [[254, 204, 203],[223, 194, 125],[27, 120, 55],[166, 219, 160],[116, 173, 209],[125, 150, 150],[203, 194, 125],[17, 120, 55],[186, 219, 160],[136, 173, 209]]
order_values = [0,1,2,3,4,5,6,7,8,9]
test_id = '01'
img_path = '/home/ubuntu/cv-asset//Satelllite_WB/SpaceNet_5Classes/data_dir/mband/{}.tif'.format(test_id)
train_img_path = '/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/mband'
annotated_img_path = '/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/gt_mband/train'
directory_out = "/home/ubuntu/01_Satelllite_WB/Landsat_8Bands/data_dir/pani/"
directory_input = "/home/ubuntu/01_Satelllite_WB/Landsat_8Bands/data_dir/clippedbands/"


log_dir='/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/model_dir/logdir'
logs='/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/model_dir/logdir/'
model_dir = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/model_dir/model2"
result_dir ="/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/pred_masks" 
weights_path = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/model_dir/model/unet_weights_spacenet_50_classes_new.hdf5"
configfile = '/home/ubuntu/cv-asset//Satelllite_WB/SpaceNet_5Classes/config_dir/data_config.py'
configfile_powerplant = '/home/ubuntu/cv-asset/Satelllite_WB/PowerPlant_3Classes/config_dir/data_config.py'
configfile_lansat = "/home/ubuntu/cv-asset/Satelllite_WB/Landsat_8Bands/config_dir/data_config_land.py"
confignew = "/home/ubuntu/cv-asset/Satelllite_WB/punedam/config.py"
config_indices ="/home/ubuntu/cv-asset/Satelllite_WB/Indices_2Classes/config_dir/config.py"

test_run = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/pred_masks/unet_weights20191031_172825"
gt_folder = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/gt_mband/test"
plots_dir = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/model_dir/plots"

data_dir = "/home/ubuntu/cv-asset/Satelllite_WB/PowerPlant_3Classes/data_dir"  
data_dir_spacenet = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir"  

input_dir_name = "mband/train"
mask_dir_single = "gt_mband_classes"
mask_dir_name = "gt_mband/train"
stacked_dir_name = "stacked"
jpegs_dir_name = "jpeg"

indices_dir = "/home/ubuntu/cv-asset/Satelllite_WB/config_src/indices"
indices_raw_dir = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/indices_raw"
indices_color_dir = "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/indices_color"

clip_dir= "/home/ubuntu/cv-asset/Satelllite_WB/SpaceNet_5Classes/data_dir/clippedbands/"
#out_dir = "/home/ubuntu/Satelllite_WB/SpaceNet_5Classes/data_dir/out"

classes_raw_bands = ["B1","B2","B3","B4","B5","B6","B7"] 
classes_gt_mband = ["mine","water"]
#images_count = 2

