import shutil
import os



def train_test_split(test_size,project_id):
    source = "/home/ubuntu/cv-asset/Satelllite_WB/" +str(project_id) +"/data_dir/gt_mband/"
    dest1 = source +"test/"
    dest2 = source +"train/"
    test_size=float(test_size)
    if not(os.path.isdir(dest1)):
        os.mkdir(dest1)
    if not(os.path.isdir(dest2)):
        os.mkdir(dest2)
    shutil.rmtree(dest1)
    shutil.rmtree(dest2)    
    if not(os.path.isdir(dest1)):
        os.mkdir(dest1)
    if not(os.path.isdir(dest2)):
        os.mkdir(dest2)
    
    files = os.listdir(source)
    train_size = 1 - test_size
    count = len(files) - 2
    w = 3
    r = test_size * count
    if (type(w) == type(r)):
        count_test = r
    else:
        count_test = int(r)
    print("no of test images --",count_test)
    count_test = count_test
    moved = []
    for i in range(count_test):
        #print("name--",files[i])
        if(os.path.isfile(source + files[i])):
            print("Test---if is file name-----",source + files[i])
            shutil.copy((source + files[i]), dest1)
            moved.append(files[i])
        else:
            print("Test---else name--",files[i])
    train_files = os.listdir(source)
    count_tr = len(train_files) - count_test
    #print("no of train images --",count_tr)
    count_train  = len(train_files) 
    #print(count_train) 
    for i in range(count_train):
        if(os.path.isfile(source + train_files[i])):
            if (train_files[i] not in moved):
                print("Train---if is file name-----",source + train_files[i])
                shutil.copy((source + train_files[i]), dest2)
        else:
            print("Train---else name--",train_files[i])
    return "success"
