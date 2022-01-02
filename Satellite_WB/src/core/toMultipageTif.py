import subprocess


'''
    trainIds = [str(i).zfill(2) for i in range(1, 3)]  # all availiable ids: from "01" to "xx"
    print(trainIds)
    for img_id in trainIds:
        print(img_id)
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