import os
import subprocess


def tif_multipagetif(img_id):
    i1 = "{}_1.tif".format(img_id)          
    i2 = "{}_2.tif".format(img_id)
    i3 = "{}_3.tif".format(img_id)          
    i4 = "{}_4.tif".format(img_id)
    outfile = "C:\\Users\\chava.sindhu\\Documents\\Satellites\\data_power\\mband\\{}.tif".format(img_id)
    print(outfile)
    command = 'C:\\"Program Files"\\IrfanView\\i_view64.exe' + ' /multitif=(C:\\Users\\chava.sindhu\\Documents\\Satellites\\data_power\\mband\\{}.tif,'.format(img_id) + i1 + "," + i2 + "," + i3 +"," + i4 + ")/killmesoftly"
    print(command)
    status = subprocess.run(command, shell=True)
    print("converted to multipage tif")
    return status

trainIds = [str(i).zfill(2) for i in range(1, 3)]  # all availiable ids: from "01" to "xx"
print(trainIds)

for img_id in trainIds:
    print(img_id)

    os.system("gdal_translate -b 1 C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_git\\gt_mband_2_sin\\01.tif C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_coalpile\\split\\{}_1.tif".format(img_id,img_id))
    os.system("gdal_translate -b 2 C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_git\\gt_mband_2_sin\\01.tif C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_coalpile\\split\\{}_2.tif".format(img_id,img_id))
    os.system("gdal_translate -b 3 C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_git\\gt_mband_2_sin\\01.tif C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_coalpile\\split\\{}_3.tif".format(img_id,img_id))
    os.system("gdal_translate -b 4 C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_git\\gt_mband_2_sin\\01.tif C:\\Users\\chava.sindhu\\Documents\\Satellites_coalpile\\data_coalpile\\split\\{}_4.tif".format(img_id,img_id))
    #tif_multipagetif(img_id)

imag = '01_1.tif'
def tojpg(imag):
    #parent_dir= "custom_projects/resources/satellite/deep_unet/"
    for name in glob.glob(imag):
        im = Image.open(name)
        name = str(name).rstrip(".tif")
        logger.info(str(name))
        print(name)
        im.save(name + '.jpg', 'JPEG')
    print("Conversion from tif/tiff to jpg completed!")


#tif_multipagetif(01)
#x = tojpg('01_1.tif')
#print(x)