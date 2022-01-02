# imports
import rasterio
# from osgeo import gdal
import numpy, gdal, gdalconst
import os
import glob
import subprocess

# Stack bands
def stack_single_bands(band_paths, dest_stacked_tiff):
    # Read metadata of first file
    with rasterio.open(band_paths[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(band_paths))

    # Read each layer and write it to stack
    with rasterio.open(dest_stacked_tiff, 'w', **meta) as dst:
        for id, layer in enumerate(band_paths, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))

# Stack multi-page bands
def stack_multipage_bands(multipage_tiff_path, dest_stacked_tiff):
    dataset = gdal.Open(multipage_tiff_path, gdalconst.GA_ReadOnly)
    # data = numpy.array([gdal.Open(name, gdalconst.GA_ReadOnly).ReadAsArray()
    #                     for name, descr in dataset.GetSubDatasets()])
    tifs = []
    for name, descr in dataset.GetSubDatasets():
        tiff_page = gdal.Open(name, gdalconst.GA_ReadOnly)
        tifs.append(name)

    outvrt = '/vsimem/stacked.vrt' #/vsimem is special in-memory virtual "directory"

    outds = gdal.BuildVRT(outvrt, tifs, separate=True)
    outds = gdal.Translate(dest_stacked_tiff, outds)
    outds = None
    gdal.Unlink('/vsimem/stacked.vrt')

def do_pansharpening():
    pass

def tiff_to_jpeg(tiff_file, jpeg_file):
    options_list = [
        '-ot Byte',
        '-of JPEG',
        '-b 1',
        '-b 2',
        '-b 3',
        '-scale min_val max_val'
    ]
    options_string = " ".join(options_list)
    gdal.Translate(jpeg_file,
                   tiff_file,
                   options=options_string)

def rgb_multipage(bands_dir,rgb_bands_dir , img_id):
    lis=glob.glob(bands_dir+"/"+"*.tif")
    x = ".tif"
    if(len(lis)<=5):
        lis=glob.glob(bands_dir+"/"+"*.TIF")
        x = ".TIF"
    output = img_id + ".tif"
    red = bands_dir + "/" + "*_B4"+x
    blue = bands_dir + "/" + "*_B3"+x
    green = bands_dir + "/" + "*_B2"+x
    multiple = red +" "+ blue +" "+ green
    command = "tiffcp {} ".format(multiple) + rgb_bands_dir + "/" + "{}".format(output)
    os.system(command)
    print("___",command)
