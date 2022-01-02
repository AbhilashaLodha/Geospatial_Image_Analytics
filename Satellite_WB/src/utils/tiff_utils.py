# imports
#import rasterio


def read_tiff():
    pass

def create_tiff():
    pass

def update_tiff():
    pass

def tiff_profiles(band_paths):

    for band in band_paths:
        print(band)
        with rasterio.open(band) as src:
            print(src.profile)