# Geospatial_Image_Analytics
An end-to-end solution for geospatial image analytics is developed to segment land use parameters like buildings, road networks, vegetation cover, water bodies, etc. The central ideology is to derive insights on economic growth, urbanisation, and changes in natural resources using satellite images and computer vision. 
This is a Tensorflow (wrapped with Keras) based implementation using UNET as the deep learning architecture to perform satellite image segmentation.

![Screen Shot 2022-01-02 at 5 48 59 PM](https://user-images.githubusercontent.com/77407100/147876327-5933d5be-e888-4859-982a-84a012422e88.png)

## Dataset
The dataset can be obtained from Landsat 8 satellite images using Earth Explorer API.
The satellite images consist of 11 bands including Red, Blue, Green, Near Infrared (NIR), and Shortwave Infrared (SWIR). Apart from the reflectance measurements obtained from these bands, standard indices including Normalized Difference Vegetation Index (NDVI), Normalized Difference Moisture Index (NDMI), Normalized Difference Water Index (NDWI), etc. are also used as the features to train the models. 

The folder structure is shown in 'data' folder.

Also available are correctly segmented images of each training location, called mask. These files contain information about 5 different classes: buildings, roads, trees, crops and water (note that original Kaggle contest had 10 classes).
Resolution for satellite images s 16-bit. However, mask-files are 8-bit.
Implementation
Deep Unet architecture is employed to perform segmentation.
The ground truth for the training dataset has been created by manually annotating these images for five classes: vegetation (dark green), ground (light green), roads and parking (gray), buildings (black) and water (blue).
Image augmentation is used for input images to significantly increase train data.
Image augmentation is also done while testing, mean results are exported to result.tif image. examples

Prediction Example
