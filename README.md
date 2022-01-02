# Geospatial_Image_Analytics
An end-to-end solution for geospatial image analytics is developed to segment land use parameters like buildings, road networks, vegetation cover, water bodies, etc. The central ideology is to derive insights on economic growth, urbanisation, and changes in natural resources using satellite images and computer vision. 
This is a Tensorflow (wrapped with Keras) based implementation using UNET as the deep learning architecture to perform satellite image segmentation.

![Screen Shot 2022-01-02 at 5 48 59 PM](https://user-images.githubusercontent.com/77407100/147876327-5933d5be-e888-4859-982a-84a012422e88.png)

## Dataset
* The dataset can be obtained from Landsat 8 satellite images using Earth Explorer API.
* The satellite images consist of 11 bands including Red, Blue, Green, Near Infrared (NIR), and Shortwave Infrared (SWIR). 
* Apart from the reflectance measurements obtained from these bands, standard indices including Normalized Difference Vegetation Index (NDVI), Normalized Difference Moisture Index (NDMI), Normalized Difference Water Index (NDWI), etc. are also used as the features to train the models. 
* The folder structure is shown in 'data' folder.

## Implementation
* Deep Unet architecture is employed to perform segmentation.
* The ground truth for the training dataset has been created by manually annotating these images for five classes: vegetation (dark green), ground (light green), roads and parking (gray), buildings (black) and water (blue).
* Image augmentation is used for input images to significantly increase train data.

## Software and Packages 
* Anaconda with inbuilt Python 3.6 (This will automatically set required environment path variables and will also contain all required libraries)
* Environment used is tensorflow_p36. So, activate the environment using the following command before executing any python file: source activate tensorflow_p36
* Download the complete folder named ‘Satellite_WB/src’ with core and utils folders inside it.

## Post Deployment Checklist (Test through Postman) for API End Points
Check the folder "post_deployment" for api testing.

## Prediction Example


