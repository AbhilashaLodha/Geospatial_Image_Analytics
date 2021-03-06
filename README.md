# Geospatial_Image_Analytics
An end-to-end solution for geospatial image analytics is developed to segment land use parameters like buildings, road networks, vegetation cover, water bodies, etc. The central ideology is to derive insights on economic growth, urbanisation, and changes in natural resources using satellite images and computer vision. 
This is a Tensorflow (wrapped with Keras) based implementation using UNET as the deep learning architecture to perform satellite image segmentation.

![Screen Shot 2022-01-02 at 5 48 59 PM copy](https://user-images.githubusercontent.com/77407100/147877554-d0a36fb0-307d-4e33-8d46-b35b3ab86c41.png)


## Dataset
* The dataset can be obtained from Landsat 8 satellite images using Earth Explorer API.
* The satellite images consist of 11 bands including Red, Blue, Green, Near Infrared (NIR), and Shortwave Infrared (SWIR). 
* The folder structure is shown in 'data' folder.

## Implementation
* Deep Unet architecture is employed to perform segmentation.
* The ground truth for the training dataset has been created by manually annotating these images for five classes: vegetation (dark green), ground (light green), roads and parking (gray), buildings (black) and water (blue).
* Image augmentation is used for input images to significantly increase train data.
* Apart from the reflectance measurements obtained from the 11 bands, standard indices including Normalized Difference Water Index (NDWI), Soil-Adjusted Vegetation Index (SAVI) and Enhanced Vegetation Index (EVI) are also used as the features to train the models. 

![geo2](https://user-images.githubusercontent.com/77407100/147877419-ae4f2fe1-e7d1-4239-bec6-1ba9a40401fb.jpg)


## Software and Packages 
* Anaconda with inbuilt Python 3.6 (This will automatically set required environment path variables and will also contain all required libraries)
* Environment used is tensorflow_p36. So, activate the environment using the following command before executing any python file: source activate tensorflow_p36
* Download the complete folder named ???Satellite_WB/src??? with core and utils folders inside it.


## UNET Model Architecture 

![geo4](https://user-images.githubusercontent.com/77407100/147877493-a54a7ae3-d049-4489-9fc6-ece76329be8d.png)


## Post Deployment Checklist (Test through Postman) for API End Points
Check the folder "post_deployment" for api testing.

## Prediction Example

![prediction1](https://user-images.githubusercontent.com/77407100/147877880-5e1682ad-57ac-476f-912c-8a7ab43c1d58.jpg)


