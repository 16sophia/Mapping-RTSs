# Mapping Retrogressive Thaw Slumps from Satellite Data using deep learning
This repository contains the code used for the master-thesis "Mapping Retrogressive Thaw Slumps from Satellite Data using deep learning". This thesis explores the automated mapping of retrogressive thaw slumps (RTS) on two Arctic sites using TanDEM-X-derived digital elevation models (DEMs) and addresses the subjectivity inherent in the RTS annotation process.
The assessment of labeler agreement in the annotation process involved three levels: whether the same patch in the normalized difference DEM was delineated, binary delineation masks between labelers, and RTS masks of intersecting RTSs. The model evaluation occurred on three levels: RTS detection, pixel classification of RTS masks within an image, transformed to a binary mask, pixel classification of RTS masks, only taking true positive predictions into account.

The repository includes:
- Code to train and apply a Mask R-CNN model
- Code to calculate retrogressive thaw slump (RTS) features
- Code to compare RTS delineation between different labelers
- Trained models
- Example predictions

## Repository structure
```
.
├── README.md
├──Data_generation.ipynb
├──Main_MaskRCNN.ipynb
├──environment.yml
├──src
|   ├──config_model.py
|   ├──dataset_module.py
|   ├──emptyimg_dataset.py
|   ├──engine.py
|   ├──model_module.py
|   ├──transforms.py
|   ├──utils.py
|   └──visualizations.py
├──notebooks
|   ├──IoU_calculation.ipynb
|   ├──feature_calculation.ipynb
|   ├──visualization.ipynb
|   ├──watermask.ipynb
|   └──geoprocessing.yml
├──data 
|   └──Sophia 
|       ├──data_clean
|       |   └──data_train
|       |   |   └──data_original
|       |   |   |    └──....
|       |   |   ├──data_original_plus_transformed
|       |   |   |    └──....
|       |   |   └──...
|       |   ├──data_train
|       └──dataframe
|           └──....csv files
├──model 
|   └──.....pth files
├──prediction 
|   └──.....tif, shp, cpg, dbf, prj, shx files
├──.renku
|   └──.....
├── .dockerignore
├──.gitattributes
├──.gitlab-ci.ym
├──.renkulfsignore
└──Dockerfile
```
## File description
For Mask R-CNN model building and application:
- `Data_generation.ipynb`: Creates `data` files, including data augmentation, circular encoding of aspect data, min-max normalize data, splitting data into train, val and test.  
- `Main_MaskRCNN.ipynb`: Create a Mask R-CNN model and trains it with Pytorch lightning including application on a specific input image with visualization of the result, calculation of the model performance, visualization of RTS characteristics.
- `environment.yml`: Defines the Conda environment with the dependencies required for the project regarding `Main_MaskRCNN.ipynb`, `Data_generation.ipynb` and the files in the `src` folder. 
- `config_model.py`: Defines the tested model parameter configurations (Can be accessed with the get_config() function).
- `dataset_module.py`: Creates the dataset for Mask-RCNN training: no empty images (full of 0 or NaN) or empty mask labels are allowed. It is a class used with PyTorch's data loading utilities. 
- `emptyimg_dataset.py`: Similar to `dataset_module.py` but allows empty images. Used for the model application on a dataset including empty images.
|- `engine.py`: Includes functions that are lnked with Mask R-CNN model. For example function which calcualte RTS characteristics for each predicted RTS, functions that calculate the performance metrics and functions for postprocessing.
- `model_module.py`: Creates the Mask R-CNN model based on PyTorch Lighning's base class for Lightning modules. 
- `transforms.py`: Functions for data transformation
- `utils.py`: Additional helper functions, including functions that transform the predictions to shapefiles, and functions for data handling. 
- `visualizations.py`: Functions that are responsible for the visualization of the Mask R-CNN predictions such as the visualization of the labelled vs. predicted mask and the labelled and predicted bounding boxes.

For the evaluation of the RTS labelling results (All in the notebooks folder):
- `IoU_calculation.ipynb`: Compares the delineation results of two labelers, including calculation of the intersection over Union (IoU) of all RTS pairs, 1:1 matching in order to only keep one pair per RTS. The total number of IoU pairs, total number of delineated RTSs, number of RTSs without intersection, number of RTSs after 1:1 matching ect. are entered into the RTS_detection.xlsx file in order to calculate the performance metrics between the labelers. The input of this file stems from the output of the `feature_calculation.ipynb` file.
- `feature_calculation.ipynb`: Calculates RTS features such as shape attributes like circularity and area and attributes based the normalised difference DEMs (the weighted mean is calculated for RTSs that are on multiple normalised difference DEM tiles).  
- `visualization.ipynb`: Visualizes the RTS feature distributions and the comparison between labelers (IoU between labeler vs. RTS feature). Caluclates the min, max, mean, median, std of the RTS characterstics. The input data stems from `feature_calculation.ipynb` and `IoU_calculation.ipynb`.
- `watermask.ipynb`: Responsible to merge the watermask that were split into subimages and merge all watermask that touch the input image. Extract the watermask that corresponds to the location of the input image.
- `environment2.yml`: Defines the Conda environment with the dependencies required for the files in the `notebook` folder.
- `data`: `dataframe` includes information used for the data normalization. `data_clean` includes folders with training and testing data for the Peel and Tuktoyaktuk train/ test site with different time steps and watermasks. The training and testing data contains the folders `aspect`, `images`, `masks` and `slope`, where `images` contains the normalized difference DEM values and `masks` contains the ground truth instance segmentation masks. 
- `model`: This folder contains some trained Mask R-CNN weights 
- `prediction`: This folder contains some prediction examples. For each example, a georeferenced .tif file is generated. All predictions are accumulated in one shapefile. 
- `.renku` & `.renkulfsignore`: contains files responsible for the settings of the Renku platform
- `.dockerignore`: Defines which files should be ignored when building a Docker image. 
- `.gitattributes` & `.gitlab-ci.ym`: Defines how Git and GitLab treats certain files and CI/CD pipelines.
- `Dockerfile`: Contains instructions for building a Docker image
