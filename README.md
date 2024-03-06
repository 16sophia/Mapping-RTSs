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
.
├── README.md
├──Data_generation.ipynb
├──Main_MaskRCNN.ipynb
├──environment.yml
├──notebooks
|   ├──IoU_calculation.ipynb
|   ├──feature_calculation.ipynb
|   ├──visualization.ipynb
|   ├──watermask.ipynb
|   └──geoprocessing.yml
├──src
|   ├──config_model.py
|   ├──dataset_module.py
|   ├──emptyimg_dataset.py
|   ├──engine.py
|   ├──model_module.py
|   ├──transforms.py
|   ├──utils.py
|   └──visualizations.py
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

## File description
"Data_generation.ipynb" 
