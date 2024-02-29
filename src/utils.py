import numpy as np
import torch
import os
import random
import shutil
import rasterio
import shapely
import fiona
from shapely.geometry import shape
from rasterio.features import shapes
from shapely.geometry import Polygon, mapping

def extract_shp(prediction, target, img_i, root_path, RTS_id, batch_i):
    '''
    Creates a georeferenced tif file for each prediction mask.
    Combines all predictions together into one shapefile
    
    '''
    my_schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'}
    }
    n_instance_pred = prediction[img_i]['masks'].shape[0]
    for instance in range(n_instance_pred):
        instance_mask = prediction[img_i]['masks'][instance].detach().numpy()
        instance_mask = (instance_mask >= 0.5).astype(int)
        dir_name =  root_path + '/prediction' + f'/prediction_{batch_i}_{img_i}_{instance}.tif'
        meta = target[img_i]['geo_meta']
        # Write prediction to tif
        with rasterio.open(dir_name, 'w', **meta) as dst:
                        dst.write(instance_mask)

        # Read tif info to convert to shp
        with rasterio.open(dir_name, 'r') as src:
            data = src.read(1)
            data = data.astype('int16')
            # Get the shapes of the features in the raster
            results = [{'geometry': s} for s in rasterio.features.shapes(data, mask=None, transform=src.transform)] # results in list of [RTS, img bbox]

        getdata = 0
        result_dict = results[getdata]

        # Create a Shapely geometry object
        geometry_info = result_dict['geometry']
        shapely_geometry = Polygon(geometry_info[0]['coordinates'][0])  # Access inner list

        # Create a dictionary compatible with Fiona write
        feature = {
            'geometry': mapping(shapely_geometry),
            'properties': {'id': RTS_id}  # You may need to adjust this based on your data
        }
        RTS_id+=1

        modelpath = root_path + '/prediction' + "/model_result.shp"
        if os.path.exists(modelpath):
            # Create sh
            with fiona.open(modelpath, mode='a', driver='ESRI Shapefile', schema=my_schema, crs=src.crs) as output:
                output.write(feature)
        else:
            # append RTS
            with fiona.open(modelpath, mode='w', driver='ESRI Shapefile', schema=my_schema, crs=src.crs) as output:
                output.write(feature)
    return RTS_id

def unravel_index(index, shape):
    '''
    converts a flat index or array of flat indices into a tuple of coordinate arrays based on shape
    Equals to np.unravel_index. Implemented manually to work for torch tensor on gpu
    '''
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def normalise_data(data):
    image = (data - np.min(data)) / (np.max(data) - np.min(data))
    image *= 255
    image = image.astype(int)
    return image


####### Data handling#########################################

def clean_raw_data(dataset, test_set, original_data, original_test, test_path, data_path):
    '''
    Delete dataset in data_clean folder that are invlid by using delete property generated in __getitem__ from file dictionary
    Invalid data: images that are empty (full of 0 or NaN) or corresponding mask are empty
    
    Input:
        dataset, test_set: DEM dataset generated from raw data, will be cleaned in this function through implementation in  __getitem__ 
        original_data, original_test: DEM dataset generated from raw data, uncleaned
        test_path, data_path: Path to clean directory
    
    '''
    # Delete invalid entries from DEMDataset: __getitem__ deletes invalid entries == a one time iteration 
    for file in dataset:
        pass

    for file in test_set:
        pass
    
    # Calculate intersection between original data and cleaned data 
    intersection_set = set(original_data.filename_list).intersection(dataset.filename_list)
    intersection_test = set(original_test.filename_list).intersection(test_set.filename_list)

    # Find the entries that are not in the intersection
    not_in_intersection = [entry for entry in original_data.filename_list + dataset.filename_list if entry not in intersection_set]
    not_in_intersection_test = [entry for entry in original_test.filename_list + test_set.filename_list if entry not in intersection_test]
    not_in_intersection, not_in_intersection_test
    
    # delete file form data_clean folder which are not in the intersection = data deleted in for loop above
    for i, path_test in enumerate([test_path, data_path]):
        intersection_file = [not_in_intersection_test, not_in_intersection][i]

        for file in intersection_file:
            delete_img = os.path.join(path_test, "images", file)
            if os.path.exists(delete_img):
                # Delete the file
                os.remove(delete_img)
                print(f"{file} image deleted successfully.")
            else:
                print(f"{file} image not found. No deletion performed.")
            delete_mask = os.path.join(path_test, "masks", file)
            if os.path.exists(delete_mask):
                os.remove(delete_mask)
                print(f"{file} mask deleted successfully.")
            else:
                print(f"{file} mask not found. No deletion performed.")
    return dataset, test_set

def move_images(source_dir, destination_dir, all_ = True, to_move = None):
    '''
    Move images. Either all (all_ = True) or images in to_move list
    to_move = image tile names
    '''
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if all_: # move all file
        # Get a list of all files in the source directory
        image_files = os.listdir(source_dir)
    else: 
        image_files = to_move
    print("move")
    # Move each image file to the destination directory
    for image_file in image_files:
        source_path = os.path.join(source_dir, image_file)
        destination_path = os.path.join(destination_dir, image_file)
        shutil.move(source_path, destination_path)
        
def split_train_test_mask(source_dir, destination_dir, source_dir2, destination_dir2, fraction):
    '''
    Move a fraction of random images (image and mask) from one directory to another 
    '''
    # Get the list of all files in the source directory
    all_files = os.listdir(source_dir)

    # Calculate the number of files to move
    num_files_to_move = int(len(all_files) * fraction)

    # Randomly select files to move
    files_to_move = random.sample(all_files, num_files_to_move)

    # Move selected files to the destination directory
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.move(source_path, destination_path)
        
        source_path2 = os.path.join(source_dir2, file_name)
        destination_path2 = os.path.join(destination_dir2, file_name)
        shutil.move(source_path2, destination_path2)
        
def split_dataset_by_percentage(dataset, percentage, seed_value):
    '''
    divide a dataset into a training set and a validation (or test) with random permutation
    Input:
        dataset: input dataset that you want to split.
        percentage: A float value between 0 and 1, indicating the percentage of samples that should be included in the first subset. 
    Output: train_data, test_data
    '''
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    total_samples = len(dataset)
    num_samples = int(total_samples * percentage)
    indices = torch.randperm(total_samples).tolist()
    train_data = torch.utils.data.Subset(dataset, indices[:num_samples])
    test_data = torch.utils.data.Subset(dataset, indices[num_samples:])
    return train_data, test_data

