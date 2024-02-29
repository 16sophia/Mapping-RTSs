import time
import torch
import datetime
import numpy as np
import torch.distributed as dist
import os
import random
import shutil
import copy
import sklearn.metrics as skmetrics
import rasterio
import shapely
import skimage

from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou
from collections import defaultdict, deque
from skimage.measure import regionprops
from rasterio.features import geometry_mask
from rasterio.merge import merge

from shapely.geometry import shape
import rasterio
from rasterio.features import shapes
import fiona
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


def analyze_preds(preds, img_, tile_id, threshold_mask = 0.5):
    '''
    Calcualte RTS characteristics for each predicted RTS
    '''
    mean_intensity_ = []
    sd_intensity_  = []
    pix_size_ = []
    circularity_ = []
    solidity_ = []
    max_diameter_ = []
    area_ = []
    img_height = img_[tile_id][0].detach().numpy()
    prediction = preds[tile_id]

    for i in range(len(prediction["labels"])): # Iterate through RTS/ tile
        RTS_mask = copy.deepcopy(preds[tile_id]["masks"][i].detach().numpy())
        empty_dim = 0
        # intensity property
        mask_boolean = RTS_mask[empty_dim]
        mean_intensity = np.nanmean(img_height[mask_boolean >=threshold_mask ])
        sd_intensity = np.std(img_height[mask_boolean >=threshold_mask ])
        # Make mask binary
        mask_boolean[mask_boolean >=threshold_mask ] = 1
        mask_boolean[mask_boolean <threshold_mask ] = 0
        # shape property
        if len(np.unique(mask_boolean)) != 2: # mask is empty after making it binary
            continue
        mask_prop = regionprops(mask_boolean.astype(int))[0] # We only have one instance per binary mask
        pix_size = np.count_nonzero(mask_boolean)
        RTS_perimeter = mask_prop.perimeter
        circularity = (4*np.pi*pix_size)/(RTS_perimeter**2)
        solidity = mask_prop.solidity
        max_diameter = mask_prop.feret_diameter_max
        area = mask_prop.area
        # Append values
        mean_intensity_.append(mean_intensity)
        sd_intensity_.append(sd_intensity)
        pix_size_.append(pix_size)
        circularity_.append(circularity)
        solidity_.append(solidity)
        max_diameter_.append(max_diameter)
        area_.append(area)
    return mean_intensity_, sd_intensity_, pix_size_, circularity_, solidity_, max_diameter_, area_

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



#########Calculation of performance metrics######################################################


def performance_metric(y_true, y_pred):
    '''Calculates performance metric for pixels
    If nothing was predicted: results in 0
    '''
    correct = torch.sum(y_true == y_pred).item()
    total = len(y_true)
        
    true_positives = torch.sum((y_true == 1) & (y_pred == 1)).item()
    false_positives = torch.sum((y_true == 0) & (y_pred == 1)).item()
    false_negatives = torch.sum((y_true == 1) & (y_pred == 0)).item()

    accuracy = correct / total if correct >0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    IoU = true_positives / (false_positives + true_positives + false_negatives) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1, IoU

def similarity_mask(results, pred_mask, targ_mask, mask_thr):
    '''
    Calculates metrics of mask pixels for RTS that were labelled as TP -> mask metric
    '''
    matched = copy.deepcopy(results)
    channel = 0
    acc_pixel = []
    precision_pixel = []
    recall_pixel = []
    f1_pixel = []
    IoU_pixel = []

    for pred_id, targ_id in enumerate(matched):
        if targ_id>-1: # Match 
            predicted = pred_mask[pred_id][channel].detach()
            labelled = targ_mask[targ_id.item()]
            # Make mask binary
            predicted = (predicted>= mask_thr).to(torch.int)
            labelled = (labelled>= mask_thr).to(torch.int)

            accuracy, precision, recall, f1, IoU = performance_metric(labelled.flatten(), predicted.flatten())
            acc_pixel.append(accuracy)
            precision_pixel.append(precision)
            recall_pixel.append(recall)
            f1_pixel.append(f1)
            IoU_pixel.append(IoU)
    if len(acc_pixel)==0: # Fill with one nan if nothing matched
        accuracy, precision, recall, f1, IoU = np.nan, np.nan, np.nan, np.nan, np.nan
        acc_pixel.append(accuracy)
        precision_pixel.append(precision)
        recall_pixel.append(recall)
        f1_pixel.append(f1)
        IoU_pixel.append(IoU)
            
    acc_pixel_ = torch.nanmean(torch.tensor(acc_pixel, dtype=torch.float))
    precision_pixel_ = torch.nanmean(torch.tensor(precision_pixel, dtype=torch.float))
    recall_pixel_ = torch.nanmean(torch.tensor(recall_pixel, dtype=torch.float))
    f1_pixel_ = torch.nanmean(torch.tensor(f1_pixel, dtype=torch.float))  
    IoU_pixel_ = torch.nanmean(torch.tensor(IoU_pixel, dtype=torch.float))  
    
    return acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_
    
    
def iou_acc(total_gt , pred_mask, targ_mask, mask_thr, src_boxes, pred_boxes, threshold, get_TP_ind = False):
        '''
        Computes result metrics for all RTS within an image based on iou threshold of bounding boxes
        Accuracy = (TP+TN)/(TP+FP+FN+TN) as TP/(TP+FP+FN) since we know that we have no dataset without an object 
        precision: percentage of TP in all positive labelled --> important if false positives are undesirable
        recall: ability to capture all positive instances --> important if false negatives are undesirable
        F1: harmonic mean of precision and recall (give lower weight to extreme values compared to the arithmetic mean)
        
        Input: 
            threshold: trange that will be used to evaluate whether TP ect.
            get_TP_ind: If True, RTS index will be returned with 1 if it is evaluated as TP, else as 0
'''
        # matching pairs of bounding boxes depending on IoU threshold 
        matcher = Matcher(threshold,threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes) # IoU between all pairs of bounding boxes: result in matrix with labelled boxes (y axis)and predicted boxes (x axis)
        results = matcher(match_quality_matrix) 
        
        
        # Set label to 1 (for TP) else -1 (FP)
        if get_TP_ind:
            RTS_TP_ind = copy.deepcopy(results)
            RTS_TP_ind[RTS_TP_ind>=0] = 1
            RTS_TP_ind = RTS_TP_ind.tolist()
        
        #### Calculate performance based on mask of detected RTS (BBox IoU >= 0.5): similarity_mask(matched elements): ----------------------------------------------------------------------------------------------------------------------------------
        acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_ = similarity_mask(results, pred_mask, targ_mask, mask_thr)

        # Calculate performance based on whole pixel image (Not just detected RTS) -------------------------------------------------------------------
        # Sum all predictions instances and all labels instances to one image and make it binary
        if len(pred_mask) >0:
            channel = 0
            binary_pred = (pred_mask[0][channel] > 0.5).int()
            pred_tot_img = binary_pred
            for instance in range(1,len(pred_mask)):
                binary_mask = (pred_mask[instance][channel] > 0.5).int()
                pred_tot_img+= binary_mask
        
        if len(targ_mask)>0:
            targ_tot_img = targ_mask[0]
            for instance in range(1,len(targ_mask)):
                targ_tot_img+= targ_mask[instance]
        
        # Make it binary
        pred_binary = (pred_tot_img > 0).int() 
        targ_binary = (targ_tot_img > 0).int() 
        accuracy_img, precision_img, recall_img, f1_img, IoU_img = performance_metric(targ_binary.flatten(), pred_binary.flatten())
        
        # Calculate performance based on RTS detection -----------------------------------------------------------------------------------------
        # IoU of RTS bbox 
        n_entries = max(len(src_boxes), len(pred_boxes))
        iout_RTS = list(match_quality_matrix[match_quality_matrix>0].to('cpu'))
        # Fill with 0 where no match was made
        difference = n_entries-len(iout_RTS)
        for i in range(difference):
            iout_RTS.append(torch.tensor(0.0))
        
        iout_RTS_ = torch.nanmean(torch.stack(iout_RTS))
        #in Matcher, a pred element can be matched only twice
        true_positive = max(torch.count_nonzero(results.unique() != -1),0) # number of matched bounding boxes that have a valid match
        matched_elements = results[results > -1]
        # false_positive = sum of unmatched predicted bounding boxes and predicted bounding boxes that have more than two matches
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(torch.unique(matched_elements)))
        false_negative = total_gt - true_positive
        #print("TP", true_positive, "FP", false_positive, "FN", false_negative)
        if true_positive >0:
            acc = true_positive / ( true_positive + false_positive + false_negative ) # we don't have true negatives
            precision = true_positive/ (true_positive + false_positive)
            recall = true_positive/(true_positive + false_negative)
            F1 = true_positive / (true_positive + 0.5 * ( false_positive + false_negative))
            acc = acc.item()
            precision = precision.item()
            recall = recall.item()
            F1 = F1.item()
            
        else:
            acc = 0
            precision = 0
            recall = 0
            F1 = 0
        
        
        if get_TP_ind:
            return acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_, iout_RTS_, acc, precision, recall, F1, RTS_TP_ind, accuracy_img, precision_img, recall_img, f1_img, IoU_img
        else:
            return acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_, iout_RTS_, acc, precision, recall, F1, accuracy_img, precision_img, recall_img, f1_img, IoU_img

def similarity_RTS(pred_mask, targ_mask, src_boxes, pred_boxes, iou_thresholds = [0.5], get_TP_ind = False, mask_thr=0.5):
    '''
    Input: 
    pred_mask, targ_mask: predicted and target binary mask matrix [instance, height, weight]
    src_boxes = target boxes within image, pred_boxes = box predictions within image
    iou_thresholds = list of threshold values, which will be used to calculate TP, FN ect., mased on the intersection over union metric
    get_TP_ind: If True, RTS index will be returned with 1 if it is evaluated as TP, else as 0
    
    Based on bounding box IoU
    Output = performance metrics calculated per image (iou of RTS) and per RTS (iou per pixel)
    accuracy = (TP+TN)/(TP+FP+FN+TN) as TP/(TP+FP+FN) since we know that we have no dataset without an object 
    Handles special cases where we have no ground truth (gt) labels or no predictions
    
    Uses iou_acc function
    '''
    total_gt = len(src_boxes)
    total_pred = len(pred_boxes)
    thrshs = torch.tensor(iou_thresholds)
    thrshs_mean = torch.nanmean(thrshs)
    
    # There are labelled RTS and predicted RTS
    if total_gt > 0 and total_pred > 0:
        #print('total_gt',total_gt, 'total_pred',total_pred)
        # Check how many RTS are detected
        # Metrics / RTS on image level: If RTS was detected based on IoU BBox >= 0.5
        iou_RTS_ = []
        acc_t = []
        precision_t = []
        recall_t = []
        F1_t = []
        RTS_TP_t = []
        # Check how well detected RTS mask are
        # metrics / pixel on RTS level: If pixel of RTS instance was detected (only taking binary masks of detected RTS into account)
        acc_pixel = []
        precision_pixel = []
        recall_pixel = []
        f1_pixel = []
        IoU_pixel = []
        # Check how many pixels in an imaeg are correctly detected
        acc_img = []
        precision_img = []
        recall_img = []
        f1_img = []
        IoU_img = []
        
        
        # metric for each threshold
        for t in iou_thresholds:
            if get_TP_ind:
                acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_, iou_RTS, acc, precision, recall, F1, RTS_TP,  accuracy_img_, precision_img_, recall_img_, f1_img_, IoU_img_= iou_acc(total_gt , pred_mask, targ_mask, mask_thr, src_boxes, pred_boxes, t, get_TP_ind)
                RTS_TP_t = RTS_TP_t+RTS_TP     
                
            else:
                acc_pixel_, precision_pixel_, recall_pixel_, f1_pixel_, IoU_pixel_, iou_RTS, acc, precision, recall, F1, accuracy_img_, precision_img_, recall_img_, f1_img_, IoU_img_ = iou_acc(total_gt , pred_mask, targ_mask, mask_thr, src_boxes, pred_boxes, t)
            # Metrics / RTS on image levell: If RTS was detected based on IoU BBox >= 0.5
            iou_RTS_.append(iou_RTS)
            acc_t.append(acc)
            precision_t.append(precision)
            recall_t.append(recall)
            F1_t.append(F1)
            
            # metrics / pixel on RTS level: If pixel of RTS instance was detected (only taking binary masks of detected RTS into account)
            acc_pixel.append(acc_pixel_)
            precision_pixel.append(precision_pixel_)
            recall_pixel.append(recall_pixel_)
            f1_pixel.append(f1_pixel_)
            IoU_pixel.append(IoU_pixel_)

                
            # Check how many pixels in an imaeg are correctly detected
            acc_img.append(accuracy_img_)
            precision_img.append(precision_img_)
            recall_img.append(recall_img_)
            f1_img.append(f1_img_)
            IoU_img.append(IoU_img_)

        # Metrics / RTS on image level
        iou_avg = torch.nanmean(torch.tensor(iou_RTS_, dtype=torch.float))
        acc_avg = torch.tensor(sum(acc_t) / len(iou_thresholds))
        precision_avg = torch.tensor(sum(precision_t)/len(iou_thresholds))
        recall_avg = torch.tensor(sum(recall_t)/len(iou_thresholds))
        F1_avg = torch.tensor(sum(F1_t)/len(iou_thresholds))
        
        # Mask, only pixel of detected RTS instance (only taking binary masks of detected RTS into account)
        acc_pixel_ = torch.nanmean(torch.tensor(acc_pixel, dtype=torch.float))
        precision_pixel = torch.nanmean(torch.tensor(precision_pixel, dtype=torch.float))
        recall_pixel = torch.nanmean(torch.tensor(recall_pixel, dtype=torch.float))
        f1_pixel = torch.nanmean(torch.tensor(f1_pixel, dtype=torch.float))
        IoU_pixel = torch.nanmean(torch.tensor(IoU_pixel, dtype=torch.float))
        
        # Check how many pixels in an imaeg are correctly detected
        acc_img_ = torch.nanmean(torch.tensor(acc_img, dtype=torch.float))
        precision_img = torch.nanmean(torch.tensor(precision_img, dtype=torch.float))
        recall_img = torch.nanmean(torch.tensor(recall_img, dtype=torch.float))
        f1_img = torch.nanmean(torch.tensor(f1_img, dtype=torch.float))
        IoU_img = torch.nanmean(torch.tensor(IoU_img, dtype=torch.float))
            
        if get_TP_ind:
            return acc_pixel_, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, iou_avg, acc_avg, precision_avg, recall_avg, F1_avg, RTS_TP_t, acc_img_, precision_img, recall_img, f1_img, IoU_img
        else:
            return acc_pixel_, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, iou_avg, acc_avg, precision_avg, recall_avg, F1_avg, acc_img_, precision_img, recall_img, f1_img, IoU_img

    # There are no labelled RTS 
    elif total_gt == 0: # TP= 0
        if total_pred > 0:
            if get_TP_ind: # all TP indices of predicted RTS are set to -1: accuracy and precision = 0 on image and RTS level
                print('FP on empty image')
                return np.nan, np.nan,np.nan,np.nan,np.nan,np.nan, 0.0,0.0,np.nan,np.nan, [-2]* total_pred,0.0,0.0,np.nan,np.nan,np.nan
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan,np.nan,0.0,0.0,np.nan,np.nan,0.0,0.0,np.nan,np.nan,np.nan
        else:
            if get_TP_ind: 
                return np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, [],np.nan,np.nan,np.nan,np.nan,np.nan # no predicted RTS -> TP list is empty
            else:
                return np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan 
    elif total_gt > 0 and total_pred == 0:
        if get_TP_ind: # accuracy and recall are set to zero
            return np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,0.0,np.nan,0.0,np.nan, [],0.0,np.nan,0.0,np.nan,np.nan # no predicted RTS -> TP list is empty
        else:
            return np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,0.0,np.nan,0.0,np.nan, 0.0,np.nan,0.0,np.nan,np.nan 
    else:
        print("unhandled edgecase")
        return 
    
############Postprocess###########################################
def postprocess_iou(output, threshold_mask=0.5, iou_min = 0.25):
    keep_i = []
    channel_i = 0 # image channel
    #iou_list = []
    # Get boolean mask based on iou and iou_min threshold
    for predicted_i, bbox in enumerate(output["boxes"]):
        xmin, ymin, xmax, ymax = torch.round(bbox.detach())
        bbox_area = (xmax - xmin) * (ymax - ymin)
        mask_i = output['masks'][predicted_i][channel_i].detach()
        mask_i[mask_i >= threshold_mask ] = 1
        mask_i[mask_i < threshold_mask ] = 0
        mask_area = mask_i.sum()
        ymin_i = int(ymin)
        ymax_i = int(ymax)
        xmin_i = int(xmin)
        xmax_i = int(xmax)
        intersection = mask_i[ymin_i:ymax_i + 1, xmin_i:xmax_i + 1].sum()
        union = bbox_area + mask_area - intersection

        iou = intersection / union
        #iou_list.append(iou)
        if iou < iou_min:
            keep_i.append(False)
        else:
            keep_i.append(True)

    # truncate predictions accordingly
    if len(keep_i)>0:
        boolean_mask = torch.tensor(keep_i)
        transf_box = output["boxes"][boolean_mask] 
        transf_labels = output["labels"][boolean_mask]
        transf_scores = output["scores"][boolean_mask]
        transf_mask = output["masks"][boolean_mask]
    else:
        transf_box = output["boxes"]
        transf_labels = output["labels"]
        transf_scores = output["scores"]
        transf_mask = output["masks"]
    return transf_box, transf_labels, transf_scores, transf_mask

def postprocess_slope(output, slope_img, degree_thresh = 25, threshold_mask=0.5, threshold_onslope = 0.8):
    # Create boolean high_slope where slope is too high to contain RTS.
    # RTS should not appear at slope > 20 -> degree_thresh = 25
    high_slope = slope_img >= degree_thresh
    
    keep_i = []
    for predicted_i in range(len(output["boxes"])):
        channel_i = 0 # image channel
        mask_i = output['masks'][predicted_i][channel_i].detach()
        mask_i[mask_i >= threshold_mask ] = 1
        mask_i[mask_i < threshold_mask ] = 0
        mask_area = mask_i.sum()
        mask_inslope = mask_i * high_slope
        fraction_inslope = mask_inslope.sum()/mask_area
        if fraction_inslope >= threshold_onslope:
             keep_i.append(False)
        else:
            keep_i.append(True)
    # truncate predictions accordingly
    if len(keep_i)>0:
        boolean_mask = torch.tensor(keep_i)
        transf_box = output["boxes"][boolean_mask] 
        transf_labels = output["labels"][boolean_mask]
        transf_scores = output["scores"][boolean_mask]
        transf_mask = output["masks"][boolean_mask]
    else:
        transf_box = output["boxes"]
        transf_labels = output["labels"]
        transf_scores = output["scores"]
        transf_mask = output["masks"]
    return transf_box, transf_labels, transf_scores, transf_mask

def create_watername(original_tile):
    '''
    In order to apply water mask for postprocessing.
    Some images are on multiple water tiles -> before merging we need to find the names. 
    Watertile is numbered: nord to south: descending [10,9,...1]. west to east: descending [10,9,...1]
    '''
    y1 = int(original_tile[3])
    x1 = int(original_tile[4])
    y2 = int(original_tile[1])
    x2 = int(original_tile[2])

    # below: y is lower
    if y1>2:
        y1_low = str(y1-1)
        y2_low = str(y2)
        x1_low = str(x1)
        x2_low = str(x2)
    else:
        y1_low = str(10)
        y2_low = str(y2-1)
        x1_low = str(x1)
        x2_low = str(x2)

    # above
    if y1<=9:
        y1_above = str(y1+1)
        y2_above = str(y2)
        x1_above = str(x1)
        x2_above = str(x2)
    else:
        y1_above = str(1)
        y2_above = str(y2+1)
        x1_above = str(x1)
        x2_above = str(x2)  

    # left: x is higher (for image not tile, it is the other way around)
    if x1<=9:
        y1_left = str(y1)
        y2_left = str(y2)
        x1_left = str(x1+1)
        x2_left = str(x2)
    else:
        y1_left = str(y1)
        y2_left = str(y2)
        x1_left = str(1)
        x2_left = str(x2+1)     

    # right:
    if x1>2:
        y1_right = str(y1)
        y2_right = str(y2)
        x1_right = str(x1-1)
        x2_right = str(x2)
    else:
        y1_right = str(y1)
        y2_right = str(y2)
        x1_right = str(10)
        x2_right = str(x2-1)  
        
    water_name = '_'.join(['watermask',str(y2), str(x2), str(y1), str(x1)]) + '.tif'
    water_name_below = '_'.join(['watermask',y2_low, x2_low, y1_low, x1_low]) + '.tif'
    water_name_above = '_'.join(['watermask',y2_above, x2_above, y1_above, x1_above]) + '.tif'
    water_name_left = '_'.join(['watermask',y2_left, x2_left, y1_left, x1_left]) + '.tif'
    water_name_right = '_'.join(['watermask',y2_right, x2_right, y1_right, x1_right]) + '.tif'
    
    water_name_above_left = '_'.join(['watermask',y2_above, x2_left, y1_above, x1_left]) + '.tif'
    water_name_above_right = '_'.join(['watermask',y2_above, x2_right, y1_above, x1_right]) + '.tif'
    water_name_below_left = '_'.join(['watermask',y2_low, x2_left, y1_low, x1_left]) + '.tif'
    water_name_below_right = '_'.join(['watermask',y2_low, x2_right, y1_low, x1_right]) + '.tif'    
    return water_name, water_name_below,water_name_above,water_name_left,water_name_right, water_name_above_left, water_name_above_right, water_name_below_left, water_name_below_right

def apply_watermask(target, prediciton, site_name, test_path, threshold_outside_water = 0.5):
    '''
    If RTS is less than threshold_outside_water % outside of water, it is removed
    Uses create_watername () function
    All watertiles that are touching the image are merged. The range of the image is cropped out from the watertile. This cropped watertile is multiplied with the predicted mask to 
    calculate how much of the RTS is outside of water
    '''
                    
    # Read tilename
    image_name =target['tile']
    name_part = image_name.split('_')
    tile_name = ('_').join(name_part[1:6])

    # create watermask file name
    original_tile = tile_name.split('_')

    water_name, water_name_below,water_name_above,water_name_left,water_name_right, water_name_above_left, water_name_above_right, water_name_below_left, water_name_below_right = create_watername(original_tile)

    # Check if image is on multiple tiles, read all itles which are intersecting the image
    image_part = image_name.split('_')
    img_y = image_part[-2]
    img_x = image_part[-1][:-4]
    image_part

    # Read water tiles and neighbour water tiles
    if site_name == 'peel':
        water_dir = 'data/Sophia/data_clean/watermask_Peel_merged'
    elif site_name == 'tukto':
        water_dir = 'data/Sophia/data_clean/watermask_tuktoyaktuk_merged'
    else:
        print('Sitename is invalid')
    water_merge = [] # to store all tiles to be merged
    water_file = os.path.join(water_dir, water_name)
    water_tile = rasterio.open(water_file)
    water_merge.append(water_tile)

    # img is on two tiles at the bottom:
    if  int(img_y) ==0: # below
        water_file_below = os.path.join(water_dir, water_name_below)
        water_tile_below = rasterio.open(water_file_below)
        water_merge.append(water_tile_below)
        #print('below')
        if int(img_x) == 8: # belowright
            water_file_below_right = os.path.join(water_dir, water_name_below_right)
            water_tile_below_right = rasterio.open(water_file_below_right)
            water_merge.append(water_tile_below_right)
            #print('below')
        elif int(img_x) == 0: #belowleft
            water_file_below_left = os.path.join(water_dir, water_name_below_left)
            water_tile_below_left = rasterio.open(water_file_below_left)
            water_merge.append(water_tile_below_left) 
            #print('below')

    if int(img_y) == 8: # above
        water_file_above = os.path.join(water_dir, water_name_above)
        water_tile_above = rasterio.open(water_file_above)
        water_merge.append(water_tile_above)
        #print('above')
        if int(img_x) == 8: # aboveright
            water_file_above_right = os.path.join(water_dir, water_name_above_right)
            water_tile_above_right = rasterio.open(water_file_above_right)
            water_merge.append(water_tile_above_right)
            #print('above')
        elif int(img_x)==0: #above left
            water_file_above_left = os.path.join(water_dir, water_name_above_left)
            water_tile_above_left = rasterio.open(water_file_above_left)
            water_merge.append(water_tile_above_left)
            #print('above')

    if int(img_x)== 8: # right
        water_file_right = os.path.join(water_dir, water_name_right)
        water_tile_right = rasterio.open(water_file_right)
        water_merge.append(water_tile_right)    
        #print('right')

    if int(img_x) == 0: #left
        water_file_left = os.path.join(water_dir, water_name_left)
        water_tile_left = rasterio.open(water_file_left)
        water_merge.append(water_tile_left)
        #print('right')

    # merge neighbour water tiles
    merged_water, merged_transf = merge(water_merge)
    # Make water tiles binary
    merged_water = merged_water[0]
    merged_water[merged_water>0]= 2 
    merged_water[merged_water<= -9999]= 2 # For undefined values
    merged_water[merged_water <= 0] = 1
    
    # Create a cropping mask with the boundary of input image
    image_file = os.path.join(test_path, 'images',image_name)
    image_tile = rasterio.open(image_file)
    mask_geometry = [shapely.geometry.box(*image_tile.bounds)]
    mask_array = geometry_mask(mask_geometry, out_shape=merged_water.shape, transform = merged_transf, invert=True)   
    croped_water = merged_water*mask_array
    # Find the top-left, top-right, bottom-left, and bottom-right of the image corners
    # Find indices where the value is 1
    indices = np.where(croped_water >0)

    top_left = (min(indices[0]), min(indices[1]))
    bottom_right = (max(indices[0]), max(indices[1]))
    extracted_water = croped_water[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    # resize and make water = 0 rest 1
    water_mask_img = skimage.transform.resize(extracted_water, image_tile.shape, mode='constant', order=1)
    water_mask_img[water_mask_img<1.5]=0
    water_mask_img[water_mask_img>0] = 1
    
    keep_i = []
    for predicted_i in range(len(prediciton['labels'])):
        predicted_RTS = prediciton['masks'][predicted_i][0].detach().numpy() # 0 is for channel
        outside_water = water_mask_img * predicted_RTS
        outside_water_fraction = outside_water.sum() / predicted_RTS.sum()
        if outside_water_fraction < threshold_outside_water:
            keep_i.append(False)
            print('remove')
        else:
            keep_i.append(True)

    if len(keep_i)>0:
        boolean_mask = torch.tensor(keep_i)
        transf_box = prediciton["boxes"][boolean_mask] 
        transf_labels = prediciton["labels"][boolean_mask]
        transf_scores = prediciton["scores"][boolean_mask]
        transf_mask = prediciton["masks"][boolean_mask]
    else:
        transf_box = prediciton["boxes"]
        transf_labels = prediciton["labels"]
        transf_scores = prediciton["scores"]
        transf_mask = prediciton["masks"]
    return transf_box, transf_labels, transf_scores, transf_mask
#######################################################
      
def get_model_instance_segmentation(num_classes):
    '''
    modifying the architecture of pre-trained model: 
        Replace classification head & mask prediction head
        Set number of hidden layer to 256

    Input: num_classes = number of classes to be predicted in new model

    Return: model = Adapted model
    '''
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT") #TODO: Choose other pre-trained model

    # get number of input features for the classifier layer
        # roi_heads: regions of interest (ROIs) -> candidate bounding box regions within an image that are considered potential objects.
        # box_predictor: predicting class scores and bounding box regressions for each ROI 
        # cls_score:layer for predicting class scores
        # in_features: number of input features (dimensions) expected by the cls_score layer
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
        # FastRCNNPredictor constructor creates new classification head where num_classes = number of classes to predict
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
        # mask_predictor: predicting segmentation masks for object instances within ROIs
        # conv5_mask: convolutional layer for processing features 
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256 # value is typically chosen based on the model's architecture and the specific requirements of the task.
    # Replace the mask predictor with a new one
        # MaskRCNNPredictor constructor is used to create a new mask prediction head
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def normalise_data(data):
    image = (data - np.min(data)) / (np.max(data) - np.min(data))
    image *= 255
    image = image.astype(int)
    return image


# def get_transform():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
#     ])
#     return transform


def tb_logging(name):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger(f"/home/maierk/cds/working/Kathrin/deep_learning_data/tensorboard/log_{now}",
                               name=name)
    return logger

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


'''
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
ef setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


'''