import numpy as np
import torch.nn.functional as Fn
from torchvision.ops import boxes as box_ops

import torch
import os
import copy
import rasterio
import shapely
import skimage

from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou
from skimage.measure import regionprops
from rasterio.features import geometry_mask
from rasterio.merge import merge
from shapely.geometry import shape
from rasterio.features import shapes
'''
Functions to run MaskRCNN model, caculation of performance metrics, postprocessing steps
'''

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

def custom_postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = Fn.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels 