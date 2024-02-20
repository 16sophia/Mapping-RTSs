import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from skimage.color import label2rgb
from matplotlib.colors import Normalize
from skimage.color import label2rgb
from torchvision.ops.boxes import box_iou
from matplotlib.colors import LinearSegmentedColormap

def combine_masks(masks):
    """
    Input: masks [RTS instance, y dimension, x dimension], binary mask for each RTS instance
    Return: mask [y dimension, x dimension]: instance mask
    
    Combine the masks labeled with their sequence number so that all RTS instances are on one plane."""
    if len(masks.shape) > 2:
        shape_mask = masks.shape
        all_masks = np.zeros((shape_mask[1], shape_mask[2])) # Create matrix filled with zeros with size = mask
        # Fill mask with different number
        for i, mask in enumerate(masks, 1): # count from 1 because background is 0
            all_masks[mask == True] = i
        return all_masks
    else: 
        return masks

def rgb_to_grayscale(rgb_matrix):
    '''Change RGB to Greyscale image according to luminosity method
    Inut [Height, Width, RGB]
    Output [Height, Width]
    '''
    grayscale_matrix = 0.299 * rgb_matrix[:,:,0] + 0.587 * rgb_matrix[:,:,1] + 0.114 * rgb_matrix[:,:,2]
    grayscale_matrix = grayscale_matrix.astype(np.uint8)
    return grayscale_matrix

def viz_bounding_box(ds, sample_index, threshold_mask = 0.5, batch_pred = None): 
    ''' 
    Visualizes all predicted bounding box and the corresponding mask in red. If targeted bounding box matches (according to bounding box iou), it will be shown in green. If it does not match, it will be shown in third pannel (unmatched labelled mask). Images are ordered according to the iou between boundary boxes (descending).
    
    Input: 
        ds: dataset consisting of tuple: image [tensor](tuple of batches containing image)[b, c, h,w] annotation [tuple of batches containing dictionary] 
        sample_index: image sample we want to analyze, index that accesses a specific tile in ds
        threshold_mask: threshold of predicted mask (logits) to decide which mask pixel will be visualized
        batch_pred: predicted result
        
    
    There is no guarantee that model prediction is in same order as targeted labels --> we need to match them by calculating the iou 
    Find match between predicted bounding box and labelled bounding box: 
    - Calculate iou between all predicted and targeted bounding boxes: result in matrix with predicted boxes (y axis)and labelled boxes (x axis) 
    - Calculate max number of RTS that can overlap = max(n_RTS_predicted, n_RTS_labelled) -> number of RTS we will keep in order to find match
    - Extract n = min_l highest values from iou matrix and get the corresponding y, x matrix id (predicted boxes (y axis)and labelled boxes (x axis))
    - Append those matrix id to indices_y, indices_x. If iou = 0, we add them to unmatched_y, unmatched_x
    - Append y matrix id of the remaining iou values to unmatched_y (predicted boxes)
    - Because we go through whole iou matrix, some RTS id can be in both unmatched and matched (E.g. matches with one label but not with the other one), Therefore: Remove elements in 'unmatched_y' that are in 'indices_y'        
'''
    # Extract data--------------------------------------
    img_batch, targets_batch = ds
    prediction = batch_pred[sample_index]  

    # Deepcopy to make sure we don't change original data.
    img = copy.deepcopy(img_batch)[sample_index][0].numpy()
    target = copy.deepcopy(targets_batch)[sample_index]
    res = {
    "masks": copy.deepcopy(prediction["masks"].detach()),
    "boxes": copy.deepcopy(prediction["boxes"].detach()),
    "labels": copy.deepcopy(prediction["labels"].detach())
    }

    # Get number of predicted and labelled RTS
    n_RTS_pred = len(batch_pred[sample_index]["labels"])
    n_RTS_label = len(target["labels"])
    print(f"Number of predicted RTS: {n_RTS_pred}")
    print(f"Number of labelled RTS: {n_RTS_label}")

    # Scale image for visualization: has no negative number and in valid range
    min_val = np.min(img)
    #if min_val<0:
    img_viz = img - min_val
    min_val = np.min(img_viz)
    max_val = np.max(img_viz)
    img_scaled = (img_viz - min_val) / (max_val - min_val)

    # Get rid of batch dimension
    res["masks"] = res["masks"][:, 0, :, :]

    # Match predicted bounding box with targeted bounding box-----------------------------------
    # Calculate iou because we only show bounding box if there is an intersection between predicted & labelled RTS
    iou = box_iou(res["boxes"],target["boxes"]).detach().numpy()

    # Calculate max number of RTS that overlap -> number of RTS we will keep
    min_l = min(target["labels"].size()[0], res["labels"].size()[0])

    # extract indices of RTS with the highest iou (where the number corresponds to min_l = min. number of labelled or predicted RTS)
    sorted_indices = np.argsort(iou.flatten())[::-1] # [::-1] sort in descending order, we start with max iou value
    indices_y = [] 
    indices_x = []
    iou_i = []
    unmatched_y = []
    unmatched_x = []
    # Add valid RTS pairs to indices. We iterate through n biggest iou values
    for i in sorted_indices[:min_l]:
        idy, idx = np.unravel_index(i, iou.shape)
        if iou[idy, idx] <=0: # IoU is 0 
            unmatched_y.append(idy)
            unmatched_x.append(idx) 
        else:
            iou_i.append(iou[idy, idx])
            indices_y.append(idy)
            indices_x.append(idx)

    # Add unvalid pairs to unmatched by iterating through the rest of iou values (not n biggest) and extract predicted bounding box id's
    for i in sorted_indices[min_l:]:
        idy, idx = np.unravel_index(i, iou.shape)
        unmatched_y.append(idy)
        #unmatched_x.append(idx) 
    
    # Extract bounding box, mask from matched predicted value, matched bounding box from labelled value. Make mask binary for visualization
    matched_pred = res["boxes"][indices_y]
    matched_label = target["boxes"][indices_x]
    predmask_matched = res["masks"][[indices_y]].detach().numpy()
    mask_thresh_matched = predmask_matched.copy()
    mask_thresh_matched[mask_thresh_matched >= threshold_mask ] = 1
    
    # Because we go through whole iou matrix, some RTS id can be in both unmatched and matched (E.g. matches with one label but not with the other one)
    # Therefore: Remove elements in 'unmatched_y' that are in 'indices_y'
    # Extract unmatched box and mask, make mask binary
    unmatched_y = np.unique([y for y in unmatched_y if y not in indices_y])
    unmatched_pred = res["boxes"][unmatched_y]
    #unmatched_label = target["boxes"][unmatched_x]
    predmask_unmatched = res["masks"][[unmatched_y]].detach().numpy()
    mask_thresh_unmatched = np.squeeze(predmask_unmatched.copy()) # get rid of dimension with size 1
    mask_thresh_unmatched[mask_thresh_unmatched >= threshold_mask ] = 1
    mask_thresh_unmatched[mask_thresh_unmatched < threshold_mask ] = 0

    # Visualize result--------------------------------------------------------------------------
    # RTS has been predicted:
    # print iou values of matched boxes
    if len(res["labels"]) > 0: 
        for box_i in range(len(matched_pred)):
            iou_print = iou_i[box_i]
            print(f"IoU: {iou_i}, box {indices_y[box_i]}")
                  
        plot_id = 0 # for subplot index
        fig, axs = plt.subplots(1, max(2,len(res["labels"])), figsize=(10, 10))
        fig.set_figheight(10)
        fig.set_figwidth(10+max(2,len(res["labels"])))

        # Visualize matched RTS pair
        for box_i in range(len(matched_pred)):
            axs[box_i].set_title(f'Predicted box {indices_y[box_i]}')
            if len(res["labels"])==1:
                axs[box_i].imshow(label2rgb(mask_thresh_matched[box_i], img_scaled, bg_label=0))
            else:
                axs[box_i].imshow(label2rgb(mask_thresh_matched[box_i], img_scaled, bg_label=0))

            # Plot bounding box of predicted RTS
            x1, y1, x2, y2 = matched_pred[box_i].detach().numpy()
            width = x2 - x1
            height = y2 - y1
            rect = matplotlib.patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axs[box_i].add_patch(rect)

            # Plot bounding box of labelled RTS
            x_1, y_1, x_2, y_2 = matched_label[box_i].detach().numpy()
            width_ = x_2 - x_1
            height_ = y_2 - y_1
            rect = matplotlib.patches.Rectangle((x_1, y_1), width_, height_, linewidth=3, edgecolor='g', facecolor='none')
            axs[box_i].add_patch(rect)

        # Visualized unmatched predicted RTS: only predicted RTS bounding box
        for box_i in range(len(unmatched_pred)):
            plot_id = box_i+ len(matched_pred)
            if len(unmatched_pred) ==1:
                axs[plot_id].set_title(f'Predicted box {unmatched_y[box_i]}')
                axs[plot_id].imshow(label2rgb(mask_thresh_unmatched, img_scaled, bg_label=0))
            else:
                axs[plot_id].set_title(f'Predicted box {unmatched_y[box_i]}')
                axs[plot_id].imshow(label2rgb(mask_thresh_unmatched[box_i], img_scaled, bg_label=0))
            # Plot bounding box of predicted RTS
            x1, y1, x2, y2 = unmatched_pred[box_i].detach().numpy()
            width = x2 - x1
            height = y2 - y1

            rect = matplotlib.patches.Rectangle((x1, y1), width, height, linewidth=5, edgecolor='none', facecolor='red', alpha = 0.3, linestyle='dotted',clip_on=False)
            axs[plot_id].add_patch(rect)

        visualized_i = max(plot_id, box_i)
        if  visualized_i == 0:
            for i in range(1, len(axs)):
                fig.delaxes(axs[i])

        # Plot all unmatched labelled RTS
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.set_figheight(5)
        fig.set_figwidth(5)

        all_id = np.arange(0, len(target["boxes"]))
        label_unmatched = [y for y in all_id if y not in indices_x]

        axs.set_title(f'Unmatched labelled boxes ({len(label_unmatched)} instances)')
        axs.imshow(img_scaled)

        for idx in label_unmatched:
            x1, y1, x2, y2 =  target["boxes"][idx].detach().numpy()
            width = x2 - x1
            height = y2 - y1

            rect = matplotlib.patches.Rectangle((x1, y1), width, height, linewidth=5, edgecolor='g', facecolor='none', linestyle='dotted',clip_on=False)
            axs.add_patch(rect)

    else: # No RTS was predicted
        print("No RTS was predicted")
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.set_figheight(5)
        fig.set_figwidth(5)

        label_unmatched = np.arange(0, len(target["boxes"]))

        axs.set_title(f'Unmatched labelled masks ({len(label_unmatched)} instances)')
        axs.imshow(img_scaled)

        for idx in label_unmatched:
            x1, y1, x2, y2 =  target["boxes"][idx].detach().numpy()
            width = x2 - x1
            height = y2 - y1

            rect = matplotlib.patches.Rectangle((x1, y1), width, height, linewidth=5, edgecolor='g', facecolor='none', linestyle='--',clip_on=False)
            axs.add_patch(rect)
     
    
def vize_sample(ds, sample_index, model_viz = None, threshold_mask = 0.5, predictions = None):
    '''
    Visualizes original image vs annotated label vs predicted label
    ds: dataset consisting of tuple: image [tensor], annotation [dictionary]
    sample_index: image sample we want to analyze
    model_viz: model that will be applied to predict labels: If not none, predictions will be implemented, otherwise we'll use provided preditions
    threshold_mask: threshold for assigning the predictions to mask 
    '''
    img_batch, targets_batch = ds
    img = img_batch[sample_index]
    targets = targets_batch[sample_index]
    # First channel is always image channel
    img_viz = img[0].numpy()
    masks_labelled = combine_masks(targets["masks"].numpy()) # combine all instances together if on same image

    # Scale image
    min_val = np.min(img_viz)
    if min_val<0:
        img_viz = img_viz - min_val
    min_val = np.min(img_viz)
    max_val = np.max(img_viz)
    img_scaled = (img_viz - min_val) / (max_val - min_val)

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].set_title('Original image')
    axs[0].imshow(img_viz)
    axs[1].set_title('Annotated label')
    axs[1].imshow(label2rgb(masks_labelled, img_scaled, bg_label=0)) 
    
    # transform labelled instance mask to binary mask of annotated RTS
    all_masks_labelled = combine_masks(masks_labelled)
    annotation_flat = np.copy(all_masks_labelled)
    annotation_flat[annotation_flat>0] = 1

    if model_viz is not None:
        model_viz.eval()
        res = model_viz.detector(img[np.newaxis, ... ]) # add batch axis    
        masks_pred = res["masks"].detach().numpy() # detach from pytorch computation graph
        mask_thresh = np.squeeze(masks_pred.copy()) # get rid of dimension with size 1, channel dimension?
        mask_thresh[mask_thresh > threshold_mask ] = 1
        all_masks = combine_masks(mask_thresh)
        mask_thresh = all_masks # combine all instances together if on same image
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].set_title('Original image')
        axs[0].imshow(img_viz)
        axs[1].set_title('Predicted label')
        # axs[1].imshow(res[0]["masks"][0,0].detach().numpy())
        axs[1].imshow(label2rgb(mask_thresh, img_scaled, bg_label=0))
        num_obj = len(res["masks"])
        n_obj = targets["masks"].shape[0]
        print(f"Number of detected objects = {num_obj}. Number of real objects = {n_obj}")
    else:
        if predictions == None:
            print("Provide either model or predictions")

        else:
            res = predictions[sample_index]    
            masks_pred = res["masks"].detach().numpy() # detach from pytorch computation graph
            mask_thresh = np.squeeze(masks_pred.copy()) # get rid of dimension with size 1, channel dimension?
            mask_thresh[mask_thresh > threshold_mask ] = 1
            all_masks = combine_masks(mask_thresh)
            mask_thresh = all_masks # combine all instances together if on same image
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].set_title('Predicted vs annotated')
            axs[0].imshow(annotation_flat)
            axs[0].imshow(mask_thresh, alpha=0.5)
            axs[1].set_title('Predicted label')
            # axs[1].imshow(res[0]["masks"][0,0].detach().numpy())
            axs[1].imshow(label2rgb(mask_thresh, img_scaled, bg_label=0))
            num_obj = len(res["masks"])
            n_obj = targets["masks"].shape[0]
            print(f"Number of detected objects = {num_obj}. Number of real objects = {n_obj}")            

            
def viz_mask(img_org, targ_org, preds_org, img_index, viz_max = 3, viz_min = -3):
    '''
    Visualizes labeled Mask, predicted mask and its comparison
    Input: img_org: Tuple of images in batch. Each image has shape [channel, H, W], targ_org: Tuple of labels in batch. One label consists of dictionary with mask entry. Mask shape is [RTS instance, H,W]
    preds_org: List of predictions, each prediction contains dictionary with mask entry. Mask shape is [RTS instances, channel, H,W]
    img_index = index of image within batch, viz_max, viz_min = capping value of image for divergent value, default is 3,= -3. otherwise will be scaled: (-3 -img_min)/ (img_max-img_min)
    
    '''
    html_red = '#d7191c'    # Red
    html_red2 = '#fdae61'
    html_white = '#FFFFFF'  # White
    html_green = '#1a9641'  # Green
    html_green2 = '#a6d96a'

    # Convert HTML color codes to RGB
    red = tuple(int(html_red[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    red2 = tuple(int(html_red2[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    white = tuple(int(html_white[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    green = tuple(int(html_green[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    green2 = tuple(int(html_green2[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    cmap = LinearSegmentedColormap.from_list('rg', [red, red2, white, green2, green], N=256)

    channel = 0
    img = img_org[img_index][channel]
    pred = preds_org[img_index]

    # Set dark grey color for 1 and black color for. 0
    bool_True = [1,1,1]
    bool_false = [0,0,0]
    dark_grey = [50, 50, 50]
    black = [0, 0, 0]
    dark_blue = [0,0, 180]

    # Make Grayscale image to RGB with diverging color
    gray_image = np.clip(img, viz_min, viz_max) 
    #cmap = plt.get_cmap('RdYlGn')
    norm = Normalize(vmin=gray_image.min(), vmax=gray_image.max())
    image_RGB = cmap(norm(gray_image), bytes=True)[:,:,:3]

    # Combine all target masks to one img & make it RGB
    targ_mask = targ_org[img_index]["masks"]
    targ_thr = np.squeeze(targ_mask.detach().numpy().copy())
    targ_thr[targ_thr >= 0.5 ] = 1
    targ_thr[targ_thr < 0.5 ] = 0
    targ_tot_img = combine_masks(targ_thr)

    pred_mask = pred["masks"]
    pred_thr = np.squeeze(pred_mask.detach().numpy().copy())
    pred_thr[pred_thr >= 0.5 ] = 1
    pred_thr[pred_thr < 0.5 ] = 0
    pred_tot_img = combine_masks(pred_thr)
    
    n_pred = len(np.unique(pred_tot_img))-1
    n_targ = len(np.unique(targ_tot_img))-1

    print('Number of predicted RTS:', n_pred,  'Number of labeled RTS:', n_targ)

    # Calculate metrics
    y_true = (targ_tot_img>0).flatten()
    y_pred = (pred_tot_img>0).flatten()
    correct = np.sum(y_true == y_pred).item()
    total = len(y_true)
    true_positives = np.sum((y_true == 1) & (y_pred == 1)).item()
    false_positives = np.sum((y_true == 0) & (y_pred == 1)).item()
    false_negatives = np.sum((y_true == 1) & (y_pred == 0)).item()

    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    IoU = true_positives / (false_positives + true_positives + false_negatives) if (precision + recall) > 0 else 0
    accuracy,precision, recall, f1, IoU = np.round([accuracy,precision, recall, f1, IoU], 2)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, IoU: {IoU}')

    # Create a 3D NumPy array for the RGB image------------------------------------------
    pred_rgb_mask = np.zeros((*pred_tot_img.shape, 3), dtype=np.uint8)
    pred_rgb_mask[pred_tot_img > 0] = bool_True
    pred_rgb_mask[pred_tot_img == 0] = bool_false

    pred_rgb_viz = np.zeros((*pred_tot_img.shape, 3), dtype=np.uint8)
    pred_rgb_viz[pred_tot_img > 0] = dark_blue
    pred_rgb_viz[pred_tot_img == 0] = black

    targ_rgb_mask = np.zeros((*targ_tot_img.shape, 3), dtype=np.uint8)
    targ_reshaped = cmap(targ_tot_img)[:,:,:3]
    targ_rgb_mask[targ_tot_img > 0] = bool_True
    targ_rgb_mask[targ_tot_img == 0] = bool_false

    targ_rgb_viz = np.zeros((*targ_tot_img.shape, 3), dtype=np.uint8)
    targ_reshaped = cmap(targ_tot_img)[:,:,:3]
    targ_rgb_viz[targ_tot_img > 0] =  dark_grey
    targ_rgb_viz[targ_tot_img == 0] = black

    # Blend the images using the mask
    blended_prediction = image_RGB.copy()
    blended_prediction[:, :, :3] = np.where(pred_rgb_mask, pred_rgb_viz[:, :, :3], image_RGB[:, :, :3])

    blended_label = image_RGB.copy()
    blended_label[:, :, :3] = np.where(targ_rgb_mask, targ_rgb_viz[:, :, :3], image_RGB[:, :, :3])


    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].set_title('Labelled mask')
    axs[0].imshow(blended_label[:,:,:])
    axs[1].set_title('Predicted mask')
    axs[1].imshow(blended_prediction[:,:,:])   

    grayscale_matrix = rgb_to_grayscale(blended_label)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.set_title('Labelled vs. predicted')
    axs.imshow(label2rgb(pred_tot_img, grayscale_matrix, bg_label=0))