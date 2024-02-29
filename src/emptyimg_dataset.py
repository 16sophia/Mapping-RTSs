import numpy as np
import os
import torch
import wandb
import torchmetrics
import warnings
import rasterio

from PIL import Image
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.transforms import functional as F
'''
Creates Dataset for Mask-RCNN application: enables empty images. Can only be used for testing, not training.

'''


class emptyimg_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, n_channel, channel_list):
        '''
        Constructor, creating initial object of DEMDataset(root, transforms)
        Generating sorted list of image file names (self.imgs) and mask file names (self.masks) 

        Input:  self:
                root: dataset's root directory
        '''
        
        self.root = root
        # load all image files, sorting them to ensure that they are aligned by name
        self.imgs_all = [item for item in os.listdir(os.path.join(root, "images"))
                           if item.endswith(".tif")]
        self.masks_all = [item for item in os.listdir(os.path.join(root, "masks"))
                           if item.endswith(".tif")]
        
        # Make sure to only keep tiles that appear both in images and masks folder
        set1 = set(self.imgs_all)
        filename_list = list(set1.intersection(self.masks_all))
        self.filename_list = sorted(filename_list)
        self.n_channel = n_channel
        self.channel_list = channel_list
        
    def __len__(self):
        '''
        Define the length of an object, called by len(obj)
        A Python magic method
        
        Return: length of DEMDataset image 
        '''
        return len(self.filename_list)

    def __getitem__(self, idx):
        ''' Called on []
         Loading and processing data for a given image index (idx)
         
         Deletes images that are empty (full of 0 or NaN) or corresponding mask are empty

         Return: img, annotations
            img: (transformed) RGB image with index idx
            annotations: dictionary containing all the processed information    
                boxes = bounding box coordinates for each mask [xmin, ymin, xmax, ymax]
                labels = object class, initialized as all one (=people)
                masks = 3D tensor, set of binary masks. z= mask index, yz = 2D mask
                        0 = background, 1 = object
                image_id = Image id idx as tensor
                area = box area
                iscrowd = tensor with len(obj) showing if instance is crowd. Initialized with all zero 
        '''
        # n channel image, where first channel is DEM image
        img_path = os.path.join(self.root, "images", self.filename_list[idx])
        # Add other channel content e.g. slope, aspect
        if self.n_channel >1: 
            image = np.array(Image.open(img_path),dtype = float)[...,np.newaxis] # convert the PIL Image into a numpy array and add third dimension axis
            # concatenate image where dimensions are (h, w, c). Because to_tensor transforms dimensions to (c,h,w) later on
            for channel_content in self.channel_list[1:]: # skip images in list since it is read before
                path = os.path.join(self.root, channel_content, self.filename_list[idx])
                content = np.array(Image.open(path),dtype = float)[..., np.newaxis]
                image = np.concatenate([image, content], axis=2)  

        else:
            image = np.array(Image.open(img_path),dtype = float) # convert the PIL Image into a numpy array

        # instance mask
        mask_path = os.path.join(self.root, "masks", self.filename_list[idx])
        mask = Image.open(mask_path) # image has PIL format: "Python Imaging Library"
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        
        # Get number of objects
        # create an array of unique values: each object instance is encoded as different color
        obj_ids = np.unique(mask)
        # first id is the background-> remove it
        obj_ids = obj_ids[1:]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        
        # split the color-encoded mask into a set of binary masks (Each id has a mask in third dimension)
        # By first inserting two axis to 1 dimensional obj_ids with [:, None, None] to enable elementwise comparison between obj_ids nd mask
        masks = mask == obj_ids[:, None, None]
        
        # Catch case where image is empty (full of 0 or NaN) or mask is empty: Give arbitary values. 
        if image.max() <= 0 or np.isnan(image.max()) or num_objs < 1: 
            box = []
            for i in range(num_objs):
                xmin = 0
                xmax = 1
                ymin = 0
                ymax = 1
                box.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor, no deep copy
            boxes = torch.as_tensor(box, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            # Calculate box area
            # product of height (ymax - ymin) & width (xmax - xmin) 
            if num_objs > 0:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                area = 0

            # Create class labels and iscrowd label
            # Each obj belongs to RTS class, therefore labels is a tensor filled with 1
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # deep copy with torch.tensor
            image_id = int(idx)
            # suppose all instances are not-crowd: RTS are not heavily overlapping
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Tilename of image
            tile = self.filename_list[idx]
            
            # Get geoinformation of tif image
            reference =  rasterio.open(img_path, nodata=-9999) 
            meta = {
                'driver': 'GTiff',
                'count': 1, # save mask as single band image
                'height': 256,
                'width': 256,
                'crs': reference.crs,
                # Transform raster to georeferenced cooridnate: define min max coordinate 
                'transform': rasterio.transform.from_bounds(
                    west= reference.bounds[0],
                    south= reference.bounds[1],
                    east=reference.bounds[2],
                    north=reference.bounds[3],
                    width=reference.width,
                    height=reference.height
                ),
                'dtype': 'int64'
            }
            
            
            # Create annotation dictionary containing all the processed information
            annotations = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
               "iscrowd": iscrowd, "tile": tile, "num objs": num_objs, "geo_meta":meta}
            
            # Transforms image from np.array to tensor & transforms dimension from (h, w, c) to (c,h,w) with torchvision transform function
            image = F.to_tensor(image).float()

            return image, annotations
 
        # Calculate boundary box
        else: # image is not empty
            box = []
            for i in range(num_objs):
                pos = np.nonzero(masks[i]) # calls specific binary mask from masks set and extracts the indices of the non-zero elements 
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmax<=xmin) | (ymax<=ymin): # box is invalid: delete data
                    warnings.warn(f"Bounding box at index {idx} is invalid. Skipped {self.filename_list[idx]}.")
                    # print(f"skipped {self.filename_list[idx]}")
                    del self.filename_list[idx]
                    return np.nan
                else:
                    box.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor, no deep copy
            boxes = torch.as_tensor(box, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            # Calculate box area
            # product of height (ymax - ymin) & width (xmax - xmin) 
            if num_objs > 0:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                area = 0

            # Create class labels and iscrowd label
            # Each obj belongs to RTS class, therefore labels is a tensor filled with 1
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # deep copy with torch.tensor
            image_id = int(idx)
            # suppose all instances are not-crowd: RTS are not heavily overlapping
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Tilename of image
            tile = self.filename_list[idx]
            
            # Get geoinformation of tif image
            reference =  rasterio.open(img_path, nodata=-9999) 
            meta = {
                'driver': 'GTiff',
                'count': 1, # save mask as single band image
                'height': 256,
                'width': 256,
                'crs': reference.crs,
                # Transform raster to georeferenced cooridnate: define min max coordinate 
                'transform': rasterio.transform.from_bounds(
                    west= reference.bounds[0],
                    south= reference.bounds[1],
                    east=reference.bounds[2],
                    north=reference.bounds[3],
                    width=reference.width,
                    height=reference.height
                ),
                'dtype': 'int64'
            }
            
            
            # Create annotation dictionary containing all the processed information
            annotations = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
               "iscrowd": iscrowd, "tile": tile, "num objs": num_objs, "geo_meta":meta}
            
            # Transforms image from np.array to tensor & transforms dimension from (h, w, c) to (c,h,w) with torchvision transform function
            image = F.to_tensor(image).float()

            return image, annotations

