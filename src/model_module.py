import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import src.transforms as T
import src.utils as utils
import pytorch_lightning as pl
import torch
import numpy as np
import wandb
import torchmetrics
import copy



from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
#from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops.boxes import box_iou
from src.engine import train_one_epoch, evaluate, lambda_function, custom_postprocess_detections
from src.utils import unravel_index
from torch import nn


class Model(pl.LightningModule):
    
    def __init__(self, num_classes, pretrained_network, config, change_min_box_axis = False):
        '''
        Initializes MaskRCNN with pytorchlightning obj
        
        Input:
            num_classes: #classeses to be predicted
        '''
        lr = config["lr"]
        min_lr_config = config["min_lr"]
        optimizer_config = config["optimizer_config"]
        momentum_config = config["momentum"]
        weight_decay_config = config["weight_decay"]
        nesterov_config = config["nesterov"]
        beta_config = config['betas']
        eps_config = config['eps']
        amsgrad_config = config['amsgrad']
        n_channel = config["n_channel"]
        img_mean = config["img_mean"]
        img_std = config["img_std"]
        img_size = config["img_size"]
        hidden_layer = config['hidden_layer']
        min_box_axis = config['min_box_axis']
        lrscheduler = config['lrscheduler']
        drop_out_apply = config['drop_out_apply']
        adapt_nms = config['adapt_nms']

        super().__init__()
        # Set model parameter.
        self.detector =  pretrained_network # pretrained_network: pretrained network for transfer learning.
        self.learning_rate = lr # lr: step size at which the model's weights are updated
        self.optimizer = optimizer_config # sgd or adam 
        self.momentum = momentum_config #  How much past gradients are considered
        self.weight_decay=weight_decay_config  #weight_decay: regularization technique
        self.nesterov=nesterov_config #  nesterov: Nesterov momentum
        self.betas = beta_config # betas: coefficients used for computing running averages of gradient
        self.eps = eps_config # eps: term added to the denominator to improve numerical stability
        self.amsgrad = amsgrad_config # amsgrad: whether to use the AMSGrad variant
        self.min_lr = min_lr_config # min_lr: minimum learning rate
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.lrscheduler = lrscheduler
        self.min_axis_box = min_box_axis
 
        
        # replace the pre-trained head with a new one
        # Set number of input features for the classifier layer
          # roi_heads: regions of interest (ROIs) -> candidate bounding box regions within an image that are considered potential objects.
          # box_predictor: predicting class scores and bounding box regressions for each ROI 
          # cls_score:layer for predicting class scores
          # in_features: number of input features (dimensions) expected by the cls_score layer
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        
          # Model constructor creates new classification head where num_classes = number of classes to predict
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                                                                  
        # now get the number of input features for the mask classifier
        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one 
        self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        # modify model with monkey patching
        self.detector.backbone.body.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.detector.transform.image_mean = img_mean #img_mean
        self.detector.transform.image_std= img_std #img_std
        self.detector.transform.min_size = (img_size,) 
        self.detector.transform.max_size = img_size
        torchvision.ops.boxes.remove_small_boxes.min_size = min_box_axis # change remove_small_boxes which maskrcnn roi_heads calls 
        
        # Change min_axis in remove_small_boxes: https://github.com/pytorch/vision/blob/e12d200c97d7aab668b976e92b46513c9ca7a0d8/torchvision/models/detection/roi_heads.py#L668
        if change_min_box_axis:
            self.detector.roi_heads.postprocess_detections = custom_postprocess_detections.__get__(self.detector.roi_heads)
        
        # adapt_nms
        if adapt_nms:
            # number of proposals that are kept after NMS
            self.detector.rpn._post_nms_top_n = config['post_nms_top_n']
            # box_nms_thresh final bounding box predictions 
            self.detector.roi_heads.nms_thresh = config['box_nms_thresh'] 
            # rpn_nms_thresh: NMS threshold used for postprocessing the RPN proposals
            self.detector.rpn.nms_thresh = config['box_nms_thresh'] 
        
        if drop_out_apply:
            dropout_prob = 0.5 # 0.5 = default

            # Place drop out layers according to: https://www.researchgate.net/figure/Different-configurations-of-MC-dropout-in-Mask-RCNN-We-place-dropout-layers-at-the-end_fig1_368935045#:~:text=We%20place%20dropout%20layers%20at,the%20box%20branch%20(purple).
            if drop_out_apply:
                # backbone
                self.detector.backbone.body.layer4.append(torch.nn.Dropout(p=dropout_prob))

                 # mask branch
                self.detector.roi_heads.mask_head.append(torch.nn.Dropout(p=dropout_prob))

                # box_head
                self.detector.roi_heads.box_head.fc7 = torch.nn.Sequential(
                        torch.nn.Linear(in_features=1024, out_features=1024),
                          torch.nn.Dropout(p=dropout_prob))            
    
        # TODO: check if we can set number of boxes before Non-Maximum Suppression (NMS)
        # self.detector.roi_heads.mask_predictor = MaskRCNN(in_features_mask, hidden_layer, num_classes,  box_detections_per_img=10)
        
        # Freeze layers
        # first layer was replaced. Set it to untrainable
        for param in self.detector.backbone.body.conv1.parameters():
            param.requires_grad = False
            
        if config['freeze_layer']: # freeze layer other than backbone
            if config['freeze_fpn']:
                # set all layers of fpn to untrainable
                i = 0
                for name, param in self.detector.backbone.fpn.named_parameters():
                    if param.requires_grad == True:
                        param.requires_grad = False
                    i+=1      
            if config['freeze_rpn']:
                # set all layers of rpn to untrainable
                i = 0
                for name, param in self.detector.rpn.named_parameters():
                    if param.requires_grad == True:
                        param.requires_grad = False
                    i+=1
            if config['freeze_roi']:
                # Set all but last predicting layers of roi to untrainable
                i = 0
                for name, param in self.detector.roi_heads.named_parameters():
                    if (name == 'box_head.fc7.weight' or name == 'box_head.fc7.bias' or name == 'box_predictor.cls_score.weight' or name == 'box_predictor.cls_score.bias' or 
                        name == 'box_predictor.bbox_pred.weight' or name == 'box_predictor.bbox_pred.bias' or name ==  'mask_head.3.0.weight' or name ==  'mask_head.3.0.bias' or 
                        name ==  'mask_predictor.conv5_mask.weight'or name == 'mask_predictor.conv5_mask.bias' or 
                        name == 'mask_predictor.mask_fcn_logits.weight' or name ==  'mask_predictor.mask_fcn_logits.bias'):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    i+=1                
        
        

    def full_train(self): #TODO: implement later
        '''
        Sets pretrained_network to training mode by enabling gradient computation
        '''
        self.detector.requires_grad = True
    
    def configure_optimizers(self):
        ''' Part of PyTorch Lightning, sets optimizer and scheduler
        optimizer: Either Adam or stg optimizer
            Adjust the model's parameters according to gradient
        scheduler: CosineAnnealingWarmRestarts scheduler is a learning rate scheduler: can help escape local minima 
            -> periodically increasse the learning rate for T_0 iterations, then follow a cosine annealing schedule to gradually decrease it to a minimum value eta_min. 
                After each cycle, it will restart from a higher learning rate. 
        '''
        
        if self.optimizer == "adamW":
            # TODO: implement Adam parameters
            optimizer = torch.optim.AdamW(self.parameters(),lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad = self.amsgrad)

        elif self.optimizer == "adam":
            # TODO: implement Adam parameters
            optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad = self.amsgrad)
        else:
            optimizer = torch.optim.SGD(self.parameters(), self.learning_rate, momentum= self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
            
        # scheduler parameter: optimizer, T_0 – Number of iterations for first restart, T_mult – A factor increases T after start, eta_min– Minimum learning rate, last_epoch-The index of last epoch
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3, 1, self.min_lr, verbose=True)
        if self.lrscheduler == 'onplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        elif self.lrscheduler == 'cosinelr':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3, 1, self.min_lr, verbose=True)
        else:
            print("Not valid lr scheduler. ReduceLROnPlateau is chosen")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "sum_loss_training" # metric determining when to adjust the learning rate
                }}
    
    
    def forward(self, x):
        ''' Part of PyTorch Lightning, defines the forward pass of your model
        Input: x = image
        '''
        self.detector.eval() # Set to evaluation mode for inference or validation
        return self.detector(x) # runs object detection

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = len(images)
        loss_dict = self.detector(images, targets) # Gets loss_classifier, loss_box_reg, loss_mask
        preds = self.forward(images)
        self.detector.train()
        
        # Loss
        sum_loss = sum(loss_dict.values())
        self.log("sum_loss_training", sum_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        loss_dict = {k:(v.detach() if hasattr(v, "detach") else v) for k, v in loss_dict.items()} # detach PyTorch tensors from the computation graph   
        for key, value in loss_dict.items():
            self.log(f'training_{key}', value, on_step=True, on_epoch=True, batch_size=batch_size)
     
        # Metrics calculated from Confusion matrix (TP, FP, FN) with iou_thresholds. TN is not taken into account since maskRCNN does not have empty images as input.----------------------------
        # Performance metrics on image level, RTS_accuracy = TP/(TP+FP+FN) , mean over different thresholds and mean over batch
        # Performance metrics on RTS/ image level: If RTS was detected based on IoU BBox >= 0.5
        accuracy_ = []
        precision_ = []
        recall_ = []
        F1_ = []
        
        # Performance metrics on pixel per RTS level: If pixel of RTS instance was detected (only taking binary masks of detected RTS into account)
        accuracy_pixel_ = []
        precision_pixel_ = []
        recall_pixel_ = []
        F1_pixel_ = []
        iou_pixel_ = []
        
        # Check how many pixels in an imaeg are correctly detected
        accuracy_img_ = []
        precision_img_ = []
        recall_img_ = []
        F1_img_ = []
        iou_img_ = []
        
        # Iterate over each batch
        for labels, output in zip(targets,preds):
            acc_pixel, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, accuracy_RTS, precision_RTS, recall_RTS, F1_RTS, acc_img, precision_img, recall_img, f1_img, IoU_img = torch.Tensor(utils.similarity_RTS(output["masks"], labels["masks"], labels["boxes"],output["boxes"])).to(self.device_)
            
            # Performance metrics on image level
            accuracy_.append(accuracy_RTS)
            precision_.append(precision_RTS)
            recall_.append(recall_RTS)
            F1_.append(F1_RTS)
            # Performance metrics on RTS level
            accuracy_pixel_.append(acc_pixel)
            precision_pixel_.append(precision_pixel)
            recall_pixel_.append(recall_pixel)
            F1_pixel_.append(f1_pixel)
            iou_pixel_.append(IoU_pixel)
            # Performance metrics on pixel/ image
            accuracy_img_.append(acc_img)
            precision_img_.append(precision_img)
            recall_img_.append(recall_img)
            F1_img_.append(f1_img)
            iou_img_.append(IoU_img)
            
        
        # Performance metrics on image level
        train_accuracy = torch.nanmean(torch.stack(accuracy_)) 
        train_precision = torch.nanmean(torch.stack(precision_))
        train_recall = torch.nanmean(torch.stack(recall_)) 
        train_F1 = torch.nanmean(torch.stack(F1_))
        self.log("RTS_Accuracy_train", train_accuracy, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_Precision_train", train_precision, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_Recall_train", train_recall, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_F1_train", train_F1, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Performance metrics on RTS level
        accuracy_pixel = torch.nanmean(torch.tensor(accuracy_pixel_, dtype=torch.float))
        precision_pixel = torch.nanmean(torch.tensor(precision_pixel_, dtype=torch.float))
        recall_pixel = torch.nanmean(torch.tensor(recall_pixel_, dtype=torch.float))
        F1_pixel = torch.nanmean(torch.tensor(F1_pixel_, dtype=torch.float))
        IoU_pixel = torch.nanmean(torch.tensor(iou_pixel_, dtype=torch.float))
        self.log("Mask_accuracy_train", accuracy_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_precision_train", precision_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_recall_train", recall_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_F1_train", F1_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_IoU_train", IoU_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        
        accuracy_img = torch.nanmean(torch.tensor(accuracy_img_, dtype=torch.float))
        precision_img =torch.nanmean(torch.tensor(precision_img_, dtype=torch.float))
        recall_img = torch.nanmean(torch.tensor(recall_img_, dtype=torch.float))
        F1_img = torch.nanmean(torch.tensor(F1_img_, dtype=torch.float))
        IoU_img = torch.nanmean(torch.tensor(iou_img_, dtype=torch.float))
        self.log("Image_accuracy_train", accuracy_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_precision_train", precision_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_recall_train", recall_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_F1_train", F1_img, on_step=True, on_epoch=True, batch_size=batch_size) 
        self.log("Image_IoU_train", IoU_img, on_step=True, on_epoch=True, batch_size=batch_size)

        return {"loss": sum_loss, "log": loss_dict}
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        batch_size = len(imgs)
        # Get loss dictionary
        self.detector.train()
        loss_dict = self.detector(imgs, targets)
        for key, value in loss_dict.items():
            self.log(f'validation_{key}', value, on_step=True, on_epoch=True, batch_size=batch_size)
        sum_loss = sum(loss_dict.values())
        self.log("sum_loss_validation", sum_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.detector.eval()
        
        # rest of validation_step
        preds = self.forward(imgs)
        batch_size = len(imgs)
        
        classification_loss_ = []
        box_regression_loss_ = []
        mask_loss_ = []
        iou_RTS_ = [] # mean iou that do not take false positives and false negatives into acount
        
        # Calculate detection metrics--------------------------------------------------
        # Performance metrics on RTS/ image level: If RTS was detected based on IoU BBox >= 0.5
        accuracy_ = []
        precision_ = []
        recall_ = []
        F1_ = []
        
        # Performance metrics on pixel per RTS level: If pixel of RTS instance was detected (only taking binary masks of detected RTS into account)
        accuracy_pixel_ = []
        precision_pixel_ = []
        recall_pixel_ = []
        F1_pixel_ = []
        iou_pixel_ = []
        
        # Check how many pixels in an imaeg are correctly detected
        accuracy_img_ = []
        precision_img_ = []
        recall_img_ = []
        F1_img_ = []
        iou_img_ = []
            
        # Iterate over images in batch
        for labels, prediction in zip(targets,preds):
            acc_pixel, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, accuracy_RTS, precision_RTS, recall_RTS, F1_RTS, acc_img, precision_img, recall_img, f1_img, IoU_img = torch.Tensor(utils.similarity_RTS(prediction["masks"], labels["masks"], labels["boxes"],prediction["boxes"])).to(self.device_)
            
            # Performance metrics on image level
            accuracy_.append(accuracy_RTS)
            precision_.append(precision_RTS)
            recall_.append(recall_RTS)
            F1_.append(F1_RTS)
            # Performance metrics on pixel / detected RTS level
            accuracy_pixel_.append(acc_pixel)
            precision_pixel_.append(precision_pixel)
            recall_pixel_.append(recall_pixel)
            F1_pixel_.append(f1_pixel)
            iou_pixel_.append(IoU_pixel)
            # Performance metrics on pixel/ image
            accuracy_img_.append(acc_img)
            precision_img_.append(precision_img)
            recall_img_.append(recall_img)
            F1_img_.append(f1_img)
            iou_img_.append(IoU_img)
            
            # Calculate loss-----------------------------------------------------
            # Deep copy to make sure we don't change original data
            output = {
            'boxes': copy.deepcopy(prediction['boxes'].detach()),
            'masks': copy.deepcopy(prediction['masks'].detach()),
            'labels': copy.deepcopy(prediction['labels'].detach()),
            'scores': copy.deepcopy(prediction['scores'].detach())
            }

            # Output contains batch dimension (only 1 entry), which target does not -> get rid of it to match them for loss calculation: [n_RTS, batch, channel, h, w] to [n_RTS, channel, h, w]
            output["masks"] = output["masks"][:, 0, :, :]
            # Calculate loss-----------------------------------------------------
            # No RTS have been predicted, no RTS have been labelled: 0 loss
                                
            # TN have not been taken into account because too many empty images could skew result
            # If instead perfect TN should be taken into account: 
            '''
            classification_loss_value = 0
            box_regression_loss_value = 0
            mask_loss_value = 0
            iou_value = 1'''
            
            if labels["labels"].size()[0] == 0 and  output["labels"].size()[0] ==0:
                continue

            # RTS have been predicted but no RTS have been labelled: loss prediction not possible, skip to next batch
            elif labels["labels"].size()[0] > 0 and  output["labels"].size()[0] ==0:
                continue

            # RTS have been predicted and RTS have been labelled 
            else: 
                # To calculate loss, both inputs have to have same length and IoU!= 0. We transform data accordingly (Truncate the ones with smalles IoU)-----------------------------------------------------
                # intersection over union: result in matrix with predicted boxes (y axis)and labelled boxes (x axis)
                iou = box_iou(output["boxes"],labels["boxes"]).detach()

                # Truncate output or labels if number of predicted RTS does not equal to number of labelled RTS or where IoU=0-> otherwise losses are skewed because we cannot match the pairs
                # Calculate max number of RTS that overlap -> number of RTS we will keep
                min_l = min(labels["labels"].size()[0], output["labels"].size()[0])

                # extract indices of RTS with n-max IoU where n = min_l = min. number of labelled or predicted RTS 
                sorted_indices = torch.flip(torch.argsort(iou.flatten()), [0]) # sort in descending order
                
                indices_y = []
                indices_x = []
                iou_i = []
                for i in sorted_indices[:min_l]:
                    
                    idy, idx = unravel_index(i.item(), iou.shape)
                    if iou[idy, idx] <=0: # IoU is 0 -> we don't extract this RTS
                        break
                    else:
                        iou_i.append(iou[idy, idx])
                        indices_y.append(idy)
                        indices_x.append(idx)
                
                if len(indices_y) == 0: # IoU was always 0 -> no pair found-> no loss can be calculated
                    continue

                iou_value = torch.nanmean(torch.tensor(iou_i))

            # Append values
            iou_RTS_.append(iou_value)
  
            
        if len(iou_RTS_)==0: # no valid pairs: set to nan so that we don't return empty values
            iou_RTS = np.nan
        else:
            iou_RTS = np.nanmean(iou_RTS_) 
            
        self.log("validation_bbox_iou", iou_RTS, on_step=True, on_epoch=True, batch_size=batch_size)
  
        
        
        # Performance metrics on image level
        validation_accuracy = torch.nanmean(torch.stack(accuracy_)) 
        validation_precision = torch.nanmean(torch.stack(precision_))
        validation_recall = torch.nanmean(torch.stack(recall_)) 
        validation_F1 = torch.nanmean(torch.stack(F1_))
        self.log("RTS_validation_accuracy", validation_accuracy, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_validation_precision", validation_precision, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_validation_recall", validation_recall, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("RTS_validation_F1", validation_F1, on_step=True, on_epoch=True, batch_size=batch_size)

        
        # Performance metrics on RTS level
        accuracy_pixel = torch.nanmean(torch.tensor(accuracy_pixel_, dtype=torch.float))
        precision_pixel =torch.nanmean(torch.tensor(precision_pixel_, dtype=torch.float))
        recall_pixel = torch.nanmean(torch.tensor(recall_pixel_, dtype=torch.float))
        F1_pixel = torch.nanmean(torch.tensor(F1_pixel_, dtype=torch.float))
        IoU_pixel = torch.nanmean(torch.tensor(iou_pixel_, dtype=torch.float))
        self.log("Mask_accuracy_validation", accuracy_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_precision_validation", precision_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_recall_validation", recall_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Mask_F1_validation", F1_pixel, on_step=True, on_epoch=True, batch_size=batch_size)   
        self.log("Mask_IoU_validation", IoU_pixel, on_step=True, on_epoch=True, batch_size=batch_size)
           
        accuracy_img = torch.nanmean(torch.tensor(accuracy_img_, dtype=torch.float))
        precision_img =torch.nanmean(torch.tensor(precision_img_, dtype=torch.float))
        recall_img = torch.nanmean(torch.tensor(recall_img_, dtype=torch.float))
        F1_img = torch.nanmean(torch.tensor(F1_img_, dtype=torch.float))
        IoU_img = torch.nanmean(torch.tensor(iou_img_, dtype=torch.float))
        self.log("Image_accuracy_validation", accuracy_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_precision_validation", precision_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_recall_validation", recall_img, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("Image_F1_validation", F1_img, on_step=True, on_epoch=True, batch_size=batch_size)   
        self.log("Image_IoU_validation", IoU_img, on_step=True, on_epoch=True, batch_size=batch_size)
        return validation_accuracy
    
    
    def test_step(self, img_org, targ_org, preds, iou_thresholds= [.5], get_TP_ind = False):
        img = copy.deepcopy(img_org)
        targets = copy.deepcopy(targ_org)
        batch_size = len(img)

        # Calculate RTS detection metrics: RTS was detected based on IoU BBox >= 0.5--------------------------------------------------
        # Performance metrics on image level
        accuracy_ = []
        precision_ = []
        recall_ = []
        F1_ = []
        
        # Performance metrics on pixel per RTS level: If pixel of RTS instance was detected (only taking binary masks of detected RTS into account)
        accuracy_pixel_ = []
        precision_pixel_ = []
        recall_pixel_ = []
        F1_pixel_ = []
        iou_pixel_ = []
        
        # performance level of binary mask
        accuracy_img_ = []
        precision_img_ = []
        recall_img_ = []
        F1_img_ = []
        iou_img_ = []
        
        # Loss of 
        classification_loss_ = []
        box_regression_loss_ = []
        mask_loss_ = []
        #iou_tot = []
        iou_RTS_ = [] # mean iou that do not take false positives and false negatives into acount
        RTS_TP_ = []
        round = 0
        
        # Iterate over images in batch
        for labels, prediction in zip(targets,preds): # iterate trough tile / batch
            if get_TP_ind:
                acc_pixel, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, accuracy_RTS, precision_RTS, recall_RTS, F1_RTS, TP_ind, acc_img, precision_img, recall_img, f1_img, IoU_img = utils.similarity_RTS(prediction["masks"], labels["masks"], labels["boxes"],prediction["boxes"], iou_thresholds, get_TP_ind)
                RTS_TP_ = RTS_TP_ + TP_ind
            else:
                acc_pixel, precision_pixel, recall_pixel, f1_pixel, IoU_pixel, accuracy_RTS, precision_RTS, recall_RTS, F1_RTS, acc_img, precision_img, recall_img, f1_img, IoU_img = torch.Tensor(utils.similarity_RTS(prediction["masks"], labels["masks"],labels["boxes"],prediction["boxes"], iou_thresholds)).to(self.device_)
            # Performance metrics on image level
            accuracy_.append(accuracy_RTS)
            precision_.append(precision_RTS)
            recall_.append(recall_RTS)
            F1_.append(F1_RTS)
            
            # Performance metrics on RTS level
            accuracy_pixel_.append(acc_pixel)
            precision_pixel_.append(precision_pixel)
            recall_pixel_.append(recall_pixel)
            F1_pixel_.append(f1_pixel)
            iou_pixel_.append(IoU_pixel)
            
            accuracy_img_.append(acc_img)
            precision_img_.append(precision_img)
            recall_img_.append(recall_img)
            F1_img_.append(f1_img)
            iou_img_.append(IoU_img)
            
            # Calculate loss: classification, box regression, mask-----------------------------------------------------------
            iou_value = 0

            # Deep copy to make sure we don't change original data
            output = {
            'boxes': copy.deepcopy(prediction['boxes'].detach()),
            'masks': copy.deepcopy(prediction['masks'].detach()),
            'labels': copy.deepcopy(prediction['labels'].detach()),
            'scores': copy.deepcopy(prediction['scores'].detach())
            }

            # Output contains batch dimension (only 1 entry), which target does not -> get rid of it to match them for loss calculation: n_RTS, batch, channel, h, w
            output["masks"] = output["masks"][:, 0, :, :]

            # No RTS have been predicted, no RTS have been labelled: 0 loss
            if labels["labels"].size()[0] == 0 and  output["labels"].size()[0] ==0:
                    classification_loss_value = 0
                    box_regression_loss_value = 0
                    mask_loss_value = 0
                    iou_value = 1
            # RTS have been predicted but no RTS have been labelled: loss prediction not possible, skip to next batch
            elif labels["labels"].size()[0] > 0 and  output["labels"].size()[0] ==0:
                continue

            # RTS have been predicted and RTS have been labelled 
            else: 
                # Truncate prediction and labels to have same size-------------------------------------------------------
                # To calculate loss, predicted RTS have to be matched with labelled RTS (Otherwise skew of loss)
                # both inputs have to have same length and IoU!= 0. 
                # Truncate output or labels if number of predicted RTS does not equal to number of labelled RTS or where IoU=0
                
                # intersection over union: result in matrix with predicted boxes (y axis)and labelled boxes (x axis)
                iou = box_iou(output["boxes"],labels["boxes"]).detach().numpy()

                # Calculate max number of RTS that overlap -> number of RTS we will keep
                min_l = min(labels["labels"].size()[0], output["labels"].size()[0])

                # extract indices of RTS with n-max IoU where n = min_l = min. number of labelled or predicted RTS 
                sorted_indices = np.argsort(iou.flatten())[::-1] # sort in descending order
                indices_y = []
                indices_x = []
                iou_i = []
                for i in sorted_indices[:min_l]:
                    idy, idx = np.unravel_index(i, iou.shape)
                    if iou[idy, idx] <=0: # IoU is 0 -> we don't extract this RTS
                        break
                    else:
                        iou_i.append(iou[idy, idx])
                        indices_y.append(idy)
                        indices_x.append(idx)

                if len(indices_y) == 0: # IoU was always 0 -> no pair found-> no loss can be calculated
                    continue

                iou_value = np.nanmean(iou_i)
                # truncate predicted RTS 
                output["boxes"] = output["boxes"][indices_y]
                output["labels"] = output["labels"][indices_y]
                output["scores"] = output["scores"][indices_y]
                # Extract matrix dimension
                output["masks"] = output["masks"][[indices_y]]

                # truncate labelled RTS
                labels["labels"] = labels["labels"][indices_x]
                labels["boxes"] = labels["boxes"][indices_x]
                # Extract matrix dimension. First dimension is # predicted labels indices
                labels["masks"] = labels["masks"][[indices_x]]

                # Get losses-----------------------------------------------------
                # Extract values needed
                class_logits = output['scores'].detach()
                box_regression = output['boxes'].detach()
                mask_prediction = output['masks'].detach()

                # Transoform class_logits. 
                # class_logits: only logit of class 1 given -> We have to calculate logits for  background class               
         
                for obj in range(output['labels'].size()[0]):
                    background = 1 - class_logits[obj]
                    class_logits_i = torch.cat([torch.unsqueeze(class_logits[obj], dim=0), torch.unsqueeze(background, dim=0)], dim=0)
                    # Create first tensor
                    if obj == 0: 
                        class_logits_ = class_logits_i
                    elif obj ==1:
                        class_logits_ = torch.stack([class_logits_, class_logits_i], dim=0)
                    else:
                         class_logits_ = torch.cat([class_logits_, class_logits_i.unsqueeze(0)], dim=0)

                # class_logits: Add another dimension to fit crossentropyloss input dimension. Get rid of batch dimension in predicted RTS mask
                while class_logits_.dim() < 2:
                    class_logits_ = torch.unsqueeze(class_logits_, dim=0)

                # Calculate loss
                classification_loss_value = nn.functional.cross_entropy(class_logits_, labels['labels'])
                #box_regression_loss_value = box_regression_loss(box_regression, labels['boxes'])
                box_regression_loss_value = nn.functional.smooth_l1_loss(box_regression, labels['boxes'],beta=1 / 9, reduction="sum")
                mask_loss_value = nn.functional.binary_cross_entropy_with_logits(mask_prediction.squeeze(), labels['masks'].squeeze().float())

            # Performance metrics on image level
            classification_loss_.append(classification_loss_value)
            box_regression_loss_.append(box_regression_loss_value)
            mask_loss_.append(mask_loss_value)
            iou_RTS_.append(iou_value)
            round+=1


        # Average detection metrics
        # Performance metrics on image level
        test_accuracy = np.nanmean([value for value in accuracy_])
        test_precision = np.nanmean([value for value in precision_])
        test_recall = np.nanmean([value for value in recall_])
        test_F1 = np.nanmean([value for value in F1_])
        # Performance metrics on RTS level
        accuracy_pixel = np.nanmean(accuracy_pixel_)
        precision_pixel = np.nanmean(precision_pixel_)
        recall_pixel = np.nanmean(recall_pixel_)
        F1_pixel = np.nanmean(F1_pixel_)
        IoU_pixel = np.nanmean(iou_pixel_)
        # performance metric for binary image
        accuracy_img = np.nanmean(accuracy_img_)
        precision_img = np.nanmean(precision_img_)
        recall_img = np.nanmean(recall_img_)
        F1_img = np.nanmean(F1_img_)
        IoU_img = np.nanmean(iou_img_)
        
        if len(iou_RTS_)==0: # no valid pairs: set to nan so that we don't return empty values
            classification_loss = np.nan
            box_regression_loss = np.nan
            mask_loss = np.nan
            iou_RTS = np.nan
        else:
            # Average loss metrics
            classification_loss = torch.mean(torch.stack(classification_loss_))
            box_regression_loss = torch.mean(torch.stack(box_regression_loss_))
            mask_loss = torch.mean(torch.stack(mask_loss_))
            iou_RTS = np.nanmean(iou_RTS_)    
        if get_TP_ind:
            return test_accuracy, test_precision, test_recall, test_F1, classification_loss, box_regression_loss, mask_loss, iou_RTS, RTS_TP_, accuracy_pixel, precision_pixel, recall_pixel, F1_pixel, IoU_pixel, accuracy_img, precision_img, recall_img, F1_img, IoU_img
        else:
            return test_accuracy, test_precision, test_recall, test_F1, classification_loss, box_regression_loss, mask_loss, iou_RTS, accuracy_pixel, precision_pixel, recall_pixel, F1_pixel, IoU_pixel, accuracy_img, precision_img, recall_img, F1_img, IoU_img
       