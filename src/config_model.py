import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

def get_config(mean_sd, model = "baseline"):
    '''
    Baseline: trained on 2016 data. Default 
    '''
    if model == "frozen_model":
        config_baseline = dict(
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "scaled_DEM_difference",
            lr=0.00001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 4, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "sgd", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = True, # freeze layer other than backbone. Following defines which ones
            freeze_fpn = True,
            freeze_rpn = True,
            freeze_roi = True)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 4 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        drop_out = False
        change_min_box_axis = False
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=0) 
        transformed_data = False
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
   
    
    if model == "baseline":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "scaled_DEM_difference",
            lr=0.00001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 4, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 4
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=5) 
        transformed_data = True
        change_min_box_axis = False
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
    
    if model == "default_drop_out":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "scaled_DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "adam", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = True,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        drop_out = True
        change_min_box_axis = False
        transformed_data = False
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
    
    
    if model == "random_weights":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "unscaled_DEM_difference",
            lr=0.0001, #0.0001 step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        drop_out = False
        change_min_box_axis = False
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        transformed_data = True
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
    
    if model == "baseline_cosinelr":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  regularization technique that discourages large weights 
            
            
            optimizer_config = "adam", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "cosinelr",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=3) 
        transformed_data = False
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data
       
    if model == "aspect_slope_default":
        config_baseline = dict(
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces. 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  regularization technique that discourages large weights 
            
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 4,
            channel_list = ["images", "slope", "x_aspect", "y_aspect"], 
            img_mean = [mean_sd['img_mean'][0],mean_sd['slope_mean'][0],mean_sd['x_mean'][0],mean_sd['y_mean'][0]],
            img_std = [mean_sd['img_sd'][0],mean_sd['slope_sd'][0],mean_sd['x_sd'][0],mean_sd['y_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
               
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing.
        drop_out = False
        transformed_data = False
        change_min_box_axis = False
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
    
    if model == "slope_default":
        config_baseline = dict(
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  regularization technique that discourages large weights 
            
            
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 2,
            channel_list = ["images", "slope"], 
            img_mean = [mean_sd['img_mean'][0],mean_sd['slope_mean'][0]],
            img_std = [mean_sd['img_sd'][0],mean_sd['slope_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
               
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        drop_out = False
        transformed_data = False
        change_min_box_axis=False
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis
    
    if model == "aspect_default":
        config_baseline = dict(
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  regularization technique that discourages large weights 
            
            
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 3,
            channel_list = ["images", "x_aspect", "y_aspect"], 
            img_mean = [mean_sd['img_mean'][0],mean_sd['x_mean'][0],mean_sd['y_mean'][0]],
            img_std = [mean_sd['img_sd'][0],mean_sd['x_sd'][0],mean_sd['y_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        drop_out = False
        transformed_data = False
        change_min_box_axis = False
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis 
    
    if model == "default_min_box_axis":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "scaled_DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 1,
            lrscheduler = "onplateau",
            adapt_nms = False,
            drop_out_apply = False,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        transformed_data = False
        change_min_box_axis = True
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis 
        
    
    if model == "default_non_maximum_supression":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "scaled_DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  L2 regularization (discourages large weights): loss = loss + weight decay parameter * L2 norm of the weights
            optimizer_config = "adamW", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],#, "aspect", "slope"], #, "aspect", "slope"], # "aspect, slope
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            min_box_axis = 0.01,
            lrscheduler = "onplateau",
            drop_out_apply = False,
            adapt_nms = True,
            post_nms_top_n = {'training': 700, 'testing': 250},
            rpn_nms_thresh = 0.5,
            box_nms_thresh = 0.25,
            freeze_layer = False)
                  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2 # background, RTS
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        change_min_box_axis = False
        transformed_data = False
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data, change_min_box_axis 
    
    if model == "transformed_data":
        config_baseline = dict (
            project = "RTS_detection",
            architecture = "maskrcnn_resnet50_fpn",
            dataset_id = "DEM_difference",
            lr=0.0001, # step size at which the model's weights are updated usually [0,1]
            min_lr=0.0000001, # minimum learning rate if learning rate gradually reduces 
            epochs=150, # number of times the model will process the entire training dataset
            batch_size= 5, # number of data samples processed together in each training iteration (forward and backward pass)
            nesterov=False, # enables Nesterov momentum
            momentum=0.9, # considering past gradients: set too high-> slow convergence /divergence. Set too low-> no significant improvements over standard
            weight_decay=0.001, #  regularization technique that discourages large weights 
            
            
            optimizer_config = "adam", # adam/ sgd
            betas=(0.9, 0.999), # Adam optimizer: coefficients used for computing running averages of gradient and its square 
            eps = 1e-8, #term added to the denominator to improve numerical stability 
            amsgrad = False, # whether to use the AMSGrad variant of this algorithm
            hidden_layer=256,
            n_channel = 1,
            channel_list = ["images"],
            img_mean = [mean_sd['img_mean'][0]],
            img_std = [mean_sd['img_sd'][0]],
            img_size = 256,
            adapt_nms = False,
            min_box_axis = 0.01,
            freeze_layer = False)
        transformed_data = True          
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 2
        worker_processes = 2 
        gpu=  torch.cuda.is_available() # for trainer definition
        gpu_device = torch.cuda.device_count() # for trainer definition
        threshold_mask = 0.5 # threshold for preformance metric during testing
        
        #trainable_backbone_layers: number of trainable layers starting from final block (ranging from 0 - 5)
        pretrained_network = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=3) 
        return config_baseline, device, num_classes, worker_processes, gpu, gpu_device, threshold_mask, pretrained_network, transformed_data
        
    else:
        print("not a valid model config")
