import time
import torch
import datetime
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou
from collections import defaultdict, deque

def split_dataset_by_percentage(dataset, percentage):
    '''
    divide a dataset into a training set and a validation (or test) with random permutation
    Input:
        dataset: input dataset that you want to split.
        percentage: A float value between 0 and 1, indicating the percentage of samples that should be included in the first subset. 
    Output: train_data, test_data
    '''
    total_samples = len(dataset)
    num_samples = int(total_samples * percentage)
    indices = torch.randperm(total_samples).tolist()
    train_data = torch.utils.data.Subset(dataset, indices[:num_samples])
    test_data = torch.utils.data.Subset(dataset, indices[num_samples:])
    return train_data, test_data

def similarity_RTS(src_boxes, pred_boxes):
    iou_thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    total_gt = len(src_boxes)
    total_pred = len(pred_boxes)
    thrshs = torch.tensor(iou_thresholds)
    thrshs_mean = torch.mean(thrshs)
    
    def iou_acc(threshold):
      # matching pairs of bounding boxes depending on IoU threshold 
        matcher = Matcher(threshold,threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes) # computes the IoU values between all pairs of bounding boxes
        results = matcher(match_quality_matrix) 

        true_positive = torch.count_nonzero(results.unique() != -1) # number of matched bounding boxes that have a valid match
        matched_elements = results[results > -1]

        #in Matcher, a pred element can be matched only twice
        # false_positive = sum of unmatched predicted bounding boxes and predicted bounding boxes that have more than two matches
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(torch.unique(matched_elements)))
        false_negative = total_gt - true_positive

        acc = true_positive / ( true_positive + false_positive + false_negative ) # we don't have true negatives

        return acc
    
    # Return avg accuracy
    if total_gt > 0 and total_pred > 0:
        return torch.tensor(sum([iou_acc(t) for t in iou_thresholds]) / len(iou_thresholds))

    elif total_gt == 0:
        if total_pred > 0:
            return torch.tensor(0.)
        else:
            return torch.tensor(1.)
    elif total_gt > 0 and total_pred == 0:
        return torch.tensor(0.)
      
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