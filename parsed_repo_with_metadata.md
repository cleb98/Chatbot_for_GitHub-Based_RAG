# config.py

## imports

```python
import os
import re
import glob
import itertools
from click.core import batch
from path import Path
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from torch.fx.experimental.proxy_tensor import snapshot_fake
```

## global: ex

ex = Experiment('PANet')

## global: source_folders

source_folders = ['.', './dataloaders', './models', './util']

## global: sources_to_save

sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

## function: def cfg

def cfg():
    ################ SET TRAINING PARAMETERS or TEST PARAMETERS ################
    input_size = (417, 417)
    seed = 1234
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 0
    batch_size = 1
    n_ways = 1
    n_shots = 5
    n_queries = 1
    label_sets = 0 #'all' #see at utils.py, is CLASS_LABELS
    net = 'resnet50' # 'resnet50'
    mode = 'test' #'train' or 'test'
    demo = False
    dataset = 'COCO'  # 'VOC' or 'COCO'
    tensorboard_tag = 'official'
    ##############################################################################

    if net == 'vgg':
        init_path = './pretrained_model/vgg16-397923af.pth'
        if n_shots == 1:
            snapshot = './runs/PANet_COCO_align_sets_0_1way_1shot_vgg[train]/8/snapshots/30000.pth'
        elif n_shots == 5:
            snapshot = './runs/PANet_COCO_align_sets_0_1way_5shot_[train]/9/snapshots/30000.pth'

    elif net == 'resnet50':
        init_path = './pretrained_model/resnet50-19c8e357.pth'
        if n_shots == 1:
            snapshot = './runs/PANet_COCO_align_sets_0_1way_1shot_[train]_model_resnet50/3/snapshots/30000.pth'
        elif n_shots == 5:
            snapshot = './runs/PANet_COCO_sets_0_1way_5shot_[train]_model_resnet50/1/snapshots/30000.pth'
            # snapshot = './runs/PANet_prova_label_train/3/snapshots/30000.pth' #coco with al classes

    log_tensorboard = f'./runs/{mode}_{dataset}_{n_ways}way_{n_shots}shot_{n_queries}query_{net}_{tensorboard_tag}'
    events_folder = f'./runs/'

#____________________________________________________________________________________________#

    if mode == 'train':
        dataset = dataset
        n_steps = 30000
        label_sets = label_sets
        batch_size = batch_size
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 10000

        model = {
            'align': True,
            'net': net,  # 'vgg' or 'resnet50'
        }

        task = {
            'n_ways': n_ways,
            'n_shots': n_shots,
            'n_queries': n_queries,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        notrain = False
        snapshot = snapshot
        n_runs = 5
        n_steps = 1000
        batch_size = batch_size
        scribble_dilation = 0
        bbox = False
        scribble = False

        # Set dataset config from the snapshot string
        '''
        if 'VOC' in snapshot:
            dataset = 'VOC'
        elif 'COCO' in snapshot:
            dataset = 'COCO'
        else:
            raise ValueError('Wrong snapshot name !')
        '''
        dataset = dataset
        # Set model config from the snapshot string
        '''
        model = {'net': net}
        for key in ['align',]:
            model[key] = key in snapshot
        '''

        model = {
            'align': True,
            'net': net,  # 'vgg' or 'resnet50'
        }

        # Set label_sets from the snapshot string
        '''
        label_sets = int(snapshot.split('_sets_')[1][0]) #dovrebbero essere quelle usate in train che saranno tolte in test
        print(f"label_sets_test: {label_sets}")
        '''
        label_sets = label_sets
        # Set task config from the snapshot string
        # task = {
        #     'n_ways': int(re.search("[0-9]+way", snapshot).group(0)[:-3]),
        #     'n_shots': int(re.search("[0-9]+shot", snapshot).group(0)[:-4]),
        #     'n_queries': 1,
        # }
        task = {
            'n_ways': n_ways,
            'n_shots': n_shots,
            'n_queries': n_queries,
        }

    else:
        raise ValueError('Wrong configuration for "mode" !')

    print(f"Training mode: {mode}, Dataset: {dataset}, Batch_size: {batch_size}, Network: {net}, Task: {task}")


    exp_str = '_'.join(
        [dataset,]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]']
        + [f'model_{net}'])
    if demo:
        exp_str = f'demo_{exp_str}'

    # exp_str = 'prova_label_' + mode


    path = {
        'log_dir': './runs',
        'init_path': init_path,
        'VOC':{'data_dir': '../../data/Pascal/VOCdevkit/VOC2012/',
               'data_split': 'trainaug',},

        # 'COCO':{'data_dir': '../../data/COCO/',
        #         'data_split': 'train',},
        'COCO':{'data_dir': '/work/tesi_cbellucci/coco',
                'data_split': 'train',},

    }

## function: def add_observer

def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    if config['mode'] == 'test':
        if config['notrain']:
            exp_name += '_notrain'
        if config['scribble']:
            exp_name += '_scribble'
        if config['bbox']:
            exp_name += '_bbox'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config

# demo.py

# demo1shot.py

## imports

```python
import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.fewshot import FewShotSeg
from dataloaders.common import BaseDataset
from dataloaders.transforms import ToTensorNormalize, Resize
from util.utils import get_bbox
from util.visual_utils import apply_mask_overlay
from pycocotools.coco import COCO
from config import ex
import numpy as np
```

## function: def getMask

def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        class_id:
            semantic class of interest

        class_ids:
            all class id in this episode

    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,}

## class: class QueryDataset

class QueryDataset(Dataset):
    def __init__(self, image_paths, input_size):
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

## class: class SupportDataset

class SupportDataset(BaseDataset):
    def __init__(self, base_dir, annotation_file, input_size, to_tensor=None):
        super().__init__(base_dir)
        self.coco = COCO(annotation_file)
        self.ids = self.coco.getImgIds()
        t = [Resize(size=input_size)]
        self.transforms = Compose(t)
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_meta = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self._base_dir, img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in masks:
                masks[catId][mask == 1] = catId
            else:
                semantic_mask = torch.zeros((img_meta['height'], img_meta['width']), dtype=torch.uint8)
                semantic_mask[mask == 1] = catId
                masks[catId] = semantic_mask

        instance_mask = torch.zeros_like(semantic_mask)
        scribble_mask = torch.zeros_like(semantic_mask)

        sample = {'image': image, 'label': masks, 'inst': instance_mask, 'scribble': scribble_mask}

        #debugging
        print("Type of self.transforms:", type(self.transforms))
        print("Type of Resize:", type(Resize))
        # print("Applying Resize to sample:", sample)
        ###
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = img_id
        return sample

## function: def denormalize_tensor

def denormalize_tensor(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizza un'immagine (C, H, W) applicando l'operazione inversa della normalizzazione.
    Se necessario, l'immagine viene clipppata per garantire che i valori siano compresi tra [0, 1].
    """
    batch = False
    if len(img.shape) == 4:
        batch = True
        img = img.squeeze(0)
    mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
    std = torch.tensor(std).view(3, 1, 1).to(img.device)
    img = img * std + mean 
    img = torch.clamp(img, 0, 1)  
    if batch:
        img = img.unsqueeze(0)
    return img

## function: def plot_image

def plot_image(query_images: torch.Tensor, desc = " "):
    """
    Plots input images.

    :param query_images: Input images tensor of shape (B, 3, H, W) with values in [0, 1].
    """
    # Converti il tensore in un array NumPy
    np_images = query_images.detach().cpu().numpy()
    # Se il batch contiene più di una immagine, prendi la prima (oppure potresti iterare su tutte)
    img = np_images[0]  # Forma: (3, H, W)
    # Trasponi in modo da avere (H, W, 3) per plt.imshow()
    img = np.transpose(img, (1, 2, 0))
    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.set_title(desc)
    ax.axis("off")

    plt.show()

## function: def main

def main(_config):
    torch.cuda.set_device(_config['gpu_id'])
    torch.set_num_threads(1)

    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id']])
    model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    # input_folder = './demo/queries'
    # annotation_path = './demo/support/support_annotation.json'
    # support_base_dir = './demo/support/'


    files = './ripetuto' #spal #demo #ripetuto
    input_folder = os.path.join(files, 'queries')
    support_base_dir = os.path.join(files, 'support')
    annotation_path = os.path.join(support_base_dir, 'support_annotation.json')

    n_shot = _config['n_shots']
    n_ways = _config['n_ways']


    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    query_dataset = QueryDataset(image_paths, _config['input_size'])
    query_loader = DataLoader(query_dataset, batch_size=_config['task']['n_queries'], shuffle=False)

    support_dataset = SupportDataset(support_base_dir, annotation_path, _config['input_size'], to_tensor=ToTensorNormalize())
    support_loader = DataLoader(support_dataset, batch_size=_config['task']['n_shots'], shuffle=False)

    with torch.no_grad():
        support_samples = next(iter(support_loader))

        # Organizza le immagini di supporto nel formato
        # support images way x shot x [B x 3 x H x W], list of lists of tensors
        support_images = [
            [support_samples['image'][shot_idx].cuda().unsqueeze(0) for shot_idx in range(len(support_samples['image']))]
        ]

        # Converte le maschere in una lista ordinata e le organizza per shot (deve essere lista di liste di tensori)
        support_masks = [
            [list(support_samples['label'].values())[shot_idx].cuda() for shot_idx in range(n_shot)]
        ]

        class_ids = list(support_samples.get('label').keys())
        if len(class_ids) != n_ways:
            raise ValueError(f"Expected {n_ways} classes, but got {len(class_ids)}")

        mask_classes = [] # lista di dict contenente le foreground masks and background masks
        for i, class_id in enumerate(class_ids):
            for j, shot in enumerate(support_masks[i]):
                shot = getMask(shot, class_id, class_ids)
                mask_classes.append(shot) # list of dicts, each dict contains fg_mask and bg_mask for a class

        #foreground and background masks for support images
        support_fg_mask = [[shot.float().cuda() for shot in way['fg_mask']]
                           for way in mask_classes] #  way x shot x [B x H x W], list of lists of tensors
        support_bg_mask = [[shot.float().cuda() for shot in way['bg_mask']] # way x shot x [B x H x W], list of lists of tensors
                           for way in mask_classes]

        # print('mask:', support_fg_mask[0][0].shape)
        # print('mask:', support_bg_mask[0][0].shape)

        for i, query_images in enumerate(query_loader):
            query_images = query_images.cuda() #N x [B x 3 x H x W], tensors ( N is # of queries x batch)

            # Passa i supporti nel formato corretto al modello
            query_preds, _ = model(support_images, support_fg_mask, support_bg_mask, [query_images])
            query_preds = torch.where(query_preds > 0.75, torch.ones_like(query_preds), torch.zeros_like(query_preds))
            if type(query_images) == list:
                query_images = query_images[0]

            query_images = denormalize_tensor(query_images)
            query_preds = apply_mask_overlay(query_images, query_preds.argmax(dim=1))
            plot_image(query_preds, desc=f'Query Image {i+1}')

        for i, (images, masks) in enumerate(zip(support_images, support_masks)):
            for img, mask in zip(images, masks):
                img = denormalize_tensor(img)
                img = apply_mask_overlay(img, mask.squeeze(0))
                plot_image(img, desc=f"Support Image {i+1}")

    print("Inference completed!")

# demo_fewshot.py

## imports

```python
import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.fewshot import FewShotSeg
from dataloaders.common import BaseDataset
from dataloaders.transforms import ToTensorNormalize, Resize
from util.utils import get_bbox
from util.visual_utils import apply_mask_overlay
from pycocotools.coco import COCO
from config import ex
import numpy as np
```

## function: def getMask

def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        class_id:
            semantic class of interest

        class_ids:
            all class id in this episode

    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,}

## class: class QueryDataset

class QueryDataset(Dataset):
    def __init__(self, image_paths, input_size):
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

## class: class SupportDataset

class SupportDataset(BaseDataset):
    def __init__(self, base_dir, annotation_file, input_size, to_tensor=None):
        super().__init__(base_dir)
        self.coco = COCO(annotation_file)
        self.ids = self.coco.getImgIds()
        t = [Resize(size=input_size)]
        self.transforms = Compose(t)
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def getitem(self):
        pass

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_meta = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self._base_dir, img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in masks:
                masks[catId][mask == 1] = catId
            else:
                semantic_mask = torch.zeros((img_meta['height'], img_meta['width']), dtype=torch.uint8)
                semantic_mask[mask == 1] = catId
                masks[catId] = semantic_mask

        instance_mask = torch.zeros_like(semantic_mask)
        scribble_mask = torch.zeros_like(semantic_mask)

        sample = {'image': image, 'label': masks, 'inst': instance_mask, 'scribble': scribble_mask}

        #debugging
        print("Type of self.transforms:", type(self.transforms))
        print("Type of Resize:", type(Resize))
        # print("Applying Resize to sample:", sample)
        ###
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = img_id
        return sample

## function: def denormalize_tensor

def denormalize_tensor(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizza un'immagine (C, H, W) applicando l'operazione inversa della normalizzazione.
    Se necessario, l'immagine viene clipppata per garantire che i valori siano compresi tra [0, 1].
    """
    batch = False
    if len(img.shape) == 4:
        batch = True
        img = img.squeeze(0)
    mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
    std = torch.tensor(std).view(3, 1, 1).to(img.device)
    img = img * std + mean 
    img = torch.clamp(img, 0, 1)  
    if batch:
        img = img.unsqueeze(0)
    return img

## function: def plot_image

def plot_image(query_images: torch.Tensor, desc = " "):
    """
    Plots input images.

    :param query_images: Input images tensor of shape (B, 3, H, W) with values in [0, 1].
    """
    # Converti il tensore in un array NumPy
    np_images = query_images.detach().cpu().numpy()
    # Se il batch contiene più di una immagine, prendi la prima (oppure potresti iterare su tutte)
    img = np_images[0]  # Forma: (3, H, W)
    # Trasponi in modo da avere (H, W, 3) per plt.imshow()
    img = np.transpose(img, (1, 2, 0))
    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.set_title(desc)
    ax.axis("off")

    plt.show()

## function: def main

def main(_config):
    torch.cuda.set_device(_config['gpu_id'])
    torch.set_num_threads(1)

    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id']])
    print(f"Loading model from snapshot: {_config['snapshot']}")
    model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    files = './spal' #'./demo' #
    input_folder = os.path.join(files, 'queries')
    support_base_dir = os.path.join(files, 'support')


    annotation_path = os.path.join(support_base_dir, 'support_annotation.json')

    # net = _config['net']
    n_shot = _config['n_shots']
    # n_ways = _config['n_ways']

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    query_dataset = QueryDataset(image_paths, _config['input_size'])
    query_loader = DataLoader(query_dataset, batch_size=_config['task']['n_queries'], shuffle=False)

    support_dataset = SupportDataset(
        support_base_dir, # support_classes_dir,
        annotation_path,
        _config['input_size'],
        to_tensor=ToTensorNormalize()
    )
    support_loader = DataLoader(support_dataset, batch_size=_config['task']['n_shots'], shuffle=False)


    with torch.no_grad():

        support_samples = []
        support_images = []
        support_masks = []
        class_ids = []
        mask_classes = []
        for i, support_sample in enumerate(support_loader):
            if i == n_shot:
                break
            support_samples.append(support_sample)
            support_images.append(support_sample['image'].cuda())
            support_masks.append(list(support_sample['label'].values()))
            # class_ids.append(list(support_sample.get('label').keys())[i])
            # if len(class_ids[i]) != n_ways:
            #     raise ValueError(f"Expected {n_ways} classes, but got {len(class_ids)}")

        for mask in support_masks: #è lista di maschere per ogni support sample li devo mettere su device i tensori
            for i, m in enumerate(mask):
                mask[i] = m.cuda()

        # lista contenente liste di classi per ogni support sample
        class_ids = [list(support_sample['label'].keys()) for i, support_sample in enumerate(support_samples)]
        for i, class_id in enumerate(class_ids):
            for shot in support_masks[i]:
                masks = []
                for j in range(len(class_id)):
                    mask = getMask(shot, class_id[j], class_id)
                    masks.append(mask)
                mask_classes.append(masks)

        support_fg_masks = []
        support_bg_masks = []
        for shot in mask_classes:
            support_fg_mask = []
            support_bg_mask = []
            for way in shot:
                support_fg_masks.append(way['fg_mask'].squeeze(1))
                support_bg_masks.append(way['bg_mask'].squeeze(1))
            # support_fg_masks.append(support_fg_mask)
            # support_bg_masks.append(support_bg_mask)


        # print('mask:', support_fg_mask[0].shape)
        # print('mask:', support_bg_mask[0].shape)

        support_images = [
            list(support_image.split(1, dim=0)) for support_image in support_images
        ]

        for i in range(len(support_images)):
            support_masks[i] = list(support_masks[i][0].split(1, dim=0))

        support_fg_masks = [
            list(support_fg_mask.split(1, dim=0)) for support_fg_mask in support_fg_masks
        ]

        support_bg_masks = [
            list(support_bg_mask.split(1, dim=0)) for support_bg_mask in support_bg_masks
        ]
        # support_fg_masks = list(support_fg_masks.split(1, dim=0))
        # support_bg_masks = list(support_bg_masks.split(1, dim=0))

        for i, query_images in enumerate(query_loader):
            query_images = query_images.cuda() #N x [B x 3 x H x W], tensors ( N is # of queries x batch)

            # Passa i supporti nel formato corretto al modello
            query_preds, _ = model(support_images, support_fg_masks, support_bg_masks, [query_images])
            query_preds = torch.where(query_preds > 0.5, torch.ones_like(query_preds), torch.zeros_like(query_preds))
            if type(query_images) == list:
                query_images = query_images[0]

            query_images = denormalize_tensor(query_images)
            query_preds = apply_mask_overlay(query_images, query_preds.argmax(dim=1))
            plot_image(query_preds, desc=f'Query Image {i+1}')

        # for images, masks in zip(support_images, support_masks):
        #     for j, (img, mask) in enumerate(zip(images, masks)):
        #         img = denormalize_tensor(img)
        #         img = apply_mask_overlay(img, mask.squeeze(0))
        #         plot_image(img, desc=f"Support Image {j+1}")


    print("Inference completed!")

# README.md

## block 0

# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

## block 1

This repo contains code for our ICCV 2019 paper [PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](https://arxiv.org/abs/1908.06391).

## block 4

* Python 3.6 +
* PyTorch 1.0.1
* torchvision 0.2.1
* NumPy, SciPy, PIL
* pycocotools
* sacred 0.7.5
* tqdm 4.32.2

## block 6

### Data Preparation for VOC Dataset

## block 7

1. Download `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and put them under `VOCdevkit/VOC2012`.

## block 8

2. Download `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and use it to replace `VOCdevkit/VOC2012/ImageSets/Segmentation`.

## block 12

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `PANet/pretrained_model` folder.

## block 13

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. You can use `sacred` features, e.g. `python train.py with gpu_id=2`.

## block 15

### Citation
Please consider citing our paper if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
title = {PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

# test.py

## imports

```python
import os
import shutil
import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from util.visual_utils import decode_and_apply_mask_overlay, apply_mask_overlay
import matplotlib.pyplot as plt
from config import ex
```

## function: def plot_images

def plot_images(np_query_images, np_query_pred):
    #type: (torch.Tensor, torch.Tensor) -> None
    """
    Plot input images and predicted masks overlayed on them.
    :param np_query_images: Input images to plot (B, 3, H, W), values in range [0, 1], type: torch.Tensor
    :param np_query_pred: Predicted masks to overlay on images (B, H, W), values in range [0, 1], type: torch.Tensor
    """
    np_query_images = np_query_images.detach().cpu().numpy()
    np_query_pred = np_query_pred.detach().cpu().numpy()
    # save plot of images with matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].imshow(np_query_images)
    ax[1].imshow(np_query_pred)
    ax[0].title.set_text('Input Image')
    ax[1].title.set_text('Predicted Mask')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    # plt.savefig(f'output_{run}.png')
    plt.show()

## function: def main

def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    print('test_classes: ', labels)

    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)

    # init logging stuffs for tensorboard
    log_path = _config['log_tensorboard']
    event_path = _config['events_folder']
    _log.info(f'tensorboard --logdir={event_path}\n')
    sw = SummaryWriter(log_path)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['n_steps'] * _config['batch_size'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries']
            )
            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader (dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")

            it = 0 #to count the number of iterations
            for sample_batched in tqdm.tqdm(testloader):
                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                suffix = 'scribble' if _config['scribble'] else 'mask'

                if _config['bbox']:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                #visualization
                if type(query_images) == list:
                    query_images = query_images[0]

                inp = apply_mask_overlay(query_images, query_labels)
                out = apply_mask_overlay(query_images, query_pred.argmax(dim=1))
                grid = torch.cat([inp, out], dim=0)
                grid = tv.utils.make_grid(
                    grid, normalize=True, value_range=(0, 1),
                    nrow=grid.size(0) ##check this se mette le img su una riga o su due
                )
                sw.add_image(
                    tag=f'results_{it}',
                    img_tensor=grid, global_step=run
                )
                it += 1
                # #overlay mask on images, convert to numpy and permute to get a plot of the images
                # pred = query_pred.argmax(dim=1)
                # np_query_pred = apply_mask_overlay(query_images, pred).squeeze().permute(1, 2, 0)
                # np_query_images = apply_mask_overlay(query_images, query_labels).squeeze().permute(1, 2, 0)
                # plot_images(np_query_images, np_query_pred)
                #INTEGRA TENSORBOARD MUPOVITI!!!!
                #visualization end
                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')

# train.py

## imports

```python
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from tqdm import tqdm  # Progress bar
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from util.visual_utils import decode_and_apply_mask_overlay
from config import ex
```

## function: def main

def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')

    labels = CLASS_LABELS[data_name][_config['label_sets']]
    print('classes: ', labels)

    transforms = Compose([Resize(size=_config['input_size']), RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    _log.info('###### Training Start ######')
    log_loss = {'loss': 0, 'align_loss': 0}
    total_batches = len(trainloader)

    start_time = time.time()

    for i_iter, sample_batched in enumerate(tqdm(trainloader, desc="Training Progress", total=total_batches)):
        batch_start_time = time.time()

        # Prepare input
        support_images = [[shot.cuda() for shot in way] for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
        query_loss = criterion(query_pred, query_labels) #

        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss


        # Monitor training time and memory usage
        batch_time = time.time() - batch_start_time
        gpu_mem = torch.cuda.memory_allocated(_config['gpu_id']) / (1024 ** 2)  # MB

        # Print logs at interval
        if (i_iter + 1) % _config['print_interval'] == 0:
            avg_loss = log_loss['loss'] / (i_iter + 1)
            avg_align_loss = log_loss['align_loss'] / (i_iter + 1)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (i_iter + 1)) * total_batches
            remaining_time = estimated_total_time - elapsed_time

            print(f"[Iter {i_iter+1}/{total_batches}] "
                  f"Loss: {avg_loss:.4f}, Align Loss: {avg_align_loss:.4f} "
                  f"| Time per batch: {batch_time:.2f}s "
                  f"| GPU Mem: {gpu_mem:.2f} MB "
                  f"| Remaining time: {remaining_time/60:.2f} min")
        # Save model periodically
        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
    _log.info(f"Training Completed in {(time.time() - start_time) / 60:.2f} minutes")

# dataloaders\coco.py

## imports

```python
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from dataloaders.common import BaseDataset
    import numpy as np
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
```

## class: class COCOSeg

class COCOSeg(BaseDataset):
    """
    Modified Class for COCO Dataset

    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use (default is 2014 version)
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        annFile = f'{base_dir}/annotations/instances_{self.split+'2017'}.json'
        self.coco = COCO(annFile)

        self.ids = self.coco.getImgIds()
        self.transforms = transforms
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch meta data
        id_ = self.ids[idx]
        img_meta = self.coco.loadImgs(id_)[0]
        annIds = self.coco.getAnnIds(imgIds=img_meta['id'])

        # Open Image
        image = Image.open(f"{self._base_dir}/images/{self.split}/{img_meta['file_name']}")
        if image.mode == 'L':
            image = image.convert('RGB')

        # Process masks
        anns = self.coco.loadAnns(annIds)
        semantic_masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in semantic_masks:
                semantic_masks[catId][mask == 1] = catId
            else:
                semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                semantic_mask[mask == 1] = catId
                semantic_masks[catId] = semantic_mask
        semantic_masks = {catId: Image.fromarray(semantic_mask)
                          for catId, semantic_mask in semantic_masks.items()}

        # No scribble/instance mask
        instance_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))
        scribble_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))

        sample = {'image': image,
                  'label': semantic_masks,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without mean subtraction/normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

## function: def main

def main():
    import numpy as np
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    # Initialize dataset
    ds = COCOSeg(
        base_dir='/work/tesi_cbellucci/coco',
        split='val',
        transforms=None,
        to_tensor=ToTensor()  # Add basic ToTensor conversion
    )

    # Create DataLoader
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True

    )

    # Visualization parameters
    num_batches_to_show = 2
    images_per_batch = 4

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches_to_show:
            break

        plt.figure(figsize=(20, 10))

        for sample_idx, sample in enumerate(batch[:images_per_batch]):
            # Get image and masks
            img = sample['image_t'].permute(1, 2, 0).numpy().astype(np.uint8)

            # Combine all semantic masks
            mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
            for cat_id, cat_mask in sample['label'].items():
                mask = np.maximum(mask, np.array(cat_mask))

            # Create subplots
            # Image
            plt.subplot(2, images_per_batch, sample_idx + 1)
            plt.imshow(img)
            plt.title(f"Image {sample_idx + 1}\n{sample['id']}")
            plt.axis('off')

            # Mask
            plt.subplot(2, images_per_batch, sample_idx + 1 + images_per_batch)
            plt.imshow(mask, cmap='jet', vmin=0, vmax=90)  # COCO has 80 classes
            plt.title(f"Semantic Mask {sample_idx + 1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

## if: __main__

if __name__ == '__main__':
    main()

# dataloaders\common.py

## imports

```python
import random
from torch.utils.data import Dataset
                from each dataset respectively.
```

## class: class BaseDataset

class BaseDataset(Dataset):
    """
    Base Dataset

    Args:
        base_dir:
            dataset directory
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self.aux_attrib = {}
        self.aux_attrib_args = {}
        self.ids = []  # must be overloaded in subclass

    def add_attrib(self, key, func, func_args):
        """
        Add attribute to the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        if key in self.aux_attrib:
            raise KeyError("Attribute '{0}' already exists, please use 'set_attrib'.".format(key))
        else:
            self.set_attrib(key, func, func_args)

    def set_attrib(self, key, func, func_args):
        """
        Set attribute in the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        self.aux_attrib[key] = func
        self.aux_attrib_args[key] = func_args

    def del_attrib(self, key):
        """
        Remove attribute in the data sample dict

        Args:
            key:
                key in the data sample dict
        """
        self.aux_attrib.pop(key)
        self.aux_attrib_args.pop(key)

    def subsets(self, sub_ids, sub_args_lst=None):
        """
        Create subsets by ids

        Args:
            sub_ids:
                a sequence of sequences, each sequence contains data ids for one subset
            sub_args_lst:
                a list of args for some subset-specific auxiliary attribute function
        """

        indices = [[self.ids.index(id_) for id_ in ids] for ids in sub_ids]
        if sub_args_lst is not None:
            subsets = [Subset(dataset=self, indices=index, sub_attrib_args=args)
                       for index, args in zip(indices, sub_args_lst)]
        else:
            subsets = [Subset(dataset=self, indices=index) for index in indices]
        return subsets

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

## class: class PairedDataset

class PairedDataset(Dataset):
    """
    Make pairs of data from dataset

    When 'same=True',
        a pair contains data from same datasets,
        and the choice of datasets for each pair is random.
        e.g. [[ds1_3, ds1_2], [ds3_1, ds3_2], [ds2_1, ds2_2], ...]
    When 'same=False',
            a pair contains data from different datasets,
            if 'n_elements' <= # of datasets, then we randomly choose a subset of datasets,
                then randomly choose a sample from each dataset in the subset
                e.g. [[ds1_3, ds2_1, ds3_1], [ds4_1, ds2_3, ds3_2], ...]
            if 'n_element' is a list of int, say [C_1, C_2, C_3, ..., C_k], we first
                randomly choose k(k < # of datasets) datasets, then draw C_1, C_2, ..., C_k samples
                from each dataset respectively.
                Note the total number of elements will be (C_1 + C_2 + ... + C_k).

    Args:
        datasets:
            source datasets, expect a list of Dataset
        n_elements:
            number of elements in a pair
        max_iters:
            number of pairs to be sampled
        same:
            whether data samples in a pair are from the same dataset or not,
            see a detailed explanation above.
        pair_based_transforms:
            some transformation performed on a pair basis, expect a list of functions,
            each function takes a pair sample and return a transformed one.
    """
    def __init__(self, datasets, n_elements, max_iters, same=True,
                 pair_based_transforms=None):
        super().__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.n_elements = n_elements
        self.max_iters = max_iters
        self.pair_based_transforms = pair_based_transforms
        if same:
            if isinstance(self.n_elements, int):
                datasets_indices = [random.randrange(self.n_datasets)
                                    for _ in range(self.max_iters)]
                self.indices = [[(dataset_idx, data_idx)
                                 for data_idx in random.choices(range(self.n_data[dataset_idx]),
                                                                k=self.n_elements)]
                                for dataset_idx in datasets_indices]
            else:
                raise ValueError("When 'same=true', 'n_element' should be an integer.")
        else:
            if isinstance(self.n_elements, list):
                self.indices = [[(dataset_idx, data_idx)
                                 for i, dataset_idx in enumerate(
                                     random.sample(range(self.n_datasets), k=len(self.n_elements)))
                                 for data_idx in random.sample(range(self.n_data[dataset_idx]),
                                                               k=self.n_elements[i])]
                                for i_iter in range(self.max_iters)]
            elif self.n_elements > self.n_datasets:
                raise ValueError("When 'same=False', 'n_element' should be no more than n_datasets")
            else:
                self.indices = [[(dataset_idx, random.randrange(self.n_data[dataset_idx]))
                                 for dataset_idx in random.sample(range(self.n_datasets),
                                                                  k=n_elements)]
                                for i in range(max_iters)]

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.indices[idx]]
        if self.pair_based_transforms is not None:
            for transform, args in self.pair_based_transforms:
                sample = transform(sample, **args)
        return sample

## class: class Subset

class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset:
            The whole Dataset
        indices:
            Indices in the whole set selected for subset
        sub_attrib_args:
            Subset-specific arguments for attribute functions, expected a dict
    """
    def __init__(self, dataset, indices, sub_attrib_args=None):
        self.dataset = dataset
        self.indices = indices
        self.sub_attrib_args = sub_attrib_args

    def __getitem__(self, idx):
        if self.sub_attrib_args is not None:
            for key in self.sub_attrib_args:
                # Make sure the dataset already has the corresponding attributes
                # Here we only make the arguments subset dependent
                #   (i.e. pass different arguments for each subset)
                self.dataset.aux_attrib_args[key].update(self.sub_attrib_args[key])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# dataloaders\customized.py

## imports

```python
import os
import random
import torch
import numpy as np
from .pascal import VOC
from .coco import COCOSeg
from .common import PairedDataset
```

## function: def attrib_basic

def attrib_basic(_sample, class_id):
    """
    Add basic attribute

    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}

## function: def getMask

def getMask(label, scribble, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    # Scribble Mask
    bg_scribble = scribble == 0
    fg_scribble = torch.where((fg_mask == 1)
                              & (scribble != 0)
                              & (scribble != 255),
                              scribble, torch.zeros_like(fg_mask))
    scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
    if scribble_cls_list:  # Still need investigation
        fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
    else:
        fg_scribble[:] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'fg_scribble': fg_scribble.long(),
            'bg_scribble': bg_scribble.long()}

## function: def fewShot

def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)]

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
                          for i in range(n_ways)]
    support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)]
                         for i in range(n_ways)]
    support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)]
                     for i in range(n_ways)]



    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]
    if coco:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
                        for j in range(cnt_query[i])]
    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels]


    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot],
                             class_ids[way], class_ids)
                     for shot in range(n_shots)] for way in range(n_ways)]


    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)


    return {'class_ids': class_ids,

            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts,

            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }

## function: def voc_fewshot

def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1):
    """
    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            object class labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())
    # Create sub-datasets and add class_id attribute
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query})])
    return paired_data

## function: def coco_fewshot

def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                 n_queries=1):
    """
    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]
    # Create sub-datasets and add class_id attribute
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]

    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True})])
    return paired_data

# dataloaders\pascal.py

## imports

```python
import os
import numpy as np
from PIL import Image
import torch
from .common import BaseDataset
```

## class: class VOC

class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')
        self._id_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.transforms = transforms
        self.to_tensor = to_tensor

        with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        scribble_mask = Image.open(os.path.join(self._scribble_dir, f'{id_}.png'))
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

# dataloaders\transforms.py

## imports

```python
import random
from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F
```

## class: class RandomMirror

class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

## class: class Resize

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.resize(img, self.size, interpolation=Image.BILINEAR)
        # if isinstance(label, dict):
        #     label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
        #              for catId, x in label.items()}
        # else:
        #     label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        if isinstance(label, dict):
            label = {
                catId: self._resize_tensor(x, interpolation=Image.NEAREST)
                for catId, x in label.items()
            }
        else:
            label = self._resize_tensor(label,
                                        interpolation=Image.NEAREST)
        # inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        # scribble = tr_F.resize(scribble, self.size, interpolation=Image.LANCZOS) #prima c'era anti-aliasing
            # For instance and scribble masks (they are tensors)
        inst = self._resize_tensor(inst,
                                   interpolation=Image.NEAREST)
        scribble = self._resize_tensor(scribble,
                                       interpolation=Image.BILINEAR)  # if "lanczos" causes issues, try "bilinear"

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

    def _resize_tensor(self, x, interpolation):
        # If x is a 2D tensor (mask), add a channel dimension.
        if torch.is_tensor(x) and x.dim() == 2:
            x = x.unsqueeze(0)  # from (H, W) to (1, H, W)
            x = tr_F.resize(x, self.size, interpolation=interpolation)
            # x = x.squeeze(0)    # back to (H, W)
        else:
            x = tr_F.resize(x, self.size, interpolation=interpolation)
        return x

## class: class DilateScribble

class DilateScribble(object):
    """
    Dilate the scribble mask

    Args:
        size: window width
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        scribble = sample['scribble']
        dilated_scribble = Image.fromarray(
            ndimage.minimum_filter(np.array(scribble), size=self.size))
        dilated_scribble.putpalette(scribble.getpalette())

        sample['scribble'] = dilated_scribble
        return sample

## class: class ToTensorNormalize

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
        if inst is not None :
            inst = torch.Tensor(np.array(inst)).long()
        if scribble is not None:
            scribble = torch.Tensor(np.array(scribble)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

# dataloaders\__init__.py

# models\base.py

## imports

```python
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
import torch
from path import Path
from torch import nn
```

## global: ConfDict

ConfDict = Dict[str, Any]

## class: class BaseModel

class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        # type: () -> None
        super().__init__()
        self.cnf_dict = None  # type: Optional[ConfDict]


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        :param x: input tensor
        """
        ...


    @property
    def n_param(self):
        # type: () -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def device(self):
        # type: () -> str
        """
        Check the device on which the model is currently located.

        :return: string that represents the device on which the model
            is currently located
            ->> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        Check if the model is on a CUDA device.

        :return: `True` if the model is on CUDA; `False` otherwise
        """
        return 'cuda' in self.device


    def save_w(self, path, cnf=None):
        # type: (Union[str, Path], Optional[ConfDict]) -> None
        """
        Save model weights to the specified path.

        :param path: path of the weights file to be saved.
        :param cnf: configuration dictionary (optional) to be saved
            along with the weights.
        """
        torch.save({'state_dict': self.state_dict(), 'cnf': cnf}, path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        Load model weights from the specified path. It also loads the
        configuration dictionary (if available).

        :param path: path of the weights file to be loaded.
        """
        d = torch.load(path, map_location=torch.device(self.device), weights_only=False)
        self.load_state_dict(d['state_dict'])
        self.cnf_dict = d['cnf']
        if d['cnf'] is not None:
            device = d['cnf'].get('device', self.device)
            '''
            il device di cnf è sempre cuda, 
            ma io voglio potrlo eseguire in locale
            aggiungo il seguente controllo
            '''
            if device != self.device:
                d['cnf']['device'] = self.device
                self.to(self.device)
            else:
                self.to(device)


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        Set the `requires_grad` attribute of all model parameters to `flag`.

        :param flag: True if the model requires gradient, False otherwise.
        """
        for p in self.parameters():
            p.requires_grad = flag

# models\decoder.py

# models\fewshot.py

## imports

```python
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import Encoder
from .ResNet import resnet50
```

## class: class FewShotSeg

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        if self.config['net'] == 'vgg':
            # Encoder
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', Encoder(in_channels, self.pretrained_path)),]))

        elif self.config['net'] == 'resnet50':
            self.encoder = resnet50(pretrained_path=self.pretrained_path, num_classes=None)


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] #[B * n_ways * n_shot, 3, H, W] join the support to the batch axes
                                + [torch.cat(qry_imgs, dim=0),], dim=0)      # [B * n_queries, 3, H, W], join tutte le query lungo la dimensione del batch
                                                                             #img concat: [B*(n_ways * n_shot + n_queries), 3, H, W]
        img_fts = self.encoder(imgs_concat) #[B*(n_ways * n_shot + n_queries), C', H', W']
        fts_size = img_fts.shape[-2:] #store (H', W')

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            pred = nn.Softmax(dim=1)(pred)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss

# models\ResNet.py

## imports

```python
from typing import Any, Callable, List, Optional, Type, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
from torchsummary import summary
from .base import BaseModel
```

## function: def conv3x3

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    # type: (int, int, int, int, int) -> nn.Conv2d
    """
    3x3 convolution with 3x3 kernel.
    Provide also groups and dilation parameters
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding = dilation,
        groups = groups,
        bias = False,
        dilation = dilation,
    )

## function: def conv1x1

def conv1x1(in_channels, out_channels, stride=1):
    # type: (int, int, int) -> nn.Conv2d
    """
    1x1 convolution with 1x1 kernel
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

## class: class Bottleneck

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # The output is Y = F(x) + x, where F(x) is the residual mapping to be learned.
    # The block that learn F(x) is composed by 3 convolutions (plus batchnorm and relu after each convolution):
    # 1x1 conv -> 3x3 conv -> 1x1 conv
    # 1. The first 1x1 convolution is used to reduce the number of channels:
    # 2. The 3x3 convolution is used to extract features
    # 3. The last 1x1 convolution is used to expand the number of channels
    # conv1x1, batchnorm -> no relu
    # 4. The block is completed by the residual connection:
    # -> Y = F(x) + x
    # -> if x and F(x) have different number of channels, x is downsampled to match the number of channels of F(x)
    # 5. The block is completed by the activation function if arg use_relu is True
    # -> Output = relu(Y)

    expansion = 4 #expand the number of channels

    def __init__(
            self,
            in_channels: int, #i dont like to call planes the number of channels
            channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module]=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            use_relu: bool = True
    )-> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # the number of channels in the convolutions if groups > 1
        out_channels = int(channels * (base_width / 64.)) * groups

        #half the number of channels in the first 1x1 convolution
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        #continue with 3x3 convolution to extract features
        self.conv2 = conv3x3(out_channels, out_channels, stride, groups, dilation)
        self.bn2 = norm_layer(out_channels)
        #expand the number of channels in the last 1x1 convolution
        self.conv3 = conv1x1(out_channels, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)

        #activation function
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_relu = use_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if the number of channels of the input is different from the output need to be downsampled
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity #residual connection
        if self.use_relu:
            out = self.relu(out)
        return out

## class: class ResNetV1_5

class ResNetV1_5(BaseModel):
    """
    ResNetV1.5 model
    """
    def __init__(
            self,
            block: Type[Bottleneck],
            layers: List[int],
            num_classes: int = None,
            pretrained_path: Optional[str] = None,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetV1_5, self).__init__()
        self.pretrained_path = pretrained_path
        self.zero_init_residual = zero_init_residual
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, 2, 2] #it uses dilation instead of stride to increase the receptive field
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        #input_layer: conv7x7, BN, ReLU, MaxPool
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Bottleneck blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if num_classes is not None:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_relu=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        else: #if num_classes is None, resnet is used as encoder for feature extraction
            print("ResNet used as encoder")
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_relu=False)


        self._init_weights()


    def _make_layer(self, block, channels, blocks, stride=1, dilate=False, last_relu=True):
        # type: (type[Bottleneck], int, int, int, bool, bool) -> nn.Sequential

        """
        Make a layer of blocks, each block is a Bottleneck block.
        The function:
        1. creates the downsampling layer (if stride > 1 or dilate is True) to match the number of channels of the input and the output
        2. if is necessary to increase the receptive field and maintain the spatial resolution, use dilation instead of stride
        3. create the layer: Stack the first block (which can modify the dimensions) and then the other blocks that maintain the configuration.
        Args:
            block:
                block to be used (e.g., Bottleneck block)
            channels:
                number of channels of the block (e.g., 64)
            blocks:
                number of blocks (to define the depth of the network)
            stride:
                stride of the first convolution
            dilate:
                if True, apply dilation to the blocks to increase the receptive field
            last_relu:
                last_relu, if True, apply relu to the output (default is True).
                Typically, the last layer of a resnet used as a feature extractor does not have the last relu to extract embeddings.

        Returns:
            nn.Sequential


        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        #downsampling layer to match the number of channels of the input and the output
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )

        layers = []
        #first block
        layers.append(
            block(self.in_channels, channels, stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        #after the first Bottleneck, the number of channels for the next blocks is expanded by the block.expansion factor
        self.in_channels = channels * block.expansion #update the number of channels for the next blocks
        for i in range(1, blocks):
            if i < blocks - 1:
                use_relu = True
            else:
                use_relu = last_relu
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_relu=use_relu)
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        #initialize the weights of the model, following the initialization of the weights of the original ResNet.
        #In addition, load the weights of the model if a pretrained model is provided.

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

                # uncomment if you want to zero-initialize the weights of the BasicBlock
                # actually BasicBlock not used in this model
                # elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                #     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        if self.pretrained_path is not None:
            pretrained_dict = torch.load(self.pretrained_path, map_location='cpu')
            #get the model state dictionary, aka the weights of the model
            model_dict = self.state_dict()
            #filter out unnecessary keys
            #when the resnet is used in encoder mode, it doesn't have the 'fc' layer so need to filter it out
            compatible_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            #update the model state dictionary and load it
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict)

    def _encoder_forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        features = self._encoder_forward(x)
        if self.num_classes is not None:
            x = self.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        return features

## function: def resnet50

def resnet50(pretrained_path=None, num_classes=None):
    # type: (Optional[str], Optional[int]) -> ResNetV1_5
    """
    ResNet-50 model:
    number of block for each of the 4 layers: [3, 4, 6, 3]
    """
    return ResNetV1_5(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained_path=pretrained_path)

## function: def resnet101

def resnet101(pretrained_path=None, num_classes=None):
    # type: (Optional[str], Optional[int]) -> ResNetV1_5
    """
    ResNet-101 model:
    number of block for each of the 4 layers: [3, 4, 23, 3]
    """
    return ResNetV1_5(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, pretrained_path=pretrained_path)

## if: __main__

if __name__ == '__main__':

    pretrained_path = "../pretrained_model/resnet50-19c8e357.pth"
    model50 = resnet50(num_classes=None, pretrained_path=pretrained_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model50.to(device)
    x = torch.randn(1, 3, 417, 417).to(device)
    out = model50(x)
    print("Output ResNet50:", out.shape, "\n")
    # Assicurati che l'input_size corrisponda alle dimensioni attese dal tuo modello
    summary(model50, (3, 417, 417), device=str(device))

# models\resnet_paper.py

## imports

```python
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
```

## function: def conv3x3

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

## function: def conv1x1

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

## class: class BasicBlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

## class: class Bottleneck

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, NL=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.nolinear = NL

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.nolinear:
            out = self.relu(out)

        return out

## class: class ResNet

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, 2, 2]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], lastRelu=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, lastRelu=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for b_id in range(1, blocks):

            NL = True
            if not lastRelu and b_id == blocks-1:
                NL= False

            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, NL=NL))

        return nn.Sequential(*layers)

## class: class ResNetSemShare4

class ResNetSemShare4(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSemShare4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, 2, 2]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], lastRelu=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_fewshot = self.layer4(x)
        feat_semantic = F.relu(feat_fewshot)

        return feat_fewshot, feat_semantic


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, lastRelu=True, sem=None):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for b_id in range(1, blocks):

            NL = True
            if not lastRelu and b_id == blocks-1:
                NL= False
            # if lastRelu:
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, NL=NL))

        return nn.Sequential(*layers)

## function: def resnet50

def resnet50(cfg=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3])
    init_path = './resnet50-19c8e357.pth'
    print(f'load: {init_path}')

    pretrained_weight = torch.load(init_path, map_location='cpu')
    model_weight = model.state_dict()

    pretrained_dict = {}
    for weight in model_weight:
        if weight in pretrained_weight:
            pretrained_dict[weight] = pretrained_weight[weight]

    model_weight.update(pretrained_dict)
    model.load_state_dict(model_weight)

    return model

## function: def resnet50Sem

def resnet50Sem(cfg=None, pretrained_path=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # if cfg['share'] == 1:
    #     model = ResNetSemShare1(Bottleneck, [3, 4, 6, 3])
    #     print('Encoder: Share1')
    # elif cfg['share'] == 2:
    #     model = ResNetSemShare2(Bottleneck, [3, 4, 6, 3])
    #     print('Encoder: Share2')
    # elif cfg['share'] == 3:
    #     model = ResNetSemShare3(Bottleneck, [3, 4, 6, 3])
    #     print('Encoder: Share3')
    # elif cfg['share'] == 4:
    if cfg['resnet'] == 101:
        model = ResNetSemShare4(Bottleneck, [3, 4, 23, 3])
        print('Encoder: resnet101')
    else:
        model = ResNetSemShare4(Bottleneck, [3, 4, 6, 3])
        print('Encoder: resnet50')

    if cfg['resnet'] == 101:
        init_path = f'./FewShotSeg-dataset/cache/resnet101-5d3b4d8f.pth'
    else:
        init_path = './resnet50-19c8e357.pth'

    if pretrained_path is not None:
        init_path = f'{cfg["ckpt_dir"]}/best.pth'
    print(f'load: {init_path}')

    pretrained_weight = torch.load(init_path, map_location='cpu')

    model_weight = model.state_dict()

    for key, weight in model_weight.items():
        if key in pretrained_weight:
            model_weight[key] = pretrained_weight[key]
        if key[:3] == 'sem' and key[3:] in pretrained_weight:
            model_weight[key] = pretrained_weight[key[3:]]

    if pretrained_path is not None:
        print("**load eval model**")
        for key, weight in model_weight.items():
            if 'module.encoder.' + key in pretrained_weight:
                model_weight[key] = pretrained_weight['module.encoder.' + key]
            if key[:3] == 'sem' and key[3:] in pretrained_weight:
                model_weight[key] = pretrained_weight[key[3:]]

    model.load_state_dict(model_weight)

    return model

## if: __main__

if __name__ == '__main__':

    model = resnet50(pretrained=True)
    x = torch.ones(1,3,417,417)
    out = model(x)
    print(out.size())

# models\vgg.py

## imports

```python
import torch
import torch.nn as nn
```

## class: class Encoder

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)

# models\__init__.py

## imports

```python
from models.base import BaseModel
```

# util\metric.py

## imports

```python
import numpy as np
```

## class: class Metric

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_runs:
            number of test runs
    """
    def __init__(self, max_label=20, n_runs=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_runs = 1 if n_runs is None else n_runs

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]

    def record(self, pred, target, labels=None, n_run=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            if target_idx_j:  # if ground-truth contains this class
                tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)

    def get_mIoU(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mIoU_binary(self, n_run=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
                      for run in range(self.n_runs)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

# util\sbd_instance_process.py

## imports

```python
import os
from scipy.io import loadmat
from PIL import Image
```

## global: voc_dir

voc_dir = '../Pascal/VOCdevkit/VOC2012/'

## global: sbd_dir

sbd_dir = '../SBD/'

## global: inst_path

inst_path = os.path.join(voc_dir, 'SegmentationObject')

## global: inst_aug_path

inst_aug_path = os.path.join(sbd_dir, 'inst')

## global: target_path

target_path = os.path.join(voc_dir, 'SegmentationObjectAug')

## global: inst_files

inst_files = os.listdir(inst_path)

## global: palette

palette = im.getpalette()

## global: inst_aug_files

inst_aug_files = os.listdir(inst_aug_path)

# util\scribbles.py

## imports

```python
from __future__ import absolute_import, division
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph
```

## function: def bezier_curve

def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it.
    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points: Integer. Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`. Maximum value 10000.
    # Returns
        ndarray: Array of shape (1000, 2) with the bezier curve of the
            given path of points.
    """
    nb_points = min(nb_points, 1000)

    points = np.asarray(points, dtype=np.float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            '`points` should be two dimensional and have shape: (N, 2)')

    n_points = len(points)
    if n_points > nb_points:
        # We are downsampling points
        return points

    t = np.linspace(0., 1., nb_points).reshape(1, -1)

    # Compute the Bernstein polynomial of n, i as a function of t
    i = np.arange(n_points).reshape(-1, 1)
    n = n_points - 1
    polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

    bezier_curve_points = polynomial_array.T.dot(points)

    return bezier_curve_points

## function: def bresenham

def bresenham(points):
    """ Apply Bresenham algorithm for a list points.
    More info: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the number
            if points and the second coordinate representing the (x, y)
            coordinates.
    # Returns
        ndarray: Array of points after having applied the bresenham algorithm.
    """

    points = np.asarray(points, dtype=np.int)

    def line(x0, y0, x1, y1):
        """ Bresenham line algorithm.
        """
        d_x = x1 - x0
        d_y = y1 - y0

        x_sign = 1 if d_x > 0 else -1
        y_sign = 1 if d_y > 0 else -1

        d_x = np.abs(d_x)
        d_y = np.abs(d_y)

        if d_x > d_y:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            d_x, d_y = d_y, d_x
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * d_y - d_x
        y = 0

        line = np.empty((d_x + 1, 2), dtype=points.dtype)
        for x in range(d_x + 1):
            line[x] = [x0 + x * xx + y * yx, y0 + x * xy + y * yy]
            if D >= 0:
                y += 1
                D -= 2 * d_x
            D += 2 * d_y

        return line

    nb_points = len(points)
    if nb_points < 2:
        return points

    new_points = []

    for i in range(nb_points - 1):
        p = points[i:i + 2].ravel().tolist()
        new_points.append(line(*p))

    new_points = np.concatenate(new_points, axis=0)

    return new_points

## function: def scribbles2mask

def scribbles2mask(scribbles,
                   output_resolution,
                   bezier_curve_sampling=False,
                   nb_points=1000,
                   compute_bresenham=True,
                   default_value=0):
    """ Convert the scribbles data into a mask.
    # Arguments
        scribbles: Dictionary. Scribbles in the default format.
        output_resolution: Tuple. Output resolution (H, W).
        bezier_curve_sampling: Boolean. Weather to sample first the returned
            scribbles using bezier curve or not.
        nb_points: Integer. If `bezier_curve_sampling` is `True` set the number
            of points to sample from the bezier curve.
        compute_bresenham: Boolean. Whether to compute bresenham algorithm for the
            scribbles lines.
        default_value: Integer. Default value for the pixels which do not belong
            to any scribble.
    # Returns
        ndarray: Array with the mask of the scribbles with the index of the
            object ids. The shape of the returned array is (B x H x W) by
            default or (H x W) if `only_annotated_frame==True`.
    """
    if len(output_resolution) != 2:
        raise ValueError(
            'Invalid output resolution: {}'.format(output_resolution))
    for r in output_resolution:
        if r < 1:
            raise ValueError(
                'Invalid output resolution: {}'.format(output_resolution))

    size_array = np.asarray(output_resolution[::-1], dtype=np.float) - 1
    m = np.full(output_resolution, default_value, dtype=np.int)
    
    for p in scribbles:
        p /= output_resolution[::-1]
        path = p.tolist()
        path = np.asarray(path, dtype=np.float)
        if bezier_curve_sampling:
            path = bezier_curve(path, nb_points=nb_points)
        path *= size_array
        path = path.astype(np.int)

        if compute_bresenham:
            path = bresenham(path)
        m[path[:, 1], path[:, 0]] = 1

    return m

## class: class ScribblesRobot

class ScribblesRobot(object):
    """Robot that generates realistic scribbles simulating human interaction.
    
    # Attributes:
        kernel_size: Float. Fraction of the square root of the area used
            to compute the dilation and erosion before computing the
            skeleton of the error masks.
        max_kernel_radius: Float. Maximum kernel radius when applying
            dilation and erosion. Default 16 pixels.
        min_nb_nodes: Integer. Number of nodes necessary to keep a connected
            graph and convert it into a scribble.
        nb_points: Integer. Number of points to sample the bezier curve
            when converting the final paths into curves.

    Reference:
    [1] Sergi et al., "The 2018 DAVIS Challenge on Video Object Segmentation", arxiv 2018
    [2] Jordi et al., "The 2017 DAVIS Challenge on Video Object Segmentation", arxiv 2017
    
    """
    def __init__(self,
                 kernel_size=.15,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):
        if kernel_size >= 1. or kernel_size < 0:
            raise ValueError('kernel_size must be a value between [0, 1).')

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.
        # Arguments
            mask: Numpy Array. Error mask
        Returns:
            skel: Numpy Array. Skeleton mask
        """
        mask = np.asarray(mask, dtype=np.uint8)
        side = np.sqrt(np.sum(mask > 0))

        mask_ = mask
        # kernel_size = int(self.kernel_size * side)
        kernel_radius = self.kernel_size * side * .5
        kernel_radius = min(kernel_radius, self.max_kernel_radius)
        # logging.verbose(
        #     'Erosion and dilation with kernel radius: {:.1f}'.format(
        #         kernel_radius), 2)
        compute = True
        while kernel_radius > 1. and compute:
            kernel = disk(kernel_radius)
            mask_ = rank.minimum(mask.copy(), kernel)
            mask_ = rank.maximum(mask_, kernel)
            compute = False
            if mask_.astype(np.bool).sum() == 0:
                compute = True
                prev_kernel_radius = kernel_radius
                kernel_radius *= .9
                # logging.verbose('Reducing kernel radius from {:.1f} '.format(
                #     prev_kernel_radius) +
                #                 'pixels to {:.1f}'.format(kernel_radius), 1)

        mask_ = np.pad(
            mask_, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        skel = medial_axis(mask_.astype(np.bool))
        skel = skel[1:-1, 1:-1]
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph
        Args:
            skeleton_mask (ndarray): Skeleton mask
        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph and the second element is an array of xy coordinates
                indicating the coordinates for each Graph node.
                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        T = nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.
        Args:
            G (nx.Graph): Graph
        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for g in nx.connected_component_subgraphs(G):

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            if len(g) < self.min_nb_nodes:
                # Prune small subgraphs
                # logging.verbose('Remove a small line with {} nodes'.format(
                #     len(g)), 1)
                continue

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        Given an undirected tree graph, compute the longest path and return it.
        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal
        Args:
            G (nx.Graph): Graph which should be an undirected tree graph
        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)


    def generate_scribbles(self, mask):
        """Given a binary mask, the robot will return a scribble in the region"""

        # generate scribbles
        skel_mask = self._generate_scribble_mask(mask)
        G, P = self._mask2graph(skel_mask)
        S = self._acyclics_subgraphs(G)
        longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
        longest_paths = [P[idx] for idx in longest_paths_idx]
        scribbles_paths = [
                bezier_curve(p, self.nb_points) for p in longest_paths
        ]

        output_resolution = tuple([mask.shape[0], mask.shape[1]])
        scribble_mask = scribbles2mask(scribbles_paths, output_resolution)

        return scribble_mask

# util\utils.py

## imports

```python
import random
import torch
import numpy as np
```

## function: def set_seed

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## global: CLASS_LABELS

CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    },
    'COCO': {
        'all': set(range(1, 81)),
        0: set(range(1, 81)) - set(range(1, 21)),
        1: set(range(1, 81)) - set(range(21, 41)),
        2: set(range(1, 81)) - set(range(41, 61)),
        3: set(range(1, 81)) - set(range(61, 81)),
    }
}

## function: def get_bbox

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

# util\visual_utils.py

## imports

```python
from typing import List, Tuple
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
```

## global: DEFAULT10

DEFAULT10 = [
    (196, 229, 56),
    (253, 167, 223),
    (247, 159, 31),
    (18, 137, 167),
    (181, 52, 113),
    (0, 148, 50),
    (153, 128, 250),
    (234, 32, 39),
    (76, 108, 252),
    (220, 76, 252),
]

## function: def get_random_colors

def get_random_colors(n_colors):
    # type: (int) -> List[Tuple[int, int, int]]
    """
    Generate a list of `n_colors` random bright colors in RGB format.
    Colors are different from each other (as much as possible).

    :param n_colors: number of colors to generate
    :return: list of RGB colors, where each color is a tuple of (R, G, B)
        and each channel is in the range [0, 255]
    """

    # generate a pool of random bright colors
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 256, size=(n_colors * 128, 3), dtype=int)

    # select colors iteratively to maximize minimum distance
    selected_colors = DEFAULT10[:n_colors]
    for _ in range(len(selected_colors), n_colors):
        distance_matrix = cdist(np.array(selected_colors), colors)
        min_distances = np.min(distance_matrix, axis=0)
        next_color_index = np.argmax(min_distances)
        selected_colors.append(colors[next_color_index])

    return [tuple(color) for color in selected_colors]

## function: def apply_mask_overlay

def apply_mask_overlay(img, masks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Applies colored overlays to an image based on binary masks.

    :param img: The input image to which the masks will be applied.
        ->> shape: (3, H, W)
        ->> values in range float[0, 1]
    :param masks: Binary masks indicating regions to overlay.
        ->> shape: (N_masks, H, W)
        ->> values in range bin{0, 1}
    :return: The image with colored mask overlays applied.
        ->> shape: (3, H, W)
        ->> values in range float[0, 1]
    """
    colors = get_random_colors(masks.shape[0])
    overlay = img.clone() * 0.4

    for class_idx in range(masks.shape[0]):
        # m0: binary mask for class `idx`; shape: (3, H, W)
        m0 = (masks[class_idx][None, ...] > 0).repeat((3, 1, 1)).unsqueeze(0)

        # m1: colored mask for class `idx`; shape: (3, H, W)
        m1 = masks[class_idx][None, ...].repeat((3, 1, 1)).unsqueeze(0)

        if m1.dtype != torch.float32:
            m1 = m1.float()

        colored_mask = torch.tensor(colors[class_idx], dtype=torch.float32, device=img.device).reshape(-1, 1, 1) / 255.
        m1 *= colored_mask

        if m1.device != img.device:
            m1 = m1.to(img.device)

        overlay[m0] = 0.25 * img[m0] + 0.75 * m1[m0]

    return overlay

## function: def tensor_to_cv2

def tensor_to_cv2(x):
    # type: (torch.Tensor) -> np.ndarray
    """
    Convert a torch tensor representing an image to a numpy array in
    the format expected by OpenCV.

    :param x: torch tensor representing an image
        ->> shape: (3, H, W)
        ->> values in range [0, 1]
    :return: image as a numpy array (OpenCV format)
        ->> shape: (H, W, 3)
        ->> values in range [0, 255] (np.uint8)
    """
    img = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

## function: def decode_and_apply_mask_overlay

def decode_and_apply_mask_overlay(img, masks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Decodes binary masks into RGB images and applies colored overlays to a batch of images.

    :param img: Input images to which the masks will be applied.
        ->> shape: (B, 3, H, W)
        ->> values in range float[0, 1]
    :param masks: Binary masks indicating regions to overlay.
        ->> shape: (B, N_masks, H, W)
        ->> values in range bin{0, 1}
    :return: Images with colored mask overlays applied.
        ->> shape: (B, 3, H, W)
        ->> values in range float[0, 1]
    """
    B, _, H, W = img.shape
    _, N_masks, _, _ = masks.shape

    # Generate a colormap
    colormap = get_random_colors(N_masks)
    class_colors = torch.tensor(colormap, dtype=torch.uint8, device=img.device)

    # Decode masks into RGB
    background_color = torch.tensor([0, 0, 0], dtype=torch.uint8, device=img.device).unsqueeze(0)
    rgb_colors = torch.cat([background_color, class_colors], dim=0)

    # Background pixels (B, H, W)
    background_mask = torch.all(masks == 0, dim=1)
    class_map = masks.argmax(dim=1)  # (B, H, W)
    class_map[background_mask] = -1  # Set background to -1

    # Decode into RGB (B, H, W, 3) -> (B, 3, H, W)
    decoded_mask = rgb_colors[class_map + 1].permute(0, 3, 1, 2).float() #/ 255.0

    # Apply overlay
    overlay = img.clone() * 0.4 + decoded_mask * 0.6

    return overlay

## function: def demo

def demo():
    # generate a grid of random colors
    n_rows = 5
    n_cols = 8
    colors = get_random_colors(n_rows * n_cols)
    grid = np.zeros((n_rows * 100, n_cols * 100, 3), dtype=np.uint8)
    for i in range(n_rows):
        for j in range(8):
            color = colors[i * n_cols + j]
            grid[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color

    # display the grid
    cv2.imshow('grid', grid[..., ::-1])
    cv2.waitKey()

## if: __main__

if __name__ == '__main__':
    demo()

# util\voc_classwise_filenames.py

## imports

```python
import os
import numpy as np
from PIL import Image
```

## global: voc_dir

voc_dir = '../../data/Pascal/VOCdevkit/VOC2012/'

## global: seg_dir

seg_dir = os.path.join(voc_dir, 'SegmentationClassAug')

## global: trainaug_path

trainaug_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'trainaug.txt')

## global: trainval_path

trainval_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'trainval.txt')

## global: train_path

train_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt')

## global: val_path

val_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt')

## global: filenames

filenames = os.listdir(seg_dir)

## global: filenames_dic

filenames_dic = {'train': train,
                 'val': val,
                 'trainval': trainval,
                 'trainaug': trainaug}

## global: dic

dic = {'train': {},
       'val': {},
       'trainval': {},
       'trainaug': {}}

# util\__init__.py

