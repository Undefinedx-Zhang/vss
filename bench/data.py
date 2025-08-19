"""
Semantic Segmentation Dataset Loaders
Supports Cityscapes and CamVid datasets
"""
import os
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CityscapesDataset(Dataset):
    """Cityscapes Dataset for Semantic Segmentation"""
    
    # Cityscapes class info
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    # Mapping from trainId to color for visualization
    PALETTE = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    
    def __init__(self, root: str, split: str = 'train', transform=None, target_transform=None):
        """
        Args:
            root: Root directory of Cityscapes dataset
            split: 'train', 'val', or 'test'
            transform: Transform for input images
            target_transform: Transform for target masks
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root, 'gtFine', split)
        
        self.images = []
        self.targets = []
        
        # Collect all image and target paths
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            
            for filename in os.listdir(img_dir):
                if filename.endswith('_leftImg8bit.png'):
                    # Image path
                    img_path = os.path.join(img_dir, filename)
                    
                    # Corresponding label path
                    target_filename = filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    target_path = os.path.join(target_dir, target_filename)
                    
                    if os.path.exists(target_path):
                        self.images.append(img_path)
                        self.targets.append(target_path)
        
        print(f"Found {len(self.images)} {split} images in Cityscapes dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target





class CamVidDataset(Dataset):
    """CamVid Dataset for Semantic Segmentation"""
    
    CLASSES = [
        'Sky', 'Building', 'Column_Pole', 'Road', 'Sidewalk', 
        'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
    ]
    
    PALETTE = [
        [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
        [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
        [64, 0, 128], [64, 64, 0], [0, 128, 192]
    ]
    
    def __init__(self, root: str, split: str = 'train', transform=None, target_transform=None):
        """
        Args:
            root: Root directory of CamVid dataset
            split: 'train', 'val', or 'test'
            transform: Transform for input images
            target_transform: Transform for target masks
        """
        self.rgb2id = {tuple(val): i for i, val in enumerate(CamVidDataset.PALETTE)}
        self.IGNORE = 11

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # CamVid directory structure
        if split == 'train':
            self.images_dir = os.path.join(root, 'train')
            self.targets_dir = os.path.join(root, 'train_labels')
        elif split == 'val':
            self.images_dir = os.path.join(root, 'val')
            self.targets_dir = os.path.join(root, 'val_labels')
        else:  # test
            self.images_dir = os.path.join(root, 'test')
            self.targets_dir = os.path.join(root, 'test_labels')
        
        # Collect image and target pairs
        self.images = []
        self.targets = []
        
        if os.path.exists(self.images_dir):
            for filename in sorted(os.listdir(self.images_dir)):    
                if filename.endswith('.png'):
                    img_path = os.path.join(self.images_dir, filename)
                    target_path = os.path.join(self.targets_dir, filename.split(".")[0] + "_L.png")   
                    
                    if os.path.exists(target_path):
                        self.images.append(img_path)
                        self.targets.append(target_path)
        
        print(f"Found {len(self.images)} {split} images in CamVid dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __color_mask_to_id__(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("RGB"))  
        h, w, _ = mask.shape
        idmask = np.full((h, w), self.IGNORE, dtype=np.uint8)
        for rgb, i in self.rgb2id.items():
            hits = (mask[:,:,0]==rgb[0]) & (mask[:,:,1]==rgb[1]) & (mask[:,:,2]==rgb[2])
            idmask[hits] = i
        return Image.fromarray(idmask, mode='L')

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB')
        target = self.__color_mask_to_id__(target_path)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


class SegmentationTransforms:
    """Transforms for segmentation that apply the same transformation to both image and target"""
    
    def __init__(self, size: Tuple[int, int], is_training: bool = True):
        self.size = size
        self.is_training = is_training
        
        # Normalization for common pretrained models
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, image: Image.Image, target: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize both image and target
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        target = TF.resize(target, self.size, interpolation=Image.NEAREST)
        
        if self.is_training:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
            
            # Random crop (if needed)
            # Could add more augmentations here
        
        # Convert to tensor
        image = TF.to_tensor(image)
        image = self.normalize(image)
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        
        return image, target


def build_cityscapes(data_root: str, batch_size: int = 8, num_workers: int = 4, 
                    image_size: Tuple[int, int] = (512, 1024)) -> Tuple[DataLoader, DataLoader]:
    """Build Cityscapes dataloaders"""
    
    train_transforms = SegmentationTransforms(image_size, is_training=True)
    val_transforms = SegmentationTransforms(image_size, is_training=False)
    
    def train_collate_fn(batch):
        images, targets = zip(*batch)
        processed_batch = []
        for img, tgt in zip(images, targets):
            img_tensor, tgt_tensor = train_transforms(img, tgt)
            processed_batch.append((img_tensor, tgt_tensor))
        
        images, targets = zip(*processed_batch)
        return torch.stack(images), torch.stack(targets)
    
    def val_collate_fn(batch):
        images, targets = zip(*batch)
        processed_batch = []
        for img, tgt in zip(images, targets):
            img_tensor, tgt_tensor = val_transforms(img, tgt)
            processed_batch.append((img_tensor, tgt_tensor))
        
        images, targets = zip(*processed_batch)
        return torch.stack(images), torch.stack(targets)
    
    train_dataset = CityscapesDataset(data_root, split='train')
    val_dataset = CityscapesDataset(data_root, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=val_collate_fn
    )
    
    return train_loader, val_loader


def build_camvid(data_root: str, batch_size: int = 8, num_workers: int = 4,
                image_size: Tuple[int, int] = (360, 480)) -> Tuple[DataLoader, DataLoader]:
    """Build CamVid dataloaders"""
    
    train_transforms = SegmentationTransforms(image_size, is_training=True)
    val_transforms = SegmentationTransforms(image_size, is_training=False)
    
    def train_collate_fn(batch):
        images, targets = zip(*batch)
        processed_batch = []
        for img, tgt in zip(images, targets):
            img_tensor, tgt_tensor = train_transforms(img, tgt)
            processed_batch.append((img_tensor, tgt_tensor))
        
        images, targets = zip(*processed_batch)
        return torch.stack(images), torch.stack(targets)
    
    def val_collate_fn(batch):
        images, targets = zip(*batch)
        processed_batch = []
        for img, tgt in zip(images, targets):
            img_tensor, tgt_tensor = val_transforms(img, tgt)
            processed_batch.append((img_tensor, tgt_tensor))
        
        images, targets = zip(*processed_batch)
        return torch.stack(images), torch.stack(targets)
    
    train_dataset = CamVidDataset(data_root, split='train')
    val_dataset = CamVidDataset(data_root, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=val_collate_fn
    )
    
    return train_loader, val_loader


def get_loaders(dataset: str, data_root: str, batch_size: int, num_workers: int = 4):
    """Get segmentation dataloaders"""
    dataset = dataset.lower()
    if dataset == "cityscapes":
        return build_cityscapes(data_root, batch_size, num_workers)
    elif dataset == "camvid":
        return build_camvid(data_root, batch_size, num_workers)
    else:
        raise ValueError(f"Unsupported segmentation dataset: {dataset}")


def get_dataset_info(dataset: str) -> dict:
    """Get dataset information"""
    dataset = dataset.lower()
    if dataset == "cityscapes":
        return {
            "num_classes": 19,
            "classes": CityscapesDataset.CLASSES,
            "palette": CityscapesDataset.PALETTE,
            "ignore_index": 255
        }
    elif dataset == "camvid":
        return {
            "num_classes": 12,
            "classes": CamVidDataset.CLASSES,
            "palette": CamVidDataset.PALETTE,
            "ignore_index": 11  # Unlabelled class
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

        