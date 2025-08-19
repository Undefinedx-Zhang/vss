"""
Semantic Segmentation Models
Supports DeepLabV3, FCN, and other segmentation architectures
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101
from torchvision.models import resnet50, resnet101
from typing import Optional

# 设置PyTorch缓存目录到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(project_root, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)
if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = cache_dir


class DeepLabV3(nn.Module):
    """DeepLabV3 with ResNet backbone"""
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'resnet50', 'resnet101'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model(x)
        if isinstance(result, dict):
            return result['out']
        return result


class FCN(nn.Module):
    """FCN with ResNet backbone"""
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        if backbone == 'resnet50':
            self.model = fcn_resnet50(pretrained=pretrained, num_classes=num_classes)
        elif backbone == 'resnet101':
            self.model = fcn_resnet101(pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'resnet50', 'resnet101'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model(x)
        if isinstance(result, dict):
            return result['out']
        return result


class SimpleSegNet(nn.Module):
    """Simple SegNet-like architecture for lightweight segmentation"""
    
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Decoder
        self.dec4 = self._make_layer(512, 256)
        self.dec3 = self._make_layer(256, 128)
        self.dec2 = self._make_layer(128, 64)
        self.dec1 = self._make_layer(64, 64)
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def _make_layer(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original size
        orig_size = x.shape[2:]
        
        # Encoder
        x1 = self.enc1(x)
        x1_pool, idx1 = self.pool(x1)
        
        x2 = self.enc2(x1_pool)
        x2_pool, idx2 = self.pool(x2)
        
        x3 = self.enc3(x2_pool)
        x3_pool, idx3 = self.pool(x3)
        
        x4 = self.enc4(x3_pool)
        x4_pool, idx4 = self.pool(x4)
        
        # Decoder - ensure size compatibility for unpooling
        x4_up = self.unpool(x4_pool, idx4, output_size=x4.size())
        x4_dec = self.dec4(x4_up)
        
        x3_up = self.unpool(x4_dec, idx3, output_size=x3.size())
        x3_dec = self.dec3(x3_up)
        
        x2_up = self.unpool(x3_dec, idx2, output_size=x2.size())
        x2_dec = self.dec2(x2_up)
        
        x1_up = self.unpool(x2_dec, idx1, output_size=x1.size())
        x1_dec = self.dec1(x1_up)
        
        # Final classification
        out = self.classifier(x1_dec)
        
        # Resize to original size if needed
        if out.shape[2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
        
        return out


class UNet(nn.Module):
    """U-Net architecture for segmentation"""
    
    def __init__(self, num_classes: int, in_channels: int = 3, features: list = [64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet
        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature))
            in_ch = feature
        
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResNetSegmentation(nn.Module):
    """ResNet backbone with simple segmentation head"""
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Load ResNet backbone
        if backbone == 'resnet50':
            resnet = resnet50(pretrained=pretrained)
            backbone_channels = 2048
        elif backbone == 'resnet101':
            resnet = resnet101(pretrained=pretrained)
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final FC layer and avgpool
        self.backbone_net = nn.Sequential(*list(resnet.children())[:-2])
        
        # Simple segmentation head
        self.classifier = nn.Sequential(
            nn.Conv2d(backbone_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        
        # Backbone feature extraction
        features = self.backbone_net(x)
        
        # Classification
        out = self.classifier(features)
        
        # Upsample to input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


# Model registry for segmentation
SEG_MODEL_REGISTRY = {
    "deeplabv3_resnet50": lambda num_classes: DeepLabV3(num_classes, 'resnet50'),
    "deeplabv3_resnet101": lambda num_classes: DeepLabV3(num_classes, 'resnet101'),
    "fcn_resnet50": lambda num_classes: FCN(num_classes, 'resnet50'),
    "fcn_resnet101": lambda num_classes: FCN(num_classes, 'resnet101'),
    "resnet50_seg": lambda num_classes: ResNetSegmentation(num_classes, 'resnet50'),
    "resnet101_seg": lambda num_classes: ResNetSegmentation(num_classes, 'resnet101'),
    "simple_segnet": lambda num_classes: SimpleSegNet(num_classes),
    "unet": lambda num_classes: UNet(num_classes),
}


def build_seg_model(name: str, num_classes: int):
    """Build segmentation model"""
    name = name.lower()
    if name in SEG_MODEL_REGISTRY:
        return SEG_MODEL_REGISTRY[name](num_classes)
    raise ValueError(f"Unsupported segmentation model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    }
