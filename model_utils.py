import torch
import torch.nn as nn
from torchvision import models
import timm

def get_model_info(model_name):
    """Get model metadata"""
    model_info = {
        "LeafNet_CNN": {
            "architecture": "Custom CNN with Residual Blocks",
            "input_size": "224x224",
            "pretrained": False,
            "description": "Custom architecture with attention mechanism"
        },
        "EfficientNet_B0": {
            "architecture": "EfficientNet-B0",
            "input_size": "224x224", 
            "pretrained": True,
            "description": "Efficient convolutional neural network"
        },
        "MobileNet_V3": {
            "architecture": "MobileNetV3-Small",
            "input_size": "224x224",
            "pretrained": True,
            "description": "Lightweight mobile-optimized network"
        },
        "ResNet18": {
            "architecture": "ResNet-18",
            "input_size": "224x224",
            "pretrained": True,
            "description": "Residual network with 18 layers"
        },
        "ViT_Small": {
            "architecture": "Vision Transformer Small",
            "input_size": "224x224",
            "pretrained": True,
            "description": "Transformer-based vision model"
        }
    }
    return model_info.get(model_name, {})

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
