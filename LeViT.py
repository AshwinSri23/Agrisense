import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from transformers import LevitForImageClassification
import os
import time
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Data directory
data_dir = 'content/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'

# Create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

# Create dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}

# Get class names
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f'Classes: {class_names}')

# Define dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Retrieve class sizes
for phase in ['train', 'valid']:
    dataset = image_datasets[phase]
    class_indices = dataset.targets
    class_counts = Counter(class_indices)
    print(f"\nNumber of images per class in the '{phase}' dataset:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = dataset.classes[class_idx]
        print(f"  Class '{class_name}': {count} images")


