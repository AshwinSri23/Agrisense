# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load Pre-trained ResNet50 Model
from torchvision.models import ResNet50_Weights
model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
from collections import Counter
import time
import copy

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
data_dir = '/content/Augmented_NPDD'

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

# Modify the final layer
num_ftrs = model_ft.fc.in_features
num_classes = len(class_names)
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# Move model to gpu
model_ft = model_ft.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0001, weight_decay=0.01)

# Set Learning Rate Scheduler
num_epochs = 10
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs)

import os
import time
import copy
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure 'device' is defined
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Utility function to compute precision, recall, and F1-score
def compute_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return precision, recall, f1

# Define training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics_history = []  # List to store metrics for each epoch
    lr_history = []  # List to store learning rates for each epoch

    checkpoint_dir = '/content/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Collect predictions and labels for metrics computation
                if phase == 'valid':
                    all_preds.append(preds)
                    all_labels.append(labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Compute and log precision, recall, F1-score for validation
            if phase == 'valid':
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                precision, recall, f1 = compute_metrics(all_preds, all_labels)
                print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
                metrics_history.append({
                    'epoch': epoch,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # Log the learning rate
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append({'epoch': epoch, 'lr': current_lr})
        print(f'Learning Rate: {current_lr}')

        # Step the scheduler
        scheduler.step()

        # Save checkpoint after each epoch
        checkpoint_path = f'/content/checkpoints/checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

        # Epoch time
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch} completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s\n')

    # Time elapsed
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, metrics_history, lr_history

# Train the model
num_epochs = 10
model_ft, hist, metrics_history, lr_history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

# Save the trained model
torch.save(model_ft.state_dict(), 'resnet50_finetuned.pth')

"""Model Traning done"""

!zip -r resnet_aug_data_checkpoints.zip checkpoints/

"""Model evaluation on test dataset"""



!unzip -q 'content/new_plant_disease_ds.zip'

import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader  # Ensure Dataset is imported
import torch
import torchvision.transforms as transforms
import os
from torchvision.models import resnet50


# Function to normalize class names and file names
def normalize_file_name(name):
    # Remove digits (e.g., '1', '2', '3') from the name
    name = re.sub(r'\d+', '', name)

    # Replace underscores with spaces for consistency
    name = name.replace("_", "").lower().strip()

    # You can also add any other normalization you might need depending on your naming conventions.
    return name

def normalize_crop_name(crop_name):
    # Remove anything in parentheses (e.g., "(maize)")
    crop_name = re.sub(r'\(.*?\)', '', crop_name)
    # Remove underscores and extra spaces
    crop_name = crop_name.replace("_", "").replace(",", "").strip()
    # Convert to lowercase
    return crop_name.lower()

# Function to normalize class names and file names
def normalize_predicted_name(class_names):
    normalized_classes = {}

    for name in class_names:
        # Split by "___" to separate crop name and disease
        parts = name.split("___")

        if len(parts) > 1:
            crop_name = normalize_crop_name(parts[0])  # Normalize crop name to lowercase
            disease_name = parts[1].replace(" ", "").replace("_", "").replace("-", "").lower() # Normalize disease name to lowercase
            # Remove any bracketed content from the disease name
            disease_name = re.sub(r"\(.*?\)", "", disease_name)
            # print(disease_name)

            # Remove duplicate crop name in the disease name
            if disease_name.startswith(crop_name):
                disease_name = disease_name[len(crop_name):]

            if disease_name == 'cedarapplerust':
                disease_name = 'cedarrust'

            if disease_name == 'yellowleafcurlvirus':
                disease_name = 'yellowcurlvirus'

            # Concatenate crop name and processed disease name without spaces
            normalized_name = crop_name + disease_name

        if name not in normalized_classes:
            normalized_classes[name] = normalized_name

    return normalized_classes

normalized_class_names = normalize_predicted_name(class_names)
print(normalized_class_names)

# Custom Dataset for Test Images
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
        print(f"Found {len(self.image_files)} images in test directory.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Open image as RGB
        label = self.image_files[idx].split('.')[0]  # Extract label from file name
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
test_data_transforms = transforms.Compose([
    transforms.Resize((224,224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize
                         [0.229, 0.224, 0.225])
])

# Test data directory
test_data_dir = 'test/test'

# Create test dataset and DataLoader
test_dataset = TestDataset(test_data_dir, transform=test_data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = resnet50(pretrained=False, num_classes=len(class_names))  # Adjust num_classes as needed

# Load the model weights from the .pth file
model_path = "resnet50_finetuned.pth"  # Replace with the path to your .pth file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Proceed with prediction and comparison
correct = 0
total = 0

with torch.no_grad():
    for inputs, file_names in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels = [normalized_class_names.get(class_names[pred]) for pred in preds.cpu()]  # Normalize predicted class names
        actual_labels = [normalize_file_name(file_name) for file_name in file_names]  # Normalize actual class names
        print(f"Predicted labels: {predicted_labels}")
        print(f"Actual labels: {actual_labels}")

        for pred, actual in zip(predicted_labels, actual_labels):
            if pred == actual:
                correct += 1
            total += 1

accuracy = correct / total * 100
print(f"Accuracy on unseen test data: {accuracy:.2f}%")

"""Test on github data"""

!unzip rw_test.zip

test_real_world_data_dir = 'rw_test'
test_real_world_dataset = TestDataset(test_real_world_data_dir, transform=test_data_transforms)
test_real_world_dataloader = DataLoader(test_real_world_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
plant_correct = 0
disease_correct = 0

with torch.no_grad():
    for inputs, file_names in test_real_world_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels = [normalized_class_names.get(class_names[pred]) for pred in preds]  # Normalize predicted class names
        actual_labels = [normalize_file_name(file_name) for file_name in file_names]  # Normalize actual class names
        print(f"Predicted labels: {predicted_labels}")
        print(f"Actual labelssss: {actual_labels}")

        for pred, actual in zip(predicted_labels, actual_labels):
            if pred[0:4] == actual[0:4]:
                plant_correct += 1
                correct += 1
                if pred == actual:
                  disease_correct += 1
            total += 1

accuracy = correct / total * 100
print(f"Accuracy on real world test data: {accuracy:.2f}%")
print(f"Accuracy given the condition that it predicts the plant right: {disease_correct / plant_correct * 100}")