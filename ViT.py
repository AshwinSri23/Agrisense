import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vit_b_32
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import copy
from PIL import Image
from torch.utils.data import Dataset 
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        # Make sure that all images are resized to a consistent size and then centered to retain the most important part of the image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Ensure that the image values are properly scaled for model input
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Data directory
data_dir = '/content/content/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'

# Create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

# Create dataloaders
# Each iteration will have 32 images and these images will be randomly shuffled
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

model = vit_b_32(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)  # Adjust for our dataset classes

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer (the classification head)
for param in model.heads.head.parameters():
    param.requires_grad = True

# Move the model to the device (GPU or CPU)
model = model.to(device)

from torch.optim import lr_scheduler
# Use Adam optimizer with different learning rates for different layers
optimizer = optim.AdamW(model.heads.head.parameters(), lr=0.0001, weight_decay=0.01)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the learning rate scheduler
num_epochs = 10
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



# Utility function to compute precision, recall, and F1-score
def compute_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return precision, recall, f1

# Directory to save checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
metrics_history = []

epochs = 10
patience = 5  # Number of epochs to wait before stopping
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients before backward pass
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss

    # Print average loss for the epoch
    avg_loss = running_loss / len(dataloaders['train'])
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():  # No gradient calculation during validation
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect predictions and labels for metrics computation
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)

    avg_val_loss = val_loss / len(dataloaders['valid'])
    print(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

    # Compute and log precision, recall, F1-score for validation
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    precision, recall, f1 = compute_metrics(all_preds, all_labels)
    print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    metrics_history.append({
        'epoch': epoch + 1,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

    # Early stopping logic
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss

        # Save the best model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1}")

        epochs_without_improvement = 0  # Reset the counter since we have improved
    else:
        epochs_without_improvement += 1  # Increment the counter if no improvement

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1} due to no improvement for {patience} epochs.")
        break  # Stop training if patience is reached

    # Update the learning rate using the scheduler
    scheduler.step()
    print(f"Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), "vitB32_plant_model.pth")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloaders['valid']:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")

# Define the model architecture
model = vit_b_32(pretrained=False, num_classes=38)  # Replace `your_num_classes` with the number of classes in your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the correct device (e.g., GPU or CPU)

model.load_state_dict(torch.load('/content/drive/MyDrive/aa/vitB32_plant_model.pth'))

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
test_data_dir = '/content/new-plant-diseases-dataset/test/test'
test_real_world_data_dir = '/content/drive/MyDrive/aa/test'


# Create test dataset and DataLoader
test_dataset = TestDataset(test_data_dir, transform=test_data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_real_world_dataset = TestDataset(test_real_world_data_dir, transform=test_data_transforms)
test_real_world_dataloader = DataLoader(test_real_world_dataset, batch_size=32, shuffle=False)

# Predict and compare labels
correct = 0
total = 0

model.eval()  # Ensure model is in evaluation mode

with torch.no_grad():
    for inputs, file_names in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        probabilities_numpy = probabilities.cpu().detach().numpy()
        print(f"probabilitiesNumpy: {probabilities_numpy}")
        print(f"rows: {outputs[0]}, columns: {outputs.shape[1]}")
        for i, prob in enumerate(probabilities_numpy[0]):
          print(f"Class {i}: {prob:.6f}")
        _, preds = torch.max(outputs, 1)
        predicted_labels = [normalized_class_names.get(class_names[pred]) for pred in preds]  # Normalize predicted class names
        actual_labels = [normalize_file_name(file_name) for file_name in file_names]  # Normalize actual class names
        print(f"Predicted labels: {predicted_labels}")
        print(f"Actual labelssss: {actual_labels}")

        for pred, actual in zip(predicted_labels, actual_labels):
            if pred == actual:
                correct += 1
            total += 1

accuracy = correct / total * 100
print(f"Accuracy on unseen test data: {accuracy:.2f}%")

correct = 0
total = 0

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
                correct += 1
            total += 1

accuracy = correct / total * 100
print(f"Accuracy on real world test data: {accuracy:.2f}%")


# Initialize lists to store true labels and predicted labels
all_preds = []
all_labels = []

# Set model to evaluation mode
model.eval()

# Disable gradient calculation for evaluation
with torch.no_grad():
    for inputs, labels in dataloaders['valid']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without Normalization')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i < len(classes) and j < len(classes):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')

# Calculate overall metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# Classification report
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# Calculate per-class metrics
f1_per_class = f1_score(all_labels, all_preds, average=None)
precision_per_class = precision_score(all_labels, all_preds, average=None)
recall_per_class = recall_score(all_labels, all_preds, average=None)

for idx, class_name in enumerate(class_names):
    print(f'Class: {class_name}')
    print(f'  Precision: {precision_per_class[idx] * 100:.2f}%')
    print(f'  Recall: {recall_per_class[idx] * 100:.2f}%')
    print(f'  F1 Score: {f1_per_class[idx] * 100:.2f}%')

# Visualize metrics per class
def plot_metric_per_class(metric, metric_name, classes):
    plt.figure(figsize=(12, 6))
    plt.bar(classes, metric)
    plt.xlabel('Classes')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Class')
    plt.xticks(rotation=90)
    plt.show()

plot_metric_per_class(f1_per_class, 'F1 Score', class_names)
plot_metric_per_class(precision_per_class, 'Precision', class_names)
plot_metric_per_class(recall_per_class, 'Recall', class_names)





