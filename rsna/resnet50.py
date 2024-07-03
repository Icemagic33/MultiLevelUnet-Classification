import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Paths for data
csv_file = "/kaggle/input/processed-rsna/processed_csv/df_train_OneHot.csv"
root = "/kaggle/input/processed-rsna/rsna_dataset_dcm2Image/train_images"

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((576, 576)),
    transforms.ToTensor(),
])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load the CSV file containing image paths and labels
        self.transform = transform  # Store the transformations to be applied on images
    
    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset
    
    def __getitem__(self, idx):
        # Construct the image file path based on the CSV file data
        image_id = os.path.join(root,
                                self.data.iloc[idx, 0].astype(str),
                                self.data.iloc[idx, 1].astype(str),
                                f"{self.data.iloc[idx, 2]}.jpg")
        if not os.path.exists(image_id):
            raise FileNotFoundError(f"File {image_id} does not exist")
        image = Image.open(image_id).convert('L')  # Open the image file and convert to grayscale
        
        # Convert grayscale image to three-channel image by duplicating the channel
        image = image.convert('RGB')
        
        # Apply transformations if any are specified
        if self.transform:
            image = self.transform(image)
        
        # Load the labels, assuming the 4th column is the label index
        label = self.data.iloc[idx, 3:].values.astype(float)  # Ensure label is an integer representing the class index
        
        return image, label  # Return the image and its corresponding label

# Load dataset
dataset = CustomImageDataset(csv_file, transform=transform)

# Function to create a sampler for a subset of the dataset
def get_subset_sampler(dataset, num_samples):
    indices = torch.randperm(len(dataset)).tolist()[:num_samples]
    return SubsetRandomSampler(indices)

# Set the subset size (e.g., 500 images per epoch)
subset_size = 500

# Create data loaders with subset sampler
batch_size = 16
train_sampler = get_subset_sampler(dataset, subset_size)
val_sampler = get_subset_sampler(dataset, subset_size // 10)  # Smaller validation subset

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Load pretrained ResNet50 model
model = models.resnet50()
# Modify the first layer to accept single-channel (grayscale) input
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# Modify the final layer to match the number of classes
num_classes = len(dataset[0][1])
model.fc = nn.Linear(model.fc.in_features, num_classes)


# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize training parameters
min_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
patience = 5
wait = 0
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
activate = nn.Sigmoid()

# Training the model
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    total_batches = len(train_loader)
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_accuracy += (outputs.round() == labels).float().mean().item() * images.size(0)
        postfix = {'loss': f'{loss:.4f}', 'batch': f'{batch_idx + 1}/{total_batches}'}
        progress_bar.set_postfix(postfix)

    epoch_loss = running_loss / subset_size  # Adjust based on the subset size
    epoch_accuracy = running_accuracy / subset_size
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    val_loss, val_accuracy = validate(model=model, criterion=criterion, val_loader=val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if val_loss < min_loss:
        min_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model.pth')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

model.load_state_dict(best_model_wts)

# Plot training and validation loss
import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Validate the model
def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_accuracy += (outputs.round() == labels).float().mean().item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy

# Perform validation
validate_loss, validate_accuracy = validate(model=model, criterion=criterion, val_loader=val_loader)
print(f'Validation Loss: {validate_loss:.4f}, Validation Accuracy: {validate_accuracy:.4f}')
