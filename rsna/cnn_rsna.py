import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths for data
csv_file = "/home/adam/processed_data/processed_csv/df_train_OneHot.csv"
root = "/home/adam/processed_data/rsna-dataset-dcm2image/train_images"


# Data preprocessing with augmentation
transform = transforms.Compose([
    transforms.Resize((576, 576)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
# Function to create a sampler for a subset of the dataset
def get_subset_sampler(dataset, num_samples):
    indices = torch.randperm(len(dataset)).tolist()[:num_samples]
    return SubsetRandomSampler(indices)


# Set the subset size 
subset_size = 300

# Create data loaders with subset sampler
batch_size = 30
train_sampler = get_subset_sampler(dataset, subset_size)
val_sampler = get_subset_sampler(dataset, subset_size // 10)  # Smaller validation subset
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)


# Define a simple model (example)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=75):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Instantiate the SimpleCNN model
num_classes = len(dataset[0][1])
model = SimpleCNN(num_classes=num_classes)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize training parameters
min_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
patience = 20
wait = 0
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
activate = nn.Sigmoid()


# Lists to store losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    preds = torch.sigmoid(outputs) > 0.5
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy


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
            total_accuracy += calculate_accuracy(outputs, labels).item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy


# Training the model
num_epochs = 25

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
        running_accuracy += calculate_accuracy(outputs, labels).item() * images.size(0)
        postfix = {'loss': f'{loss:.4f}', 'batch': f'{batch_idx + 1}/{total_batches}'}
        progress_bar.set_postfix(postfix)

    epoch_loss = running_loss / subset_size  # Adjust based on the subset size
    epoch_accuracy = running_accuracy / subset_size
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    # Validate after each epoch
    validate_loss, validate_accuracy = validate(model=model, criterion=criterion, val_loader=val_loader)
    val_losses.append(validate_loss)
    val_accuracies.append(validate_accuracy)
    
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model.pth')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    print(f'Validation Loss: {validate_loss:.4f}, Validation Accuracy: {validate_accuracy:.4f}')


model.load_state_dict(best_model_wts)


# Plotting the results
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the figure
plt.savefig('plots/cnn_training_validation_metrics.png')
plt.show()



# def dcm_to_image(image_path):
#     """Convert DICOM file to PIL image and preprocess."""
#     dicom = pydicom.dcmread(image_path)
#     data = dicom.pixel_array
    
#     data = (data - np.min(data)) / (np.max(data) - np.min(data))
#     data = (data * 255).astype(np.uint8)
    
#     image = Image.fromarray(data)
#     image = image.convert('RGB')
#     return image

# def predict_image(image_path):
#     """Predict the class probabilities for a single image."""
#     image = dcm_to_image(image_path)
#     image = transform(image).unsqueeze(0)
#     image = image.to(device)
    
#     with torch.no_grad():
#         outputs = model(image)
#         probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
#     return probabilities

# def predict_folder_recursive(folder_path, all_probabilities):
#     """Recursively predict class probabilities for all images in a folder."""
#     for item in os.listdir(folder_path):
#         item_path = os.path.join(folder_path, item)
        
#         if os.path.isdir(item_path):
#             predict_folder_recursive(item_path, all_probabilities)
#         elif item_path.endswith('.dcm'):
#             probabilities = predict_image(item_path)
#             all_probabilities.append(probabilities)

# def predict_folder(folder_path):
#     """Predict the class probabilities for all images in the folder and return the results."""
#     results = {}
    
#     for sub_folder in os.listdir(folder_path):
#         sub_folder_path = os.path.join(folder_path, sub_folder)
        
#         if os.path.isdir(sub_folder_path):
#             all_probabilities = []
#             predict_folder_recursive(sub_folder_path, all_probabilities)
            
#             all_probabilities = np.array(all_probabilities)
#             results[sub_folder] = all_probabilities.mean(axis=0).reshape(25, 3)
    
#     return results

# def save_results_to_csv(results, output_csv_path):
#     """Save the prediction results to a CSV file."""
#     labels = [
#         "left_neural_foraminal_narrowing_l1_l2",
#         "left_neural_foraminal_narrowing_l2_l3",
#         "left_neural_foraminal_narrowing_l3_l4",
#         "left_neural_foraminal_narrowing_l4_l5",
#         "left_neural_foraminal_narrowing_l5_s1",
#         "left_subarticular_stenosis_l1_l2",
#         "left_subarticular_stenosis_l2_l3",
#         "left_subarticular_stenosis_l3_l4",
#         "left_subarticular_stenosis_l4_l5",
#         "left_subarticular_stenosis_l5_s1",
#         "right_neural_foraminal_narrowing_l1_l2",
#         "right_neural_foraminal_narrowing_l2_l3",
#         "right_neural_foraminal_narrowing_l3_l4",
#         "right_neural_foraminal_narrowing_l4_l5",
#         "right_neural_foraminal_narrowing_l5_s1",
#         "right_subarticular_stenosis_l1_l2",
#         "right_subarticular_stenosis_l2_l3",
#         "right_subarticular_stenosis_l3_l4",
#         "right_subarticular_stenosis_l4_l5",
#         "right_subarticular_stenosis_l5_s1",
#         "spinal_canal_stenosis_l1_l2",
#         "spinal_canal_stenosis_l2_l3",
#         "spinal_canal_stenosis_l3_l4",
#         "spinal_canal_stenosis_l4_l5",
#         "spinal_canal_stenosis_l5_s1"
#     ]
    
#     data = []
    
#     for folder_name, average_probabilities in results.items():
#         for i, label in enumerate(labels):
#             row_id = f"{folder_name}_{label}"
#             probabilities = average_probabilities[i]
#             normalized_probabilities = probabilities / probabilities.sum()
#             row = [row_id] + normalized_probabilities.tolist()
#             data.append(row)
    
#     df = pd.DataFrame(data, columns=["row_id", "normal_mild", "moderate", "severe"])
#     df.to_csv(output_csv_path, index=False)
#     print(f"Results saved to {output_csv_path}")

# # Create submission file
# folder_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images"
# output_csv_path = 'submission.csv'

# results = predict_folder(folder_path)
# save_results_to_csv(results, output_csv_path)
