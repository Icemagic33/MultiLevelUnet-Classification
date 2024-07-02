import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from check_data import im_list_dcm


fd = "/home/adam/data"


# Define mean and std
mean = 0.6116309762001038
std = 0.24248747527599335

# Define transformation pipeline with data augmentation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])
])

# Load the label coordinates CSV file
df_coor = pd.read_csv(f'{fd}/train_label_coordinates.csv')

# Define mappings for conditions and levels
condition_map = {
    'Spinal Canal Stenosis': 0,
    'Left Neural Foraminal Narrowing': 1,
    'Right Neural Foraminal Narrowing': 2,
    'Left Subarticular Stenosis': 3,
    'Right Subarticular Stenosis': 4
}

level_map = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}

severity_map = {
    'normal/mild': 0,
    'moderate': 1,
    'severe': 2
}

# Placeholder for dummy severity mapping
severity = 'normal/mild'

# Define a function to map severity to indices (modify according to actual severity data in your df_coor)
def map_severity(x):
    return severity_map[severity]  # Replace with actual severity column data

class SpineDataset(Dataset):
    def __init__(self, im_list_dcm, df_coor, transform=None, max_images=54):
        self.im_list_dcm = im_list_dcm
        self.df_coor = df_coor
        self.study_ids = list(im_list_dcm.keys())
        self.transform = transform
        self.max_images = max_images

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        data = self.im_list_dcm[study_id]
        
        series_images = []
        for image_dict in data['images']:
            img = image_dict['dicom'].pixel_array
            img = img.astype(np.float32)  # Work with float32 to preserve precision
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255  # Normalize to 0-255
            img = img.astype(np.uint8)  # Convert to uint8 for the transformation
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.repeat(img, 3, axis=-1)  # Repeat the channel to convert to 3 channels
            if self.transform:
                img = self.transform(img)
            series_images.append(img)

        # Pad with zeros if the number of images is less than max_images
        while len(series_images) < self.max_images:
            series_images.append(torch.zeros_like(series_images[0]))

        # Stack images along the batch dimension and flatten to shape (max_images, 3, 256, 256)
        all_images = torch.stack(series_images, dim=0)
        
        # Extract labels for this study
        labels = self.df_coor[self.df_coor['study_id'] == int(study_id)]
        label_tensor = torch.zeros((25,), dtype=torch.long)  # Placeholder for 25 conditions

        for _, row in labels.iterrows():
            condition_idx = condition_map[row['condition']]
            level_idx = level_map[row['level']]
            severity_idx = map_severity(row)  # Use actual severity data here
            label_tensor[condition_idx * 5 + level_idx] = severity_idx

        return all_images, label_tensor

# Custom collate function to handle variable number of images per study
def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.cat(images, dim=0)
    labels = torch.stack(labels)
    return images, labels


# Initialize the model with pretrained weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the last layer of the model to match the number of output classes
num_conditions = 5  # spinal_canal_stenosis, left_neural_foraminal_narrowing, etc.
num_levels = 5  # L1/L2, L2/L3, L3/L4, L4/L5, L5/S1
num_classes = 3  # normal/mild, moderate, severe

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_conditions * num_levels * num_classes)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create dataset and dataloader
dataset = SpineDataset(im_list_dcm, df_coor, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)  # Using batch_size=1 to handle each study separately

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# List to store loss values
loss_values = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images, labels = batch
        images = images.view(-1, 3, 256, 256).to(device)  # Flatten batch of studies into batch of images
        labels = labels.to(device).view(-1)  # Flatten the labels to match the number of images
        
        # Repeat labels to match the number of images
        labels = labels.repeat_interleave(images.size(0) // len(labels))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    loss_values.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Finished Training')
torch.save(model.state_dict(), 'resnet50_spine.pth')

# Plot the loss values
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
