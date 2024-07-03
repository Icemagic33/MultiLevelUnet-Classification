import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_vis import SpineDataset, transform
from model_multi_level import DoubleConv, MultiLevelUnet
from check_data import im_list_dcm

########## Max Width: 384, Max Height: 540, Max Images: 54 ###########

# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define mean and std
mean = 0.6116309762001038
std = 0.24248747527599335

# Define transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

# Load the label coordinates CSV file
fd = "/home/adam/data"
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

# Define a function to map severity to indices
def map_severity(row):
    severity_str = row['severity']
    return severity_map[severity_str]

# Initialize the model
model = MultiLevelUnet(in_channels=1, num_heads=25, num_classes_per_head=3).to(device)

# Create dataset and split into training and validation sets
dataset = SpineDataset(im_list_dcm, df_coor, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Lists to store loss values
train_loss_values = []
val_loss_values = []

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        images, labels = batch
        images = images.view(-1, 1, 256, 256).to(device)  # Flatten batch of studies into batch of images
        labels = labels.to(device)
        
        # Repeat labels to match the number of images
        labels = labels.repeat_interleave(images.size(0) // labels.size(0))

        optimizer.zero_grad()
        outputs = model(images)
        
        # Concatenate the outputs from each head for the loss calculation
        outputs = torch.cat(outputs, dim=0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_loss_values.append(epoch_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.10f}')

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.view(-1, 1, 256, 256).to(device)
            labels = labels.to(device)
            
            # Repeat labels to match the number of images
            labels = labels.repeat_interleave(images.size(0) // labels.size(0))
            
            outputs = model(images)
            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    val_loss_values.append(epoch_val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_val_loss:.10f}')

print('Finished Training')
# torch.save(model.state_dict(), 'multi_level_unet.pth')

plt.figure()
# Plot the loss values
plt.plot(train_loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.savefig('training_and_validation_loss_over_epochs.png')
plt.show()
