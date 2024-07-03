import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_vis import SpineDataset
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
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_heads=25, num_classes_per_head=3):
        super(SimpleCNN, self).__init__()
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)  # Assuming input images are resized to 256x256
        self.fc2 = nn.Linear(1024, 512)

        # Define the heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_head) for _ in range(num_heads)
        ])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))

        # Collect outputs from each head
        outputs = [head(x) for head in self.heads]
        return outputs

model = SimpleCNN(in_channels=1, num_heads=25, num_classes_per_head=3).to(device)

# Create dataset and dataloader
dataset = SpineDataset(im_list_dcm, df_coor, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# List to store loss values
train_loss_values = []
val_loss_values = []

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images, labels = batch
        images = images.view(-1, 1, 256, 256).to(device)  # Flatten batch of studies into batch of images
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Concatenate the outputs from each head for the loss calculation
        outputs = torch.cat(outputs, dim=0)
        labels = labels.repeat_interleave(images.size(0) // labels.size(0)).to(device)  # Repeat labels to match the number of images
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    train_loss_values.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Finished Training')
torch.save(model.state_dict(), 'simple_cnn.pth')

plt.figure()
# Plot the loss values
plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('baseline_training_loss_over_epochs.png')
plt.show()
