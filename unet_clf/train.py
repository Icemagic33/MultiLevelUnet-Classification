import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import numpy as np
from data_vis import SpineDataset, transform
from model_multi_level import DoubleConv, MultiLevelUnet
from check_data import im_list_dcm


# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Pad((0, 0, 255 - width, 255 - height)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6116], std=[0.2425])
])

# Load dataset
train_dataset = SpineDataset(im_list_dcm, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define model
model = MultiLevelUnet(in_channels=1, num_heads=25, num_classes_per_head=3).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = sum(criterion(output, labels[:, i]) for i, output in enumerate(outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

print('Finished Training')
torch.save(model.state_dict(), 'multi_level_unet.pth')

