import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
from data_vis import SpineDataset, transform
from model_multi_level import DoubleConv, MultiLevelUnet
from check_data import im_list_dcm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the transformation pipeline for test images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6116309762001038], std=[0.24248747527599335])
])

# Define the SpineTestDataset class for the test dataset
class SpineTestDataset(Dataset):
    def __init__(self, test_img_dir, transform=None, max_images=54):
        self.test_img_dir = test_img_dir
        self.transform = transform
        self.max_images = max_images
        self.study_ids = os.listdir(test_img_dir)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        study_path = os.path.join(self.test_img_dir, study_id)
        series_ids = os.listdir(study_path)

        series_images = []
        for series_id in series_ids:
            series_path = os.path.join(study_path, series_id)
            dcm_files = glob.glob(f"{series_path}/*.dcm")
            for dcm_file in sorted(dcm_files, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
                img = pydicom.dcmread(dcm_file).pixel_array
                img = img.astype(np.float32)
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = img.astype(np.uint8)
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
                if self.transform:
                    img = self.transform(img)
                series_images.append(img)

        while len(series_images) < self.max_images:
            series_images.append(torch.zeros_like(series_images[0]))

        all_images = torch.stack(series_images, dim=0)
        return study_id, all_images

# Load the trained model
model = MultiLevelUnet(in_channels=1, num_heads=25, num_classes_per_head=3).to(device)
model.load_state_dict(torch.load('multi_level_unet.pth'))
model.eval()

# Create the test dataset and dataloader
test_img_dir = '/home/adam/data/test_images'  # replace with your test image path
test_dataset = SpineTestDataset(test_img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the condition and level names
conditions = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']
levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

# Generate predictions
submission = []

with torch.no_grad():
    for study_id, images in tqdm(test_loader):
        images = images.view(-1, 1, 256, 256).to(device)
        outputs = model(images)
        outputs = torch.cat(outputs, dim=0)
        outputs = nn.functional.softmax(outputs, dim=1).cpu().numpy()

        # Format the predictions
        for i, condition in enumerate(conditions):
            for j, level in enumerate(levels):
                row_id = f"{study_id[0]}_{condition}_{level}"
                normal_mild = outputs[i * 5 + j][0]
                moderate = outputs[i * 5 + j][1]
                severe = outputs[i * 5 + j][2]
                submission.append([row_id, normal_mild, moderate, severe])

# Save the predictions to a CSV file
submission_df = pd.DataFrame(submission, columns=['row_id', 'normal_mild', 'moderate', 'severe'])
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
