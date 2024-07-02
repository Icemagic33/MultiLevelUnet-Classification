import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# File location
fd = "/content/drive/MyDrive/rsna-2024-lumbar-spine-degenerative-classification"
train = pd.read_csv(f'{fd}/train.csv')


# Check how the training files are organized
print("Total Cases: ", len(train))
print(train.columns)


# Visualize the distribution of different diagnostic categories
figure, axis = plt.subplots(1, 3, figsize=(20, 5))

# three main diagnostic categories
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    # inlucde only the filters that are related to the current diagnostic category
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))

    dff = train[diagnosis]
    with warnings.catch_warnings():
        # calculate the frequency of each diagnostic severity for each condition within the current category
        # and transpose the result for plotting
        warnings.simplefilter(action='ignore', category=FutureWarning)
        value_counts = dff.apply(pd.value_counts).fillna(0).T
    # plot the value counts as a stacked bar chart on the corresponding subplot
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')

# List out all of the Studies we have on patients.
part_1 = os.listdir(f'{fd}/train_images')
# Filter out any system files that might be present (e.g., '.DS_Store')
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))

# Load metadata from the CSV file
df_meta_f = pd.read_csv(f'{fd}/train_series_descriptions.csv')

# Create a list of tuples containing study IDs and their corresponding folder paths
p1 = [(x, f"{fd}/train_images/{x}") for x in part_1]

# Initialize a dictionary to hold metadata for each study
meta_obj = {p[0]: {'folder_path': p[1], 'SeriesInstanceUIDs': []} for p in p1}

# Iterate over each study in the meta_obj dictionary
for m in meta_obj:
    # List all directories (series) within the study folder, filtering out system files
    meta_obj[m]['SeriesInstanceUIDs'] = list(
        filter(lambda x: x.find('.DS') == -1,
               os.listdir(meta_obj[m]['folder_path'])
               )
    )

# Iterate over each study in the meta_obj dictionary using tqdm for progress bar
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        # Initialize the 'SeriesDescriptions' list if not already present
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            # Append the series description to the 'SeriesDescriptions' list
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) &
                          (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            # Print an error message if the series description cannot be found
            print("Failed on", s, k)

# Access, retrieve, and display the metadata for the second study
print(meta_obj[list(meta_obj.keys())[1]])

# Retrieve data for the second patient in the train DataFrame
patient = train.iloc[1]

# Retrieve the metadata for the study ID associated with the second patient in the train DataFrame from the meta_obj dictionary
ptobj = meta_obj[str(patient['study_id'])]
print(ptobj)

# Get data into the format
"""
im_list_dcm = {
    '{SeriesInstanceUID}': {
        'images': [
            {'SOPInstanceUID': ...,
             'dicom': PyDicom object
            },
            ...,
        ],
        'description': # SeriesDescription
    },
    ...
}
"""

# Initialize the dictionary to hold the DICOM images and series descriptions
im_list_dcm = {}

# Iterate over each SeriesInstanceUID in the patient's study
for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
    # Initialize the dictionary for each series with an empty list for images and the series description
    im_list_dcm[i] = {'images': [],
                      'description': ptobj['SeriesDescriptions'][idx]}

    # Get the list of all DICOM files in the current series directory
    images = glob.glob(
        f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")

    # Iterate over the sorted list of DICOM files
    for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
        # Append the SOPInstanceUID and the PyDicom object to the list of images for the current series
        im_list_dcm[i]['images'].append({
            'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''),
            'dicom': pydicom.dcmread(j)})


# Function to display images
def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row -
                1) // max_images_per_row  # Ceiling division

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(
        5 * max_images_per_row, 5 * num_rows))

    # Flatten axes array for easier looping if there are multiple rows
    axes = axes.flatten()

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        # Assuming grayscale for simplicity, change cmap as needed
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


# Function to load DICOM images from a given directory
def load_dicom_images_from_dir(directory):
    dicom_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                dicom = pydicom.dcmread(dicom_path)
                dicom_images.append(dicom.pixel_array)
    return dicom_images


# Display
directory = f'{fd}/train_images/4003253/702807833/'
images = load_dicom_images_from_dir(directory)
display_images(images, title="Sample DICOM Images")


# Base directory for images
base_dir = f'{fd}/train_images/'

# Load the CSV files containing training data and label coordinates
train_df = pd.read_csv(f'{fd}/train.csv')
train_coords_df = pd.read_csv(f'{fd}/train_label_coordinates.csv')

# Display the first few rows of the training data DataFrame to verify the data
print(train_df.head())
# Display the first few rows of the train_coords_df DataFrame to verify the data
print(train_coords_df.head(20))


# Function to get the lengths of series (number of images in each series)
def get_series_lengths(base_dir, train_df, coords_df):
    lengths = []

    # Iterate over each study in train_df
    for idx in range(len(train_df)):
        study_id = train_df.iloc[idx]['study_id']

        # Ensure the study_id exists in coords_df
        if study_id in coords_df['study_id'].values:
            # Get the corresponding series_id and series directory
            series_id = coords_df[coords_df['study_id']
                                  == study_id].iloc[0]['series_id']
            series_dir = os.path.join(base_dir, str(study_id), str(series_id))
            # Count the number of DICOM images in the series directory
            num_images = len([name for name in os.listdir(
                series_dir) if name.endswith('.dcm')])
            # Append the count to the lengths list
            lengths.append(num_images)
        else:
            # Print a warning message if study_id is not found in coords_df
            print(f"Study ID {study_id} not found in coords_df")

    return lengths


# Calculate series lengths
series_lengths = get_series_lengths(base_dir, train_df, train_coords_df)

# Find the shortest and longest series
shortest_length = min(series_lengths)
longest_length = max(series_lengths)

# Print the results (debugging)
print(f'Shortest number of images in a series: {shortest_length}')
print(f'Longest number of images in a series: {longest_length}')


# Convert labels to numerical format
condition_mapping = {
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
}

# Set image size
image_size = (256, 256)

# Transform to preprocess images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size),
    transforms.Normalize((0.5,), (0.5,))
])


# Convert labels to numerical format
condition_mapping = {
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
}

# Transform to preprocess images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize((0.5,), (0.5,))
])


# Function to load DICOM images from a given directory
def load_dicom_images_from_dir(directory):
    dicom_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                dicom = pydicom.dcmread(dicom_path)
                dicom_images.append(dicom.pixel_array)
    return dicom_images


# Preprocess images
def preprocess_images(images, max_length=176):
    processed_images = [transform(image) for image in images]
    if len(processed_images) < max_length:
        # Pad with zeros if less than max_length
        padding = [torch.zeros_like(processed_images[0])
                   for _ in range(max_length - len(processed_images))]
        processed_images.extend(padding)
    elif len(processed_images) > max_length:
        # Truncate if more than max_length
        processed_images = processed_images[:max_length]
    return torch.stack(processed_images)


# Function to get labels for a patient
def get_labels(patient_row):
    labels = []
    for col in patient_row.index:
        if col != 'study_id':
            labels.append(condition_mapping[patient_row[col]])
    return labels


# Load, preprocess, and structure the data correctly for further analysis or training

# Create a dictionary to hold images and labels for each study
im_list_dcm = {}

for idx, row in train_df.iterrows():
    study_id = row['study_id']
    patient_labels = get_labels(row)

    # Find all series for this study in the coords DataFrame
    series_ids = train_coords_df[train_coords_df['study_id']
                                 == study_id]['series_id'].unique()

    im_list_dcm[study_id] = {'images': [],
                             'labels': patient_labels, 'series': {}}

    for series_id in series_ids:
        series_dir = os.path.join(base_dir, str(study_id), str(series_id))
        images = load_dicom_images_from_dir(series_dir)

        # Only preprocess and add if there are images
        if images:
            preprocessed_images = preprocess_images(images)
            im_list_dcm[study_id]['series'][series_id] = preprocessed_images

# ------------------------------DEBUGGING-------------------------------------------
# Print a summary
for study_id, data in im_list_dcm.items():
    print(f"Study ID: {study_id}")
    print(f"Labels: {data['labels']}")
    for series_id, images in data['series'].items():
        print(f"  Series ID: {series_id}, Number of images: {len(images)}")
# ------------------------------DEBUGGING-------------------------------------------

# Save the processed data for later use
torch.save(im_list_dcm, 'processed_data.pt')


# SpineDataset class for loading, preprocessing, and serving batches of spinal imaging data with corresponding labels.
class SpineDataset(Dataset):
    def __init__(self, im_list_dcm):  # Initializes the dataset object
        self.im_list_dcm = im_list_dcm
        self.study_ids = list(im_list_dcm.keys())

    def __len__(self):  # Returns the number of studies in the dataset
        return len(self.study_ids)

    def __getitem__(self, idx):
        # Get the study ID for the given index
        study_id = self.study_ids[idx]
        # Retrieve the data for this study from the im_list_dcm dictionary
        data = self.im_list_dcm[study_id]

        # Initialize a list to hold images from all series for this study
        series_images = []
        # Iterate over each series in this study and collect the preprocessed images
        for series_id, images in data['series'].items():
            series_images.append(images)

        # Check if there are any images collected, raise an error if none found
        if len(series_images) == 0:
            raise RuntimeError(
                f"No valid images found for study ID {study_id}")

        # Concatenate all series images into one tensor along the first dimension
        all_images = torch.cat(series_images, dim=0)

        # Convert the labels to a tensor
        labels = torch.tensor(data['labels'], dtype=torch.long)

        # Return the combined images and labels as a tuple
        return all_images, labels


# Create dataset and dataloader
im_list_dcm = torch.load('processed_data.pt')  # Load preprocessed data
train_dataset = SpineDataset(im_list_dcm)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ------------------------------DEBUGGING-------------------------------------------
# Print a batch of data to verify
for batch in train_loader:
    images, labels = batch
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    break
# ------------------------------DEBUGGING-------------------------------------------
