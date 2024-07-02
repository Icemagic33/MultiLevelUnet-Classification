import torch 
import pandas as pd
import matplotlib.pyplot as plt
# import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fd = '/home/adam/data'
train = pd.read_csv(f'{fd}/train.csv')

# print(train)
# print("Total Cases: ", len(train))
# print(train.columns)


# Plot data distribution
figure, axis = plt.subplots(1, 3, figsize=(20, 5))
# Iterate through the diagnostic categories and plot data
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    # Filter the columns related to the current diagnostic category
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))
    # Filter the DataFrame to include only the relevant columns
    dff = train[diagnosis]
    # Use a context manager to suppress future warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # Count values, transpose, and fill missing values with zero
        value_counts = dff.apply(pd.value_counts).fillna(0).T
    
    # Plot a stacked bar chart for the current diagnostic category
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')

# # Save the plot(chart) as png image 
# plt.tight_layout() # Set the layout to be tight (optional but often improves layout issues)
# plt.savefig('diagnostic_distribution.png') # Save the figure to a file
# plt.show() # Display the plot in the script if you want to see it while running the script

# List out all of the Studies we have on patients.
part_1 = os.listdir(f'{fd}/train_images')
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))
df_meta_f = pd.read_csv(f'{fd}/train_series_descriptions.csv')

p1 = [(x, f"{fd}/train_images/{x}") for x in part_1]
meta_obj = { p[0]: { 'folder_path': p[1], 
                    'SeriesInstanceUIDs': [] 
                   } 
            for p in p1 }

for m in meta_obj:
    meta_obj[m]['SeriesInstanceUIDs'] = list(
        filter(lambda x: x.find('.DS') == -1, 
               os.listdir(meta_obj[m]['folder_path'])
              )
    )

# # grabs the correspoding series descriptions
# for k in tqdm(meta_obj):
#     for s in meta_obj[k]['SeriesInstanceUIDs']:
#         if 'SeriesDescriptions' not in meta_obj[k]:
#             meta_obj[k]['SeriesDescriptions'] = []
#         try:
#             meta_obj[k]['SeriesDescriptions'].append(
#                 df_meta_f[(df_meta_f['study_id'] == int(k)) & 
#                 (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
#         except:
#             print("Failed on", s, k)


meta_obj_check = meta_obj[list(meta_obj.keys())[1]]
# print(meta_obj_check)
''' {'folder_path': '/home/adam/data/train_images/3824003946', 
	'SeriesInstanceUIDs': ['2295292164', '1823339975', '3261324781', '3511463550'], 
	'SeriesDescriptions': ['Sagittal T2/STIR', 'Axial T2', 'Axial T2', 'Sagittal T1']} '''


patient = train.iloc[1]
ptobj = meta_obj[str(patient['study_id'])]
# print(ptobj)
''' {'folder_path': '/home/adam/data/train_images/4646740', 
	'SeriesInstanceUIDs': ['3201256954', '3666319702', '3486248476']}'''


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
im_list_dcm = {}
for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
    im_list_dcm[i] = {'images': [], 'description': ptobj['SeriesDescriptions'][idx]}
    images = glob.glob(f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")
    for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
        im_list_dcm[i]['images'].append({
            'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''), 
            'dicom': pydicom.dcmread(j) })


# Function to display images
def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))
    
    # Flatten axes array for easier looping if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for consistency

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')  # Assuming grayscale for simplicity, change cmap as needed
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()