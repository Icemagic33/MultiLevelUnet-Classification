import torch 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

# grabs the correspoding series descriptions
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            print("Failed on", s, k)


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
def display_images(images, title, max_images_per_row=4, save_path=None):
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row

    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5 * max_images_per_row, 1.5 * num_rows))

    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, image in enumerate(images):
        if idx < len(axes):
            ax = axes[idx]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)  # Close the plot to free up memory

# # Save all of the first patient's images 
# for i in im_list_dcm:
#     images = [x['dicom'].pixel_array for x in im_list_dcm[i]['images']]
#     description = " ".join(im_list_dcm[i]['description'])  # Assuming descriptions is a list
#     file_name = f"{i}.png"  # Use the study ID as the file name
#     display_images(images, description, save_path=f"{file_name}")


df_coor = pd.read_csv(f'{fd}/train_label_coordinates.csv')
print(df_coor.head(20))
''' study_id   series_id  instance_number                         condition  level           x           y
0    4003253   702807833                8             Spinal Canal Stenosis  L1/L2  322.831858  227.964602
1    4003253   702807833                8             Spinal Canal Stenosis  L2/L3  320.571429  295.714286
2    4003253   702807833                8             Spinal Canal Stenosis  L3/L4  323.030303  371.818182
3    4003253   702807833                8             Spinal Canal Stenosis  L4/L5  335.292035  427.327434
4    4003253   702807833                8             Spinal Canal Stenosis  L5/S1  353.415929  483.964602
5    4003253  1054713880                4  Right Neural Foraminal Narrowing  L4/L5  187.961759  251.839388
6    4003253  1054713880                4  Right Neural Foraminal Narrowing  L5/S1  198.240918  285.613767
7    4003253  1054713880                5  Right Neural Foraminal Narrowing  L3/L4  187.227533  210.722753
8    4003253  1054713880                6  Right Neural Foraminal Narrowing  L1/L2  194.569790  127.755258
9    4003253  1054713880                6  Right Neural Foraminal Narrowing  L2/L3  191.632887  165.934990
10   4003253  1054713880               11   Left Neural Foraminal Narrowing  L1/L2  196.070671  126.021201
11   4003253  1054713880               11   Left Neural Foraminal Narrowing  L4/L5  186.504472  251.592129
12   4003253  1054713880               11   Left Neural Foraminal Narrowing  L5/S1  197.100569  289.457306
13   4003253  1054713880               12   Left Neural Foraminal Narrowing  L2/L3  191.321555  170.120141
14   4003253  1054713880               12   Left Neural Foraminal Narrowing  L3/L4  187.878354  217.245081
15   4003253  2448190387                3        Left Subarticular Stenosis  L1/L2  179.126448  161.235521
16   4003253  2448190387                4       Right Subarticular Stenosis  L1/L2  145.288771  158.624642
17   4003253  2448190387               11        Left Subarticular Stenosis  L2/L3  180.979730  158.764479
18   4003253  2448190387               11       Right Subarticular Stenosis  L2/L3  145.900042  157.096466
19   4003253  2448190387               19        Left Subarticular Stenosis  L3/L4  176.037645  157.528958'''

def display_coor_on_img(c, i, title, save_path='only_severe_cases.png'):
    center_coordinates = (int(c['x']), int(c['y']))
    radius = 10
    color = (255, 0, 0)  # Red color in RGB

    # Normalize the DICOM image
    IMG = i['dicom'].pixel_array.astype(float)
    normalized = (IMG - IMG.min()) * (255.0 / (IMG.max() - IMG.min()))
    IMG_normalized = np.uint8(normalized)  # Convert to unsigned byte format

    # Create a PIL image from the numpy array in RGB mode
    pil_img = Image.fromarray(IMG_normalized)
    pil_img = pil_img.convert('RGB')  # Convert to RGB for color drawing
    draw = ImageDraw.Draw(pil_img)

    # Draw the circle on the image
    draw.ellipse((center_coordinates[0] - radius, center_coordinates[1] - radius, 
                  center_coordinates[0] + radius, center_coordinates[1] + radius), 
                 outline=color, width=2)

    # Save the image to a file
    pil_img.save(save_path)

    # Display the image with matplotlib
    plt.imshow(pil_img)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title(title)
    plt.show()

coor_entries = df_coor[df_coor['study_id'] == int(patient['study_id'])]

# Save the severe image in png
print("Saving an image for the severe cases for this patient")
for idc, c in coor_entries.iterrows():
    for i in im_list_dcm[str(c['series_id'])]['images']:
        if int(i['SOPInstanceUID']) == int(c['instance_number']):
            try:
                patient_severity = patient[
                    f"{c['condition'].lower().replace(' ', '_')}_{c['level'].lower().replace('/', '_')}"
                ]
            except Exception as e:
                patient_severity = "unknown severity"
            title = f"{i['SOPInstanceUID']} \n{c['level']}, {c['condition']}: {patient_severity} \n{c['x']}, {c['y']}"
            if patient_severity == 'Severe':
                display_coor_on_img(c, i, title)


