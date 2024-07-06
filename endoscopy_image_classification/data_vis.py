import matplotlib.pyplot as plt
from PIL import Image
import os


# Define the labels and directory
labels = ['Diverticulosis', 'Neoplasm', 'Peritonitis', 'Ureters']
DIR = '/home/adam/endoscopy_data'
IMG_SIZE = 224


# Load one image from each class
fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Adjust figure size

for i, label in enumerate(labels):
    folderPath = os.path.join(DIR, label)
    img_name = os.listdir(folderPath)[0]  # Take the first image in the directory
    img_path = os.path.join(folderPath, img_name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    axes[i].imshow(img)
    axes[i].set_title(label, fontsize=16)  # Increase the font size of the labels
    axes[i].axis('off')

# Save the visualization
plt.subplots_adjust(wspace=0.1, hspace=0)  # Adjust space between subplots
plt.savefig('cleaned_images_visualization.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
