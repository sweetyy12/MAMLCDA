
from __future__ import print_function
import glob
import os


from PIL import Image

path_to_images = 'train5/pos/'
#path_to_images = 'train5/neg/'

all_images = glob.glob(path_to_images + '*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)  # read
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)
