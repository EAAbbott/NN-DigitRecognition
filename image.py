# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:12:20 2018

@author: eansl
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import numpy as np


np.set_printoptions(precision=2, suppress=True, linewidth=100,
                    threshold=500)

img = (255 - ndimage.imread("usr_img/7.png", flatten=True)) / 255
#print(img.shape)
#
#curr_width = img.shape[1]
#curr_height = img.shape[0]
#new_width = 28
#new_height = 28
#
#width_shrink = int(curr_width / new_width)
#height_shrink = int(curr_height / new_height)
#
#print(curr_width / width_shrink)
#print(curr_height / height_shrink)
print(img.shape)


