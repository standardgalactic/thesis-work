# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tifffile
import skimage
import skimage.exposure
import matplotlib.pyplot as plt

img=tifffile.imread(r'C:\Users\Mark Zaidi\Documents\QuPath\CODEX\Breast cancer\reg010.ome.tiff')
#%%
for x in range(img.shape[0]):
    plt.imshow(skimage.exposure.adjust_gamma(img[x,:,:],gamma=0.1),cmap='gray')
    plt.title('Channel:' + str(x+1))
    plt.figure()

