# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:07:31 2021
- apply a despeckle filter to select IMC channels. Current despeckle filter will likely be a median filter with a 3x3 kernel size
@author: Mark Zaidi
"""
#%% load libraries

import tifffile
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
#%% declare constants
fdir=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Aug 16 2021 - updated panel\ome_tiffs'
output_folder='despeckled'
output_path=os.path.join(fdir,output_folder)
#fdir=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Aug 16 2021 - updated panel\ome_tiffs'
fext='.tiff'
ch_to_process=['Nd(143)_143Nd-GFAP','Nd(145)_145Nd-CD31','Nd(146)_146Nd-Nestin','Nd(150)_150Nd-SOX2']
#%% Get list of ome.tif to read and rewrite
filelist=[]
for file in os.listdir(fdir):
    if file.endswith(fext):
        filelist.append(os.path.join(fdir, file))
#%%Create folder to save new images in
if not os.path.exists(output_path):
    os.makedirs(output_path)
#%% begin batch despeckling
#########put for loop on filelist here
for curr_file in filelist:
    #create file path for writing to
    output_impath=os.path.join(output_path,curr_file.rsplit('\\',maxsplit=1)[1])
    # read image
    curr_img=tifffile.imread(curr_file)
    # read metadata
    curr_metadata=tifffile.xml2dict(tifffile.TiffFile(curr_file).ome_metadata)
    # derive channel names from metadata
    ch_names=[]
    for i in range(len(curr_metadata['OME']['Image']['Pixels']['Channel'])):
        ch_names.append(curr_metadata['OME']['Image']['Pixels']['Channel'][i]['Name'])
    ######put for loop on ch_to_process here
    for curr_ch_name in ch_to_process:
    #curr_ch_name=ch_to_process[0]
        #extract channel to process based on channel name
        curr_ch=curr_img[ch_names.index(curr_ch_name),:,:]
        #apply the 3x3 median filter
        curr_ch=ndimage.median_filter(curr_ch,size=3)
        #insert channel back into whole image
        curr_img[ch_names.index(curr_ch_name),:,:]=curr_ch
        
    tags_to_add={'Channel':curr_metadata['OME']['Image']['Pixels']['Channel'],
                     'PhysicalSizeX':curr_metadata['OME']['Image']['Pixels']['PhysicalSizeX'],
                     'PhysicalSizeY':curr_metadata['OME']['Image']['Pixels']['PhysicalSizeY']}
    with tifffile.TiffWriter(output_impath) as tif:
        tif.write(curr_img,compression='ADOBE_DEFLATE',metadata=tags_to_add)









#%% preview channel before and after
# curr_ch=curr_img[ch_names.index(ch_to_process[2]),:,:]
# #before img
# plt.subplot(231)
# plt.imshow(curr_ch,vmin=0,vmax=50,cmap='gray')
# plt.title("Mean: "+ curr_ch.mean().round(3).astype('str') + ", Std: "+ curr_ch.std().round(3).astype('str') + ", Min: "+ curr_ch.min().round(3).astype('str') + ", Max: "+ curr_ch.max().round(3).astype('str'))
# #before hist
# plt.subplot(232)
# plt.hist(curr_ch.flatten(),bins=range(1000), edgecolor='black',linewidth=1)
# plt.gca().set_ylim([0,1000]),plt.gca().set_xlim([0,500])
# plt.title("Original Nestin")

# plt.subplot(233)
# plt.hist(curr_ch.flatten(),bins=range(1000), edgecolor='black',linewidth=1)
# plt.gca().set_ylim([0,30]),plt.gca().set_xlim([300,700])
# #after img


# ### median filter happens on line below ###
# curr_ch=ndimage.median_filter(curr_ch,size=3)

# plt.subplot(234)
# plt.imshow(curr_ch,vmin=0,vmax=50,cmap='gray')
# plt.title("Mean: "+ curr_ch.mean().round(3).astype('str') + ", Std: "+ curr_ch.std().round(3).astype('str') + ", Min: "+ curr_ch.min().round(3).astype('str') + ", Max: "+ curr_ch.max().round(3).astype('str'))

# #before hist
# plt.subplot(235)
# plt.hist(curr_ch.flatten(),bins=range(1000), edgecolor='black',linewidth=1)
# plt.gca().set_ylim([0,1000]),plt.gca().set_xlim([0,500])
# plt.title("3x3 Median Filtered Nestin")


# plt.subplot(236)
# plt.hist(curr_ch.flatten(),bins=range(1000), edgecolor='black',linewidth=1)
# plt.gca().set_ylim([0,30]),plt.gca().set_xlim([300,700])
#%% TO DO
#bring "preview channel before and after" to end as a commented out block
#run for all 3 image sets as a batch