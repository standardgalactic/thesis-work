# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:25:03 2021

@author: Mark Zaidi
Trying out and installing Napari
"""
#%% import packages
import napari
from skimage.data import astronaut
import tifffile

#%% Create viewer, display image
viewer = napari.view_image(astronaut(), rgb=True)
#napari.Viewer.close(viewer)
#%% Read IMC image
im_path=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Feb 2, 2021_SOA-3 txt files\ome tiffs\Feb 2, 2021_SOA-3_ROI_001_1.ome.tiff'
img=tifffile.imread(im_path)
txt_filepath=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Feb 2, 2021_SOA-3 txt files\Feb 2, 2021_SOA-3_ROI_001_1.txt'
tif=tifffile.TiffFile(im_path)

#%% Get a list of IMC channel names
metadata_dict=tifffile.xml2dict(tif.ome_metadata)
ch_names=[]
#viewer=napari.view_image(img,channel_axis=0)
for i in range(len(metadata_dict['OME']['Image']['Pixels']['Channel'])):
    print(i)
    ch_names.append(metadata_dict['OME']['Image']['Pixels']['Channel'][i]['Name'])
#%% show IMC image
viewer=napari.view_image(img,channel_axis=0,name=ch_names)
    