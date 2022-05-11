# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:35:36 2022

@author: Mark Zaidi

Script to automatically rename extracted .ome.tiffs. In this case, remove the ROIx_ prefix from each image, to match
what is being provided by in Kevin's list
"""

#%% Import libraries
import os
#%% declare variables
img_dir=r'E:\Img_storage\Histowiz\20220418_Priya_2ndBatch'
file_ext='.ome.tiff'
#%% First, check to see that the renaming operation will not yield 2 images with the same name
img_list_orig=[]
img_list_renamed=[]

for file in os.listdir(img_dir):
    if file.endswith(file_ext):
        img_list_orig.append(file)
        new_name=file.split('_',1)[1]
        img_list_renamed.append(new_name)
unique_orig=len(img_list_orig)
unique_renamed=len(list(set(img_list_renamed)))
if (unique_orig != unique_renamed):
    raise Exception('Number of unique renamed images does not match original number of images. Check to see you are avoiding any duplicate names after renaming')
#%% Perform the renaming once error checking is passed
img_list_orig=[]
img_list_renamed=[]

for file in os.listdir(img_dir):
    if file.endswith(file_ext):
        img_list_orig.append(file)
        new_name=file.split('_',1)[1]
        img_list_renamed.append(new_name)
        orig_path=os.path.join(img_dir, file)
        new_path=os.path.join(img_dir, new_name)
        #Rename file here:
        os.rename(orig_path,new_path) #Note, this should still error if new_path matches an existing orig_path file