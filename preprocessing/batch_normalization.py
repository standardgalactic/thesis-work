# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:30:20 2021

@author: Mark Zaidi

The purpose of this script is to normalize intensities of a batch of images, such that the intensities
lie on a scale of 0-1. Lack of normalization is causing issues when calculating Haralick texture features on IMC data.

This script will be subdivided into the following steps, and may be further split into one or more functions

- obtain cumulative histograms. Iterate over each image in a folder, and for each channel, collect the histogram.
Combine histograms for all images in the dataset (folder). Output will be an m*n array, where m is a channel, and n is the histogram bin.
values of the array are the counts of pixels for given channel m that fall in bin n. Input parameters will be a folder of images, bin 
parameters, and a filename to write the array to

- Apply normalization. Rescale data to fall in range of 0-1 (or a specified percentile). Input is a folder and cumulative histogram array.
Output is a folder containing normalized images

Note, normalization doesn't mean all the data must lie in 0-1. Rather, the top nth percentile will be
scaled to 1 and the bottom pth percentile will be scaled to 0. You can still have values exceeding 0-1, particularly
the global minimum and maximums.

"""
#%% load libraries
import tifffile
import os
import numpy as np
import matplotlib.pyplot as plt

#%% declare constants
fdir=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Aug 16 2021 - updated panel\ome_tiffs\norm_test'
output_folder='batch_normalized'
output_path=os.path.join(fdir,output_folder)
#fdir=r'C:\Users\Mark Zaidi\Documents\QuPath\images\IMC\Aug 16 2021 - updated panel\ome_tiffs'
fext='.tiff'
bins=np.linspace(0,100000,1000001) #basically, have a range and increment so large that you're 100% certain all pixel values will fall somewhere in it
low_pct=0.01 #lower percentile range for normalization
high_pct=0.99 #higher percentile range for normalization
#%% Get list of ome.tif to read and rewrite
filelist=[]
for file in os.listdir(fdir):
    if file.endswith(fext):
        filelist.append(os.path.join(fdir, file))
#%%Create folder to save new images in
if not os.path.exists(output_path):
    os.makedirs(output_path)
#%% Get a list of IMC channel names
#get channel names from first image in dataset
metadata_dict=tifffile.xml2dict(tifffile.TiffFile(filelist[0]).ome_metadata)
ch_names=[]
for i in range(len(metadata_dict['OME']['Image']['Pixels']['Channel'])):
    ch_names.append(metadata_dict['OME']['Image']['Pixels']['Channel'][i]['Name'])

#%% obtain histogram
dataset_hists=[]
dataset_min=[]
dataset_max=[]
#for each image:
for curr_file in filelist:

    curr_img=tifffile.imread(curr_file)
    curr_img_hist=[]
    curr_img_range=[]
    #curr_img_edge=[]
    #for each channel
    for i in range(len(curr_img)):
        #compute histogram using predefined bins, and append to a list
        histogram, bin_edges = np.histogram(curr_img[i,:,:], bins=bins)
        curr_img_hist.append(histogram)
        #curr_img_edge.append(bin_edges)
        #plt.bar(bins[:1000],curr_img_hist[i][:1000],width=0.1,fill=False,edgecolor=np.random.randint(0,255,size=3)/255)

    #curr_img_range=np.concatenate((curr_img.min(axis=(1,2)),curr_img.max(axis=(1,2))),axis=2)
    #calculate min and max on a per-channel basis
    curr_img_range=np.vstack((curr_img.min(axis=(1,2)),curr_img.max(axis=(1,2)))).transpose()
    #append histograms, min, and max to a list covering all images in dataset.
    dataset_hists.append(curr_img_hist)
    dataset_min.append(curr_img_range[:,0])
    dataset_max.append(curr_img_range[:,1])
#%% Generate master measurements, summarizing the dataset
#aggregate results by summing histogram, finding global channel-wise min and max values.
#only use for min and max is to see if bins limits are okay-ish, and to verify that calling the 0th and 100th percentile
#counts from the cumulative histogram function from master_hist equal master_min and master_max, respectively
master_hist=np.sum(dataset_hists,0)
master_max=np.max(dataset_max,0)
master_min=np.min(dataset_min,0)
total_pixels=total_pixels=np.sum(master_hist,axis=1) #all values should be the same (total pixels in image). If not, you done diddly screwed up
master_hist_prob=master_hist/total_pixels[:,None]#probability of pixel to fall in given bin
master_cum=np.cumsum(master_hist_prob,axis=1)#not as dirty as it sounds
#%% TO DO: refer to old MATLAB code and see how bottom nth and top pth percentiles can be taken from master_hist.

#Understanding from MATLAB: take master_hist, divide each element by total number of pixels [DONE]
#run cumsum on the normalized values. First element should be ~very low, last should be 1. This is your CDF [DONE]
#find the first case where cdf is greater than OR EQUAL to 0.99 (99st percentile). The index of that will be used
#to find the corresponding `bins` value of intensities
#find the first case where cdf is greater than 0.01 (1st percentile). repeat subsequent steps
#find first occurence in channel 0's PDF is greater than 0.01
#%%Identify upper and lower limits to equalize channels based on percentiles
limit_low=[]
limit_high=[]
for p in range(len(master_cum)):
    #find first value in CDF that meets lower percentile
    low_first_occ=next(i for i in master_cum[p,:] if i > low_pct)
    #find the index of that first occurence, and identify the corresponding bin of that index
    limit_low.append(bins[master_cum[p,:].tolist().index(low_first_occ)])
    
    #same as above, but for the upper limit
    high_first_occ=next(i for i in master_cum[p,:] if i >= high_pct)
    limit_high.append(bins[master_cum[p,:].tolist().index(high_first_occ)])

#%% Apply normalization to images
for f in filelist:
    #read image
    current_image=tifffile.imread(f)
    #get file name for downstream writing
    filename=f.rsplit(sep='\\',maxsplit=1)[1]
    #imnormalize formula in MATLAB: norm_img(:,:,i) = (input_img(:,:,i)-minmax(2,i))/(minmax(1,i)-minmax(2,i));
    #create empty numpy array of same shape and dtype as input image
    normalized_image=np.empty(current_image.shape,dtype=current_image.dtype)
    #for each channel:
    for i in range(len(current_image)):
        #normalize channel i to upper and lower limits defined in limit_high and limit_low, respectively
        #clip is used to set any negative values to 0
        normalized_image[i,:,:]=((current_image[i,:,:]-limit_low[i])/((limit_high[i]-limit_low[i]))).clip(min=0)
    
    #Obtain metadata from original image
    metadata=tifffile.xml2dict(tifffile.TiffFile(f).ome_metadata)
    #Specify which tags to carry over from original image
    tags_to_add={'Channel':metadata['OME']['Image']['Pixels']['Channel'],
                 'PhysicalSizeX':metadata['OME']['Image']['Pixels']['PhysicalSizeX'],
                 'PhysicalSizeY':metadata['OME']['Image']['Pixels']['PhysicalSizeY']}
    with tifffile.TiffWriter(os.path.join(output_path,filename)) as tif:
        tif.write(normalized_image,compression='ADOBE_DEFLATE',metadata=tags_to_add)









