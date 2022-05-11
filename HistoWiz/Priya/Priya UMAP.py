# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:49:26 2022

@author: Mark Zaidi

I WANT TO ADD A FEATURE WHERE YOU GIVE IT AN IMAGE, AND RETURNS THE TOP N MOST SIMILIAR IMAGES
BUT DON'T USE EUCLIDEAN DISTANCE, LOOK FOR SOME HYPERDIMENSIONAL THING LIKE ARC SIN
"""

#%% load libraries
import pandas
import pandas as pd

import math
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import stats
import time
from scipy.stats import spearmanr
import winsound
import umap

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220124_Sandra\processed_cell_measurements.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220124_Sandra\figures\UMAP\Sandra_no_DRUG-AIN'

data=pandas.read_csv(csv_path)
#force str type on Patient column to avoid weird discretization issues
# data=data.astype({'Patient': 'str'})
# a_unique_patients=data['Patient'].unique()
col_names=data.columns

#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data["Patient"])
# slide_IDs.sort()
data=data[~data['Sandra_4'].str.contains("Drug-AIN|OMIT|Missing")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively
measures_to_drop=['DNA193','DNA191'] #remove these from any percent positive or intensity comparisons


seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2

#Priya
#measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(150)_150Sm-PD-L1: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean']                                      
#Sandra
measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean','Pt(195)_195Pt-PLAT: Cell: Mean']                                      
measurements_of_interest.remove('Pt(195)_195Pt-PLAT: Cell: Mean')                               


#drop specific measurements
# for measure in measures_to_drop:
#     measurements_of_interest[:] = [x for x in measurements_of_interest if measure not in x]

plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
#data.sort_values(param_Parent,inplace=True)

#Drop rows containing a specific value in the target_variable
data=data[~data['Sandra_4'].str.contains('OMIT|Missing')]
#%%Get shortened names of measurements
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]
#%% Remove rows containing NAN
data=data.dropna(subset=measurements_of_interest)
#%% set up UMAP
startTime = datetime.now()
reducer = umap.UMAP(random_state=seed,low_memory=False)
reduced_data = data[measurements_of_interest].dropna().values #only perform clustering on columns listed in measurements_of_interest

scaled_data = StandardScaler().fit_transform(reduced_data) #convert intensities into normalized z scores
#overwrite original data as the scaled data, if planning to visualize
data[measurements_of_interest]=scaled_data
# calculate the UMAP
#compute UMAP

embedding = reducer.fit_transform(scaled_data)
#append to original dataframe
data['UMAP_X']=embedding[:,0]
data['UMAP_Y']=embedding[:,1]
print(datetime.now() - startTime)
#%% Plot a subplot of every IHC marker intensity as specified in measurements_of_interest
plt.close('all')
print ('Plotting per-IHC marker plot')
nrows=4
ncols=7
pt_size=10000/len(data)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(ncols*2.5, nrows*2))
#linearize ax into axz so that you have a variable you can iterate over
axs=ax.ravel()
for curr_ax,curr_measure,curr_short_measure in zip(axs, measurements_of_interest,measure_short_for_fname):
    curr_ax.set_facecolor((0, 0, 0))
    curr_ax.set_xticks([])
    curr_ax.set_yticks([])
    #curr_ax.set_title(curr_short_measure + ' ',fontsize=12,y=1,pad=-15,loc='right',color='white')
    curr_ax.set_title(curr_short_measure,fontsize=12)
    h=sns.scatterplot(data=data,ax=curr_ax, x="UMAP_X", y="UMAP_Y",hue=curr_measure,linewidth=0,s=pt_size,palette='viridis',hue_norm=(-1,2),legend=False)
    curr_ax.set_aspect('equal')

#Hide unused axis
if nrows*ncols>len(measurements_of_interest):
    num_subplots_to_hide=nrows*ncols-len(measurements_of_interest)    
    for to_hide in range(num_subplots_to_hide):
        #Yeah, this math down here doesn't make sense to me either. But it works
        axs[len(axs)-to_hide-1].set_visible(False)
        
plt.tight_layout()
plt.pause(2)

plt.savefig(figpath + '\\UMAP_per_marker.png',dpi=800,pad_inches=0.1,bbox_inches='tight')


#%%Plot scatterplot of KR
print('Plotting KR-colored plot')
plt.close('all')

coloring='KR'

plt.gca().set_facecolor((0, 0, 0))
h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.3)

#h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
plt.gca().set_aspect('equal')
#plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
plt.title(coloring)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)
plt.tight_layout()
plt.savefig(figpath + '\\' + coloring + '.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%%Plot scatterplot of code
print('Plotting Code-colored plot')
plt.close('all')

coloring='Sandra_4'

plt.gca().set_facecolor((0, 0, 0))
h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.3)

#h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
plt.gca().set_aspect('equal')
#plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
plt.title(coloring)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)
plt.tight_layout()
plt.legend(title='Group')

plt.savefig(figpath + '\\' + coloring + '.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%%Plot Scatterplot of Code_full
print('Plotting Code-colored plot')
plt.close('all')

coloring='Code_full'

plt.gca().set_facecolor((0, 0, 0))
h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.3)

#h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
plt.gca().set_aspect('equal')
#plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
plt.title(coloring)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)
plt.tight_layout()
plt.savefig(figpath + '\\' + coloring + '.png',dpi=800,pad_inches=0.1,bbox_inches='tight')


#%%Plot scatterplot of IDH_status
# plt.close('all')

# coloring='IDH_status'

# plt.gca().set_facecolor((0, 0, 0))
# h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.7)

# #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
# plt.gca().set_aspect('equal')
# #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
# plt.title(coloring)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.pause(1)
# plt.tight_layout()
# plt.savefig(figpath + '\\' + coloring + '.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%%Plot scatterplot of primary_recurrent
# plt.close('all')

# coloring='primary_recurrent'

# plt.gca().set_facecolor((0, 0, 0))
# h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.7)

# #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
# plt.gca().set_aspect('equal')
# #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
# plt.title(coloring)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.pause(1)
# plt.tight_layout()
# plt.savefig(figpath + '\\' + coloring + '.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%%Plot scatterplot of Cell area
# plt.close('all')

# coloring='Cell: Area µm^2'

# plt.gca().set_facecolor((0, 0, 0))
# h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.7,palette='viridis',hue_norm=(0,300),legend=False)

# #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
# plt.gca().set_aspect('equal')
# #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
# plt.title(coloring)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.pause(1)
# plt.tight_layout()
# plt.savefig(figpath + '\\CellArea.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%%Plot scatterplot of CXCR4
# plt.close('all')

# coloring='Lu(175)_175Lu-CXCR4: Cell: Mean'

# plt.gca().set_facecolor((0, 0, 0))
# h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.7,palette='viridis',hue_norm=(-1,2),legend=False)

# #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
# plt.gca().set_aspect('equal')
# #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
# plt.title(coloring)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.pause(1)
# plt.tight_layout()
# plt.savefig(figpath + '\\CXCR4.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%%Plot scatterplot of HK2
# plt.close('all')

# coloring='Dy(163)_163Dy-HK2: Cell: Mean'

# plt.gca().set_facecolor((0, 0, 0))
# h=sns.scatterplot(data=data, x="UMAP_X", y="UMAP_Y",hue=coloring,linewidth=0,s=0.7,palette='viridis',hue_norm=(-1,2),legend=False)

# #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
# plt.gca().set_aspect('equal')
# #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
# plt.title(coloring)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.pause(1)
# plt.tight_layout()
# plt.savefig(figpath + '\\HK2.png',dpi=800,pad_inches=0.1,bbox_inches='tight')