# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:53:06 2021

@author: Mark Zaidi
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
from statannot import add_stat_annotation
import time
from scipy.stats import spearmanr
import winsound
#%% Read data
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\14541\2021_11_04_data_visualization\Pavi-desktop_GAD67Detection_measurements.csv')
col_names=data.columns
data_orig=data
data[['Slide','second_slide_name']] = data.Image.str.split('_',1,expand=True)
data[['Mouse','Layer']] = data.Slide.str.split('-',expand=True)
data['Mouse']=data['Mouse'].str.upper()
#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data_orig["Parent"])
# slide_IDs.sort()
# data=data_orig[data_orig['Parent'].str.contains("PathAnnotationObject")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'

figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\14541\2021_11_04_data_visualization\figures'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2
#%% create scatterplots
slides=data['Mouse'].unique()
hueorder=['Nucleus GFP- neurons','Nucleus GFP- neurons: NeuN+','Nucleus GFP+ neurons: NeuN+']
for slide in slides:
#slide=slides[0]
    curr_data=data[data['Mouse'].str.contains(slide,case=False,regex=False)]
    #curr_data.sort_values('Class',inplace=True,ascending=False)

    
    plt.close('all')
    

    plt.gca().set_facecolor((0, 0, 0))
    #h=sns.scatterplot(data=data, x="Opal 620: Nucleus: Mean", y="Opal 520: Cell: Mean",hue=coloring,linewidth=0,s=0.7,palette='viridis',hue_norm=(0,5),legend=False)
    #h=sns.scatterplot(data=curr_data, x="Opal 570: Cell: Mean", y="Opal 520: Cell: Mean",alpha=0.2,linewidth=0,s=3,hue='Class',hue_order=hueorder,palette='bright')
    h=sns.scatterplot(data=curr_data, x="Opal 570: Cell: Mean", y="Opal 520: Cell: Mean",alpha=0.2,linewidth=0,s=3,hue='Class',palette='bright')

    plt.xlim(0,255)
    plt.ylim(0,255)
    plt.legend(loc='upper right')
    plt.gca().set(xlabel='GAD67 in Cell', ylabel='GFP in Cell')
    #h=plt.scatter(data['UMAP_X'],data['UMAP_Y'],c=data[coloring],norm=plt.Normalize(vmin=None, vmax=3, clip=False),edgecolors='None',alpha=0.5,s=2.5)
    plt.gca().set_aspect('equal')
    #plt.colorbar() #FIGURE OUT HOW TO SHOW COLORBAR WITHOUT ALPHA BLENDING INTERFERING
    plt.title(slide)
    plt.pause(2)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    plt.pause(2)

    plt.savefig(figpath +'\\'+ slide +'.png',dpi=800,pad_inches=0.1,bbox_inches='tight')