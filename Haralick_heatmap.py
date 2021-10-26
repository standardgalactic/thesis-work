# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:57:35 2021

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
from scipy.spatial import distance

#%% Read data
data=pd.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Haralick feature projects\Aug 16 2021 - haralick\haralick_norm_0_1_100.csv')
#Specify figure plotting path
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Haralick feature projects\Aug 16 2021 - haralick\figures'

#%% Prune data
data=data.drop(["Image","Name","Class","Parent","ROI","Centroid X µm","Centroid Y µm","Area µm^2","Perimeter µm"],axis=1)
# reduce to mean
data=data.mean(axis=0)
#%% string processing for row and column names
measurement_names=data.index.values.tolist()
marker_short=list(set([i.split('-', 1)[1].split(':', 1)[0] for i in measurement_names]))
haralick_short=list(set([i.split('Haralick ', 1)[1].split('(', 1)[0] for i in measurement_names]))
#%% Create and populate matrix
matrix = np.empty(shape=(len(haralick_short),len(marker_short)))
for i in range(len(haralick_short)):
    for j in range(len(marker_short)):
        matrix[i][j]=99999.0 #a nifty little red flag if you screwed something up
        
        df2=data[data.index.str.contains(haralick_short[i],regex=False)]
        df3=df2[df2.index.str.contains(marker_short[j],regex=False)]
        matrix[i][j]=df3[0]
    matrix[i]=(matrix[i]-min(matrix[i]))/(max(matrix[i])-min(matrix[i]))    #normalize each measurement to 0-1
        
#%% Construct dataframe
matrix_df=pd.DataFrame(data=matrix,index=haralick_short,columns=marker_short)
#%% Hypoxia texture score (HT score)
# #calculate the euclidean distance of each gene to PIMO using texture features as coordinates
# euclid_dist=[distance.euclidean(matrix_df['PIMO'],matrix_df[i]) for i in matrix_df.columns]
# #normalize score to 0-1. Subtract from 1 so that the shortest distance will have the highest score
# #also consider using cosine similarity, or something else in the scipy.spatial.distance library
# euclid_dist=[1-(euclid_dist[i]-min(euclid_dist))/(max(euclid_dist)-min(euclid_dist)) for i in range(len(euclid_dist))]
# euclid_dist_df=pd.Series(euclid_dist,index=matrix_df.columns,name='Hypoxia Texture Score')
# matrix_df=matrix_df.append(euclid_dist_df)
#%% Visualize as heatmap

sns.heatmap(matrix_df.sort_values(by='PIMO',axis=0).sort_values(by='Entropy ',axis=1),annot=True,fmt='.2f',vmin=0,vmax=1)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
matplotlib.pyplot.pause(1)
plt.tight_layout()
plt.savefig(figpath + r'\haralick_norm_0_1_100.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
