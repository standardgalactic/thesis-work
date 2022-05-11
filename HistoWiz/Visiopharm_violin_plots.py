# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:09:40 2022

@author: Mark Zaidi
Purpose of script is to create violin plots of a single marker, grouped by some grouping variable. In this case, the grouping
variable will be the parent image name. 
"""
#%% load libraries
import pandas
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
#%% set constants
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Temp_visiopharm_violin_demo\HTT_Positive_Cells_Intensity.tsv' #This the full path of the .tsv file from Visiopharm
delimiter='\t' #This does NOT need to be changed, unless the files are no longer .tsv
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Temp_visiopharm_violin_demo\figures' #The folder where the violin plots should be created. If the folder does not exist, you need to make it
groupby='Name' #Bar grouping for violin plot, basically x axis
measure='HTT Positive Intensity (per cell)' #Y axis for violin plot
#%% read csv
data=pd.read_csv(csv_path,sep=delimiter)

#%% start generating violin plot
order=data[groupby].unique()
plt.close('all')

num_std_to_include=3
scaletype='width'
ax = sns.violinplot(x=groupby,y=measure,data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], scale=scaletype,linewidth=1,order=order)
plt.xticks(rotation=90)
ax.set_xticklabels(ax.get_xticklabels(), size=5)

plt.tight_layout()

plt.savefig(figpath + '\\' + measure + '_violin_histograms.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

plt.close()
