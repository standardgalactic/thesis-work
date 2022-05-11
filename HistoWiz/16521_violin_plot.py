# -*- coding: utf-8 -*-
"""
Created on Nov 11 12:43:28 2021

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
#%% read data
txt_folder_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\16051\exports'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\16051\figures'
#%% get list of files to process
fext='.csv'
filelist=[]
for file in os.listdir(txt_folder_path):
    if file.endswith(fext):
        filelist.append(os.path.join(txt_folder_path, file))
#%% get image name
file_part_1=[i.split('$', 1)[1] for i in filelist]
file_part_2=[i.split('$', 1)[0] for i in file_part_1]
#%% Read data
for txt_path,name in zip(filelist,file_part_2):
    data=pandas.read_csv(txt_path,sep='\,', lineterminator='\r')
    col_names=data.columns
    data_orig=data
    ##%% data preprocessing
    data['Row']=data["TMA core"].str.rstrip('-0123456789')
    ##%% plot violin plot
    #for each row
    data=data.dropna(axis=0,subset=['Row'])
    rows=data['Row'].unique()
    #row=rows[0]
    for row in rows:
        #filter all rows that do not contain row in its Row column
        curr_data=data[data['Row'].str.contains(row)]
        #ax = sns.violinplot(x="TMA core",y="Cell: DAB OD mean", data=curr_data, scale='area',linewidth=0)
        ax = sns.boxplot(x="TMA core",y="Cell: DAB OD mean", data=curr_data,showfliers = False)

        ax.set_ylim([0, 1])  
        #ax.set_xticklabels(ax.get_xticklabels(), size=2)
        #plt.xticks(rotation=90)
        plt.title("Cell: DAB OD mean of row " + str(row))
        plt.tight_layout()
        plt.savefig(figpath + '\\' + name + '_' + row + '_boxplot.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
        plt.close()
#%% get descriptive statistics on a per-core basis
for txt_path,name in zip(filelist,file_part_2):
    
    #txt_path=filelist[0]
    data=pandas.read_csv(txt_path,sep='\,', lineterminator='\r')
    data=data.dropna(axis=0,subset=['TMA core'])
    col_names=data.columns
    
    core_list=data["TMA core"].unique()
    c_mean=[]
    c_median=[]
    c_std=[]
    c_min=[]
    c_max=[]
    c_name=[]
    #core=core_list[0]
    for core in core_list:
        curr_core=data[data['TMA core'].str.contains(core)]
        c_mean.append(np.mean(curr_core["Cell: DAB OD mean"]))
        c_median.append(np.median(curr_core["Cell: DAB OD mean"]))
        c_std.append(np.std(curr_core["Cell: DAB OD mean"]))
        c_min.append(np.min(curr_core["Cell: DAB OD mean"]))
        c_max.append(np.max(curr_core["Cell: DAB OD mean"]))
        c_name.append(core)
    summary_df=pd.DataFrame([c_name,c_mean,c_median,c_std,c_min,c_max],index=["name","mean","median","std","min","max"]).transpose()
    summary_df.to_csv(figpath + '\\' + name +'_summary.csv')



