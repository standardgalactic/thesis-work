# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:46:26 2021

@author: Mark Zaidi

Script for visualizing per-cell csv of histograms for DAPI, Opal 690 (NeuN), Opal 480 (GFP), Opal 620 (GAD67), and Sample AF (autofluorescence)
to evaluate if their intensities vary considerably or not.
Input: per-cell csv from QuPath containing the mean measurement for the above channel names, and the "Image" column for separate images
Output: violin plot figure per image, with 5 bars corresponding to each of the channels' mean measurements
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
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Alpheus\measurements_with_dog_6.csv')
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Alpheus\figures_Alpheus\dog1-6'

col_names=data.columns
data_orig=data
groupby='Dog'
#groupby='Slide'

#%% clean up image names
data[['Slide','second_slide_name']] = data.Image.str.split('_',expand=True)
data['Image'].unique()
#%% Specify metadata in dict format to include. Should be in format {patient_name:[gender, IDH_status, primary_recurrent ]}
metadata_dict={'PPIX 1_Scan1.qptiff - resolution #1':['Dog 1'], #PANEL 1
               'PPIX 2_Scan1.qptiff - resolution #1':['Dog 1'],
               'PPIX 3_Scan1.qptiff - resolution #1':['Dog 1'],
               'Dog2':['Dog 2'],
               'Dog 3':['Dog 3'], #PANEL 3
               'dog 3':['Dog 3'],
               'Germany B':['Germany B'],
               'Germany A':['Germany A'],
               'Dog 4':['Dog 4'],
               'Dog 5':['Dog 5'],
               'Dog 6':['Dog 6']
               }

#%% append new columns to dataframe from dict
data['Dog']='placeholder'
#dict_entry=list(metadata_dict.keys())[6]
for dict_entry in metadata_dict:
    data['Dog'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][0], data['Dog'])
#%%Optional filtering of data
#Filter by patient
slide_IDs=pandas.unique(data_orig["Dog"])
slide_IDs.sort()
data=data_orig[data_orig['Dog'].str.contains("Dog 1|Dog 2|Dog 3|Dog 4|Dog 5|Dog 6")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'

seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2
#measurement names for Feb 2021 batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
measurements_of_interest=['Opal 620: Cell: Mean','DAPI: Nucleus: Mean','Opal 620: Cell: Mean','Sample AF: Cell: Mean']
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')

#%% create violin plots for each channel
marker_short=[i.split(':', 1)[0] for i in measurements_of_interest]

for measure,measure_short in zip(measurements_of_interest, marker_short):
    ax = sns.violinplot(x=groupby,y=measure, data=data, scale='area',hue='Parent',linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), size=5)
    plt.xticks(rotation=90)
    plt.title(measure_short)
    plt.tight_layout()
    plt.savefig(figpath + '\\' + measure_short + '_violin_histograms.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
    plt.close()
#%% get descriptive statistics on a per-core basis

#txt_path=filelist[0]
tumor_or_stroma='Tumor'
core_list=data[groupby].unique()
c_mean=[]
c_median=[]
c_std=[]
c_min=[]
c_max=[]
c_name=[]
#core=core_list[0]
for core in core_list:
    curr_core=data[data[groupby].str.contains(core)]
    curr_core=curr_core[curr_core['Parent'].str.contains(tumor_or_stroma)]

    c_mean.append(np.mean(curr_core["Opal 620: Cell: Mean"]))
    c_median.append(np.median(curr_core["Opal 620: Cell: Mean"]))
    c_std.append(np.std(curr_core["Opal 620: Cell: Mean"]))
    c_min.append(np.min(curr_core["Opal 620: Cell: Mean"]))
    c_max.append(np.max(curr_core["Opal 620: Cell: Mean"]))
    c_name.append(core)
summary_df=pd.DataFrame([c_name,c_mean,c_median,c_std,c_min,c_max],index=["name","mean","median","std","min","max"]).transpose()
summary_df.to_csv(figpath + '\\' + tumor_or_stroma +'_summary.csv')
