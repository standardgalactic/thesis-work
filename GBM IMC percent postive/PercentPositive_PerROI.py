# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:40:05 2021

@author: Mark Zaidi

Script is geared towards calculating percent positive scores for each marker and for each ROI. Rows will be unque ROIs.
Columns will be percent [marker] positive in pimo positive, percent [marker] positive in pimo negative, percent [marker] positive overall
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
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\\despeckle_cell_measurements.csv')
col_names=data.columns
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
param_image='Image'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively


figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\figures'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2

#measurement names for Feb 2021 batch
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurements names for Aug 2021 batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
data.sort_values(param_Parent,inplace=True)
#%% get unique IHC marker names
#split class name by :, listing all classes a cell belongs to
df2=data[param_Name].str.split(':',expand=True)
#Identify unique names, filtering out the unused class name and any NoneType cases
df3=pandas.unique(df2[df2.columns].values.ravel('K'))
df3=df3[(df3 != param_UnusedClass)]
marker_list = [x for x in df3 if x != None]
#remove spaces, cast to list
marker_list=[x.strip(' ') for x in marker_list]
marker_list=list(set(marker_list))
#get unique annotation class names. Should be equal to the value set by param_pos_kwd and param_neg_kwd
annotation_list=data[param_Parent].unique().tolist()
#get shortened marker name (for figure plotting)
marker_short=[i.split('-', 1)[1] for i in marker_list]
#%% calculate percent positive per ROI
annotation_list=data[param_Parent].unique().tolist()
annotation_list.append('all')

#declare empty lists
pct_pos=[]
pos_in_pimo_neg=[]
pos_in_pimo_pos=[]
pct_val=[]
pct_name=list()
summary_array=np.zeros((1,len(marker_list)*len(annotation_list)))
#identify unique ROI names
ROI_list=data[param_image].unique()
#append a new annotation in annotation_list called `all`
#simulating first case of for loop
for curr_ROI in ROI_list:
    #curr_ROI=ROI_list[0]
    for curr_marker in marker_list:
        #curr_marker=marker_list[0]
        for curr_annotation in annotation_list:
            #curr_annotation=annotation_list[0]
            #create filters to include cells from original data that belong to current annotation AND marker AND ROI classification
            #condition to select rows with cells belonging to current ROI
            cond_ROI=data[param_image].str.contains(curr_ROI,regex=False)
            #condition to select rows with cells positive for marker
            cond_marker=data[param_Name].str.contains(curr_marker,regex=False)
            #condition to select rows with cells belonging to the given annotation
            cond_annotation=data[param_Parent].str.contains(curr_annotation,regex=False)
            #if annotation is 'all', just select the same rows already selected in cond_ROI
            if curr_annotation=='all':
                cond_annotation=cond_ROI
            #in the dataset, find the sum total of rows that meet all 3 conditions, sum them, and divide by number of rows that meet the annotation and ROI condition, then multiply by 100    
            curr_pct_value=((cond_marker&cond_annotation&cond_ROI).sum())/((cond_annotation&cond_ROI).sum())*100
            
            curr_pct_name='Percent ' + curr_marker + ' positive in ' + curr_annotation
            #append results to a list
            pct_name.append(curr_pct_name)
            pct_val.append(curr_pct_value)
    #after calculating each measurement in each annotation, append vertically where rows correspond to ROI
    summary_array=np.vstack((summary_array,pct_val))
    temp_name=pct_name
    #clear pct_name and pct_val after each iteration, so it doesn't keep appending to a longer and longer list
    pct_name=[]
    pct_val=[]
#delete first row of summary_array, seems to be a duplicate caused by the for looping
summary_array=np.delete(summary_array,0,0)
summary_df=pd.DataFrame(data=summary_array,index=ROI_list,columns=temp_name)
summary_df.to_csv(figpath + r'\Percent single positive across all ROIs.csv')










#%% iteratively get percent positive scores
#define variables
# count=0
# pct_name=list()
# marker_count=0
# pct_pos=[]
# pos_in_pimo_neg=[]
# pos_in_pimo_pos=[]
# for marker in marker_short:
#     #identify full marker name
#     full_marker=marker_list[marker_count]
#     marker_count=marker_count+1
#     for annotation in annotation_list:
#         #label current entry as the current iterators for marker and annotation
#         pct_name.append(['Percent ' + marker + ' positive in ' + annotation])
#         #create filters to include cells from original data that belong to current annotation and marker classification
#         cond1= data[param_Name].str.contains(full_marker,regex=False)
#         cond2= data[param_Parent].str.contains(annotation,regex=False)
#         #If conditions are met, calculate the total number of cells that meet cond1&cond2, divide by total cells in cond2(annotation) and multiply by 100
#         #pct_name corresponds to pct_pos value
#         pct_pos.append((cond1&cond2).sum()/cond2.sum()*100)
#         if param_pos_kwd in annotation:
#             pos_in_pimo_pos.append((cond1&cond2).sum()/cond2.sum()*100)
#         elif param_neg_kwd in annotation:
#             pos_in_pimo_neg.append((cond1&cond2).sum()/cond2.sum()*100)

#         count=count+1

