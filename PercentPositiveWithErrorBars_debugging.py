# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:47:13 2021

@author: Mark Zaidi
Fragment of code derived from and to be merged with Data Visualization v2.py
Code focuses on comparing groups of patients to assess patient variability in hypoxia-regulated gene expression
Can be adapted to group by supplementary info such as IDH mutation status
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
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\cell_measurements.csv')
annotation_data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\annotation_measurements.csv')
col_names=data.columns
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
param_Image='Image'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\figures\all patients'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2
#measurement names for Feb 2021 batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Median','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
categories=['45A','28','29B','39']
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')
#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data["Image"])
# data=data[data['Image'].str.contains("39")]
#%% count the number of cells in different regions of interest
cells_in_pimo_pos=sum(data.apply(lambda x: 1 if x[param_Parent] == param_pos_kwd else 0 , axis=1))
cells_in_pimo_neg=sum(data.apply(lambda x: 1 if x[param_Parent] == param_neg_kwd else 0 , axis=1))
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
#%% iteratively get percent positive scores
#define variables
count=0
pct_name=list()
marker_count=0
pct_pos=[]
pos_in_pimo_neg=[]
pos_in_pimo_pos=[]
for marker in marker_short:
    #identify full marker name
    full_marker=marker_list[marker_count]
    marker_count=marker_count+1
    for annotation in annotation_list:
        #label current entry as the current iterators for marker and annotation
        pct_name.append(['Percent ' + marker + ' positive in ' + annotation])
        #create filters to include cells from original data that belong to current annotation and marker classification
        cond1= data[param_Name].str.contains(full_marker,regex=False)
        cond2= data[param_Parent].str.contains(annotation,regex=False)
        #If conditions are met, calculate the total number of cells that meet cond1&cond2, divide by total cells in cond2(annotation) and multiply by 100
        #pct_name corresponds to pct_pos value
        pct_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        if param_pos_kwd in annotation:
            pos_in_pimo_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        elif param_neg_kwd in annotation:
            pos_in_pimo_neg.append((cond1&cond2).sum()/cond2.sum()*100)

        count=count+1
pct_pos_df=pandas.DataFrame([marker_short,pos_in_pimo_pos,pos_in_pimo_neg,[i / j for i, j in zip(pos_in_pimo_pos, pos_in_pimo_neg)]]).transpose().sort_values(3,ascending=False)
#%% try and calculate percent positive on a per case basis for each IHC marker
score_df= pd.DataFrame(columns = ['pos_score','patient_grouping','hypoxic_grouping'])
score_list=[]
hypox_list=[]
patient_list=[]
markershort_list=[]
positive_count_list=[]
total_count_list=[]
for marker, full_marker in zip(marker_short,marker_list):
    for patient in categories:
        for annotation in annotation_list:
             print('Percent '+marker+' positive in '+annotation+' for case '+patient)
             #create filters to include cells from original data that belong to current annotation and marker classification
             cond1= data[param_Name].str.contains(full_marker,regex=False) #check if full_marker string is present in the list of markers a cell is positive for
             cond2= data[param_Parent].str.contains(annotation,regex=False) #check if cell belongs to annotation
             cond3= data[param_Image].str.contains(patient,regex=False) #check if cell belongs to patient
             score=((cond1&cond2&cond3).sum()/(cond2&cond3).sum()*100)
             
             score_list.append(score) #return percent of cells positive for a given `marker` in a given `hypoxic_grouping` in a given `patient_grouping`
             hypox_list.append(annotation) #return `hypoxic_grouping`
             patient_list.append(patient) #return `patient_grouping`
             markershort_list.append(marker) #return `marker`
             positive_count_list.append((cond1&cond2&cond3).sum()) #return absolute count of positive cells for a given `marker` in a given `hypoxic_grouping` in a given `patient_grouping`
             total_count_list.append((cond2&cond3).sum()) #return total count of cells for a given `hypoxic_grouping` in a given `patient_grouping`

score_df=pd.DataFrame({'marker':markershort_list,'pos_score':score_list,'hypoxic_grouping':hypox_list,'patient_grouping':patient_list,'positive_cells':positive_count_list,'total_cells':total_count_list})
#%% make the box plot
ax = sns.boxplot(x="marker", y="pos_score", hue="hypoxic_grouping", data=score_df)
plt.legend([],[], frameon=False) #remove legend
plt.xticks(rotation=90) #rotate x ticks vertically
plt.tight_layout()
