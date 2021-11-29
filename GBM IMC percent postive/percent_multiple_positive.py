# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:01:16 2021

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
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\despeckle_cell_measurements.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\figures\despeckled\all_patients'

data=pandas.read_csv(csv_path)
#annotation_data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\annotation_measurements.csv')
col_names=data.columns
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively



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
#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data["Image"])
# slide_IDs.sort()
# data=data[data['Image'].str.contains("28")]
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
#%% percent quadruple positive
denominators=["CD68","CD163","Iba1"]
# denominators=["CD68"]
# denominators=["CD163"]
# denominators=["Iba1"]
numerators=["HK2","GLUT1","LDHA","CA9","ICAM1","Ki67"]
pct_dp_PIMO_pos=[]
pct_dp_PIMO_neg=[]
denominator_used=[]
numerator_used=[]
cells_num_pos=[]
cells_denum_pos=[]
cells_num_neg=[]
cells_denum_neg=[]
for numerator in numerators:
#numerator=numerators[0]
    
    denominator_mask = np.vstack((data[param_Name].str.contains(string) for string in denominators)).all(axis=0)
    
    numerator_mask = np.vstack((data[param_Name].str.contains(string) for string in denominators+[numerator])).all(axis=0)
    pos_mask=data[param_Parent].str.contains(param_pos_kwd,regex=False)
    neg_mask=data[param_Parent].str.contains(param_neg_kwd,regex=False)
    
    pos_value=(pos_mask&numerator_mask).sum()/(pos_mask&denominator_mask).sum()*100
    neg_value=(neg_mask&numerator_mask).sum()/(neg_mask&denominator_mask).sum()*100
    #MARK NOTE: CHANGE NUMERATOR_MASK TO DENOMINATOR_MASK BELOW
    if numerator_mask.sum()==0:
        pos_value= float("NaN")
        neg_value= float("NaN")
#Append various measurements to a list
    cells_num_pos.append((pos_mask&numerator_mask).sum()) #number of cells used in calculating numerator of pimo positive cells    
    cells_denum_pos.append((pos_mask&denominator_mask).sum()) #number of cells used in calculating denominator of pimo positive cells
    cells_num_neg.append((neg_mask&numerator_mask).sum()) #number of cells used in calculating numerator of pimo negative cells
    cells_denum_neg.append((neg_mask&denominator_mask).sum()) #number of cells used in calculating denominator of pimo negative cells

    
    
    pct_dp_PIMO_neg.append(neg_value) #percent multiple positive in pimo negative area
    pct_dp_PIMO_pos.append(pos_value)#percent multiple positive in pimo positive area
    
    denominator_used.append(denominators)
    numerator_used.append(numerator)

pair_df=pandas.DataFrame(list(zip(numerator_used,denominator_used,pct_dp_PIMO_pos,pct_dp_PIMO_neg,[i / j for i, j in zip(pct_dp_PIMO_pos, pct_dp_PIMO_neg)],cells_num_pos,cells_denum_pos,cells_num_neg,cells_denum_neg)),columns =['numerator_used','denominator_used', 'pct_dp_PIMO_pos','pct_dp_PIMO_neg','ratio','cells_num_pos','cells_denum_pos','cells_num_neg','cells_denum_neg']).sort_values('ratio',ascending=False)
