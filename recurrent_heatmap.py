# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:38:08 2021

@author: Mark Zaidi
Request: If we could find a way to combine the 3 recurrent samples would be great. No rush, whenever you have it is great.
Regarding the correlation plots for marker intensities in panel 3, we need to separate samples PIMO 46 and 47 from the rest, as they are recurrent GBMs, and we need to generate a correlation matrix separately. PIMO50 is also recurrent but not sure if you can combine its data with 46 and 47.

What needs to be done:
    - Load in panel 2 and 3 per cell .csv files
    - filter to include only rows with recurrent cases (46A and 47A in panel 3, 50 in panel 2 which is all ROIs in panel 2)
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
panel2=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\despeckle_cell_measurements.csv')
colnames_p2=panel2.columns
panel3=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\August 16 2021 - updated panel\cell_measurements.csv')
colnames_p3=panel3.columns
#Filter panel 3 to keep only the two recurrent cases (46A and 47)
# slide_IDs=pandas.unique(panel3["Image"])
# slide_IDs.sort()
panel3=panel3[panel3['Image'].str.contains("46A|47A")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively


figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\August 16 2021 - updated panel\figures\panel2_3 recurrent'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2

measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
#%% find common columns
common_columns=list(set(colnames_p2).intersection(colnames_p3))
#measurements names for markers of interest that overlap in both panel 3 and panel 2
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
is_subset = set(measurements_of_interest).issubset(set(common_columns))
#%% concatenate dataframe
data=pd.concat([panel2,panel3]).dropna(axis=1)
slide_IDs=pandas.unique(data["Image"])
slide_IDs.sort()
data.sort_values(param_Parent,inplace=True)

#is_subset = set(measurements_of_interest).issubset(set(data.columns)) check if measurements of interest exist in concatenated dataframe
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
#%% TEMPORARY PRUNING OF DATA
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]

# testvar_measures=['Gd(155)_155Gd-PIMO: Cell: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean']
# testvar_measures_short=['PIMO','DNA193','CD68']
#testvar_measures=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean']
#testvar_measures_short=['aSMA','GFAP','CD31']
# testvar=data[[param_Name,param_Parent]+testvar_measures]

testvar_measures=measurements_of_interest
testvar_measures_short=measure_short_for_fname
testvar=data
colors=sns.color_palette("tab10")
#%% spearmann correlation and p value calculation
#PIMO negative
PIMO_neg = testvar[testvar["Parent"]==param_neg_kwd]
PIMOneg_corr = []
PIMOneg_p = []

for i in range(0,len(testvar_measures)):
    data1 = PIMO_neg[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_neg[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            PIMOneg_corr.append(round(corr,3))
            PIMOneg_p.append(pval)


spearmann_corr = pd.DataFrame(PIMOneg_corr,columns=['PIMOneg'])
p_values = pd.DataFrame(PIMOneg_p,columns=['PIMOneg'])

PIMO_pos = testvar[testvar["Parent"]==param_pos_kwd]
PIMOpos_corr = []
PIMOpos_p = []

for i in range(0,len(testvar_measures)):
    data1 = PIMO_pos[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_pos[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            PIMOpos_corr.append(round(corr,3))
            PIMOpos_p.append(pval)


            
spearmann_corr = spearmann_corr.assign(PIMOpos= PIMOpos_corr)
p_values = p_values.assign(PIMOpos = PIMOpos_p)

all_corr = []
all_p = []

for i in range(0,len(testvar_measures)):
    data1 = testvar[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = testvar[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            all_corr.append(round(corr,3))
            all_p.append(pval)


spearmann_corr = spearmann_corr.assign(All= all_corr)
p_values = p_values.assign(All = all_p)
#%% HEATMAPS
##heatmap dimensions:
    #use testvar_measures for the temp short data
    #use measurements_of_interest for real deal
dims = len(testvar_measures)

#reshape the spearrmann correlations (each column in spearmann_corr) into 2D arrays to be used by seaborn heatmaps
PIMOneg_2D = np.reshape(PIMOneg_corr, (dims, dims))
PIMOpos_2D = np.reshape(PIMOpos_corr, (dims, dims))
allcorr_2D = np.reshape(all_corr, (dims, dims))

#Convert 2D heatmap arrays into dataframe with labelled axes
df_PIMOneg_2D = pandas.DataFrame(data=PIMOneg_2D,index=testvar_measures_short,columns=testvar_measures_short)
df_PIMOpos_2D = pandas.DataFrame(data=PIMOpos_2D,index=testvar_measures_short,columns=testvar_measures_short)
df_allcorr_2D = pandas.DataFrame(data=allcorr_2D,index=testvar_measures_short,columns=testvar_measures_short)
#stick PIMO to the start
temp=testvar_measures_short[:]
temp.remove('PIMO')
temp.insert(0, 'PIMO')
#reorder dataframes with PIMO at the start
df_allcorr_2D=df_allcorr_2D.reindex(temp)
df_allcorr_2D=df_allcorr_2D.reindex(columns=temp).sort_values('PIMO',ascending=False)
#now, sort the columns by the same way the rows were sorted
new_order=df_allcorr_2D.index
df_allcorr_2D=df_allcorr_2D.reindex(columns=new_order)
#apply this order to the df_PIMOneg_2D and df_PIMOpos_2D dataframes
df_PIMOneg_2D=df_PIMOneg_2D.reindex(columns=new_order).reindex(index=new_order)
df_PIMOpos_2D=df_PIMOpos_2D.reindex(columns=new_order).reindex(index=new_order)
#Write out corr .csv
df_allcorr_2D.to_csv(figpath + r'\all_corr.csv')
df_PIMOneg_2D.to_csv(figpath + r'\PIMOneg_corr.csv')
df_PIMOpos_2D.to_csv(figpath + r'\PIMOpos_corr.csv')

#%% repeat above, but for creating an array of the p values
#reshape the p values (each column in p_values) into 2D arrays to be used by seaborn heatmaps
p_PIMOneg_2D = np.reshape(PIMOneg_p, (dims, dims))
p_PIMOpos_2D = np.reshape(PIMOpos_p, (dims, dims))
p_allcorr_2D = np.reshape(all_p, (dims, dims))
#Convert 2D heatmap arrays into dataframe with labelled axes
p_df_PIMOneg_2D = pandas.DataFrame(data=p_PIMOneg_2D,index=testvar_measures_short,columns=testvar_measures_short)
p_df_PIMOpos_2D = pandas.DataFrame(data=p_PIMOpos_2D,index=testvar_measures_short,columns=testvar_measures_short)
p_df_allcorr_2D = pandas.DataFrame(data=p_allcorr_2D,index=testvar_measures_short,columns=testvar_measures_short)
#stick PIMO to the start
temp=testvar_measures_short[:]
temp.remove('PIMO')
temp.insert(0, 'PIMO')
#reorder dataframes with PIMO at the start              #This was commented because we want to use the original sort order of the spearman corr array, else p values will not match up to their respective spearman corr
# df_allcorr_2D=df_allcorr_2D.reindex(temp)
# df_allcorr_2D=df_allcorr_2D.reindex(columns=temp).sort_values('PIMO',ascending=False)
#now, sort the columns by the same way the rows were sorted
new_order=df_allcorr_2D.index
p_df_allcorr_2D=p_df_allcorr_2D.reindex(columns=new_order)
#apply this order to the df_PIMOneg_2D and df_PIMOpos_2D dataframes
p_df_PIMOneg_2D=p_df_PIMOneg_2D.reindex(columns=new_order).reindex(index=new_order)
p_df_PIMOpos_2D=p_df_PIMOpos_2D.reindex(columns=new_order).reindex(index=new_order)
p_df_allcorr_2D=p_df_allcorr_2D.reindex(columns=new_order).reindex(index=new_order)

p_df_allcorr_2D.to_csv(figpath + r'\all_p.csv')
p_df_PIMOneg_2D.to_csv(figpath + r'\PIMOneg_p.csv')
p_df_PIMOpos_2D.to_csv(figpath + r'\PIMOpos_p.csv')
#%% plot sorted heatmaps
fig, axes = plt.subplots(1, 3, figsize=(50,50))
#fig.suptitle('Heatmaps')
sns.set(font_scale=0.7)
# PIMO negative
sns.heatmap(ax=axes[0],data=df_allcorr_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8)
axes[0].set_title('All')
axes[0].set_yticklabels(new_order,rotation=0)

# PIMO positive
sns.heatmap(ax=axes[1],data=df_PIMOneg_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8)
axes[1].set_title('PIMO Negative')
axes[1].set_yticklabels(new_order,rotation=0)

# All
sns.heatmap(ax=axes[2],data=df_PIMOpos_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8)
axes[2].set_title('PIMO Positive')
axes[2].set_yticklabels(new_order,rotation=0)

#write heatmaps
plt.tight_layout()
matplotlib.pyplot.pause(1)
plt.savefig(os.path.join(figpath,'Heatmaps.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
plt.close()
 