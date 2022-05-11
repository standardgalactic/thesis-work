# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:07:28 2022

@author: Mark Zaidi

Code for generating heatmaps of spearman correlation coefficient for Priya
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
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220124_Sandra\figures\Heatmaps\Sandra_4'

data=pandas.read_csv(csv_path)
#force str type on Patient column to avoid weird discretization issues
# data=data.astype({'Patient': 'str'})
# a_unique_patients=data['Patient'].unique()

col_names=data.columns
#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data["Patient"])
# slide_IDs.sort()
# data=data[data['Patient'].str.contains("53B")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively
measures_to_drop=['DNA193','DNA191'] #remove these from any percent positive or intensity comparisons
groupby='Sandra_4'


seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2

#Priya
#measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(150)_150Sm-PD-L1: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean']                                      
#Sandra
measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean','Pt(195)_195Pt-PLAT: Cell: Mean']                                      


#drop specific measurements
# for measure in measures_to_drop:
#     measurements_of_interest[:] = [x for x in measurements_of_interest if measure not in x]

plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
#data.sort_values(param_Parent,inplace=True)

#%%Get shortened names of measurements
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]
#%% Remove rows containing NAN
len_data=len(data)
data=data.dropna(subset=measurements_of_interest)
print('Dropped ' + str(len_data-len(data)) + ' rows containing NaN')
#Drop rows containing a specific value in the target_variable
data=data[~data[groupby].str.contains('OMIT|Missing')]
#%% spearmann correlation and p value calculation
#PIMO negative
code_list=data[groupby].unique()
code_list_writeable=[x.replace('*','_STAR') for x in code_list] #The * cannot be used in file names
for curr_code, curr_code_writeable in zip(code_list,code_list_writeable):
    #curr_code=code_list[0]
    curr_subset = data[data[groupby]==curr_code]
    curr_subset_corr = []
    curr_subset_p = []
    
    
    for i in range(0,len(measurements_of_interest)):
        data1 = curr_subset[measurements_of_interest[i]].tolist()
        for j in range(0,len(measurements_of_interest)):
                data2 = curr_subset[measurements_of_interest[j]].tolist()
                corr, pval = spearmanr(data1, data2)
                curr_subset_corr.append(round(corr,3))
                curr_subset_p.append(pval)
    
    
    # spearmann_corr = pd.DataFrame(curr_subset_corr,columns=['PIMOneg'])
    # p_values = pd.DataFrame(curr_subset_p,columns=['PIMOneg'])
    
    dims = len(measurements_of_interest)
    
    
    #reshape the spearrmann correlations (each column in spearmann_corr) into 2D arrays to be used by seaborn heatmaps
    corr_2D = np.reshape(curr_subset_corr, (dims, dims))
    #Convert 2D heatmap arrays into dataframe with labelled axes
    df_corr_2D = pandas.DataFrame(data=corr_2D,index=measure_short_for_fname,columns=measure_short_for_fname)
    
    # PIMO_kwd='PLAT'
    # #stick PIMO to the start
    # temp=measure_short_for_fname[:]
    # temp.remove(PIMO_kwd)
    # temp.insert(0, PIMO_kwd)
    
    # #reorder dataframes with PIMO at the start
    # df_corr_2D=df_corr_2D.reindex(temp)
    # df_corr_2D=df_corr_2D.reindex(columns=temp).sort_values(PIMO_kwd,ascending=False)
    # #now, sort the columns by the same way the rows were sorted
    # new_order=df_corr_2D.index
    # df_corr_2D=df_corr_2D.reindex(columns=new_order)
    
    df_corr_2D.to_csv(figpath + '\\' + curr_code_writeable +'_corr.csv')
    
    #reshape the p values (each column in p_values) into 2D arrays to be used by seaborn heatmaps
    p_corr_2D = np.reshape(curr_subset_p, (dims, dims))
    #Convert 2D heatmap arrays into dataframe with labelled axes
    p_df_corr_2D = pandas.DataFrame(data=p_corr_2D,index=measure_short_for_fname,columns=measure_short_for_fname)
    #apply this order to the df_PIMOneg_2D and df_PIMOpos_2D dataframes
    # p_df_corr_2D=p_df_corr_2D.reindex(columns=new_order).reindex(index=new_order)
    
    p_df_corr_2D.to_csv(figpath + '\\' + curr_code_writeable +'_p.csv')
    
    #Plot heatmap
    
    plt.close('all')
    
    h=sns.heatmap(data=df_corr_2D,annot=True,cmap='bwr',center=0,vmin=-0.8,vmax=0.8,annot_kws={'fontsize':6},square=True,xticklabels=True,yticklabels=True)
    matplotlib.pyplot.pause(1)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.tight_layout()
    matplotlib.pyplot.pause(1)

    plt.savefig(figpath + '\\' + curr_code_writeable +'.png',dpi=300,pad_inches=0.1,bbox_inches='tight')
    plt.close('all')

