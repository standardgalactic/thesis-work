#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 9:18:06 2021

@author: phoebelombard
"""
#%% Calculate select double-positive combinations for bar charts in PIMO +/- areas
# load libraries
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



#%% Read data
#csv_path='/Users/phoebelombard/Desktop/UHN/Panel_1v2.csv' #Panel 1
#csv_path='/Users/phoebelombard/Desktop/UHN/Panel 2 (Feb 2021 IMC)/Panel_2v2.csv' #Panel 2
csv_path='/Users/phoebelombard/Desktop/UHN/Panel 3 (August 16 2021 - updated panel)/Panel_3v2.csv' #Panel 3
figpath='/Users/phoebelombard/Desktop/UHN/FIGS_despeckled'

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
slide_IDs=pandas.unique(data["Image"])
slide_IDs.sort()

#Have to restart kernel each time you change the data filter for some reason!
#data=data[data['Image'].str.contains("33|45|58|59")] # primary IDHwt
#data=data[data['Image'].str.contains("46|47")] # recurrent IDHwt
data=data[data['Image'].str.contains("53")] # primary IDHmut
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
#%% Set numerators and denominators

#NEW
#Works for Panel 1
#denominators=["CD68","CD163","Iba1"]
#denominators=["CD68","Iba1"]
#denominators=["CD68"]
#denominators=["CD163"]
#denominators=["Iba1"]
#numerators=["HK2","GLUT1","LDHA","CA9","ICAM1","Ki67","TMHistone", "CD11b","PD-L1","pS6"]


#Use for Panel 2 and 3 (No CD163, IBA1 instead of Iba1)
#denominators=["CD68","IBA1"]
#denominators=["CD68"]
#denominators=["IBA1"]
#numerators=["HK2","GLUT1","LDHA","CA9","ICAM","Ki67","TMHistone", "CD11b"]
#numerators=["TMHistone"] #for IBA1

#NEW
#GROUP 2 
#denominators=["HK2"]
#denominators=["CA9"]
#denominators=["LDHA"]
#denominators=["GLUT1"]
#Panel 1
#numerators=["SOX2", "Nestin", "Ki67", "GFAP", "PD-L1","pS6","CXCR4","TMHistone","CD45","CD11b","aSMA","CD31","CD68","Iba1"]
#Panels 2&3
#numerators=["SOX2", "Nestin", "Ki67", "GFAP", "PD-L1","pS6","CXCR4","TMHistone","CD45","CD11b","aSMA","CD31","CD68","Iba1"]

# tumor cells % positives (GFAP) PANEL 1
#denominators=["GFAP"]
#numerators=["CD45", "CD31", "Iba1","CA9", "GLUT1","HK2", "LDHA","ICAM1","Ki67","TMHistone"]

#For Panel 2 and 3, use:
#denominators=["GFAP"]
#numerators=["CD45", "CD31", "IBA1", "CA9", "GLUT1","HK2", "LDHA","ICAM","Ki67","TMHistone"]

#Group 3 Panels 2/3 (Panel 1 doesn't have TMHistone)
#denominators=["TMHistone"]
#numerators=["CD68", "CD45", "CD45RO", "CD11b", "CD163", "CD3", "CD4", "CD8", "IBA1", "Ki67", "SOX2", "GFAP", "Nestin", "CD31", "aSMA", "CA9", "GLUT1", "LDHA", "HK2"]

#Group 4 
#denominators=["aSMA"]
#numerators=["GLUT1", "CA9", "HK2", "LDHA", "CD31","TMHistone"]

#Group 5 panel 1
#denominators=["CD45"]
#numerators=["CD11b", "CD68", "CD163", "Iba1", "CD3", "CD4", "CD8","CA9","GLUT1","HK2","LDHA","TMHistone"]
#numerators=["CD11b"] #for CD45

#Group 5 panel 2/3
#denominators=["CD45"]
#numerators=["CD11b", "CD68", "CD163", "IBA1", "CD3", "CD4", "CD8","CA9","GLUT1","HK2","LDHA","TMHistone"]

#denominators=["Nestin"]
#numerators=["CA9", "GLUT1","HK2", "LDHA","ICAM","Ki67","TMHistone"]

#Panel 1 does NOT HAVE SOX2
#denominators=["SOX2"]
#numerators=["CA9", "GLUT1","HK2", "LDHA","ICAM","Ki67","TMHistone"]

#denominators=["CD31"]
#numerators=["Ki67","aSMA"]

#denominators=["Ki67"]
#numerators=["CD11b","CD68","CD45","CD163","CD3","CD4","CD8","CD31","aSMA","SOX2","Nestin","GFAP","TMHistone"]

#NEW
#Panels 2 & 3 (Panel 1 does not have ICAM)
denominators=["ICAM"]
numerators=["CD68","CD163","CD45","CD4","CD3","CD8","CD11b","IBA1","GFAP"]

#Panel 1
#denominators=["CD4"]
#numerators=["CA9","GLUT1","HK2","LDHA","TMHistone","CD45"]

#Panel 1
#denominators=["CD3"]
#numerators=["CA9","GLUT1","HK2","LDHA","pSTAT3"]

#NEW
#Panel 1
#denominators = ["pS6"]
#numerators = ["CD68","CD45","GFAP","SOX2","Nestin","Iba1","CA9","GLUT1","HK2","LDHA","CD163","CD3","CD4","CD8","CD45RO","CD11b","aSMA","CD31"]


pct_dp_PIMO_pos=[]
pct_dp_PIMO_neg=[]
denominator_used=[]
numerator_used=[]
cells_num_pos=[]
cells_denum_pos=[]
cells_num_neg=[]
cells_denum_neg=[]
#file_name = "multiPos"
#file_name = "Panel2multiPos"
#file_name = "Panel3multiPos"
#file_name = "P3primIDHwt"
#file_name = "P3recurrIDHwt"
file_name = "P3primIDHmut"

index = 0
denominators_string=""
numerators_string=""
for denominator in denominators:
    if denominators.index(denominator)<(len(denominators)-1):
        file_name = file_name + denominator + "_"
        denominators_string = denominators_string + denominator + ", "
    elif denominators.index(denominator) == (len(denominators)-1):
        file_name = file_name + denominator
        denominators_string = denominators_string + denominator
    index = marker_short.index(denominator)
    denominators[denominators.index(denominator)]=marker_list[index]
    
for numerator in numerators:
    if numerators.index(numerator)<(len(numerators)-1):
        numerators_string = numerators_string + numerator + ", "
    elif numerators.index(numerator) == (len(numerators)-1):
        numerators_string = numerators_string + numerator
        
    if numerator in marker_short:
        index = marker_short.index(numerator)
        numerators[numerators.index(numerator)]=marker_list[index]
    else:
        numerators[numerators.index(numerator)]="0"
    
numerators = [numerator for numerator in numerators if numerator != '0']
        
#%%Calculate % positive scores

for numerator in numerators:
    
        denominator_mask = np.vstack((data[param_Name].str.contains(string,regex=False) for string in denominators)).all(axis=0)
    
        numerator_mask = np.vstack((data[param_Name].str.contains(string,regex=False) for string in denominators+[numerator])).all(axis=0)
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


for numerator in numerator_used:
    index = marker_list.index(numerator)
    numerator_used[numerator_used.index(numerator)]=marker_short[index]

i=0
for j in denominator_used[i]:
    index = marker_list.index(j) 
    denominator_used[i][denominator_used[i].index(j)]=marker_short[index]

pair_df=pandas.DataFrame(list(zip(numerator_used,denominator_used,pct_dp_PIMO_pos,pct_dp_PIMO_neg,[i / j for i, j in zip(pct_dp_PIMO_pos, pct_dp_PIMO_neg)],cells_num_pos,cells_denum_pos,cells_num_neg,cells_denum_neg)),columns =['numerator_used','denominator_used', 'pct_dp_PIMO_pos','pct_dp_PIMO_neg','ratio','cells_num_pos','cells_denum_pos','cells_num_neg','cells_denum_neg']).sort_values('ratio',ascending=False)
pair_df.to_csv(figpath + '/' + file_name + '.csv',index=False,header=["Numerator","Denominator","Percent Positive in PIMO+ Regions","Percent Positive in PIMO- Regions","Ratio","Total Double-positive Cells in PIMO+","Total Denominator-positive Cells in PIMO+","Total Double-positive Cells in PIMO-", "Total Denominator-positive Cells in PIMO-"])


#%% visualize percent multiple positives from above as clustered bar chart

#labels = marker_short
#neg_bar = pos_in_pimo_pos
#pos_bar = pos_in_pimo_neg
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)


plt.style.use('default')

labels=pair_df["numerator_used"]
title = " ".join(denominator_used[0])
pos_bar=pair_df["pct_dp_PIMO_pos"]
neg_bar=pair_df["pct_dp_PIMO_neg"]
ratios = pair_df["ratio"]


x = np.arange(len(numerator_used))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
plt.xticks(rotation=90)
rects1 = ax.bar(x - width/2, neg_bar, width, label='PIMO Negative')
rects2 = ax.bar(x + width/2, pos_bar, width, label='PIMO Positive')

#add data labels
for rect1, rect2, ratio in zip(rects1, rects2, ratios):
    ratio="{:.2f}".format(ratio)
    #height = rect2.get_height()
    height=(max(rect1.get_height(),rect2.get_height()))
    ax.text(rect2.get_x() + rect2.get_width() * 0, height + 5, ratio,
            ha='center', va='bottom',rotation='vertical')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Positive')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
plt.ylim([0,125])
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)


#plt.savefig(figpath + '/Percent multipositive scores.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#fig.tight_layout()
#plt.savefig(figpath,dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.savefig(os.path.join(figpath,file_name),dpi=800,pad_inches=0.1,bbox_inches='tight')
#plt.savefig(figpath + '/Percent double positive scores.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.close()