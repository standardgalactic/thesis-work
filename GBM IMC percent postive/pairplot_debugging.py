# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:56:35 2021

@author: Mark Zaidi

Fragment of code derived from and to be merged with Data Visualization v2.py
Code focuses on performance and stylistic optimizations of the sns.pairplot figure, without having to run unnecessary code
Could also work on making the matrix plots here
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
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\figures\all patients'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2
#measurement names for Feb 2021 batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Median','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')
#%% TEMPORARY PRUNING OF DATA
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]

# testvar_measures=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean']
# testvar_measures=['Pr(141)_141Pr-aSMA: Cell: Median', 'Nd(143)_143Nd-GFAP: Cell: Mean', 'Nd(145)_145Nd-CD31: Cell: Mean', 'Nd(146)_146Nd-Nestin: Cell: Mean', 'Nd(148)_148Nd-Tau: Cell: Mean', 'Sm(149)_149Sm-CD11b: Cell: Mean', 'Nd(150)_150Nd-PD-L1: Cell: Median', 'Eu(151)_151Eu-CA9: Cell: Mean']
# testvar_measures_short=['aSMA','GFAP','CD31']
# testvar=data[[param_Name,param_Parent]+testvar_measures]

testvar_measures=measurements_of_interest
testvar_measures_short=measure_short_for_fname
testvar=data
colors=sns.color_palette("tab10")


#%%PAIRPLOTS
# only pass data less than 2 standard deviations above the mean 
# Code below is wrong. measure isn't being iterated upon, meaning its filtering out all measurements such that its the mean + stdev of the last `measure` iterand.
#Output will still be generated, but technically not correct. Potential solution is to either include outliers and perform no filtering, OR  find a way to modify `data` such that it labels outliers as NaN
#pair = sns.pairplot(data=testvar,vars=testvar_measures, hue='Parent',diag_kind='kde',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01))
start_time = time.time()
plt.style.use('default')
plt.rcParams.update({'font.size': 5})
#pair = sns.pairplot(data=testvar.sample(n=1000,random_state=seed).sort_values([param_Parent]),vars=testvar_measures, hue=param_Parent,plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=0.01),diag_kws=dict(common_norm=False))
pair = sns.pairplot(data=testvar.sample(n=1000,random_state=seed),vars=testvar_measures,height=2)
pair.fig.set_figheight(20)
pair.fig.set_figwidth(20)

print('It took', time.time()-start_time, 'seconds.')
#Variants for plotting only positive or negative data
#pair = sns.pairplot(data=testvar[testvar["Parent"]==param_neg_kwd],vars=testvar_measures, hue='Parent',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01),palette=[colors[0]])
#pair = sns.pairplot(data=testvar[testvar["Parent"]==param_pos_kwd],vars=testvar_measures, hue='Parent',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01),palette=[colors[1]])
# #%%
# #pair.map_lower(sns.kdeplot, levels=4, color=".2")

xlabels=testvar_measures_short
ylabels=testvar_measures_short

for i in range(len(xlabels)):
    for j in range(len(ylabels)):
        pair.axes[j,i].xaxis.set_label_text(xlabels[i])
        pair.axes[j,i].yaxis.set_label_text(ylabels[j])
      
##Goddamn, I spent 3 hours writing the two lines of code below, just so that the subplots remained in a square shape. High variable counts break the aspect parameter of pairplot
##Presumably because the figure generation is so fast, that whatever function the aspect parameter gets passed to is called too late. By adding a 1 second delay, seems to work.      
matplotlib.pyplot.pause(1)
pair.fig.set_figheight(20)
pair.fig.set_figwidth(20)
pair.tight_layout()

#plt.savefig(os.path.join(figpath,'PairPlotV2_TESTFIGURE.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
#%% spearmann correlation
#PIMO negative
PIMO_neg = testvar[testvar["Parent"]==param_neg_kwd]
PIMOneg_corr = []

for i in range(0,len(testvar_measures)):
    data1 = PIMO_neg[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_neg[testvar_measures[j]].tolist()
            corr, _ = spearmanr(data1, data2)
            PIMOneg_corr.append(round(corr,3))


spearmann_corr = pd.DataFrame(PIMOneg_corr,columns=['PIMO-'])

PIMO_pos = testvar[testvar["Parent"]==param_pos_kwd]
PIMOpos_corr = []

for i in range(0,len(testvar_measures)):
    data1 = PIMO_pos[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_pos[testvar_measures[j]].tolist()
            corr, _ = spearmanr(data1, data2)
            PIMOpos_corr.append(round(corr,3))


            
spearmann_corr = spearmann_corr.assign(PIMOpos= PIMOpos_corr)

all_corr = []

for i in range(0,len(testvar_measures)):
    data1 = testvar[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = testvar[testvar_measures[j]].tolist()
            corr, _ = spearmanr(data1, data2)
            all_corr.append(round(corr,3))



spearmann_corr = spearmann_corr.assign(All= all_corr)
#%% write correlation coefficients on subplot
for neg_coeff,all_coeff,pos_coeff,ax in zip(spearmann_corr['PIMO-'],spearmann_corr['All'],spearmann_corr['PIMOpos'],pair.axes.flatten()):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmax*0.6, ymax*0.6,pos_coeff, fontsize=9,color=colors[1]) #pimo positive
    ax.text(xmax*0.6, ymax*0.7,all_coeff, fontsize=9,color='black') #all cells
    ax.text(xmax*0.6, ymax*0.8,neg_coeff, fontsize=9,color=colors[0]) #pimo negative



#%% Write pairplot
#plt.savefig(os.path.join(figpath,'PairPlotV3_TESTING.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')    
## TO DO:
    #Calculate P values from spearman correlation. spearmanr function already does this, just need to store output in a 2nd variable instead of an underscore
    #If P <0.05, add 1 * next to number plotted on pairplot. If p <0.01, add **. If <0.001, add ***. May change p thresholds depending on how it looks
    #Create a matrix for each column in spearmann_corr, reshaped to match the shape of the pairplot (should be len(measurements_of_interest) for both the length and height)
    #Make a seaborn.matrix plot, with each square's color intensity proportional to the spearmann coefficient. Write the coefficient inside the square and add * corresponding to p values, just as it was done for the pairplot.
    #In total, you should have 3 matrix plots: pimo positive, pimo negative, and all cells
    
winsound.Beep(440, 1000)
