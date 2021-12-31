# -*- coding: utf-8 -*-
"""
Goal is to create per-channel violin plots. Channels to create plots for will be defined in measurements_of_interest.
Y axis is marker intensity, x axis is slide ID. May need to append new columns with better slide IDs
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
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\IMC_data_clustering\Panel_3v2.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\August 16 2021 - updated panel\figures\panel 3 figures and tables\all_patients\Per_ROI_violin_plots'


data=pandas.read_csv(csv_path)
#manually fix any irregular or broken column names
#data.rename(columns={"Nd(146)_146NdCD16: Cell: Mean":"Nd(146)_146Nd-CD16: Cell: Mean"},inplace=True)

colnames=data.columns

#%% Define measurements of interest
#From Priya (actually Sandra)
#measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(150)_150Sm-PD-L1: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean']                                      
#From Sheila Panel 3
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

#Trim names into something shorter so that it's easier to read on a plot or figure name
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]
#Trim image names into something more manageble
data['img_short']=[i.split('.', 1)[0] for i in data['Image']]
#%%generate violin plots for each channel

for measure,short_name in zip(measurements_of_interest,measure_short_for_fname):
# measure=measurements_of_interest[16]
# short_name=measure_short_for_fname[16]
    plt.close('all')

    num_std_to_include=3
    scaletype='width'
    
    #ax = sns.violinplot( x='img_short',y=measure,data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], scale=scaletype,linewidth=0)
    ax = sns.violinplot( x='img_short',y=measure,data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], scale=scaletype,linewidth=0,hue='Parent',split=True)

    #ax = sns.violinplot( x='img_short',y=measure,data=data, scale='area',linewidth=0)
    ax.set_xticklabels(ax.get_xticklabels(), size=5)
    plt.xticks(rotation=90)
    ax.set_title(short_name)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    #ax.set_ylim([0, 20])
    
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,short_name + '.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
