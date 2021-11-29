# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:17:06 2021

@author: Mark Zaidi

Building off of add_m

When incorporating vessel segmentations as annotations in the IMC workflow, cells that fall into vessels have their parent annotation
overwritten to the vessel class, instead of the pimo positive/negative class. This script will obtain the original parent annotation,
based on that cell's distance to a pimo positive/negative area. If it is 0 for any distance, that means it is inside of one such area.
The 'parent' label will be reverted back to what it previously was, and 'raw_parent' will be the value of 'parent' before execution of the script.

To do:
    -create raw_parent to be original parent column
    -create an if/elseif statement and only execute on cells with parent CD31+ vessel
    -create 'distance_to_border' as being the negative distance to pimo positive AND positive distance to pimo negative (need to think about it)

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
#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\cell_measurements_withVessel.csv'
out_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\figures'

data=pandas.read_csv(csv_path)

    
#force str type on Patient column to avoid weird discretization issues
col_names=data.columns
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
param_vessel='CD31+ vessel'

unique_parents=data[param_Parent].unique()
if not(param_vessel in unique_parents):
    raise ValueError('param_vessel not found in param_Parent. Are you sure you have vessel objects? If vessel segmentation has not been performed, there is no need to execute this script!')

p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively
measures_to_drop=['DNA193','DNA191'] #remove these from any percent positive or intensity comparisons
param_neg_distance='Distance to annotation with '+param_neg_kwd+' µm'
param_pos_distance='Distance to annotation with '+param_pos_kwd+' µm'

seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2

#measurement names for Feb 2021 batch (Panel 2)
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch (Panel 1)
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurements names for Aug 2021 batch (Panel 3)
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
data.sort_values(param_Parent,inplace=True)
#%% Begin data preprocessing
#Filter out cells that have parent `Image` (means that cells didn't fall into PIMO positive or negative errors, likely due to being on edge of image)
data=data[data[param_Parent]!='Image']
data['raw_parent']=data[param_Parent]
distance_to_border=[]
#revert a cell's parent to the original grandparent it fell in, if that cell also falls in a vessel
#for cells in pimo negative areas:
data[param_Parent] = np.where([data[param_Parent]==param_vessel] and [data[param_neg_distance]==0][0], param_neg_kwd, data[param_Parent])
#for cells in pimo positive areas:
data[param_Parent] = np.where([data[param_Parent]==param_vessel] and [data[param_pos_distance]==0][0], param_pos_kwd, data[param_Parent])

#create a distance_to_border variable
data['distance_to_border']=999999.0
data['distance_to_border'] = np.where(data[param_Parent]==param_pos_kwd, data[param_neg_distance], data['distance_to_border'])
data['distance_to_border'] = np.where(data[param_Parent]==param_neg_kwd, -data[param_pos_distance], data['distance_to_border'])
# plt.gca().set_facecolor((0, 0, 0))
# sns.scatterplot(data=data,x='Centroid X µm',y='Centroid Y µm',hue='distance_to_border',linewidth=0,s=5,palette='viridis',hue_norm=(-100,100),legend=False)
#%% Write .csv
data.to_csv(out_path + '\\' + csv_path.rsplit(sep='\\',maxsplit=1)[1],index=False)














