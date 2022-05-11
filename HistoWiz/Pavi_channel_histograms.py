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
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Pavi_data_processing\panel_1\Pavi_Exports_cells.csv')
col_names=data.columns
data_orig=data

#%%Optional filtering of data
#Filter by patient
slide_IDs=pandas.unique(data_orig["Parent"])
slide_IDs.sort()
data=data_orig[data_orig['Parent'].str.contains("PathAnnotationObject")]
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'

figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Pavi_data_processing\panel_1\figures'
seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2
#measurement names for Feb 2021 batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
measurements_of_interest=['DAPI: Nucleus: Mean','Opal 570: Cell: Mean','Opal 620: Cell: Mean','Opal 520: Cell: Mean','Sample AF: Cell: Mean']
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')

##% create violin plots for each channel
marker_short=[i.split(':', 1)[0] for i in measurements_of_interest]

for measure,measure_short in zip(measurements_of_interest, marker_short):
    ax = sns.violinplot(x="Image",y=measure, data=data, scale='area',linewidth=0)
    ax.set_xticklabels(ax.get_xticklabels(), size=2)
    plt.xticks(rotation=90)
    plt.title(measure_short)
    plt.tight_layout()
    plt.savefig(figpath + '\\' + measure_short + '_violin_histograms.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
    plt.close()