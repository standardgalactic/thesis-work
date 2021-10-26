# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:58:22 2021

@author: Mark Zaidi

Many of the Nd-conjugated antibodies have been using thresholds on the median cell/nucleus measurement as a means of
classifying a cell as positive or negative for that marker. Now that IMC_despeckler.py removes the speckle artifact
that forced us to use median thresholds, we can go back to using mean measurement thresholds, which is what we used for
all other IHC markers.

Now, we need to find the mean threshold that will result in roughly the same cells being classified as positive, as done using
the median threshold. To do this, we need to:
    - load in a per-cell .csv into the "data" dataframe
    - find what percent of cells are positive with the current median threshold
    - of the cells above the median measurement threshold, what is the minimum mean measurement
    - what percent of cells are above this minimum mean measurement (new threshold)
    - compare the old median threshold with the new mean threshold via percent positive scoring
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
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\August 16 2021 - updated panel\August 16 2021 - updated panel.csv')
colnames=data.columns
#%% identify pos cells
#define pos_cells as those with a measurement above threshold defined in thresholds.xlsx
pos_cells_median=data[data["Nd(150)_150Nd-SOX2: Nucleus: Median"]>6.1194]
#find what percent of cells are positive using the current median threshold
pct_pos_median=len(pos_cells_median)/len(data)*100
#of those positive cells, find what the minimum mean threshold is
min_mean_pos_cells=min(pos_cells_median["Nd(150)_150Nd-SOX2: Nucleus: Mean"])
#then on the original data, find what percent of cells are positive using the generated mean threshold
pos_cells_mean=data[data["Nd(150)_150Nd-SOX2: Nucleus: Mean"]>min_mean_pos_cells]
pct_pos_mean=len(pos_cells_mean)/len(data)*100
#%%
