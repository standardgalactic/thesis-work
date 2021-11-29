# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:26:08 2021

@author: Mark Zaidi

Goal is to add external patient data to per cell statistics, by looking up what ROI a cell came from and adding the respective metadata

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
#%% Read data
#csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\despeckle_cell_measurements.csv' #PANEL 1
#csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\despeckle_cell_measurements.csv' #PANEL 2
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\August 16 2021 - updated panel\cell_measurements_withVessels.csv' #PANEL 3
#csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Jun 2021 - Old IMC data\figures\cell_measurements_withVessel.csv' #PANEL 2and3

output_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\IMC_data_clustering\Panel_3v2.csv'
data=pandas.read_csv(csv_path)
col_names=data.columns
data['Image'].unique()
#%% Specify metadata in dict format to include. Should be in format {patient_name:[Sex, IDH_status, primary_recurrent ]}
metadata_dict={'29B':['M','WT','P'], #PANEL 1
               '39':['F','WT','P'],
               '45A':['F','WT','P'],
               '28':['M','WT','P'],
               '50':['F','WT','R'], #PANEL 2
               '53B':['M','MUT','P'], #PANEL 3
               '47A':['F','WT','R'],
               '45C':['F','WT','P'],
               '58':['F','WT','P'],
               '33':['M','WT','P'],
               '59B':['F','WT','P'],
               '46A':['M','WT','R'],
               }
#%% append new columns to dataframe from dict
data['Sex']='placeholder'
data['IDH_status']='placeholder'
data['primary_recurrent']='placeholder'
#dict_entry=list(metadata_dict.keys())[6]
for dict_entry in metadata_dict:
    data['Sex'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][0], data['Sex'])
    data['IDH_status'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][1], data['IDH_status'])
    data['primary_recurrent'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][2], data['primary_recurrent'])
#Add patient and ROI columns in case it's needed
data[['Patient','IMC_ROI']] = data.Image.str.split('_',expand=True)
#%% Write out new csv
data.to_csv(output_path,index=False)
#test=pandas.read_csv(output_path)