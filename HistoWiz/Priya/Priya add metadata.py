# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:17:20 2022

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
import time
from scipy.stats import spearmanr
import winsound

#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\annotation_measurements_withPixelClassifier.csv'
if (csv_path.rsplit('.',maxsplit=1)[1]=='csv'):
    data=pandas.read_csv(csv_path,low_memory=False)
elif (csv_path.rsplit('.',maxsplit=1)[1]=='parquet'):
    data=pandas.read_parquet(csv_path)
else:
    raise Exception('Unable to detect file format to read')

output_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2'
#See, this is what I have to deal with when channels aren't named properly...
#Easier fix for CD16:
data.columns = data.columns.str.replace("146NdCD16", "146Nd-CD16")
#Add '-' between 146Nd and CD16
# data.rename(columns={
#         'Nd(146)_146NdCD16: Nucleus: Mean':'Nd(146)_146Nd-CD16: Nucleus: Mean',
#         'Nd(146)_146NdCD16: Nucleus: Median':'Nd(146)_146Nd-CD16: Nucleus: Median',
#         'Nd(146)_146NdCD16: Nucleus: Min':'Nd(146)_146Nd-CD16: Nucleus: Min',
#         'Nd(146)_146NdCD16: Nucleus: Max':'Nd(146)_146Nd-CD16: Nucleus: Max',
#         'Nd(146)_146NdCD16: Nucleus: Std.Dev.':'Nd(146)_146Nd-CD16: Nucleus: Std.Dev.',
#         'Nd(146)_146NdCD16: Cytoplasm: Mean':'Nd(146)_146Nd-CD16: Cytoplasm: Mean',
#         'Nd(146)_146NdCD16: Cytoplasm: Median':'Nd(146)_146Nd-CD16: Cytoplasm: Median',
#         'Nd(146)_146NdCD16: Cytoplasm: Min':'Nd(146)_146Nd-CD16: Cytoplasm: Min',
#         'Nd(146)_146NdCD16: Cytoplasm: Max':'Nd(146)_146Nd-CD16: Cytoplasm: Max',
#         'Nd(146)_146NdCD16: Cytoplasm: Std.Dev.':'Nd(146)_146Nd-CD16: Cytoplasm: Std.Dev.',
#         'Nd(146)_146NdCD16: Membrane: Mean':'Nd(146)_146Nd-CD16: Membrane: Mean',
#         'Nd(146)_146NdCD16: Membrane: Median':'Nd(146)_146Nd-CD16: Membrane: Median',
#         'Nd(146)_146NdCD16: Membrane: Min':'Nd(146)_146Nd-CD16: Membrane: Min',
#         'Nd(146)_146NdCD16: Membrane: Max':'Nd(146)_146Nd-CD16: Membrane: Max',
#         'Nd(146)_146NdCD16: Membrane: Std.Dev.':'Nd(146)_146Nd-CD16: Membrane: Std.Dev.',
#         'Nd(146)_146NdCD16: Cell: Mean':'Nd(146)_146Nd-CD16: Cell: Mean',
#         'Nd(146)_146NdCD16: Cell: Median':'Nd(146)_146Nd-CD16: Cell: Median',
#         'Nd(146)_146NdCD16: Cell: Min':'Nd(146)_146Nd-CD16: Cell: Min',
#         'Nd(146)_146NdCD16: Cell: Max':'Nd(146)_146Nd-CD16: Cell: Max',
#         'Nd(146)_146NdCD16: Cell: Std.Dev.':'Nd(146)_146Nd-CD16: Cell: Std.Dev.',
#         'Pt(195)_195Pt: Nucleus: Mean':'Pt(195)_195Pt-PLAT: Nucleus: Mean',
#         'Pt(195)_195Pt: Nucleus: Median':'Pt(195)_195Pt-PLAT: Nucleus: Median',
#         'Pt(195)_195Pt: Nucleus: Min': 'Pt(195)_195Pt-PLAT: Nucleus: Min',
#         'Pt(195)_195Pt: Nucleus: Max': 'Pt(195)_195Pt-PLAT: Nucleus: Max',
#         'Pt(195)_195Pt: Nucleus: Std.Dev.': 'Pt(195)_195Pt-PLAT: Nucleus: Std.Dev.',
#         'Pt(195)_195Pt: Cytoplasm: Mean': 'Pt(195)_195Pt-PLAT: Cytoplasm: Mean',
#         'Pt(195)_195Pt: Cytoplasm: Median': 'Pt(195)_195Pt-PLAT: Cytoplasm: Median',
#         'Pt(195)_195Pt: Cytoplasm: Min': 'Pt(195)_195Pt-PLAT: Cytoplasm: Min',
#         'Pt(195)_195Pt: Cytoplasm: Max': 'Pt(195)_195Pt-PLAT: Cytoplasm: Max',
#         'Pt(195)_195Pt: Cytoplasm: Std.Dev.':'Pt(195)_195Pt-PLAT: Cytoplasm: Std.Dev.',
#         'Pt(195)_195Pt: Membrane: Mean': 'Pt(195)_195Pt-PLAT: Membrane: Mean',
#         'Pt(195)_195Pt: Membrane: Median':'Pt(195)_195Pt-PLAT: Membrane: Median',
#         'Pt(195)_195Pt: Membrane: Min':'Pt(195)_195Pt-PLAT: Membrane: Min',
#         'Pt(195)_195Pt: Membrane: Max':'Pt(195)_195Pt-PLAT: Membrane: Max',
#         'Pt(195)_195Pt: Membrane: Std.Dev.':'Pt(195)_195Pt-PLAT: Membrane: Std.Dev.',
#         'Pt(195)_195Pt: Cell: Mean':'Pt(195)_195Pt-PLAT: Cell: Mean',
#         'Pt(195)_195Pt: Cell: Median': 'Pt(195)_195Pt-PLAT: Cell: Median',
#         'Pt(195)_195Pt: Cell: Min': 'Pt(195)_195Pt-PLAT: Cell: Min',
#         'Pt(195)_195Pt: Cell: Max':'Pt(195)_195Pt-PLAT: Cell: Max',
#         'Pt(195)_195Pt: Cell: Std.Dev.': 'Pt(195)_195Pt-PLAT: Cell: Std.Dev.'
#                 },inplace=True,errors='Raise')

col_names=data.columns

#%% Extract KR and ROI number from image name
data[['KR','IMC_ROI']] = data.Image.str.split('___',expand=True)
#%% Specify metadata in dict format to include. Should be in format {patient_name:[Code]}
# metadata_dict={
#         'KR-21-4146':['TKI-TMA','TKI-TMA','ICI-Other'],
#         'KR-21-4413':['ICI-AIN','ICI-AIN 1','ICI-AIN'],
#         'KR-21-2267':['ICI-AIN', 'ICI-AIN 2','ICI-AIN'],
#         'KR-21-3215':['ICI-AIN', 'ICI-AIN 3','ICI-AIN'],
#         'KR-21-4280':['ICI-AIN', 'ICI-AIN 4','ICI-AIN'],
#         'KR-18-4025':['ICI-AIN', 'ICI-AIN 5A','OMIT'],
#         'KR-21-2822':['ICI-AIN', 'ICI-AIN','OMIT'],
#         'KR-21-3094':['Drug-AIN', 'Drug-AIN 1','Drug-AIN'],
#         'KR-21-4127':['Drug-AIN', 'Drug-AIN 2','Drug-AIN'],
#         'KR-21-4287':['Drug-AIN', 'Drug-AIN 3','Drug-AIN'],
#         'KR-20-2442':['Drug-AIN', 'Drug-AIN 4','Drug-AIN'],
#         'KR18-5037':['Control', 'Control 1','Control'],
#         'KR18-4930':['Control', 'Control 2','Control'],
#         'KR20-3003':['Control', 'Control 3','Control'],
#         'KR-21-4937':['ATN-TX', 'ATN-TX','OMIT'],
#         'KR08-48':['ATN-Other', 'ATN-Other','OMIT'],
#         'KR-21-4742':['ATN', 'ATN 1','ICI-Other'],
#         'KR-20-5610':['ATN', 'ATN 2','ICI-Other'],
#         'KR-21-2276':['ATN', 'ATN 3','ICI-Other'],
#         'KR-18-5112 ':['ICI-AIN', 'ICI-AIN 5B','OMIT']
#     }

metadata_dict={
        'KR-14-6142 A1':['cABMR'],
        'KR21-4226 A4':['Normal'],
        'KR-17-2231':['cABMR'],
        'KR-17-2602':['ABMR*'],
        'KR-17-1985':['ABMR*'],
        'KR21-4213 A2':['Normal'],
        'KR-17-182':['ABMR*'],
        'KR-21-4828 A4':['Mixed'],
        'KR21-4227 A2':['Normal'],
        'KR21-4222 A2':['Normal'],
        'KR-20-4625 A4':['cABMR'],
        'KR-19-1310 A2':['Normal'],
        'KR-19-5581 A4':['Mixed'],
        'KR-18-2117 A3':['ABMR'],
        'KR-19-4424 A1':['BK'],
        'KR-17-3526':['ABMR*'],
        'KR-17-51845 A2':['BK'],
        'KR-16-6637 A1':['C. Pyel'],
        'KR-18-4424 A3':['BK'],
        'KR-17-50724 A3':['BK'],
        'KR-19-627 A3':['BK'],
        'KR-18-5588 A3':['ACR'],
        'KR-18-6091 A3':['C. Pyel'],
        'KR-21-4225 A4':['ACR'],
        'KR-18-1537 A3':['cABMR'],
        'KR-21-4285 A4':['Mixed'],
        'KR-18-5492 A3':['ABMR'],
        'KR-16-2317 A1':['cABMR']
    
    }


# metadata_dict={'29B':['M','WT','P'], #PANEL 1
#                '39':['F','WT','P'],
#                '45A':['F','WT','P'],
#                '28':['M','WT','P'],
#                '50':['F','WT','R'], #PANEL 2
#                '53B':['M','MUT','P'], #PANEL 3
#                '47A':['F','WT','R'],
#                '45C':['F','WT','P'],
#                '58':['F','WT','P'],
#                '33':['M','WT','P'],
#                '59B':['F','WT','P'],
#                '46A':['M','WT','R'],
#                }
#%% append new columns to dataframe from dict
data['Code']='Missing'
#data['Code_full']='Missing'
#data['Sandra_4']='Missing'

# data['IDH_status']='placeholder'
# data['primary_recurrent']='placeholder'
#dict_entry=list(metadata_dict.keys())[6]
for dict_entry in metadata_dict:
    data['Code'] = np.where(data['KR'].str.contains(dict_entry), metadata_dict[dict_entry][0], data['Code'])
    #data['Code_full'] = np.where(data['KR'].str.contains(dict_entry), metadata_dict[dict_entry][1], data['Code_full'])
    #data['Sandra_4'] = np.where(data['KR'].str.contains(dict_entry), metadata_dict[dict_entry][2], data['Sandra_4'])
    # data['primary_recurrent'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][2], data['primary_recurrent'])
#Add patient and ROI columns in case it's needed
#data[['Patient','IMC_ROI']] = data.Image.str.split('_',expand=True)
#%% Write out new csv
if (csv_path.rsplit('.',maxsplit=1)[1]=='csv'):
    data.to_csv(output_path + '\\processed_annotation_measurements_v2.csv',index=False)
elif (csv_path.rsplit('.',maxsplit=1)[1]=='parquet'):
    data.to_parquet(output_path + '\\processed_cell_measurements.parquet',index=False)
else:
    raise Exception('Unable to detect file format to read')


#test=pandas.read_csv(output_path)