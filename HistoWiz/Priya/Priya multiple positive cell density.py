# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:49:31 2022

@author: Mark Zaidi


This script will load in a processed_master_sheet containing a list of different
percent multiple positive cases, as computed by their respective groovy scripts in QuPath.
We will specify which columns to create boxplots for, grouped by code. May even toss on a strip plot,

"""


#%% Import libraries
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\percent_multiple_positives\all_cells\processed_master_sheet.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\figures\Percent multiple positive\all_cells_7_codes_density'

data=pandas.read_csv(csv_path)
columns=data.columns.tolist()
data['Activated CD4 and CD8 %']=data['Activated CD4 %']+data['Activated CD8 %']
cols_to_plot=['Activated CD4 per mm^2',
 'Activated CD8 per mm^2',
 'Activated CD4 and CD8 per mm^2',
 'B Cell per mm^2',
 'CD4 Memory T Cells per mm^2',
 'CD4 Naive T Cells per mm^2',
 'CD8 Memory T Cells per mm^2',
 'CD8 Naive T Cells per mm^2',
 'Dendritic Cell per mm^2',
 'Macrophage per mm^2',
 'NK Cells per mm^2',
 'T Cytotoxic Subset per mm^2',
 'T Helper Subset per mm^2',
 'T Reg per mm^2']
data.rename(columns={'Activated CD4 %':'Activated CD4 per mm^2',
 'Activated CD8 %':'Activated CD8 per mm^2',
 'Activated CD4 and CD8 %':'Activated CD4 and CD8 per mm^2',
 'B Cell %':'B Cell per mm^2',
 'CD4 Memory T Cells %':'CD4 Memory T Cells per mm^2',
 'CD4 Naive T Cells %':'CD4 Naive T Cells per mm^2',
 'CD8 Memory T Cells %':'CD8 Memory T Cells per mm^2',
 'CD8 Naive T Cells %':'CD8 Naive T Cells per mm^2',
 'Dendritic Cell %':'Dendritic Cell per mm^2',
 'Macrophage %':'Macrophage per mm^2',
 'NK Cells %':'NK Cells per mm^2',
 'T Cytotoxic Subset %':'T Cytotoxic Subset per mm^2',
 'T Helper Subset %':'T Helper Subset per mm^2',
 'T Reg %':'T Reg per mm^2'},inplace=True,errors='Raise')
#%% Begin plotting
#order = ['Control','ICI-AIN', 'Drug-AIN', 'ATN', 'TKI-TMA', 'ATN-TX','ATN-Other']
order = ['cABMR','Normal', 'ABMR*', 'Mixed', 'ABMR', 'BK','C. Pyel','ACR']
#order.remove('cABMR')
#Replace ABMR* and cABMR as ABMR
data.Code=data.Code.replace({'cABMR':'ABMR'})
order = data['Code'].unique().tolist()
#order.remove('Missing')
for col in cols_to_plot:
    data[col]=data[col]/100*data['Num Detections']/data['Area µm^2']*1000**2
    #col=cols_to_plot[0]
    plt.close('all')
    h=sns.boxplot(data=data,x='Code',y=col,showfliers = False, order = order)
    h=sns.stripplot(data=data,x='Code',y=col,color=".25",size=3, order = order)

    plt.xticks(rotation=90)
    h.set(ylabel=col)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,col + '.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
