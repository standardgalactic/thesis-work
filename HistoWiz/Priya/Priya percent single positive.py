# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:05:35 2021

@author: Mark Zaidi

This script will read an "annotation_measurements.csv" file, with image named with a KR number delimited by some string (triple underscores)
and will quantify the percentage of cells positive for a single marker in the entire image, and append these as new columns. Then, it will append
a new column corresponding to the KR number, to help with visualization and grouping (serving as a categorical variable)

"""


#%% Import libraries
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\processed_annotation_measurements_v1.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\figures\Percent single positive\grouped by code - 7 codes'

data=pandas.read_csv(csv_path)
columns=data.columns.tolist()
total_expected_markers=26
#order = ['Control','ICI-AIN', 'Drug-AIN', 'ATN', 'TKI-TMA', 'ATN-TX','ATN-Other']
order = ['cABMR','Normal', 'ABMR*', 'Mixed', 'ABMR', 'BK','C. Pyel','ACR']
order.remove('cABMR')
#Replace ABMR* and cABMR as ABMR
data.Code=data.Code.replace({'cABMR':'ABMR'})
#%% Derive IHC marker names
#drop non IHC related columns
col1=[x for x in columns if x.startswith('Num')]
col1.remove('Num Detections')
#Split list elements by :, remove leading whitespace
col2 = [word.lstrip() for line in col1 for word in line.split(':')]
#Identify unique elements. Apparently you just convert the list to a set and back
col2=list(set(col2))
#Remove leading "Num" again
col3= [word.lstrip() for line in col2 for word in line.split(' ')]
col3=list(set(col3))
col3.remove('Num')
#Remove contents before underscore
col4=[x.split('_',1)[1] for x in col3]
#Just for added safety, make sure the number of unique markers found matches up with what we're expecting
# if (len(col4)!=total_expected_markers):
#     raise Exception('Number of markers found does not match number expected')
#%%Iterate and sum columns containing a given IHC marker
new_df=data.filter(['Image','Num Detections'])
for curr_marker in col4:
    #curr_marker=col4[3]
    cols_containing_marker=[x for x in columns if curr_marker in x]
    summed_columns=data[cols_containing_marker].sum(axis=1)
    pct_column=summed_columns/new_df['Num Detections']*100
    new_df[curr_marker]=pct_column
#Get KR number
#KR_ID=[x.split('___')[0] for x in new_df['Image']] #This will include the last two characters after the space
KR_ID=[x.split('___')[0].split(' ')[0] for x in new_df['Image']] #This will NOT include the last two characters after the space

new_df.insert(0,'Area µm^2',data['Area µm^2'])
new_df.insert(0,'Num CD45 per mm^2',(new_df['152Sm-CD45']/new_df['Area µm^2']*1000**2))

new_df.insert(0,'KR number',KR_ID)
new_df.insert(0,'Code',data['Code'])

#Write out table
new_df.to_csv(figpath + '\Percent_Single_Positive.csv',index=False)
#%% Visualize data as boxplot for each marker
for marker in col4:
    plt.close('all')
    h=sns.boxplot(data=new_df,x='Code',y=marker,showfliers = False,order=order)
    #Box plot documentation: box shows the quartiles of the dataset while the whiskers extend
    #to show the rest of the distribution, except for points that are determined to be “outliers”
    #using a method that is a function of the inter-quartile range.
    #Outliers omitted are those exceeding 1.5x the range of the 25th-75th percentiles (IQR)
    #h=sns.stripplot(data=new_df,x='Code',y=marker,color=".25",size=3,order=order)

    plt.xticks(rotation=90)
    h.set(ylabel='Percent ' + marker + ' Positive')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,marker + '.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
#%% Create a separate plot just for CD45 as per the client (Priya)'s request
plt.close('all')
h=sns.boxplot(data=new_df,x='Code',y='Num CD45 per mm^2',showfliers = False,order=order)
plt.xticks(rotation=90)
h.set(ylabel='Number of CD45 cells per mm^2')
plt.tight_layout()
plt.savefig(os.path.join(figpath,'CD45_density.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')