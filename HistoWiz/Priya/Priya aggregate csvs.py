# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:25:15 2022

@author: Mark Zaidi

Script will read the tab-delimited outputs from per_image_exporter.groovy, and concatenate them into one large csv
Only requirement is that you provide it with a directory, where there are multiple tab delimited .csv files, all containing EXACTLY the same columns and ordering

"""
#%% load libraries
import pandas
import pandas as pd
import os
import numpy as np
from datetime import datetime
import pyarrow.csv as csv
#%% set constants
csv_dir=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\per image cell measurements'
file_ext='.csv'
delimiter='\t'
output_name='aggregated_statistics.csv'
max_distance=200
use_pyarrow_engine=True
output_dir='aggregated_csv'
#%% Create output dir
if not os.path.exists(os.path.join(csv_dir, output_dir)):
    os.mkdir(os.path.join(csv_dir, output_dir))
#%% Get a list of files to aggregate
csv_list=[]
raw_csv_list=[]
for file in os.listdir(csv_dir):
    if file.endswith(file_ext):
        #print(os.path.join(csv_dir, file))
        csv_list.append(os.path.join(csv_dir, file))
        raw_csv_list.append(file)
#%% Perform a very very sketchy operation that reads in ALL csvs in the directory and concatenates them at once
#The premise here is that instead of reading one file at a time and appending, why not read them all in at once?
#Regardless, we'll have all the data loaded into memory at once, so may as well do it in one swift motion.
startTime = datetime.now()
if use_pyarrow_engine:
    combined_csv = pd.concat([pd.read_csv(f, sep=delimiter,engine='pyarrow') for f in csv_list ])
else:
    combined_csv = pd.concat([pd.read_csv(f, sep=delimiter) for f in csv_list ])
print('Time to read:   ' + str(datetime.now() - startTime) + '\n')

cols=combined_csv.columns
#%% find "distance to" columns. If nan or over max_distance, set to 200
startTime = datetime.now()

#combined_csv=combined_csv_raw.head(10000).copy()
#find column names that contain the "distance to" keywords
distance_strs=['Distance to']
distance_cols=[s for s in cols if any(xs in s for xs in distance_strs)]
cluster_strs=['Cluster mean']
cluster_cols=[s for s in cols if any(xs in s for xs in cluster_strs)]
smoothed_25strs=['Smoothed: 25']
smoothed_25_cols=[s for s in cols if any(xs in s for xs in smoothed_25strs)]
smoothed_100strs=['Smoothed: 100']
smoothed_100_cols=[s for s in cols if any(xs in s for xs in smoothed_100strs)]

#Select cell distance related columns
#df.loc[df['First Season'] > 1990, 'First Season'] = 1
for curr_col in distance_cols:
#curr_col=distance_cols[26]
    combined_csv.loc[combined_csv[curr_col].isna(),curr_col]=max_distance
    combined_csv.loc[combined_csv[curr_col]>max_distance,curr_col]=max_distance

    #combined_csv[curr_col]>max_distance=max_distance
#columns still with nan. Honestly, at this point, just set them to 0.
remaining_NaN_cols=combined_csv.columns[combined_csv.isna().any()].tolist()
for curr_col in remaining_NaN_cols:
    combined_csv.loc[combined_csv[curr_col].isna(),curr_col]=0
#Do a final pass and check if any NaNs remain
if (combined_csv.isnull().values.any()):
    still_has_NaN=combined_csv.columns[combined_csv.isna().any()].tolist()
    print('NaNs still present in:\n')
    print(*still_has_NaN,sep="\n")

print('Time to process:   ' + str(datetime.now() - startTime) + '\n')



#%% write out combined csv file
startTime = datetime.now()

if use_pyarrow_engine:
    #using pyarrow
    output_name=output_name.replace('.csv','.parquet')
    combined_csv.to_parquet(os.path.join(csv_dir,output_dir,output_name),index=False)
else:
    #using normal pandas to_csv writer
    combined_csv.to_csv(os.path.join(csv_dir,output_dir,output_name),index=False)
print('Time to write:   ' + str(datetime.now() - startTime) + '\n')

#%% check reading times
# startTime = datetime.now()
# temp=pd.read_parquet(r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\per image cell measurements\aggregated_csv\aggregated_statistics.parquet')
# print('Time to read after writing:   ' + str(datetime.now() - startTime) + '\n')

#%% Get a list of image names in a dir
# img_list=[]
# for file in os.listdir(r'C:\Users\Mark Zaidi\Documents\QuPath\images\HistoWiz\Future_Priya_Sandra\211018_New_Priya_MayoClinic\Renamed_ome_tiffs'):
#     if file.endswith('.tiff'):
#         #print(os.path.join(csv_dir, file))
#         img_list.append(file)
# raw_csv_list=[x.split(' - ')[0] for x in raw_csv_list]
# #to get elements which are in temp1 but not in temp2
# differences=list(set(img_list) - set(raw_csv_list))