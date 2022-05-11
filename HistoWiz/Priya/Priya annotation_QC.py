# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:03:40 2022

@author: mzaidi

Identifies total annotated area of each class, and if there are any unclassified annotations (and which images they came from)
"""
#%% Import libraries
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220126_Priya_Annotations\classifier_annotation_measurements_withZeroTest.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220126_Priya_Annotations\figures\with ZeroTest'

data=pandas.read_csv(csv_path)
#Drop these columns as they're missing empty space classes
data = data[data.columns.drop(list(data.filter(regex='rt_all_markers|RT_markers_gaussian')))]
columns=data.columns.tolist()
#%% Show the absolute annotated area of each class
summed_data=data.groupby('Name')['Area µm^2'].sum()
plt.close('all')
ax=plt.bar(summed_data.index,summed_data)
plt.ylabel('Area µm^2')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.title('Total annotated area in µm^2: %.0f' % (data['Area µm^2'].sum()))
plt.savefig(os.path.join(figpath,'Training_annotation_area_breakdown.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')

missing_class=data[data['Name']=='PathAnnotationObject']
if (len(missing_class) !=0):
    print('Some annotations are missing a classification. See missing_class for details')
#%% Identify which images to use in training and which to use for validating
images=data['Image'].unique()
train_fraction=0.75
num_train=int(np.floor(len(images)*train_fraction))
train_names=images[0:num_train].tolist()
#Now that we've run the script once to obtain train_names, permanently save it for future use. 38 images total
train_names=['KR-14-6142 A1___ROI1_0UU4_001.ome.tiff - ROI1_0UU4_001.ome',
             'KR-14-6142 A1___ROI2_0UU4_002.ome.tiff - ROI2_0UU4_002.ome',
             'KR-14-6142 A1___ROI3_0UU4_003.ome.tiff - ROI3_0UU4_003.ome',
             'KR-14-6142 A1___ROI4_0UU4_004.ome.tiff - ROI4_0UU4_004.ome',
             'KR-14-6142 A1___ROI7_ROI_007.ome.tiff - ROI7_ROI_007.ome',
             'KR-14-6142 A1___ROI8_ROI_008.ome.tiff - ROI8_ROI_008.ome',
             'KR-14-6142 A1___ROI9_ROI_009.ome.tiff - ROI9_ROI_009.ome',
             'KR-14-6142 A1___ROI10_ROI_010.ome.tiff - ROI10_ROI_010.ome',
             'KR-16-2317 A1___ROI1_0UUO_001.ome.tiff - ROI1_0UUO_001.ome',
             'KR-16-2317 A1___ROI2_0UUO_002.ome.tiff - ROI2_0UUO_002.ome',
             'KR-16-2317 A1___ROI5_0UUO_005.ome.tiff - ROI5_0UUO_005.ome',
             'KR-16-2317 A1___ROI6_0UUO_006.ome.tiff - ROI6_0UUO_006.ome',
             'KR-16-6637 A1___ROI2_0UU9_002.ome.tiff - ROI2_0UU9_002.ome',
             'KR-16-6637 A1___ROI5_0UU9_005.ome.tiff - ROI5_0UU9_005.ome',
             'KR-16-6637 A1___ROI6_0UU9_006.ome.tiff - ROI6_0UU9_006.ome',
             'KR-17-182___ROI1_0UWR_001.ome.tiff - ROI1_0UWR_001.ome',
             'KR-17-182___ROI2_0UWR_002.ome.tiff - ROI2_0UWR_002.ome',
             'KR-17-182___ROI3_0UWR_003.ome.tiff - ROI3_0UWR_003.ome',
             'KR-17-1985___ROI1_0UWS_001.ome.tiff - ROI1_0UWS_001.ome',
             'KR-17-1985___ROI2_0UWS_002.ome.tiff - ROI2_0UWS_002.ome',
             'KR-17-1985___ROI3_0UWS_003.ome.tiff - ROI3_0UWS_003.ome',
             'KR-17-2231___ROI1_ROI_001.ome.tiff - ROI1_ROI_001.ome',
             'KR-17-2231___ROI2_ROI_002.ome.tiff - ROI2_ROI_002.ome',
             'KR-17-2231___ROI3_ROI_003.ome.tiff - ROI3_ROI_003.ome',
             'KR-17-2231___ROI4_ROI_004.ome.tiff - ROI4_ROI_004.ome',
             'KR-17-2602___ROI1_0UWU_001.ome.tiff - ROI1_0UWU_001.ome',
             'KR-17-2602___ROI2_0UWU_002.ome.tiff - ROI2_0UWU_002.ome',
             'KR-17-2602___ROI3_0UWU_003.ome.tiff - ROI3_0UWU_003.ome',
             'KR-17-3526___ROI1_0UWV_001.ome.tiff - ROI1_0UWV_001.ome',
             'KR-18-2117 A3___ROI2_0UU2_002.ome.tiff - ROI2_0UU2_002.ome',
             'KR-18-2117 A3___ROI3_0UU2_003.ome.tiff - ROI3_0UU2_003.ome',
             'KR-18-2117 A3___ROI6_0UU2_006.ome.tiff - ROI6_0UU2_006.ome',
             'KR-18-4424 A3___ROI6_0UU7_006.ome.tiff - ROI6_0UU7_006.ome',
             'KR-18-5492 A3___ROI5_0UU3_005.ome.tiff - ROI5_0UU3_005.ome',
             'KR-18-6091 A3___ROI3_0UUA_003.ome.tiff - ROI3_0UUA_003.ome',
             'KR-19-627 A3___ROI6_0UU6_006.ome.tiff - ROI6_0UU6_006.ome',
             'KR-19-1310 A2___ROI2_ROI_002.ome.tiff - ROI2_ROI_002.ome',
             'KR-19-1310 A2___ROI4_ROI_004.ome.tiff - ROI4_ROI_004.ome']


train_data=data[data['Image'].isin(train_names)]
test_data=data[~data['Image'].isin(train_names)]
#Make sure train and test datasets both contain at least one example of each class
summed_train_data=train_data.groupby('Name')['Area µm^2'].sum()
summed_test_data=test_data.groupby('Name')['Area µm^2'].sum()
#print(train_names)
#%% Get the percentage of areas for each annotation and classifier that was correctly classified
#Here, we take all columns that contain a : in the name. If there is a : then split it and take the first half. That should be the classifier.
#Then, convert the list to a set and back to a list, which retains only unique elements of the list
class_list = list(set([s.split(':',1)[0] for s in columns if ":" in s]))
annotation_list=test_data['Name'].unique().tolist()
area_kwd=' area µm^2'
pred_area_cols=list(set([s for s in columns if area_kwd in s]))
validation_truth_area=test_data.groupby('Name')['Area µm^2'].sum()
pred_ann_areas=pd.Series(dtype='float64')


for curr_classifier in class_list:
#curr_classifier=class_list[0]
    for curr_annot in annotation_list:
    #curr_annot=annotation_list[0]
    
        search_list=[area_kwd,curr_annot,curr_classifier]
        cols_containing_search_list=columns #First, set cols_containing_search_list to be all columns in the dataset
        for curr_search in search_list: #Then, for each of the individual strings to search for...
            cols_containing_search_list=[x for x in cols_containing_search_list if curr_search in x] #Only select the columns that contain a specific search keyword. From this list, move onto the next keyword, and the next, and so on
            #Once this loop is completed, cols_containing_search_list should now ONLY be composed of ONE column, a column that tells us the predicted area of curr_annotation in curr_classifier.
        if (len(cols_containing_search_list) > 1):
            print('More than one columns detected. Verify that you don\'t have ambiguity of classifier names')
            
        #Select only the rows corresponding to the curr_annot    
        curr_test_data=test_data[test_data['Name']==curr_annot]
        #Sum cols_containing_search_list in those rows, divide by the sum of 'Area µm^2' in those rows. Multiply by 100
        curr_area_sum=curr_test_data[cols_containing_search_list].sum()/curr_test_data['Area µm^2'].sum()*100
        pred_ann_areas=pd.concat([pred_ann_areas,curr_area_sum])
#%% Format pred_ann_areas into a barplot-ready dataframe
box_df=pd.DataFrame(columns=class_list,index=annotation_list)
for index,value in pred_ann_areas.items():
    #print(f"Index : {index}, Value : {value}")
    box_col=index.split(':',1)[0]
    box_ind=index.split(':',1)[1].split(area_kwd)[0].lstrip()
    box_df[box_col][box_ind]=value
#%% Plot barplot
#Visualize all data in a single grouped bar plot
plt.close('all')
box_df.transpose().plot.bar()
plt.ylim([0,110])

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.ylabel('Percentage area correctly classified in validation set')
plt.xticks(rotation = 90)
plt.pause(1)
plt.tight_layout()
plt.savefig(os.path.join(figpath,'Classifier_validation.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')

for classifier in box_df.columns:
    plt.close('all')
    #classifier=box_df.columns[0]
    ax=plt.bar(box_df.index,box_df[classifier])
    plt.ylim([0,110])
    plt.ylabel('Percentage area correctly classified in validation set')
    plt.xticks(rotation = 30)
    plt.title('Average Accuracy: %.2f' % (box_df[classifier].mean())+'%')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,classifier + '_accuracy.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')



















