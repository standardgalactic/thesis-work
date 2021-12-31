# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:31:12 2021

@author: Mark Zaidi

Goal is to extract feature importance from some kind of ML classifier, where the end output is a categorical classification (IDH mut, primary/recurrent, etc.)
Doesn't seem like this can be done with UMAP alone. Maybe feature preprocessing can be done with UMAP, but currently unsure how.
To do:
    Train a ML classifier on a portion of the dataset (training set)
    Evaluate accuracy on an independent portion of the dataset (test set)
    Once a reasonable accuracy is achieved, calculate feature importance
"""

#%% import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas
import pandas as pd
from xgboost import XGBClassifier
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%declare constants
"""
There are two ways to specify features to use in training: specify them via measurements_of_interest, or omit everything else via cols_to_drop
If working with the 18 cell mean features, use measurements_of_interest. If working with the >400 features, use cols_to_drop
When training on Parent (PIMO positive or negative), you'll want to omit any PIMO-related markers (no point showing that PIMO can be used to predict PIMO)


"""
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\IMC_data_clustering\Panel_3v2.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\IMC_data_clustering\Panel_3\all_patients\experimental\feature_importance\20_feature'
seed=69
prune_from_dataset='BCK' #remove any measurements relating to background channels
#measurements names for Aug 2021 batch (Panel 3)
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#panel 3 but with pimo removed
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
target_variable="Parent" #name of column you want to use as classification
#Drop columns from training that we either can't use (strings) or are very obviously not going to benefit the classifier (e.g. centroid X um)
cols_to_drop=['Image', 'Name', 'Class', 'Parent', 'ROI', 'Centroid X µm',
              'Centroid Y µm','Sex','IDH_status', 'primary_recurrent', 'Patient', 'IMC_ROI', 'raw_parent',
              'Distance to annotation with pimo positive µm','distance_to_border','Distance to annotation with pimo negative µm']
# ,
#               'Gd(155)_155Gd-PIMO: Nucleus: Mean',
#         'Gd(155)_155Gd-PIMO: Nucleus: Median',
#         'Gd(155)_155Gd-PIMO: Nucleus: Min', 'Gd(155)_155Gd-PIMO: Nucleus: Max',
#         'Gd(155)_155Gd-PIMO: Nucleus: Std.Dev.',
#         'Gd(155)_155Gd-PIMO: Cytoplasm: Mean',
#         'Gd(155)_155Gd-PIMO: Cytoplasm: Median',
#         'Gd(155)_155Gd-PIMO: Cytoplasm: Min',
#         'Gd(155)_155Gd-PIMO: Cytoplasm: Max',
#         'Gd(155)_155Gd-PIMO: Cytoplasm: Std.Dev.',
#         'Gd(155)_155Gd-PIMO: Membrane: Mean',
#         'Gd(155)_155Gd-PIMO: Membrane: Median',
#         'Gd(155)_155Gd-PIMO: Membrane: Min',
#         'Gd(155)_155Gd-PIMO: Membrane: Max',
#         'Gd(155)_155Gd-PIMO: Membrane: Std.Dev.',
#         'Gd(155)_155Gd-PIMO: Cell: Mean', 'Gd(155)_155Gd-PIMO: Cell: Median',
#         'Gd(155)_155Gd-PIMO: Cell: Min', 'Gd(155)_155Gd-PIMO: Cell: Max',
#         'Gd(155)_155Gd-PIMO: Cell: Std.Dev.']
cols_to_drop.remove(target_variable) #remove the target variable from the list of variables to drop, as we already drop it later on
#%% load data
data=pandas.read_csv(csv_path)
#force str type on Patient column to avoid weird discretization issues
data=data.astype({'Patient': 'str'})
col_names=data.columns

data=data.drop(columns=[x for x in col_names if prune_from_dataset in x])
col_names=data.columns


#%% Prepare training and test data
X = data.drop(target_variable, axis=1) #Let the predictors X be all variables that aren't the target variable
X=X.drop(cols_to_drop,axis=1)



#OVERRIDE: ONLY USE MEASUREMENTS SPECIFIED UNDER MEASUREMENTS_OF_INTEREST
X=X[measurements_of_interest]



y = data[target_variable] #Let the target variable Y be, well, the target variable
#Encode target_variable classes as integer labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
y = label_encoder.transform(y)
y_orig=data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #Split the data such that 25% of the dataset is used for testing
#Perform z score normalization on the train and test predictor variables
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#%% Train model. In this case, we use the XGBoost classifier
model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)
startTime = datetime.now()
model.fit(X_train_scaled, y_train)
print(datetime.now() - startTime)
#%% Create importances dataframe
att_short=[]
for att in X_train.columns:
    if len(att.split('-'))>1:
            att_short.append(att.split('-')[1])
    else:
        att_short.append(att)

importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_,
    'Feature': att_short
})
importances = importances.sort_values(by='Importance', ascending=False) #This is the most "important" (get it?) variable, contains the list of measurements and what their importance was in training the classifier and receiving the subsequent test score

#%% Test model. Make predictions on X_test_scaled, compare output to y_test
# make predictions for test data
y_pred = model.predict(X_test_scaled)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
acc_title="Accuracy: %.2f%%" % (accuracy * 100.0)
print(acc_title)
#%% Create bar plot showing prediction scores
plt.close('all')

h=sns.barplot(data=importances[0:10],x='Feature',y='Importance')
plt.xticks(rotation=90)
plt.title('Top 10 Most Important Features Required For Prediction Of Variable: ' + target_variable + ' \n' + acc_title)
plt.tight_layout()
plt.savefig(figpath + '\\20_feature_PIMO_bar.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%% Create violin plot showing measurement comparison of different target_variable categories

#This one is a bit more complicated. We need to find the top 10 columns in data, perform the same z score normalization, bring in the target_variable
#and use that dataframe for plotting

plt.close('all')
#isolate the top 10 attributes that are most important for the classification
att_names=importances['Attribute'][0:10].tolist()
att_names_short=importances['Feature'][0:10].tolist()

#Find their values in the original dataset
top_feature_df=data[att_names]
#normalize the original dataset again (last time we subdivided the data to test/train BEFORE normalization)
top_feature_scaled=ss.fit_transform(top_feature_df)
#create a dataframe from the normalized values
top_feature_scaled=pd.DataFrame(data=top_feature_scaled,columns=att_names)
#and bring over the original classes
top_feature_scaled[target_variable]=data[target_variable]

nrows=2
ncols=5
num_std_to_include=2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(ncols*2.5, nrows*2))
axs=ax.ravel()
for curr_ax,curr_att,curr_att_short in zip(axs, att_names,att_names_short):
    # curr_ax.set_facecolor((0, 0, 0))
    # curr_ax.set_xticks([])
    # curr_ax.set_yticks([])
    #curr_ax.set_title(curr_short_measure + ' ',fontsize=12,y=1,pad=-15,loc='right',color='white')
    h = sns.violinplot(ax=curr_ax,x=target_variable,y=curr_att,data=top_feature_scaled[top_feature_scaled[curr_att]<(top_feature_scaled[curr_att].mean()+num_std_to_include*top_feature_scaled[curr_att].std())], palette="muted",scale='width',cut=0,inner="box",hue_order=data[target_variable].unique)

    curr_ax.set_title(curr_att_short,fontsize=12)
    curr_ax.set_aspect('equal')
    h.set(xlabel=None,ylabel=None)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(1)
plt.tight_layout()
plt.savefig(figpath + '\\20_feature_PIMO_violin.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%% Create an unnecessarily flamboyant circle bar plot
# initialize the figure
plt.close('all')

plt.figure(figsize=(20,10))

ax = plt.subplot(111, polar=True)
plt.axis('off')

# Constants = parameters controling the plot layout:
upperLimit = 100
lowerLimit = 30
labelPadding = 4
df=importances[0:20]
# Compute max and min in the dataset
maxval = df['Importance'].max()

# Let's compute heights: they are a conversion of each item value in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (maxval - lowerLimit) / maxval
heights = slope * df['Importance'] + lowerLimit
heights = df['Importance'] *500 #Overwrite the heights to this as their calculation sucks

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(df.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(df.index)+1))
angles = [element * width for element in indexes]
# Create color gradient for bars
data_color_normalized = [x / max(df['Importance']) for x in df['Importance']]
my_cmap = plt.cm.get_cmap('Blues')
colors = my_cmap(data_color_normalized)

# Draw bars
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white",
    color=colors,
)

# Add labels
for bar, angle, height, label in zip(bars,angles, heights, df["Feature"]):

    # Labels are rotated. Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"

    # Finally add the labels
    ax.text(
        x=angle, 
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=label, 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor") 
plt.title('Top 20 Most Important Features Required For Prediction Of Variable: ' + target_variable + ' \n' + acc_title)
plt.tight_layout()
plt.savefig(figpath + '\\20_feature_PIMO_circle.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#plt.close('all')
#%% To do:
    #Iteratively rerun the training, and test accuracy comparison on a list of the sorted features,
    #see how the model performs as you start including more features one by one. X axis is number of top importance features, y axis is accuracy on test set.
    #Have one line being only the "measurement of interest" features. Have a second line being all features derived from that given marker
#verify that we're working on the 18-feature set first
if len(importances) != len(measurements_of_interest):
    raise Exception("Script needs to be run on the 18-feature set")

#add new column that's just the marker name
importances['Marker']=[x.split(':',1)[0] for x in importances['Feature']]
importances.reset_index(inplace=True,drop=True)
#specify descriptive statistic strings to look for in column names
desc_stats=['Mean','Median','Min','Max','Std.Dev']
#find column names that contain the descriptive statistic keywords, but excluding ones relating to diameter
derived_feat_cols=[s for s in col_names if any(xs in s for xs in desc_stats)]
derived_feat_cols=[i for i in derived_feat_cols if not ('diameter' in i)]
#%% Iteratively run training, adding one more feature each time
attribute_list=[]
acc_score_list=[]
for curr_attribute in importances['Attribute']:
    #curr_attribute=importances['Attribute'][0]
    
    attribute_list.append(curr_attribute)
    
    #Prepare training and test data
    X=data[attribute_list]
    y=data[target_variable]
    #Encode target_variable classes as integer labels
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    y = label_encoder.transform(y)
    y_orig=data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #Split the data such that 25% of the dataset is used for testing
    #Perform Z score normalization
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    #Train model
    model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)
    startTime = datetime.now()
    model.fit(X_train_scaled, y_train)
    print(datetime.now() - startTime)
    #Test model
    y_pred = model.predict(X_test_scaled)
    # evaluate predictions
    acc_to_append=accuracy_score(y_test, y_pred)*100.0
    acc_score_list.append(acc_to_append)
    print("Accuracy: %.2f%%" % acc_to_append)
importances['Attribute_Cumulative_Accuracy']=acc_score_list
#%% Iteratively run training, adding all descriptive features relating to the marker each time

#To do: repeat the above, but this time include all derived features associated with the given curr_attribute marker
acc_score_list=[]
marker_list=[]
for marker in importances['Marker']:
    #marker=importances['Marker'][0]
    curr_derived_feats=[s for s in derived_feat_cols if marker in s]
    marker_list.extend(curr_derived_feats)
    #Prepare training and test data
    X=data[marker_list]
    y=data[target_variable]
    #Encode target_variable classes as integer labels
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    y = label_encoder.transform(y)
    y_orig=data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #Split the data such that 25% of the dataset is used for testing
    #Perform Z score normalization
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    #Train model
    model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)
    startTime = datetime.now()
    model.fit(X_train_scaled, y_train)
    print(datetime.now() - startTime)
    #Test model
    y_pred = model.predict(X_test_scaled)
    # evaluate predictions
    acc_to_append=accuracy_score(y_test, y_pred)*100.0
    acc_score_list.append(acc_to_append)
    print("Accuracy: %.2f%%" % acc_to_append)
importances['Descriptive_Statistics_Features_Cumulative_Accuracy']=acc_score_list
#%% Plot line plot showing how much the classifier improves when using descriptive statistics features
#construct a seaborn-compatibled dataframe with the features we intend to plot
plt.close('all')
line_df=importances[['Marker','Attribute_Cumulative_Accuracy','Descriptive_Statistics_Features_Cumulative_Accuracy']]
line_df=line_df.rename(columns={'Attribute_Cumulative_Accuracy':'Mean Intensity Only',"Descriptive_Statistics_Features_Cumulative_Accuracy":"Descriptive Statistics Features"})
line_df=line_df.melt('Marker',var_name='Feature Set',value_name='Accuracy')
h=sns.lineplot(data=line_df,x='Marker',y='Accuracy',hue='Feature Set', style="Feature Set", markers=True, dashes=False)
plt.xticks(rotation=90)
h.set(ylabel='Accuracy (%)',xlabel='Collective Markers Used')
plt.pause(1)
plt.tight_layout()
plt.savefig(figpath + '\\Accuracy_Improvement_With_Marker_Used',dpi=800,pad_inches=0.1,bbox_inches='tight')





"""
#%% Prepare training and test data
X = data.drop(target_variable, axis=1) #Let the predictors X be all variables that aren't the target variable
X=X.drop(cols_to_drop,axis=1)



#OVERRIDE: ONLY USE MEASUREMENTS SPECIFIED UNDER MEASUREMENTS_OF_INTEREST
X=X[measurements_of_interest]



y = data[target_variable] #Let the target variable Y be, well, the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #Split the data such that 25% of the dataset is used for testing
#Perform z score normalization on the train and test predictor variables
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#%% Train model. In this case, we use the XGBoost classifier
model = XGBClassifier()
startTime = datetime.now()
model.fit(X_train_scaled, y_train)
print(datetime.now() - startTime)
"""








