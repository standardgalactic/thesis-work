# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:55:48 2022

@author: Mark Zaidi

Aim is to take a classifier that predicts whether a cell came from a patient with a specific cause of transplant
rejection, and translate this to predict the patient's cause. Several considerations need to be made
1. The classifier will likely have a lower accuracy. Not all cells of a given patient will have the
same classification. As a result, 49% may be ABMR and 51% may be ACR. If we use majority rules and
the ground truth is ABMR, we'll end up predicting it as ACR

2.  We can't randomly split the test and training datasets for a 25/75 split. We need to discretely group patients
as belonging to the test or training set. Currently, my best solution is to accept a list of patient
IDs to include in the training set, and any patient not in this list will make up the test set. We can then print out
What the test/train split percentage looks like, and have it be close to our desired 25/75 split

3. Majority rules may not be the best means of translating cell predictions to the patient scale.
We may need to use a second XGBoost classifier, or a different classifier that can take multiple
datapoints to predict one label

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
#%% Declare constants
#Priya
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\per image cell measurements\aggregated_csv\processed_cell_measurements.parquet'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\figures\Classifier\patient_prediciton'
measurements_of_interest=['Pr(141)_141-SMA: Cell: Mean','Nd(142)_142Nd-CD19: Cell: Mean','Nd(143)_143Nd-Vimentin: Cell: Mean','Nd(144)_144Nd-cd14: Cell: Mean','Nd(146)_146Nd-CD16: Cell: Mean','Nd(148)_148-Pan-Ker: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Sm(150)_150Sm-PD-L1: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Sm(154)_154Sm-CD11c: Cell: Mean','Gd(155)_155Gd-FoxP3: Nucleus: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-E_Cadherin: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-Vista: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Er(166)_166Er-CD45RA: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tmp-CollagenI: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(171)_171Yb-HistoneH3: Nucleus: Mean','Yb(173)_173Yb-CD45RO: Cell: Mean','Yb(174)_174Yb-HLA-DR: Cell: Mean','Lu(175)_175Lu-Beta2M: Cell: Mean','Yb(176)_176Yb-Nak-ATPase: Cell: Mean','Ir(193)_193Ir-NA2: Nucleus: Mean']                                      
seed=69
prune_from_dataset=['BCK','Xe','Nd(145)','BKG','Dy(164)','Ho(165)','Pt(195)','Hg(202)'] #remove any measurements relating to background channels

target_variable="Code" #name of column you want to use as classification
#Drop columns from training that we either can't use (strings) or are very obviously not going to benefit the classifier (e.g. centroid X um)
cols_to_drop=['Image', 'Name', 'Class', 'Parent', 'ROI', 'Centroid X µm','Centroid Y µm',target_variable,'KR','IMC_ROI','training_mask']
patient_col='KR'

cols_to_drop.remove(target_variable) #remove the target variable from the list of variables to drop, as we already drop it later on
#MANUAL TRAINING LISTS
#One of each - Normal, cABMR, BK, C .Pyel, ACR, Mixed - one of each in that order below. About 25% of dataset
#training_list=['KR21-4226 A4','KR-20-4625 A4','KR-19-4424 A1','KR-16-6637 A1','KR-21-4225 A4','KR-21-4285 A4']
#Two of each - Normal, Normal, cABMR, cABMr, BK, BK, C. Pyel, C. Pyel, ACR, ACR, Mixed, Mixed. About 50% of dataset
#training_list=['KR21-4226 A4','KR21-4222 A2','KR-20-4625 A4','KR-14-6142 A1','KR-19-4424 A1','KR-17-50724 A3','KR-16-6637 A1','KR-18-6091 A3','KR-21-4225 A4','KR-18-5588 A3','KR-21-4285 A4','KR-21-4828 A4']
#Three of each (if available) - Normal, Normal, Normal, cABMR, cABMR,cABMR, BK, BK, BK, C. Pyel, C. Pyel, ACR, ACR, Mixed, Mixed, Mixed. About 63% of dataset
#training_list=['KR21-4226 A4','KR21-4222 A2','KR21-4213 A2','KR-20-4625 A4','KR-14-6142 A1','KR-17-2231','KR-19-4424 A1','KR-17-50724 A3','KR-17-51845 A2','KR-16-6637 A1','KR-18-6091 A3','KR-21-4225 A4','KR-18-5588 A3','KR-21-4285 A4','KR-21-4828 A4','KR-19-5581 A4']
#Four of each (if available) and special cases - Normal, Normal, Normal, Normal, cABMR, cABMR,cABMR, cABMR, BK, BK, BK, BK, C. Pyel, C. Pyel, ACR, ACR, Mixed, Mixed, Mixed, ABMR*,Normal. About 75% of dataset, just like original test/train split
training_list=['KR21-4226 A4','KR21-4222 A2','KR21-4213 A2','KR-19-1310 A2','KR-20-4625 A4','KR-14-6142 A1','KR-17-2231','KR-16-2317 A1','KR-19-4424 A1','KR-17-50724 A3','KR-17-51845 A2','KR-18-4424 A3','KR-16-6637 A1','KR-18-6091 A3','KR-21-4225 A4','KR-18-5588 A3','KR-21-4285 A4','KR-21-4828 A4','KR-19-5581 A4','KR-17-2602','KR21-4227 A2']

#%% load data
if (csv_path.rsplit('.',maxsplit=1)[1]=='csv'):
    data=pandas.read_csv(csv_path,low_memory=False)
elif (csv_path.rsplit('.',maxsplit=1)[1]=='parquet'):
    data=pandas.read_parquet(csv_path)
else:
    raise Exception('Unable to detect file format to read')

#force str type on Patient column to avoid weird discretization issues
#data=data.astype({'Patient': 'str'})
col_names=data.columns

data=data.drop(columns=[s for s in col_names if any(xs in s for xs in prune_from_dataset)])
col_names=data.columns
#Replace ABMR* and cABMR as ABMR
data.Code=data.Code.replace({'cABMR':'ABMR','ABMR*':'ABMR'})
#Drop rows containing a specific value in the target_variable
data=data[~data[target_variable].str.contains('OMIT|Missing')]
#%% Identify rows to use for training
mask=data[patient_col].str.contains('|'.join(training_list))
data['training_mask']=mask
print('train_size: ', (sum(mask)/len(mask)))
#%% Prepare training and test data
X = data.drop(target_variable, axis=1) #Let the predictors X be all variables that aren't the target variable
X=X.drop(cols_to_drop,axis=1)
#OVERRIDE: ONLY USE MEASUREMENTS SPECIFIED UNDER MEASUREMENTS_OF_INTEREST
#X=X[measurements_of_interest]
#Manually define training and test datasets
X_train=X[data['training_mask']==True]
X_test=X[data['training_mask']==False]

y = data[target_variable] #Let the target variable Y be, well, the target variable
#Encode target_variable classes as integer labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
y = label_encoder.transform(y)
y_orig=data[target_variable]
y_train=y[data['training_mask']==True]
y_test=y[data['training_mask']==False]
#Randomly split data, uncomment below if you want to use the old method
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #Split the data such that 25% of the dataset is used for testing

#Perform z score normalization on the train and test predictor variables
#And yes, this should be done after splitting. The test set represents an independent dataset, so we want to scale independently without relying mean/variance info from the training set
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
#%% Train model. In this case, we use the XGBoost classifier
model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed,tree_method='gpu_hist')
#model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)

startTime = datetime.now()
model.fit(X_train_scaled, y_train)
print(datetime.now() - startTime)
#Save model trained on all features
model.save_model(figpath + '\\All_Feature_Trained.json')
#%% Test model. Make predictions on X_test_scaled, compare output to y_test
# make predictions for test data
y_pred = model.predict(X_test_scaled)
# evaluate predictions
cell_accuracy = accuracy_score(y_test, y_pred)
cell_acc_title="Cell prediction accuracy: %.2f%%" % (cell_accuracy * 100.0)
print(cell_acc_title)
#%% Generate patient-level predictions from cell-level
masked_test_data=data[data['training_mask']==False].copy()
masked_test_data['y_pred_cell']=y_pred #this is a copy of the test dataset, with predictions for each cell appended
y_pred_patient=[] #predicted code for the patient. Right now, uses mode value
y_pred_patient_list=masked_test_data[patient_col].unique().tolist() #List of patient KR numbers
y_test_patient=[] #ground truth code for the patients
y_pred_patient_percell=[] #y_pred_patient, but extended for all cells of the given patient

for patient in masked_test_data[patient_col].unique():
    curr_data=masked_test_data[masked_test_data[patient_col]==patient]#Select only data belonging to current test patient
    y_test_patient.append(label_encoder.transform(curr_data[target_variable].unique().tolist()))#encode the patient's ground truth label into a labelencoder
    
    mode_val=curr_data['y_pred_cell'].mode().tolist() #compute the most commonly repeated value (mode) for the current patient's predicted cell types
    y_pred_patient.append(mode_val)
    y_pred_patient_percell.extend(mode_val for _ in range(len(curr_data)))
masked_test_data['y_pred_patient']=[item for sublist in y_pred_patient_percell for item in sublist]
if(len(y_test_patient)!=len(y_pred_patient_list)):
    raise Exception('Potential contamination, number of patient ground truth predictions doesnt match number of patients')
# evaluate predictions
cellmode_accuracy = accuracy_score(y_test, y_pred_patient_percell)
cellmode_acc_title="Cell prediction accuracy assigned by patient mode: %.2f%%" % (cellmode_accuracy * 100.0)
print(cellmode_acc_title) 
# evaluate predictions
patient_accuracy = accuracy_score(y_test_patient, y_pred_patient)
patient_acc_title="Patient prediction accuracy: %.2f%%" % (patient_accuracy * 100.0)
print(patient_acc_title) 
#Alright, the accuracy is good, but don't get cocky. Its high because you're STILL predicting on the cell scale, but making sure all cell predictions of a given patient are the same.
#You need to run accuracy_score on a variable of length equal to the number of patients in the test set (so like 7)
#Else, you're weighing accuracy by the number of cells predicted correctly and NOT the number of patients predicted correctly.


#AHHH, THE ACCURACY IS STILL GOOD, IN YOUR FACE, ME!
#So we do get a good accuracy, but we're only evaluating like 7 test cases. Having even 1 misclassification more,
#and the accuracy will drop even more. Also, create a dataframe containing the percentages of each predicted cell class for each patient
#%% Compute ROI prediction accuracy - if given a random ROI, how accurately can you predict transplant rejection cause
masked_test_data=data[data['training_mask']==False].copy()
masked_test_data['y_pred_cell']=y_pred #this is a copy of the test dataset, with predictions for each cell appended
y_pred_ROI=[] #predicted code for the ROI. Right now, uses mode value
y_pred_ROI_list=masked_test_data['Image'].unique().tolist() #List of ROIs
y_test_ROI=[] #ground truth code for the ROI
y_pred_ROI_percell=[] #y_pred_ROI, but extended for all cells of the given ROI

for ROI in masked_test_data['Image'].unique():
    curr_data=masked_test_data[masked_test_data['Image']==ROI]#Select only data belonging to current test ROI
    y_test_ROI.append(label_encoder.transform(curr_data[target_variable].unique().tolist()))#encode the ROI's ground truth label into a labelencoder
    
    mode_val=curr_data['y_pred_cell'].mode().tolist() #compute the most commonly repeated value (mode) for the current ROI's predicted cell types
    y_pred_ROI.append(mode_val)
    y_pred_ROI_percell.extend(mode_val for _ in range(len(curr_data)))
masked_test_data['y_pred_ROI']=[item for sublist in y_pred_ROI_percell for item in sublist]
if(len(y_test_ROI)!=len(y_pred_ROI_list)):
    raise Exception('Potential contamination, number of ROI ground truth predictions doesnt match number of ROI')
# evaluate predictions
ROI_accuracy = accuracy_score(y_test_ROI, y_pred_ROI)
ROI_acc_title="ROI prediction accuracy: %.2f%%" % (ROI_accuracy * 100.0)
print(ROI_acc_title) 
accs=[cell_accuracy * 100,cellmode_accuracy * 100,ROI_accuracy * 100.0,patient_accuracy * 100.0]

#%% Create importances dataframe
att_short=[]
for att in X_train.columns:
    if len(att.split('-'))>1:
            att_short.append(att.split('-',1)[1])
    else:
        att_short.append(att)

importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_,
    'Feature': att_short
})
importances = importances.sort_values(by='Importance', ascending=False) #This is the most "important" (get it?) variable, contains the list of measurements and what their importance was in training the classifier and receiving the subsequent test score
#%% Determine which feature belongs to which marker, and to which set (desc_stats, mean, cluster, etc.)
measure_short=[i.split('_', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]
marker_grouped_feat_df=pd.DataFrame()
#ERROR BELOW: because you never created a NA2 or NA1 class, we don't have
for measure in measure_short_for_fname:
    marker_grouped_feat_df[measure] = list(filter(lambda x: measure in x, importances['Attribute']))
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
    #model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)
    model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed,tree_method='gpu_hist')

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
    #model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed)
    model = XGBClassifier(use_label_encoder=False,verbosity=0,random_state=seed,tree_method='gpu_hist')

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
model.save_model(figpath + '\\DescStat_Feature_Trained.json')
















#%% To do


#   - Train an image classifier. This will defeat the purpose of using any QuPath processes, and will require quite a lot of work. At that point, is it really worth it? (THIRD CHOICE) 
#       -Hold on, what if we create a new project, consisting of WSI annotations around each ROI, and assign them a class based on what type of transplant rejeciton they were?
#       We have good control over training selection, markers used, and can compute measurements for each annotation to get a "percent likelihood" of an ROI belonging to a given transplant rejection cause
#       But, unlike a multichannel image classifier, we can't use GPU acceleration, so that might suck. Also, normalization can only be per-image. Think about it...



#%% Create bar plot showing prediction scores
# plt.close('all')

# h=sns.barplot(data=importances[0:10],x='Feature',y='Importance')
# plt.xticks(rotation=90)
# plt.title('Top 10 Most Important Features Required For Prediction Of Variable: ' + target_variable + ' \n' + acc_title)
# plt.tight_layout()
# plt.savefig(figpath + '\\' +str(len(importances)) + '_feature_bar.png',dpi=800,pad_inches=0.1,bbox_inches='tight')






















