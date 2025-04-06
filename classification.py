# PTO-BM (Radiomics): Baseline Classifier
# Primary Tumor Origin Classifier in Brain Metastases
# ImagineQuant Lab | Department of Radiology & Biomedical Imaging | Yale School of Medicine
# Leon Jekel, MD student (5th year)
# Created (04/01/22)
# Updated (9/13/22)

""" 
Imports 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from operator import itemgetter
from tqdm.notebook import tqdm_notebook
from itertools import product
from statistics import mean, stdev
from ast import literal_eval  

from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb


"""
Define functions and search space for hyperparameter tuning
"""

search_space = {
    'estimator__n_estimators': [10],
    'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'estimator__max_depth': range(3, 10),
    'estimator__colsample_bytree': [i/10.0 for i in range(1, 3)],
    'estimator__gamma': [i/10.0 for i in range(3)],
    'estimator__n_jobs': [4]
  }       

def mean_auc_from_dict(auc_dict):
    dict_values = {}
    for i in range(6):
        dict_values['auc_' + str(i)] = mean(list(map(itemgetter(i), auc_dict)))
        dict_values['std_' + str(i)] = stdev(list(map(itemgetter(i), auc_dict)))
    return dict_values
def return_correlated_features(feature_values, feature_names, correlation_indicator = 'pearson', correlation_threshold = 0.8):
    correlated_feature_names = []
    corr_matrix = pd.DataFrame(feature_values, columns = feature_names).corr()
    for i, feature_name in enumerate(feature_names):
        for j in np.arange(i+1,len(feature_names)):
            if (np.abs(corr_matrix.iloc[i, j]) >= correlation_threshold):
                correlated_feature_names.append(feature_name)
                break
    return correlated_feature_names
def y_to_grid(df):   
    #for OneHotEncoding
    y = pd.DataFrame(np.zeros(len(df)))
    y[['1', '2', '3', '4', '5', '6']] = [0,0,0,0,0,0]   #initialize Matrix with binary outcome for every Prediction Class

    for i in df.index: 
        for j in np.arange(1, 7, 1):
            if df['Classifier Label_x'][i] == j:
                y.iloc[i, j] = 1
    y.drop([0], axis = 1, inplace = True)
    return y
def Find_Optimal_Cutoff_CZ(TPR, FPR, thresholds):   #CZ method
    y = TPR*(1- FPR)
    max_concordance = np.argmax(y) 
    optimal_threshold = thresholds[max_concordance]
    point = [FPR[max_concordance], TPR[max_concordance]]
    return optimal_threshold, point

"""
Load dataframe
"""
PATH = 'Users/jeleke/linked_radiomics.csv'
df = pd.read_excel(PATH)



"""
Model training & validation
"""

seed = 123
np.random.seed(seed)
auc_scores1 = []
fpr_all1 = []
tpr_all1 = []
threshold_all1 = []
feature_importances = []

all_labels = pd.DataFrame()                                    
all_predictions = pd.DataFrame()

glb = globals()
for i in range(6):
    glb['tprs' + str(i)] = []
    glb['aucs' + str(i)] = []

mean_fpr = np.linspace(0, 1, 100) #For ROC Curves

for i in tqdm_notebook(np.arange(15), 'Iteration'):
    df_shuffled = shuffle(df).copy().reset_index(drop =True)
    X = df_shuffled.drop(['Classifier Label_x'], axis=1).copy()
    y = y_to_grid(df_shuffled) #OneHotEncoding of class label
    y_max = y.idxmax(axis=1)
    groups = df_shuffled['Accnum']
    
    kf = StratifiedGroupKFold(n_splits=5)
    cv_splits = list(kf.split(X,y_max, groups))
    
    for fold, (train_index, test_index) in tqdm_notebook(enumerate(cv_splits)):
        X_train, X_test = X.iloc[train_index].copy(),X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index].copy(),y.iloc[test_index].copy()
        y_max_train, y_max_test = y_max.iloc[train_index].copy(), y_max.iloc[test_index].copy()
        accnum_train, accnum_test = groups.iloc[train_index].copy(), groups.iloc[test_index].copy()
        
        test_index = X_test.index.tolist()
        features = X_train.columns
        
        # Standardization of continuous features
        scaler = StandardScaler()        
        X_train = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns = features)        
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns = features)
        
        # Removing correlated features       
        train_correlated_feature_names = return_correlated_features(feature_values = X_train, feature_names = X_train.columns) #correlated feature removal for every features apart from the last
        X_train = X_train.drop(columns = train_correlated_feature_names)        
        X_test = X_test.drop(columns = train_correlated_feature_names)


        # Feature selection with mRMR
        feat_number = 20
        selected_feat = pymrmr.mRMR(X_train_outer_loop, 'MIQ', feat_number)
        X_train_outer_loop = X_train_outer_loop[selected_feat]
        X_test_outer_loop = X_test_outer_loop[selected_feat]

        # Hyperparameter tuning
        clf = RandomizedSearchCV(estimator=OneVsRestClassifier(xgb.XGBClassifier()),
                           param_distributions=search_space)  #default 5fold splitter
        clf.fit(X_train, y_train)
        hyperparam = clf.best_params_
        
        # Predictor fitting
        predictor = clf.best_estimator_
        predictor.fit(X_train, y_max_train)
        y_pred = predictor.predict_proba(X_test)
        y_pred_hard = predictor.predict(X_test)
        
        """ 
        Store results from each iteration 
        """ 
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresh = dict()
        roc_auc = dict()
        feature_importances_dic = {}

        for i in range(6):
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test.iloc[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            feature_importances_dic[str(i)] = dict(zip(X_train.columns.tolist(), predictor.estimators_[i].feature_importances_))

        threshold_all1.append(thresh)
        auc_scores1.append(roc_auc)
        fpr_all1.append(fpr)
        tpr_all1.append(tpr)  
        feature_importances.append(feature_importances_dic) #collect class-wise feature importances
        all_predictions = all_predictions.append(pd.DataFrame(y_pred)).reset_index(drop = True)
        all_labels = all_labels.append(pd.DataFrame(y_max_test)).reset_index(drop = True)

        
       """ 
       ROC Curve Visualization 
       """

        fix, ax = plt.subplots()
        for i in range(6):            
            glb['viz' + str(i)] = RocCurveDisplay.from_predictions(y_test.iloc[:, i], y_pred[:, i],
                name="ROC fold {}".format(fold),
                alpha=0.3,
                lw=1,
                ax=ax,
                )
            interp_tpr = np.interp(mean_fpr, glb['viz' + str(i)].fpr, glb['viz' + str(i)].tpr)
            interp_tpr[0] = 0.0
            glb['tprs' + str(i)].append(interp_tpr)
            glb['aucs' + str(i)].append(glb['viz' + str(i)].roc_auc)

"""
Plot ROC curves
"""

tpr_list = [tprs0, tprs1, tprs2, tprs3, tprs4, tprs5]
labels = ['Breast', 'GI', 'SCLC', 'Melanoma', 'NSCLC', 'Others']
for i, tprno in enumerate(tpr_list):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tpr_list[i], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tpr_list[i])

    line1, = ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
        )

    std_tpr = np.std(tpr_list[i], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(   #Grey range of AUC between iterations
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
        )

    line2, = ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8) #Chance Line
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=labels[i]
        )
    ax.legend(handles = [line1, line2], loc = 'lower right', prop = {'size': 7.5})
    plt.show()


"""
Determination of optimal ROC cut-off value and average sensitivity/specificity 
"""
names = ['Breast', 'GI', 'SCLC', 'Melanoma', 'NSCLC', "Others"]
metrics = dict()
for i in range(6): 
    sens = []
    spec = []
    for j in range(len(fpr_all1)):
        fpr = fpr_all1[j][i]  
        tpr = tpr_all1[j][i]
        threshold = threshold_all1[j][i]

        thresh, coordinate = Find_Optimal_Cutoff_CZ(tpr, fpr, threshold)
        fpr_opt, tpr_opt = coordinate
        sens.append(tpr_opt)
        spec.append(1-fpr_opt)
    metrics[names[i]] = [mean(sens), stdev(sens), mean(spec), stdev(spec)]
    metrics_df = pd.DataFrame(metrics, index = ['Sens', 'SensStd', 'Spec', 'SpecStd']).T

"""
Store feature importances in Excel file with different sheets for every class
"""

PG_0, PG_1, PG_2, PG_3, PG_4, PG_5 = []
for i in range(len(feature_importances)):
    PG_0.append(feature_importances[i]['0'])
    PG_1.append(feature_importances[i]['1'])
    PG_2.append(feature_importances[i]['2'])
    PG_3.append(feature_importances[i]['3'])
    PG_4.append(feature_importances[i]['4'])
    PG_5.append(feature_importances[i]['5'])
    
def get_all_features(listdict): 
    len_array = []
    for i in range(len(listdict)):
        len_array.append(len(listdict[i]))
    max_value = max(len_array)
    pos_ax = len_array.index(max_value)
    feature_list = dict()
    for j, value in enumerate(listdict[pos_ax]):
        try:
            feature_list[value] = list(map(itemgetter(value), listdict))
        except: 
            feature_list[value] =listdict[pos_ax][value]
    return feature_list
    
def make_tuple(feature_list) :
    feature_tuple = []
    for i in range(len(feature_list)): 
        a = [list(feature_list.keys())[i], list(feature_list.values())[i]] 
        feature_tuple.append(a) 
    return feature_tuple

buffer = 'FeatureImportances.xlsx'
with pd.ExcelWriter(buffer, engine = 'openpyxl') as writer:
    pd.DataFrame(make_tuple(get_all_features(PG_0))).to_excel(writer, sheet_name = 'Base Estimator 0')
    pd.DataFrame(make_tuple(get_all_features(PG_1))).to_excel(writer, sheet_name = 'Base Estimator 1')
    pd.DataFrame(make_tuple(get_all_features(PG_2))).to_excel(writer, sheet_name = 'Base Estimator 2')
    pd.DataFrame(make_tuple(get_all_features(PG_3))).to_excel(writer, sheet_name = 'Base Estimator 3')
    pd.DataFrame(make_tuple(get_all_features(PG_4))).to_excel(writer, sheet_name = 'Base Estimator 4')
    pd.DataFrame(make_tuple(get_all_features(PG_5))).to_excel(writer, sheet_name = 'Base Estimator 5')

"""
Display and visualize feature importances
"""

def split_observations(df):
    def split_dataframe_rows(df,column_selectors):          #https://gist.github.com/jlln/338b4b0b55bd6984f883#:~:text=Efficiently%20split%20Pandas%20Dataframe%20cells%20containing%20lists%20into,%3D%20the%20symbol%20used%20to%20perform%20the%20split
        def _split_list_to_rows(row,row_accumulator,column_selector):
            split_rows = {}
            max_split = 0
            for column_selector in column_selectors:
                split_row = row[column_selector]
                split_rows[column_selector] = split_row
                if len(split_row) > max_split:
                    max_split = len(split_row)
                
            for i in range(max_split):
                new_row = row.to_dict()
                for column_selector in column_selectors:
                    try:
                        new_row[column_selector] = split_rows[column_selector].pop(0)
                    except IndexError:
                        new_row[column_selector] = ''
                row_accumulator.append(new_row)
        new_rows = []
        df.apply(_split_list_to_rows,axis=1,args = (new_rows,column_selectors))
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        return new_df

    df = df.sort_values('Mean', ascending = False)[:10] #10 highest ranking features 
    df_split = split_dataframe_rows(df = df, column_selectors = ['ValuesList']) #split dataframe   
    return df_split

sheet_names = ['Base Estimator 0', 'Base Estimator 1', 'Base Estimator 2', 'Base Estimator 3', 'Base Estimator 4', 'Base Estimator 5']
for i, val in enumerate(sheet_names): 
    glb['features_' + str(i)]= pd.read_excel(r'FeatureImportances.xlsx', sheet_name = val, converters={'1': pd.eval} )
    glb['features_' + str(i)].drop('Unnamed: 0', axis = 1, inplace = True)
    glb['features_' + str(i)].columns = ['Feature', 'Values']
    
classes = ['features_0', 'features_1', 'features_2', 'features_3', 'features_4', 'features_5']
classes_mod = []
for i in range(len(classes)): 
    classes_mod.append(classes[i] + "_mod")

for i, label in enumerate(classes_mod): 
    glb[label] = split_observations(dataframe_modification(glb[classes[i]]))
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Feature", y="ValuesList", data=glb[label], color=".25", showmeans = True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
