import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
#Impute Libraries
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn import metrics
from sklearn.impute import SimpleImputer
#Import library for logistic regression
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,accuracy_score
from math import *
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from math import *
import sys
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import matplotlib as mpl
from scipy import interp
import sweetviz as sv
import math
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from scipy.stats import randint as sp_randint
import pickle
import shap
import functools
from abc import ABC, abstractmethod
from os.path import join as pjoin
import catboost as cb
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score, cross_val_predict, LeaveOneOut, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, accuracy_score, recall_score, make_scorer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as MICE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from scipy import interp
from scipy.stats import randint as sp_randint
import sweetviz as sv
import shap
import pickle
import functools
lb = preprocessing.LabelBinarizer()








SEED = 2019
def train_test_split(df, test_size=30, random_seed=SEED):
    df = df.sample(frac=1, random_state=random_seed)
    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]
    return train_df, test_df
    
# Function to normalize data
def normalize_data(X_train, X_test):
    transformer = preprocessing.QuantileTransformer(output_distribution='normal')
    transformer.fit(X_train)
    nor_X_train = transformer.transform(X_train)
    nor_X_test = transformer.transform(X_test)
    return nor_X_train, nor_X_test
    
def convert_array(df, subtype=False):
    if not subtype:
        X = df.iloc[:, :-2].values
        
        y = df.iloc[:, -2].values
        
    else:
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2].values

    return X, y
class CVModelStorage:
    def __init__(self, auc=[], fprs=[], tprs=[], model=[]):
        self.auc = auc
        self.fprs = fprs
        self.tprs = tprs
        self.model = model
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_pred = []

    def get_mean_auc(self):
        return np.mean(self.auc)

class Model(ABC):
    def __init__(self, model_name, param_dict):
        """
        Arguments:
        model_name - name of model
        param_dict - params to be passed to model
        """

        #access actual self.model underneath for increased control if needed
        self.model_name = model_name
        self.model = self.init_model(model_name, param_dict)
        self.cv_storage = CVModelStorage([], [], [], []) #store results of each fold of cross val
        self.argmax_ind = -1 #stores the index of the model with best score, within cv_storage lists
        self.params_dict = param_dict
        self.train_splits = []

    @abstractmethod
    def init_model(self, model_name, param_dict):
        """ To be implemented by subclasses (linear and forest), big switch statement on model_name flag

            Returns model initialized with param_dict parameters
        """
        pass

    def avgd_folds_decision_function(self, test_set):
        temp = np.zeros(len(test_set))
        folds = len(self.cv_storage.model)
        for i in range(folds):
            k = pickle.loads(self.cv_storage.model[i])
            if hasattr(k, 'predict_proba'):
                temp += k.predict_proba(test_set)[:, 1]
            elif hasattr(k, 'decision_function'):
                temp += k.decision_function(test_set)
        return temp * (1 / folds)

    def avgd_folds_decision_function_multiclass(self, test_set):
        temp = []
        folds = len(self.cv_storage.model)
        print(folds)
        for i in range(folds):
            clf = pickle.loads(self.cv_storage.model[i])
            if hasattr(clf, 'predict_proba'):
                temp.append(clf.predict_proba(test_set))
            else:
                print("Multiclass currently only supports predict_proba")
        return np.dstack(temp).mean(axis=2)

    def decision_function(self, test_set):
        """ Consistency between predict_proba and decision_function
        """
        score = None
        if hasattr(self.model, 'predict_proba'):
            score = self.model.predict_proba(test_set)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(test_set)
        return score

    def decision_function_multiclass(self, test_set):
        """ Return the multiclass predict_proba values
        """
        score = None
        if hasattr(self.model, 'predict_proba'):
            score = self.model.predict_proba(test_set)
        else:
            print("Multiclass currently only supports predict_proba")
        return score

    def run_cv(self, X, y, n_splits, seed=SEED):
        """ Run cross val with seed to standardize across models
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)

        for train, test in cv.split(X, y):
            self.model.fit(X[train], y[train])
            score = self.decision_function(X[test])
            fpr, tpr, _ = roc_curve(y[test], score)
            a = auc(fpr, tpr)
            self.cv_storage.auc.append(a)
            self.cv_storage.X_train.append(X[train])
            self.cv_storage.y_train.append(y[train])
            self.cv_storage.X_test.append(X[test])
            self.cv_storage.y_test.append(y[test])
            self.cv_storage.y_pred.append(score)
            self.cv_storage.fprs.append(fpr)
            self.cv_storage.tprs.append(tpr)
            self.cv_storage.model.append(pickle.dumps(self.model))
            if isinstance(self, LinearModel):
                self.train_splits.append(X[train])

    def run_cv_multiclass(self, X, y, n_splits, seed=SEED):
        """ Run cross validation with seed to standardize across models
        Support for multiclass problems.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)

        for train, test in cv.split(X, y):
            self.model.fit(X[train], y[train])
            score = self.decision_function_multiclass(X[test])

            # make a one hot multiclass label
            y_one_hot = np.zeros(score.shape)
            for idx, label in enumerate(y[test]):
                y_one_hot[idx, label] = 1

            fpr, tpr, _ = roc_curve(y_one_hot.ravel(), score.ravel())
            a = auc(fpr, tpr)

            self.cv_storage.auc.append(a)
            self.cv_storage.fprs.append(fpr)
            self.cv_storage.tprs.append(tpr)
            self.cv_storage.model.append(pickle.dumps(self.model))
            if isinstance(self, LinearModel):
                print(len(self.cv_storage.model))
                self.train_splits.append(X[train])

    def get_prediction_stats(self, true_values, pred_values):
        """ Gets the fpr, tpr, and AUC

        Arguments:
            true {List} -- True predictions
            pred {List} -- predict_proba output
        """
        fpr, tpr, _ = roc_curve(true_values, pred_values)
        a = auc(fpr, tpr)
        return fpr, tpr, a

    def update_model_parameters(self, params_dict):
        """ Update the model parameters to the params_dict
        Clears existing cv_storage
        """
        self.model = self.init_model(self.model_name, params_dict)
        self.params_dict = params_dict
        self.cv_storage = CVModelStorage([], [], [], [])

class LinearModel(Model):
    def init_model(self, model_name, param_dict):
        # Linear model subclass implementation
        self.train_splits = []
        self.is_elastic = False
        if model_name == 'logistic':
            return LogisticRegression(**param_dict)
        elif model_name == 'ridge':
            return RidgeClassifier(**param_dict)
        elif model_name == 'lasso':
            return LogisticRegression(**param_dict) #l1 penalty instead
        elif model_name == 'svm':
            return SVC(**param_dict)
        elif model_name == 'rbf':
            return SVC(**param_dict)
        elif model_name == 'elastic':
        	self.is_elastic = True
        	return LogisticRegression(**param_dict)
        return None

    def top_n_mz_coefficients(self, col_names, n=5):
        # Feature importance based on coefficients of linear model

        coefs = np.abs(self.model.coef_.ravel())
        ind = np.argpartition(coefs, -n)[-n:]
        ind = ind[np.argsort(coefs[ind])][::-1]
        return col_names[ind]

    def grid_search(self, nfolds, X, y, seed=None, visualize=True, label_sets=None):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state = None)
        alphas = np.logspace(-1, 4, 20)
        tuned_params = [{'C': alphas}]
        if self.is_elastic:
        	l1_ratios = np.linspace(0, 1, num=20)
        	tuned_params = [{'C': alphas, 'l1_ratio' : l1_ratios}]
        if isinstance(self.model, RidgeClassifier):
            tuned_params = [{'alpha': alphas}]

        if label_sets:
            compute_auc = functools.partial(compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, scoring=compute_auc)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False)

        grid.fit(X, y)

        if visualize:
            scores = grid.cv_results_['mean_test_score']
            scores_std = grid.cv_results_['std_test_score']
            plt.figure().set_size_inches(8, 6)
            plt.semilogx(alphas, scores)
            std_error = scores_std / np.sqrt(nfolds)

            # plot error lines showing +/- std. errors of the scores
            std_error = scores_std / np.sqrt(nfolds)

            plt.semilogx(alphas, scores + std_error, 'b--')
            plt.semilogx(alphas, scores - std_error, 'b--')

            # alpha=0.2 controls the translucency of the fill color
            plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

            plt.ylabel('CV score +/- std error')
            plt.xlabel('alpha')
            plt.axhline(np.max(scores), linestyle='--', color='.5')
            plt.xlim([alphas[0], alphas[-1]])

        return grid.best_params_

    def cv_shap_values(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models

        Arguments:
            X_test {Arry} -- test samples
            y_test - useless, just for compatability with forests
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.LinearExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        # explainers = [shap.KernelExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        # [shap.KernelExplainer(svm.predict_proba, X_train, link="logit")]
        total = []

        for explainer in explainers:
            shap_values = explainer.shap_values(X_test)
            if len(total) == 0:
                total = np.zeros(shap_values.shape)
            else:
                total += shap_values
        aggregated_shap = total / len(explainers)
        return aggregated_shap

    def cv_shap_values_multiclass(self, X_test, y_test):
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.LinearExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        shap_values = [explainer.shap_values(X_test) for explainer in explainers]
        num_classes = len(shap_values[0])
        output_shap = []
        for class_num in range(num_classes):
            class_shap_values = [shaps[class_num] for shaps in shap_values]
            stacked_shap_values = np.dstack(class_shap_values).mean(axis=2)
            output_shap.append(stacked_shap_values)

        return output_shap

def compute_auc_binarized(clf, X, y_true, label_sets):
    """ Compute a multiclass AUC for the subproblems in classification.
    Average the AUC across problems to get a single score.

    clf {sklearn Classifier} -- Classifier to use for evaluation
    X {np.array} -- features array
    y_true {np.array} -- labels array
    label_sets {List} -- List of lists of lists outlining the subproblems to evaluate
        For example: 
            [[[0,1], [2,3]], [[1], [0,2,3]]]
            This defines two label_sets: label (0,1) vs (2,3)
            and label (1) vs (0,2,3). The AUC for both is averaged.
    """
    y_pred = clf.predict_proba(X)
    aucs = []
    for label_set in label_sets:
        pos = y_pred[:, label_set[0]].sum(axis=1)
        pos_label = [1 if i in label_set[0] else 0 for i in y_true]
        auc = roc_auc_score(pos_label, pos)
        aucs.append(auc)
    auc = np.mean(aucs)

    return auc


class LGBM(Model):
    def init_model(self, model_name, params_dict):
        n_jobs = os.cpu_count()

        clf = lgb.LGBMClassifier(**params_dict,n_jobs=n_jobs)
        return clf

    def cv_shap_values(self, X_test, y_test):

        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.TreeExplainer(clf) for clf in models]
        shap_values = [explainer.shap_values(X_test, y_test) for explainer in explainers]
        aggregated_shap = np.dstack([shap_matrix[1] for shap_matrix in shap_values]).mean(axis=2)
        return aggregated_shap


    def random_search(self, nfolds, X, y, label_sets=None, seed=None):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state=seed)
        np.random.seed(seed)
        tuned_params = \
            {'num_leaves': [4,8,16,32,64,128],
             'max_depth': [2,4,8],
             'min_data_in_leaf': [2,4,8,16,32],
            }

        if label_sets:
            compute_auc = functools.partial(compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42, scoring=compute_auc)
            grid.fit(X, y)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42)
            grid.fit(X, y)

        return grid.best_params_




import numpy as np


def train_lgbm_model(X_train, y_train, random_search_iter=4):
    """
    Trains an LGBM model using provided training data.
    
    Parameters:
    X_train (DataFrame or ndarray): Normalized training features.
    y_train (Series or ndarray): Training labels.
    random_search_iter (int): Number of iterations for random search.
    
    Returns:
    LGBM: Trained LGBM model.
    """
    # Initialize the LGBM model
    lgbm_model = LGBM('LGBM', {})
    
    # Perform random search to find best hyperparameters
    # best_params = lgbm_model.random_search(random_search_iter, X_train, y_train)
    best_params = {'max_depth': 4, 'min_data_in_leaf': 16, 'num_leaves': 8}
    
    # Re-initialize the LGBM model with the best parameters
    lgbm_model = LGBM('LGBM', best_params)
    
    # Run cross-validation
    lgbm_model.run_cv(X_train, y_train, random_search_iter)
    
    return lgbm_model

def get_features(shap_values, X_test, col_names, num_features=10):

    # Determine median value for each feature
    median_value = np.median(X_test, axis=0)
    # Keep where feature contributes positives to classification (shap_values > 0)
    # and the value is greater than the median value
    pos_effect = (shap_values >= 0) * (X_test >= median_value) * (X_test)
    # Percentage high value features contrinute to positive or negative effects
    percent_pos_effect = (pos_effect > 0).sum(axis=0) / pos_effect.shape[0]
    color = percent_pos_effect
    # compute feature importance
    fi_shap = abs(shap_values).sum(0)
    # Normalize
    fi_shap = fi_shap / fi_shap.sum()
    ind = (-fi_shap).argsort()[:num_features]
    return fi_shap[ind], color[ind], col_names[ind]

def feature_selection_metabolomics(all_data,col_names):
    train, test = train_test_split(all_data, test_size=50)
    
    # Convert to arrays
    X_train, y_train = convert_array(train)
    X_test, y_test = convert_array(test)
    
    # Normalize the data
    nor_X_train, nor_X_test =normalize_data(X_train, X_test)
    
    lgbm_model=train_lgbm_model(nor_X_train,y_train,4)
    shap_values = lgbm_model.cv_shap_values(nor_X_test, y_test)
    fi_shap, color, feature_names=get_features(shap_values, X_test, col_names, num_features=20)
    return fi_shap, color, feature_names


def plot_feature_selction(fi_shap, color, feature_names, title="Feature Selection"):
    """ Plot feature importance with colors using seaborn.
    
    Args:
        fi_shap (array): Normalized feature importances
        color (array): Colors indicating the effect of each feature
        feature_names (list): Names of the features
        title (str): Title of the plot
    """
    top_features = feature_names
    
    # Create a DataFrame for plotting
    feature_importance = pd.DataFrame({'Feature': top_features, 'Importance': fi_shap, 'Color': color})
    
    # Sort by importance for better visualization
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Use seaborn to create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feature_importance, 
        palette=['#505277', '#3f6c78', '#408a76', '#84b56d'],
        orient='h'
    )
    
    plt.xlabel('Importance')
    plt.title(title)
    plt.show()






def missing_imputaion(x,imputer='none'):
    xt=x
    if imputer=='knn':

        X = xt
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        Knn_data=imputer.fit_transform(X)
        X1=pd.DataFrame(Knn_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='mice':
        Mice_data=MICE().fit_transform(xt)
        X1=pd.DataFrame(Mice_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='randomforest':
        imputer = MissForest()
        Rf = imputer.fit_transform(xt)
        X1=pd.DataFrame(Rf)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    else:
        X1=xt.dropna(axis=0)
        return(X1)





def get_leave_one_out_cv(data_input_local, label_input_local, resample_condition_input_local=True, return_type='index'):
    smote_object_local = SMOTE()
    index_local = data_input_local.columns
    #data_input_local = data_input_local.values
    data_input_local = StandardScaler().fit_transform(data_input_local)
    data_input_local = np.array(data_input_local)
    label_input_local = np.array(label_input_local)

    fold_object_local = LeaveOneOut()
    fold_object_local.get_n_splits(data_input_local)

    if return_type=='index':
        fold_num_local = 0
        fold_index_dict_local = {}
        for train_index_local, test_index_local in fold_object_local.split(data_input_local, label_input_local):
            train_index_local, val_index_local = train_test_split(train_index_local, test_size=0.2)
            fold_index_dict_local[fold_num_local] = [train_index_local, val_index_local, test_index_local]
            fold_num_local = fold_num_local + 1

        return fold_index_dict_local

    if return_type == 'data':
        smote_object_local = SMOTE()

        data_train_list_local = []
        label_train_list_local = []
        data_test_list_local = []
        label_test_list_local = []
        for train_index_local, test_index_local in fold_object_local.split(data_input_local, label_input_local):
            each_data_train_list_local = data_input_local[train_index_local]
            each_label_train_list_local = label_input_local[train_index_local]

            each_data_test_list_local = data_input_local[test_index_local]
            each_label_test_list_local = label_input_local[test_index_local]

            if resample_condition_input_local == True:
                each_data_train_list_local, each_label_train_list_local = smote_object_local.fit_resample(each_data_train_list_local, each_label_train_list_local)

            data_train_list_local.append(each_data_train_list_local)
            label_train_list_local.append(each_label_train_list_local)
            data_test_list_local.append(each_data_test_list_local)
            label_test_list_local.append(each_label_test_list_local)

        fold_data_dict_local = {'data': (data_train_licomst_local, data_test_list_local, label_train_list_local, label_test_list_local), 'index': index_local}
        return fold_data_dict_local

# 

def cv_fold_2(df):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    random_seed = 2019

    # Shuffle the DataFrame
    # df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    transformer = QuantileTransformer(output_distribution='normal')
    xx1= transformer.fit_transform(datas)
    yt = df['label']

    X=np.array(xx1)

    y=np.array(yt)
    cc=datas.columns


    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

    for _ in range(5):
        # Perform the train-test split with the first 50 samples as the test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=96, random_state=2019, stratify=y)

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)
    d={'data':(xtrain,xtest,ytrain,ytest),'index':cc}
    return d

def cv_fold_sumon(df):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    transformer = QuantileTransformer(output_distribution='normal')
    xx1= transformer.fit_transform(datas)
    yt = df['label']

    X=np.array(xx1)

    y=np.array(yt)
    cc=datas.columns


    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = X[:-1254]
        y_train = y[:-1254]
        X_test = X[-1254:]
        y_test = y[-1254:]

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d




def cv_fold_head(df,head_index):#
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    yt = df['label']
    transformer = QuantileTransformer(output_distribution='normal')
    Xtr_main = datas.tail(len(datas)-int(head_index))
    Ytr_main = yt.tail(len(datas)-int(head_index))
    Xt_main =datas.head(int(head_index))
    Yt_main = yt.head(int(head_index))



    Xtr_main_n= transformer.fit_transform(Xtr_main)
    Xt_main_n = transformer.fit_transform(Xt_main)

    Xtr_main_n_a =np.array(Xtr_main_n)
    Ytr_main_a = np.array(Ytr_main)
    Xt_main_n_a = np.array(Xt_main_n)
    Yt_main_a = np.array(Yt_main)

    cc=datas.columns

    print(Xtr_main)

    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = Xtr_main_n_a
        y_train = Ytr_main_a
        X_test = Xt_main_n_a
        y_test = Yt_main_a

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d




def cv_foldcsv(n_splits=5, shuffle=False, feature_selection_model=None):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    smote = SMOTE()

    # Assuming you have fold_1_train.csv, fold_1_test.csv, ..., fold_5_train.csv, fold_5_test.csv
    fold_prefix = "fold_"
    train_suffix = "_train"
    test_suffix = "_test"
    csvpath = "D:/sumon2/Metabolomics/metabolomic_last/maindata/121data"
    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

    for fold in range(1, n_splits + 1):
        train_filename = f'{csvpath}/{fold_prefix}{str(fold)}{train_suffix}.csv'
        test_filename = f'{csvpath}/{fold_prefix}{str(fold)}{test_suffix}.csv'

        if feature_selection_model is None:
            raise ValueError("Please provide a feature selection model list.")

        # Load your data using pandas, adjust the read_csv parameters as needed
        X_train = pd.read_csv(train_filename)
        X_test = pd.read_csv(test_filename)

        # Select only the features and the target variable ('label')
        selected_columns = feature_selection_model + ['label']
        X_train = X_train[selected_columns]
        X_test = X_test[selected_columns]

        # Assuming the target variable is in the last column, adjust this if needed
        y_train = X_train['label']
        y_test = X_test['label']

        # Drop the target variable from features
        X_train = X_train.drop(columns=['label'])
        X_test = X_test.drop(columns=['label'])

        xx1 = StandardScaler().fit_transform(pd.concat([X_train, X_test]))
        x1, y1 = smote.fit_resample(xx1[:len(X_train)], y_train)

        xtrain.append(x1)
        xtest.append(xx1[len(X_train):])
        ytrain.append(y1)
        ytest.append(y_test)

    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': X_train.columns}
    return d


def cv_fold_tail(df,tail_index):#
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    yt = df['label']
    transformer = QuantileTransformer(output_distribution='normal')
    Xtr_main = datas.head(len(datas)-int(tail_index))
    Ytr_main = yt.head(len(datas)-int(tail_index))
    Xt_main =datas.tail(int(tail_index))
    Yt_main = yt.tail(int(tail_index))


    Xtr_main_n= transformer.fit_transform(Xtr_main)
    Xt_main_n = transformer.fit_transform(Xt_main)

    Xtr_main_n_a =np.array(Xtr_main_n)
    Ytr_main_a = np.array(Ytr_main)
    Xt_main_n_a = np.array(Xt_main_n)
    Yt_main_a = np.array(Yt_main)

    cc=datas.columns



    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = Xtr_main_n_a
        y_train = Ytr_main_a
        X_test = Xt_main_n_a
        y_test = Yt_main_a

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d






def feature_selection(x,y, num_of_max_feature_for_genetic_extractor):
        X_train=x
        y_train=y
        mdl=[]
        mdl.append(xgb.XGBClassifier(
                        max_depth=4
                        ,learning_rate=0.2
                        ,reg_lambda=1
                        ,n_estimators=150
                        ,subsample = 0.9
                        ,colsample_bytree = 0.9))
        mdl.append(RandomForestClassifier(n_estimators=50,max_depth=10,
                                            random_state=0,class_weight=None,
                                            n_jobs=-1))
        mdl.append(ExtraTreesClassifier())
        ml1=['XGBoost','Random_Forest','Extra_Tree']
        feat_sel=[]
        for i in range(3):

            model=mdl[i]
            model.fit(X_train, y_train)
            model.feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns
            print("Feature ranking:")
            sel_feat=[]
            for f in range(X_train.shape[1]):
                    print("%d. feature no:%d feature name:%s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
                    sel_feat.append(feat_labels[indices[f]])
            top_n=20
            feat_sel.append(sel_feat)
            indices = indices[0:top_n]
            plt.subplots(figsize=(12, 10))
            g = sns.barplot(x=importances[indices], y=feat_labels[indices], orient='h', label='big') #import_feature.iloc[:Num_f]['col'].values[indices]

            g.set_title(ml1[i]+' feature selection',fontsize=25)
            g.set_xlabel("Relative importance",fontsize=25)
            g.set_ylabel("Features",fontsize=25)
            g.tick_params(labelsize=14)
            sns.despine()
                # plt.savefig('feature_importances_v3.png')
            plt.show()
            print('-----------------------------------------------------------------')
        xgboost=feat_sel[0]
        randomforest=feat_sel[1]
        extratree=feat_sel[2]

        from genetic_selection import GeneticSelectionCV
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier()
        model = GeneticSelectionCV(
            estimator, cv=10, verbose=0,
            scoring="accuracy", max_features=num_of_max_feature_for_genetic_extractor,
            n_population=100, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=50,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.04,
            tournament_size=3, n_gen_no_change=10,
            caching=True, n_jobs=-1)

        model = model.fit(x, y)
        genetic_feature_selection = list(x.columns[model.support_])

        print('****************Genectic Evolution Based Feature Selection****************')
        print("Number of Selected Features : " + str(len(genetic_feature_selection)))
        print("Features : ")
        iter = 1
        for each_feature in genetic_feature_selection:
          print(str(iter) + ". " + each_feature)
          iter=iter+1

        return(xgboost,randomforest,extratree,genetic_feature_selection)


def feature_selection_without_gen(x,y):
        X_train=x
        y_train=y
        mdl=[]
        mdl.append(xgb.XGBClassifier(
                        max_depth=4
                        ,learning_rate=0.2
                        ,reg_lambda=1
                        ,n_estimators=150
                        ,subsample = 0.9
                        ,colsample_bytree = 0.9))
        mdl.append(RandomForestClassifier(n_estimators=50,max_depth=10,
                                            random_state=0,class_weight=None,
                                            n_jobs=-1))
        mdl.append(ExtraTreesClassifier())
        ml1=['XGBoost','Random_Forest','Extra_Tree']
        feat_sel=[]
        for i in range(3):

            model=mdl[i]
            model.fit(X_train, y_train)
            model.feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns
            print("Feature ranking:")
            sel_feat=[]
            for f in range(X_train.shape[1]):
                    print("%d. feature no:%d feature name:%s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
                    sel_feat.append(feat_labels[indices[f]])
            top_n=20
            feat_sel.append(sel_feat)
            indices = indices[0:top_n]
            plt.subplots(figsize=(12, 10))
            g = sns.barplot(importances[indices],feat_labels[indices], orient='h',label='big') #import_feature.iloc[:Num_f]['col'].values[indices]

            g.set_title(ml1[i]+' feature selection',fontsize=25)
            g.set_xlabel("Relative importance",fontsize=25)
            g.set_ylabel("Features",fontsize=25)
            g.tick_params(labelsize=14)
            sns.despine()
                # plt.savefig('feature_importances_v3.png')
            plt.show()
            print('-----------------------------------------------------------------')
        xgboost=feat_sel[0]
        randomforest=feat_sel[1]
        extratree=feat_sel[2]

        return(xgboost,randomforest,extratree)


def feature_genetic_extractor(x,y, num_of_max_feature_for_genetic_extractor, num_of_population=100, num_of_generation=50, num_of_gen_no_change=10):
        X_train=x
        y_train=y
        from genetic_selection import GeneticSelectionCV
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier()
        model = GeneticSelectionCV(
            estimator, cv=10, verbose=0,
            scoring="accuracy", max_features=num_of_max_feature_for_genetic_extractor,
            n_population=num_of_population, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=num_of_generation,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.04,
            tournament_size=3, n_gen_no_change=num_of_gen_no_change,
            caching=True, n_jobs=-1)

        model = model.fit(x, y)
        genetic_feature_selection = list(x.columns[model.support_])

        print('****************Genectic Evolution Based Feature Selection****************')
        print("Number of Selected Features : " + str(len(genetic_feature_selection)))
        print("Features : ")
        iter = 1
        for each_feature in genetic_feature_selection:
          print(str(iter) + ". " + each_feature)
          iter=iter+1

        return genetic_feature_selection


def models():

        clf=[]
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(13, 13), learning_rate='constant',
               learning_rate_init=0.001, max_iter=500, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=111,
               shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
               verbose=False, warm_start=False)
        clf.append(MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500))


        clf.append(LinearDiscriminantAnalysis())

        clf.append(xgb.XGBClassifier(
                        max_depth=85
                        ,learning_rate=0.9388440565186442,
                        min_split_loss= 0.0
                        ,reg_lambda=5.935581318908179
                        ,min_child_weight= 2.769401581888831
                        ,colsample_bylevel= 0.7878344729848824
                        ,colsample_bynode=0.4895496034538383
                        ,alpha= 7.9692927383000445
                        ,n_estimators=150
                        ,subsample = 0.2656532818978606
                        ,colsample_bytree = 0.8365485367400313))

        clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=10, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0, warm_start=False))


        clf.append(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False))


        clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto',
                kernel='linear', max_iter=100, probability=True, random_state=0,
                shrinking=True, tol=0.001, verbose=False))


        clf.append(ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=0))

        clf.append(AdaBoostClassifier(n_estimators=100, random_state=0))

        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0))

        clf.append(XGBClassifier(n_estimators=400,
                      iterations=500,
                      learning_rate=0.001,
                      loss_function='Logloss'))
        clf.append(cb.CatBoostClassifier())
        clf.append(LGBMClassifier(learning_rate=0.01))
        clf.append(AdaBoostClassifier(learning_rate=0.001))
        clf.append(SVC(probability=True))
        clf.append(RandomForestClassifier())
        clf.append(ExtraTreesClassifier(bootstrap=True))
        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(LinearDiscriminantAnalysis())
        clf.append(LogisticRegression())
        clf.append(LogisticRegression(penalty='elasticnet', l1_ratio=0.01, solver='saga'))
        #clf.append(RidgeClassifier())




        clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier','XGB_untuned', 'CatBoost_untuned', 'LGBM_untuned', 'AdaBoost_untuned', 'SVC_untuned', 'RandomForest_untuned', 'ExtraTrees_untuned', 'KNeighbors_untuned', 'LDA_untuned', 'LogisticRegression_untuned', 'ElasticNet_untuned']
        #, 'Ridge_untuned'
        #Result.to_csv
        return(clf,clff )


def models_v_2():

        clf=[]
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(13, 13), learning_rate='constant',
               learning_rate_init=0.001, max_iter=500, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=111,
               shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
               verbose=False, warm_start=False)
        clf.append(MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500))


        clf.append(LinearDiscriminantAnalysis())

        clf.append(xgb.XGBClassifier(
                        max_depth=85
                        ,learning_rate=0.9388440565186442,
                        min_split_loss= 0.0
                        ,reg_lambda=5.935581318908179
                        ,min_child_weight= 2.769401581888831
                        ,colsample_bylevel= 0.7878344729848824
                        ,colsample_bynode=0.4895496034538383
                        ,alpha= 7.9692927383000445
                        ,n_estimators=150
                        ,subsample = 0.2656532818978606
                        ,colsample_bytree = 0.8365485367400313))

        clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=10, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0, warm_start=False))


        clf.append(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False))


        clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto',
                kernel='linear', max_iter=100, probability=True, random_state=0,
                shrinking=True, tol=0.001, verbose=False))


        clf.append(ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=0))

        clf.append(AdaBoostClassifier(n_estimators=100, random_state=0))

        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0))

        clf.append(XGBClassifier(n_estimators=400,
                      iterations=500,
                      learning_rate=0.001,
                      loss_function='Logloss'))
        clf.append(cb.CatBoostClassifier())
        clf.append(LGBMClassifier(learning_rate=0.01))
        clf.append(AdaBoostClassifier(learning_rate=0.001))
        clf.append(SVC(probability=True))
        clf.append(RandomForestClassifier())
        clf.append(ExtraTreesClassifier(bootstrap=True))
        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(LinearDiscriminantAnalysis())
        clf.append(LogisticRegression())
        clf.append(LogisticRegression(penalty='elasticnet', l1_ratio=0.01, solver='saga'))
        clf.append(RidgeClassifier())




        clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier','XGB_untuned', 'CatBoost_untuned', 'LGBM_untuned', 'AdaBoost_untuned', 'SVC_untuned', 'RandomForest_untuned', 'ExtraTrees_untuned', 'KNeighbors_untuned', 'LDA_untuned', 'LogisticRegression_untuned', 'ElasticNet_untuned', 'Ridge_untuned']

        #
        #Result.to_csv
        return(clf,clff)

def classification_with_top_feature_v_2(data,feature_num,feature_selection_model,classifier,feat_increment):

        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models_v_2()


        if classifier=='all':
            l=0
            for c in range(22):

                clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]  #feature increasing
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(xt1)


                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())



                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
                    # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'




                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)



                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')

                print('---------------------------------------------------------------------')
                l=l+1

            return
        else:

                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0


#                 clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(np.array(xt1))

                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred

                    categories=list(pd.Series(y2).unique())
                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'

                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)


#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')


                return




def classification_with_top_feature(data,feature_num,feature_selection_model,classifier,feat_increment):

        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()


        if classifier=='all':
            l=0
            for c in range(21):

                clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]  #feature increasing
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(xt1)


                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())



                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
                    # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'




                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)



                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')

                print('---------------------------------------------------------------------')
                l=l+1

            return
        else:

                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0


#                 clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(np.array(xt1))

                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred

                    categories=list(pd.Series(y2).unique())
                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'

                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)


#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')


                return




def classification_with_combined_features(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        auc_all=[]
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]
        confusion_matrices = [] 
        accuracies = []
        for c in range(21):

            clf1=clf[c]

            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]

                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)
                    accuracy = accuracy_score(ytest[k], y_pr)
                    accuracies.append(accuracy)
                    
                    
                        
          



                categories=list(pd.Series(y2).unique())
                y21, y_pred1=y2,y_pred
                flat_pred_probabilities = [prob[1] for prob in probs]

                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'


                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                # print(f'{c} is {cm}')
                # Calculate confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                confusion_matrices.append({'index': c, 'confusion_matrix': cm})
                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)
                auc1 = roc_auc_score(y2, flat_pred_probabilities)

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0]
                    FP = cm_per_class[i][0][1]
                    FN = cm_per_class[i][1][0]
                    TP = cm_per_class[i][1][1]
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
              
                    Precision = round(100*(TP)/(TP+FP), 2)
                    Sensitivity = round(100*(TP)/(TP+FN), 2)
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                    Specificity = round(100*(TN)/(TN+FP), 2)

                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1)
                # create tmep excel table
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # print(f"Fold {k+1} Accuracy: {accuracies:.4f}")
              
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
                a.append(ac)
                auc_all.append(auc1)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)
          



        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
        Result.index= clff





        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred,confusion_matrices


####

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import sklearn

def classification_with_combined_features_withauc2(data, feature_num, feature_selection_model, classifier):
    xtrain, xtest, ytrain, ytest = data['data']
    ind = data['index'].to_list()
    num_feat = feature_num
    fsm = feature_selection_model
    clf, clff = models()  # Ensure models() is defined or imported
    classifier = 'all'

    if classifier == 'all':
        auc_all = []
        a = []
        p = []
        r = []
        s = []
        f = []
        prb0 = []  # Collects probabilities for all classifiers
        prb1 = []  # Collects fold-wise probabilities for each classifier
        pred = []
        tar = []
        confusion_matrices = []
        rep = []

        top_models = []  # List to store the top 3 models and their accuracies

        for c in range(21):
            clf1 = clf[c]
            probss = []  # Collects fold-wise probabilities for the current classifier

            # Initialize y_pred, y2, and probs for each classifier
            y_pred = []
            y2 = []
            probs = []

            for k in range(len(xtrain)):
                x11 = pd.DataFrame(xtrain[k])
                x11.columns = ind
                x1 = x11[fsm[0:num_feat]]
                y1 = ytrain[k]

                model = clf1.fit(np.array(x1), np.array(y1))
                xts = pd.DataFrame(xtest[k])
                xts.columns = ind
                xt1 = xts[fsm[0:num_feat]]

                y_pr = model.predict(np.array(xt1))
                y_prob = model.predict_proba(np.array(xt1))

                # Append predictions and probabilities
                y_pred.extend(y_pr)
                y2.extend(ytest[k])
                probs.extend(y_prob)
                probss.append(y_prob)

            # Append fold-wise probabilities for the current classifier to prb1
            prb1.append(probss)

            categories = list(pd.Series(y2).unique())
            y21, y_pred1 = y2, y_pred

            # Calculate metrics using Scikit-learn functions
            accuracy = accuracy_score(y21, y_pred1) * 100
            precision = precision_score(y21, y_pred1, average='weighted') * 100
            recall = recall_score(y21, y_pred1, average='weighted') * 100
            f1 = f1_score(y21, y_pred1, average='weighted') * 100

            # Calculate ROC-AUC score
            if len(categories) == 2:
                flat_pred_probabilities = [prob[1] for prob in probs]
                auc_t = roc_auc_score(y21, flat_pred_probabilities)
            else:
                flat_pred_probabilities = np.vstack(probs)
                auc_t = roc_auc_score(y21, flat_pred_probabilities, multi_class='ovr')
            auc1 = round(auc_t * 100, 2)
            auc_all.append(auc1)  # Append AUC score to auc_all

            # Calculate confusion matrix for specificity
            cm = confusion_matrix(y21, y_pred1)
            cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
            report = classification_report(y21, y_pred1)
            confusion_matrices.append({'index': c, 'confusion_matrix': cm})
            rep.append({'index': c, 'report': report})

            # Calculate specificity
            specificities = []
            for i in range(len(categories)):
                TN = cm_per_class[i][0][0]
                FP = cm_per_class[i][0][1]
                specificity_class = TN / (TN + FP) if (TN + FP) > 0 else 0
                specificities.append(specificity_class * (sum(y21 == categories[i]) / len(y21)))

            specificity = sum(specificities) * 100

            # Store metrics
            a.append(accuracy)
            p.append(precision)
            r.append(recall)
            s.append(specificity)
            f.append(f1)
            prb0.append(probs)

            pred.append(y_pred1)
            tar.append(y2)

            # Track the top 3 models
            if len(top_models) < 3:
                top_models.append((accuracy, model))
                top_models.sort(reverse=True, key=lambda x: x[0])
            else:
                if accuracy > top_models[-1][0]:
                    top_models[-1] = (accuracy, model)
                    top_models.sort(reverse=True, key=lambda x: x[0])

        print("Shapes of individual DataFrames being concatenated:")
        print(f"a: {pd.DataFrame(a).shape}, p: {pd.DataFrame(p).shape}, r: {pd.DataFrame(r).shape}")
        print(f"s: {pd.DataFrame(s).shape}, f: {pd.DataFrame(f).shape}, auc_all: {pd.DataFrame(auc_all).shape}")

        Result = pd.concat([pd.DataFrame(a), pd.DataFrame(p), pd.DataFrame(r), pd.DataFrame(s), pd.DataFrame(f), pd.DataFrame(auc_all)], axis=1)
        print(f"Shape of Result DataFrame: {Result.shape}")

        Result.columns = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-score', 'AUC']
        Result.index = clff

        print(Result)

        top_models = [model for _, model in top_models]  # Extract the models from the list

        return top_models, Result, prb1, prb0, ytest, tar, pred, confusion_matrices, rep

def classification_with_combined_featuresR(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]

        for c in range(21):

            clf1=clf[c]  
          
            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]
            
                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]   
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)





                categories=list(pd.Series(y2).unique()) 
                y21, y_pred1=y2,y_pred
                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'
              

                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                

                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)

              

                

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0] 
                    FP = cm_per_class[i][0][1]   
                    FN = cm_per_class[i][1][0]  
                    TP = cm_per_class[i][1][1]  
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                    Precision = round(100*(TP)/(TP+FP), 2)  
                    Sensitivity = round(100*(TP)/(TP+FN), 2) 
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
                    Specificity = round(100*(TN)/(TN+FP), 2)  
                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1) 
                # create tmep excel table 
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)  
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)  
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)  
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2) 
                a.append(ac)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)


        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
        Result.index= clff

        print(Result)
        
        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred
  





####

def classification_with_combined_features_mul(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        auc_all=[]
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]

        for c in range(21):

            clf1=clf[c]

            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]

                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)




                categories=list(pd.Series(y2).unique())
                y21, y_pred1=y2,y_pred
                flat_pred_probabilities = np.array([prob[1] for prob in probs])
                flat_pred_probabilities = flat_pred_probabilities.reshape((-1, 1))

                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'


                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)


                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)
                auc1 = roc_auc_score(y2, flat_pred_probabilities, multi_class='ovr')

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0]
                    FP = cm_per_class[i][0][1]
                    FN = cm_per_class[i][1][0]
                    TP = cm_per_class[i][1][1]
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                    print(f"acc is {Accuracy}")
                    Precision = round(100*(TP)/(TP+FP), 2)
                    Sensitivity = round(100*(TP)/(TP+FN), 2)
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                    Specificity = round(100*(TN)/(TN+FP), 2)

                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1)
                # create tmep excel table
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
                a.append(ac)
                auc_all.append(auc1)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)


        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
        Result.index= clff

        print(Result)



        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred




def processed_data(ml1,ml2,ml3,td2):
    xts=[]
    xtr=[]
    yts=[]
    ytr=[]


    prf=[]
    for i in range(5):
      pl=np.concatenate((ml1[i],ml2[i],ml3[i]),1)
      prf.append(pl)

    for j in range(5):
      if j==1:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[2],td2[3],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[2],prf[3],prf[4],prf[0]),0))
      elif j==2:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[4],prf[0]),0))
      elif j==3:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[2],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[2],prf[4],prf[0]),0))
      elif j==4:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[2],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[2],prf[0]),0))
        
      elif j==0:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[2],td2[4]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[2],prf[4]),0))
    return xtr,xts,ytr,yts



def stacking_classification(ml1,ml2,ml3,td2):

    xtrain,xtest,ytrain,ytest=processed_data(ml1,ml2,ml3,td2)
    clf,clff=models()
    classifier='all'
    if classifier=='all':
      l=0
      auc_all=[]
      a=[]
      p=[]
      r=[]
      s=[]
      f=[]
      prb0=[]
      prb1=[]
      pred=[]
      tar=[]
      top_models = [] 
      for c in range(21):

          clf1=clf[c]

          feat=[]
          for i in list(range(1)):

              y_pred=[]
              y2=[]
              # tl=fsm[0:num_feat]
              probs=[]
              probss=[]

              total_fold_num = len(xtrain)
              for k in range(total_fold_num):
                  x1=pd.DataFrame(xtrain[k])
                  # x11.columns=ind
                  # x1=x11[tl]
                  y1=ytrain[k]
                  model = clf1.fit(np.array(x1),np.array(y1))
                  #model = clf1.fit(x[train],y.iloc[train])
                  xt1=pd.DataFrame(xtest[k])
                  # xts.columns=ind
                  # xt1=xts[tl]
                  y_pr=model.predict(np.array(xt1))
                  y_prob=model.predict_proba(np.array(xt1))
                  y_pred.extend(y_pr)
                  y2.extend(ytest[k])
                  probs.extend(y_prob)
                  probss.append(y_prob)



              

              categories=list(pd.Series(y2).unique())
              flat_pred_probabilities = [prob[1] for prob in probs]
              y21, y_pred1=y2,y_pred
              if (i+1)!=1:
                feature_no= 'top_'+str(i+1)+'_features'
              else:
                feature_no= 'top_'+str(i+1)+'_feature'

              accuracy = accuracy_score(y21, y_pred1) * 100
              from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
              # main confusion matrix
              cm = confusion_matrix(y21, y_pred1)
              print(f'{c} is {cm}')
              # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
              # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
              cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
              # Overall Accuracy
              Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
              Overall_Accuracy = round(Overall_Accuracy*100, 2)
              auc1 = roc_auc_score(y2, flat_pred_probabilities)
              Eval_Mat = []
              # per class metricies
              for i in range(len(categories)):
                  TN = cm_per_class[i][0][0]
                  FP = cm_per_class[i][0][1]
                  FN = cm_per_class[i][1][0]
                  TP = cm_per_class[i][1][1]
                  Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                  Precision = round(100*(TP)/(TP+FP), 2)
                  Sensitivity = round(100*(TP)/(TP+FN), 2)
                  F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                  Specificity = round(100*(TN)/(TN+FP), 2)
                  Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
              # sizes of each class
              s2 = np.sum(cm,axis=1)
              # create tmep excel table
              headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
              temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
              # weighted average of per class metricies
              ac=Overall_Accuracy
              # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
              pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
              rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
              f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
              sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
              a.append(ac)
              auc_all.append(auc1)
              p.append(pr)
              r.append(rc)
              s.append(sp)
              f.append(f1)
              feat.append(feature_no)
              prb0.append(probs)
              prb1.append(probss)
              pred.append(y_pred1)
              tar.append(y2)
               # Track the top 3 models
              if len(top_models) < 3:
                    top_models.append((accuracy, model))
                    top_models.sort(reverse=True, key=lambda x: x[0])
              else:
                    if accuracy > top_models[-1][0]:
                        top_models[-1] = (accuracy, model)
                        top_models.sort(reverse=True, key=lambda x: x[0])

      Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
      Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
      Result.index= clff

      print(Result)

      l=l+1
      print('---------------------------------------------------------------------')
      return  model,Result, prb1,prb0,ytest,tar,pred






# def stacking_classification2(ml1, ml2, ml3, td2):
#     xtrain, xtest, ytrain, ytest = processed_data(ml1, ml2, ml3, td2)
#     clf, clff = models()
#     classifier = 'all'
#     if classifier == 'all':
#         l = 0
#         auc_all = []
#         a = []
#         p = []
#         r = []
#         s = []
#         f = []
#         prb0 = []
#         prb1 = []
#         pred = []
#         tar = []

#         for c in range(21):
#             clf1 = clf[c]

#             for i in range(1):
#                 y_pred = []
#                 y2 = []
#                 probs = []
#                 probss = []

#                 total_fold_num = len(xtrain)
#                 for k in range(total_fold_num):
#                     x1 = pd.DataFrame(xtrain[k])
#                     y1 = ytrain[k]
#                     model = clf1.fit(np.array(x1), np.array(y1))
#                     xt1 = pd.DataFrame(xtest[k])
#                     y_pr = model.predict(np.array(xt1))
#                     y_prob = model.predict_proba(np.array(xt1))
#                     y_pred.extend(y_pr)
#                     y2.extend(ytest[k])
#                     probs.extend(y_prob)
#                     probss.append(y_prob)

#                     # Calculate metrics for the current fold
#                     categories = list(pd.Series(y2).unique())
#                     flat_pred_probabilities = [prob[1] for prob in probs]
#                     y21, y_pred1 = y2, y_pred

#                     cm = confusion_matrix(y21, y_pred1)
#                     # main confusion matrix
#                     cm = confusion_matrix(y21, y_pred1)
#                     print(f'{c} is {cm}')
#                     cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
#                     Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
#                     Overall_Accuracy = round(Overall_Accuracy * 100, 2)
#                     auc1 = roc_auc_score(y2, flat_pred_probabilities)
#                     Eval_Mat = []
                   
#                     for i in range(len(categories)):
#                         TN = cm_per_class[i][0][0]
#                         FP = cm_per_class[i][0][1]
#                         FN = cm_per_class[i][1][0]
#                         TP = cm_per_class[i][1][1]
#                         Accuracy = round(100 * (TP + TN) / (TP + TN + FP + FN), 2)
#                         Precision = round(100 * (TP) / (TP + FP), 2)
#                         Sensitivity = round(100 * (TP) / (TP + FN), 2)
#                         F1_score = round((2 * Precision * Sensitivity) / (Precision + Sensitivity), 2)
#                         Specificity = round(100 * (TN) / (TN + FP), 2)
#                         Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])

#                     s2 = np.sum(cm, axis=1)
#                     headers = ['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
#                     temp_table = pd.DataFrame(Eval_Mat, index=categories, columns=headers)
#                     ac = Overall_Accuracy
#                     pr = round(temp_table['Precision'].dot(s2) / np.sum(s2), 2)
#                     rc = round(temp_table['Sensitivity'].dot(s2) / np.sum(s2), 2)
#                     f1 = round(temp_table['F1_score'].dot(s2) / np.sum(s2), 2)
#                     sp = round(temp_table['Specificity'].dot(s2) / np.sum(s2), 2)

#                     # Print fold results
#                     print(f"Fold {k+1} Results for model {c}: Accuracy = {ac}, Precision = {pr}, Recall = {rc}, Specificity = {sp}, F1-score = {f1}, AUC = {auc1}")

#                 a.append(ac)
#                 auc_all.append(auc1)
#                 p.append(pr)
#                 r.append(rc)
#                 s.append(sp)
#                 f.append(f1)
#                 prb0.append(probs)
#                 prb1.append(probss)
#                 pred.append(y_pred1)
#                 tar.append(y2)

#         Result = pd.concat([pd.DataFrame(a), pd.DataFrame(p), pd.DataFrame(r), pd.DataFrame(s), pd.DataFrame(f), pd.DataFrame(auc_all)], axis=1)
#         Result.columns = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-score', 'AUC']
#         Result.index = clff

#         print(Result)

#         l = l + 1
#         print('---------------------------------------------------------------------')
#         return Result, prb1, prb0, ytest, tar, pred

# Example usage (assuming `processed_data` and `models` are defined elsewhere):
# result, prb1, prb0, ytest, tar, pred = stacking_classification(ml1, ml2, ml3, td2)
