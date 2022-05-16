import numpy as np
import pandas as pd

from hyperopt import hp
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import joblib, glob, os
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

import lightgbm as lgb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

from time import time

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 25})

import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls


gaia_features = [
    'gaiaedr3_parallax',
    'gaiaedr3_parallax_error',
    'gaiaedr3_pmra',
    'gaiaedr3_pmra_error',
    'gaiaedr3_pmdec',
    'gaiaedr3_pmdec_error',
#     'gaiaedr3_pseudocolour', 
#     'gaiaedr3_pseudocolour_error',
#     'gaiaedr3_dr2_radial_velocity',
#     'gaiaedr3_dr2_radial_velocity_error'
]

features_path = '/home/nmalysheva/models_before_aggregation_j/features.pkl'
column = 'J'
j_features = ['J']

features = pd.read_pickle(features_path) + j_features

sdss_wise = [
                'sdssdr16_u_cmodel-decals8tr_Lw1',
                'sdssdr16_u_cmodel-decals8tr_Lw2',
                'sdssdr16_g_cmodel-decals8tr_Lw1',
                'sdssdr16_g_cmodel-decals8tr_Lw2',
                'sdssdr16_r_cmodel-decals8tr_Lw1',
                'sdssdr16_r_cmodel-decals8tr_Lw2',
                'sdssdr16_i_cmodel-decals8tr_Lw1',
                'sdssdr16_i_cmodel-decals8tr_Lw2',
                'sdssdr16_z_cmodel-decals8tr_Lw1',
                'sdssdr16_z_cmodel-decals8tr_Lw2' 
              ]
sdss_nwise = [
                'sdssdr16_g_cmodel-decals8tr_g',
                'sdssdr16_r_cmodel-decals8tr_r',
                'sdssdr16_z_cmodel-decals8tr_z'
]

ps_decals = [
                'psdr2_g_kron-decals8tr_Lw1',
                'psdr2_g_kron-decals8tr_Lw2',
                'psdr2_r_kron-decals8tr_Lw1',
                'psdr2_r_kron-decals8tr_Lw2',
                'psdr2_i_kron-decals8tr_Lw1',
                'psdr2_i_kron-decals8tr_Lw2',
                'psdr2_z_kron-decals8tr_Lw1',
                'psdr2_z_kron-decals8tr_Lw2',
                'psdr2_y_kron-decals8tr_Lw1',
                'psdr2_y_kron-decals8tr_Lw2'
]

sdss = [i for i in features if 'sdss' in i and 'decals' not in i and column not in i] 
decals = [i for i in features if 'decals' in i and 'sdss' not in i and 'psdr' not in i and column not in i] 
wise = [i for i in decals if 'Lw' in i and column not in i] 
ps = [i for i in features if 'psdr' in i and 'decals' not in i and column not in i]

f0 = {"sdssdr16+wise_decals8tr": sdss+wise+sdss_wise,
     "psdr2+wise_decals8tr": ps+wise+ps_decals,
     "sdssdr16+all_decals8tr": sdss+decals+sdss_wise+sdss_nwise,
     "psdr2+all_decals8tr": ps+decals+ps_decals,
     "decals8tr": decals,
     "sdssdr16+psdr2+wise_decals8tr": sdss+ps+wise+sdss_wise+ps_decals,
     "sdssdr16+psdr2+all_decals8tr": sdss+ps+decals+ps_decals+sdss_wise+sdss_nwise}

sdss_j = [
    'sdssdr16_u_psf-j',
     'sdssdr16_g_psf-j',
     'sdssdr16_r_psf-j',
     'sdssdr16_i_psf-j',
     'sdssdr16_z_psf-j',
     'sdssdr16_u_cmodel-j',
     'sdssdr16_i_cmodel-j'
]

ps_j = [
    'psdr2_i_kron-j',
     'psdr2_y_kron-j',
     'psdr2_g_psf-j',
     'psdr2_r_psf-j',
     'psdr2_i_psf-j',
     'psdr2_z_psf-j',
     'psdr2_y_psf-j'
]

wise_j = [
    'decals8tr_Lw1-j',
     'decals8tr_Lw2-j'
]

nwise_j = [
    'decals8tr_g-j',
     'decals8tr_r-j',
     'decals8tr_z-j'
]


l = [sdss_j, ps_j, wise_j, nwise_j]

f1 = {"sdssdr16+wise_decals8tr": sdss+wise+sdss_wise+sdss_j+wise_j+j_features,
     "psdr2+wise_decals8tr": ps+wise+ps_decals+ps_j+wise_j+j_features,
     "sdssdr16+all_decals8tr": sdss+decals+sdss_wise+sdss_nwise+sdss_j+wise_j+nwise_j+j_features,
     "psdr2+all_decals8tr": ps+decals+ps_decals+ps_j+wise_j+nwise_j+j_features,
     "decals8tr": decals+wise_j+nwise_j+j_features,
     "sdssdr16+psdr2+wise_decals8tr": sdss+ps+wise+sdss_wise+ps_decals+ps_j+wise_j+sdss_j+j_features,
     "sdssdr16+psdr2+all_decals8tr": sdss+ps+decals+ps_decals+sdss_wise+sdss_nwise+sdss_j+nwise_j+ps_j+wise_j+j_features}

f={'not_j': f0, 'j':f1}


def data_preparation(X, y, c=100000, test_size = 0.8):

    X1_train, X1_test, y1_train, y1_test = train_test_split(X[y==1], y[y==1], test_size=test_size, random_state = 43)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X[y==2], y[y==2], test_size=test_size, random_state = 43)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X[y==3], y[y==3], test_size=test_size, random_state = 43)
    
    count = c
    count1 = c

    X_train, X_test = np.concatenate((X1_train[:count], X2_train[:count], X3_train[:count])), np.concatenate((X1_test[:count1], X2_test[:count1], X3_test[:count1]))
    y_train, y_test = np.concatenate((y1_train[:count], y2_train[:count], y3_train[:count])), np.concatenate((y1_test[:count1], y2_test[:count1], y3_test[:count1]))

    data = np.concatenate((X_train, y_train.reshape((len(y_train), 1))), axis=1)
    np.random.shuffle(data)

    datat = np.concatenate((X_test, y_test.reshape((len(y_test), 1))), axis=1)
    np.random.shuffle(datat)
    

    return data, datat

def data_preparation_bin(X, y, c=100000, test_size = 0.8):

    X1_train, X1_test, y1_train, y1_test = train_test_split(X[y==1], y[y==1], test_size=test_size, random_state = 43) if  len(y[y==1]) > 0 else ([], [], [], [])
    X2_train, X2_test, y2_train, y2_test = train_test_split(X[y==2], y[y==2], test_size=test_size, random_state = 43) if  len(y[y==2]) > 0 else ([], [], [], [])
    #X3_train, X3_test, y3_train, y3_test = train_test_split(X[y==3], y[y==3], test_size=test_size, random_state = 43) if  len(y[y==3]) > 0 else ([[]*X.shape[1]], [[]*X.shape[1]], [[]], [[]])
    
    count = c
    count1 = c

    X_train, X_test = np.concatenate((X1_train[:count], X2_train[:count])), np.concatenate((X1_test[:count1], X2_test[:count1]))
    y_train, y_test = np.concatenate((y1_train[:count], y2_train[:count])), np.concatenate((y1_test[:count1], y2_test[:count1]))

    data = np.concatenate((X_train, y_train.reshape((len(y_train), 1))), axis=1)
    np.random.shuffle(data)

    datat = np.concatenate((X_test, y_test.reshape((len(y_test), 1))), axis=1)
    np.random.shuffle(datat)
    

    return data, datat


def data_open(path, features):
    classes = {'STAR': 1, 'QSO':2, 'GALAXY':3}
    with gzip.open(path, 'rb') as f:
        df = pickle.load(f)
    df = df[features + ['class']].dropna()
    X = df[features].values
    y = df.replace({'class':classes}, inplace = True)['class'].values
    return X, y


def scor(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

lgb_reg_params = {
    'min_child_samples':hp.randint('min_child_samples', 80)+1,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'num_leaves' :      hp.randint('num_leaves', 100)+10,
    #'min_child_weight': hp.uniform('min_child_weight', 0.001, 0.99),
    'subsample_freq':     hp.randint('subsample_freq', 20),
    'n_estimators':     10000
}
lgb_fit_params = {
    'early_stopping_rounds': 20,
    'verbose': False
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['score'] = lambda y, pred: -roc_auc_score(y, pred, multi_class='ovo')


rf_reg_params = {
    'min_samples_leaf': hp.randint('min_samples_leaf', 20)+1,
    'min_samples_split':hp.uniform('min_samples_split', 0.001, 0.1),
    #'max_features':     hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    #'learning_rate':    hp.uniform('learning_rate', 0.001, 0.1),
    'n_estimators':     hp.randint('n_estimators', 800)+100
}
rf_fit_params = {
}
rf_para = dict()
rf_para['reg_params'] = rf_reg_params
rf_para['fit_params'] = rf_fit_params
rf_para['score'] = lambda y, pred: -accuracy_score(y, pred)


class HPOpt(object):

    def __init__(self, X, y, cv=3):
        self.X = X
        self.y = y
        self.cv = cv
        #print('init')

    def process(self, fn_name, space, trials, algo, max_evals):
        #print('in process')
        fn = getattr(self, fn_name)
        try:
            #print('try')
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            #print('i can')
        except Exception as e:
            print({'status': STATUS_FAIL,
                    'exception': str(e)})
            raise e
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def rf_reg(self, para):
        reg = RandomForestClassifier(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMClassifier(**para['reg_params'])
        if self.cv>1:
            return self.train_cv_gb(reg, para)
        return self.train_reg(reg, para)


    def train_reg(self, reg, para):
        if len(para['fit_params'])>0:
            #print('start')
            reg.fit(self.X, self.y,
                  eval_set=[(self.X, self.y), (self.X, self.y)],
                  **para['fit_params'])
        else:
            reg.fit(self.X, self.y)
        pred = reg.predict(self.X)
        loss = para['score'](self.y, pred)
        return {'loss': loss, 'status': STATUS_OK}


    def train_cv_gb(self, reg, para):
        kf = KFold(n_splits=self.cv, shuffle=False)
        loss = 0 
        for train, test in kf.split(self.X):
            #print('start', type(train[0]), type(test[0]), type(self.y[0]), type(self.X))
            if len(para['fit_params'])>0:
                reg = lgb.LGBMClassifier(**para['reg_params'])
                reg.fit(self.X[train], self.y[train],
                      eval_set=[(self.X[train], self.y[train]), (self.X[test], self.y[test])],
                      **para['fit_params'])
            else:
                reg.fit(self.X[train], self.y[train])
            #print('pred')
            pred = reg.predict_proba(self.X[test])
            score = para['score'](self.y[test], pred)
            loss += score

        loss=loss/self.cv
        return {'loss': loss, 'status': STATUS_OK}
    
def create_path(path):
    import os
    try:
        os.makedirs(path)
    except FileExistsError:
        print("Директория %s уже существует" % path) 
    except OSError:
        print ("Создать директорию %s не удалось" % path)
        return 1
    return 0    

# Обучение общего классификатора по фолдам
def train_classifier_SGQ(df, version='', folds=[0, 1], overviews=None, variant='not_j', path_buf='./', check_exist=True,
                        class_weight=None):
    def get_cls_weights(n, w):
        if w is None:
            return None
        norm = np.argmin(n)
        res = {norm+1 : 1}
        for i in set([1, 2, 3]) - set([norm+1]):
            res[i] = (n[norm] * w[i-1]) / (n[i-1] * w[norm])
        print(f'Res of get_cls_weights {res}')
        return res
    
    for fold in folds:
#         for mod in variants:
            path = os.path.join(path_buf, f'{version}/models_{fold}/{variant}')
            if create_path(path):
                raise Exception('Некуда писать')

            for overview in (overviews or f0.keys()):
                print(overview)
                assert overview in f0.keys()
                if check_exist and os.path.exists(os.path.join(path, f'model_{overview}.pkl')):
                    print(f"{os.path.join(path, f'model_{overview}.pkl')} already exists")
                    continue
                if isinstance(fold, str):
                    df1 = df.loc[:, f0[overview]+['class']].dropna()
                else:
                    df1 = df.loc[df["fold"] == fold, f0[overview]+['class']].dropna()
                print('Number of features', len(f0[overview]))
                X, y = df1[f0[overview]].values, df1['class'].values
                n = [np.sum(y == 1), np.sum(y == 2), np.sum(y == 3)]
                X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state = 43, stratify=y)

                robust = RobustScaler()
                print('Train ', np.sum(y1 == 1), np.sum(y1 == 2), np.sum(y1 == 3))
                print('Test ', np.sum(y2 == 1), np.sum(y2 == 2), np.sum(y2 == 3))

                X_train_norm = robust.fit_transform(X1)
                X_test_norm = robust.transform(X2)
                y_train = y1
                y_test = y2

                obj = HPOpt(X_train_norm, y_train, cv=3)
                lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=50)
                print(lgb_opt)
                gb = lgb.LGBMClassifier( 
                                            **{
                                                  'class_weight': get_cls_weights(n, class_weight),
                                                  'colsample_bytree': lgb_opt[0]['colsample_bytree'],
                                                  'min_child_samples': lgb_opt[0]['min_child_samples']+1,
                                                  #'min_child_weight': lgb_opt[0]['min_child_weight'],
                                                  'num_leaves': lgb_opt[0]['num_leaves']+10,
                                                  'subsample_freq':lgb_opt[0]['subsample_freq'],
                                                  'n_estimators': 1000
                                              }
                                          )
                t = time()
                gb.fit(X_train_norm, y_train, eval_set=[(X_train_norm, y_train), (X_test_norm, y_test)],  **lgb_fit_params)
                print(time()-t)
                gb_test_acc = accuracy_score(y_test, gb.predict(X_test_norm))

                print(gb_test_acc)
                print(os.path.join(path, f'model_{overview}.pkl'))
                joblib.dump(gb, os.path.join(path, f'model_{overview}.pkl'))
                joblib.dump(f0[overview], os.path.join(path, f'features_{overview}.pkl'))
                joblib.dump(robust, os.path.join(path, f'{overview}_robust_for_gb.pkl'))

                
def train_classifier_Q(df, version='', folds=[0, 1], overviews=None, variant='not_j', path_buf='./', weight=None, check_exist=True):
    for fold in folds:
#         for mod in variants:
            path = os.path.join(path_buf, f'{version}/models_{fold}/{variant}')
            if create_path(path):
                raise Exception('Некуда писать')

            for overview in (overviews or f0.keys()):
                print(overview)
                assert overview in f0.keys()
                if check_exist and os.path.exists(os.path.join(path, f'model_{overview}.pkl')):
                    print(f"{os.path.join(path, f'model_{overview}.pkl')} already exists")
                    continue
                if isinstance(fold, str):
                    df1 = df.loc[:, f0[overview]+['class']].dropna()
                else:
                    df1 = df.loc[df['fold'] == fold, f0[overview]+['class']].dropna()
                print(df1, df.loc[df['fold'] == fold])
                if weight is not None:
                    if weight is True:
                        weight = np.sum(df1['class'] == 2) / np.sum(df1['class'] == 1)
                else:
                    weight = 1.0

                X, y = df1[f0[overview]].values, df1['class'].values
                n = [np.sum(y == 1), np.sum(y == 2), np.sum(y == 3)]
#                 data, datat = data_preparation_bin(X, y, test_size=0.2, c=500000)
#                 print(data.shape, datat.shape)

#                 X1, y1 = data[:, :-1], data[:, -1].astype('int')
#                 X2, y2 = datat[:, :-1], datat[:, -1].astype('int')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 43, stratify=y)

    
                robust = RobustScaler()

                X_train_norm = robust.fit_transform(X_train)
                X_test_norm = robust.transform(X_test)
#                 y_train = y1
#                 y_test = y2
                print('Train:', np.sum(y_train==1), np.sum(y_train==2), np.sum(y_train==3), '\nTest:', np.sum(y_test==1), np.sum(y_test==2), np.sum(y_test==3))
                obj = HPOpt(X_train_norm, y_train, cv=3)
                lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=50)
                print(lgb_opt)
                gb = lgb.LGBMClassifier( 
                                            **{
                                                  'scale_pos_weight': weight,
                                                  'colsample_bytree': lgb_opt[0]['colsample_bytree'],
                                                  'min_child_samples': lgb_opt[0]['min_child_samples']+1,
                                                  #'min_child_weight': lgb_opt[0]['min_child_weight'],
                                                  'num_leaves': lgb_opt[0]['num_leaves']+10,
                                                  'subsample_freq':lgb_opt[0]['subsample_freq'],
                                                  'n_estimators': 1000
                                              }
                                          )
                t = time()
                gb.fit(X_train_norm, y_train, eval_set=[(X_train_norm, y_train), (X_test_norm, y_test)],  **lgb_fit_params)
                print(time()-t)
                gb_test_acc = accuracy_score(y_test, gb.predict(X_test_norm))

                print(gb_test_acc)
                print(os.path.join(path, f'model_{overview}.pkl'))
                joblib.dump(gb, os.path.join(path, f'model_{overview}.pkl'))
                joblib.dump(f0[overview], os.path.join(path, f'features_{overview}.pkl'))
                joblib.dump(robust, os.path.join(path, f'{overview}_robust_for_gb.pkl'))

def report_plots(df):
    from matplotlib import colors
    fig, ( ax2, ax3) = plt.subplots(ncols = 2, figsize = (24, 6))
    sns.histplot(df[df['class']==1]['zoo_best-x1_z_max'], ax = ax2, bins = 50, color = 'g', binrange=(0, 7) )
    sns.histplot(df[df['class']==2]['zspec'], ax = ax3, bins = 50, color = 'b', binrange=(0, 7) )
    fig.suptitle('Распределение для полученных слабых объектов (green - STAR, blue - QSO)', fontsize=16)
    plt.show()
    print('STAR', sum(df['class']==1), '\nQSO', sum(df['class']==2), '\nGALAXY', sum(df['class']==3))

    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (24, 6))
    sns.histplot(df[df['class']==1]['zoo_best-x1_z_max'], ax = ax1, bins = 50, color = 'g', binrange=(0, 7) )
    sns.histplot(df[df['class']==2]['zoo_best-x1_z_max'], ax = ax2, bins = 50, color = 'b', binrange=(0, 7) )

    fig.suptitle('Распределение по zoo_best-x1_z_max (green - STAR, blue - QSO)', fontsize=16)
    plt.show()
    
    def plotishe(df, x, y, normal=None):
        CLASS = {1:'STAR', 2:'QSO'}
        f, axes = plt.subplots(ncols = 2, figsize=(24, 6), sharex=True, sharey = True)
        for cls in CLASS.keys():
            p = df.loc[df['class'] == cls]
            if normal is None:
                normal = []
            norm = colors.Normalize(vmin=normal[0], vmax=normal[1])
            axes[cls-1].set_title(CLASS[cls])
            axes[cls-1].set_xlabel(x)
            axes[cls-1].set_ylabel(y)
            axes[cls-1].set_xlim([3, 7])
            axes[cls-1].scatter(p[x],p[y], c=p['zoo_best-x1_z_maxConf'], norm=norm, s=1)

        plt.show()
    normal = [0.07196959430644186, 1.0000000000000002]
    plotishe(df, 'zoo_best-x1_z_max', 'psdr2_i-z_psf', normal)
    plotishe(df, 'zoo_best-x1_z_max', 'decals8tr_z', normal)
    plotishe(df, 'zoo_best-x1_z_max', 'zspec', normal)
    
class SGQModel:
    def __init__(self, version='', path='./models', folds=[0, 1], variant='not_j', binary=False, cls=None):
        self.fold_name = 'fold'
        self.version = version
        self.path = path
        if not isinstance(folds, list):
            self.folds = ['full']
        else:
            self.folds = folds
        self.variant = variant
        self.binary = binary
        self.cls = {True: {1: 'SG', 2:'Q'}, False:{1: 'S', 2: 'Q', 3:'G'}}[binary]
        if cls:
            self.cls = cls
        
    def train(self, df, overviews=None, class_weight=None, check_exist=True):
        if self.binary:
            train_classifier_Q(df, version=self.version, folds=self.folds, overviews=overviews, variant=self.variant, path_buf=self.path, weight=class_weight, check_exist=check_exist)
        else:
            train_classifier_SGQ(df, version=self.version, folds=self.folds, overviews=overviews, variant=self.variant, path_buf=self.path, check_exist=check_exist, class_weight=class_weight)
            
    def predict(self, input_data, train=True, overviews=''):
        df = input_data.copy()
        for fold in self.folds:
#             for mod in self.variant:
                path = os.path.join(self.path, f'{self.version}/models_{fold}/{self.variant}')
                for files in glob.glob(os.path.join(path, f'model_*{overviews}.pkl')):
                    overview = files.split('/')[-1][6:-4]
                    model = joblib.load(os.path.join(path, f'model_{overview}.pkl'))
                    features = joblib.load(os.path.join(path, f'features_{overview}.pkl'))
                    robust = joblib.load(os.path.join(path, f'{overview}_robust_for_gb.pkl'))
                    if self.fold_name not in df:
                        train = False
#                     print(df[features].shape)
                    if train:
                        predict = model.predict(robust.transform(df.loc[df[self.fold_name] != fold, features]))
                    else:
                        predict = model.predict(robust.transform(df.loc[:,features]))
#                     print(predict.shape)
#                     print(predict)
                    if train:
                        df.loc[df[self.fold_name] != fold, 'predict_'+overview] = predict
                    else:
                        df.loc[:, f'predict_{overview}_fold_{fold}'] = predict
        return df
    
    def predict_proba(self, input_data, train=True, overviews='', best=True):
        df = input_data.copy()
        for fold in self.folds:
#             for mod in self.variants:
                path = os.path.join(self.path, f'{self.version}/models_{fold}/{self.variant}')
                for files in glob.glob(os.path.join(path, f'model_*{overviews}.pkl')):
                    overview = files.split('/')[-1][6:-4]
                    model = joblib.load(os.path.join(path, f'model_{overview}.pkl'))
                    features = joblib.load(os.path.join(path, f'features_{overview}.pkl'))
                    robust = joblib.load(os.path.join(path, f'{overview}_robust_for_gb.pkl'))
                    if self.fold_name not in df:
                        train = False
                    if train:
                        predict = model.predict_proba(robust.transform(df.loc[df[self.fold_name] != fold, features]))
                    else:
                        predict = model.predict_proba(robust.transform(df.loc[:,features]))
                    if train:
                        df.loc[df[self.fold_name] != fold, [f'predict_proba_{self.cls[cls]}_{overview}' for cls in self.cls]] = predict
                    else:
                        df.loc[:, [f'predict_proba_{self.cls[cls]}_{overview}_fold_{fold}' for cls in self.cls]] = predict
        if best:
            for cls in self.cls:
                df.loc[:, f'predict_proba_{self.cls[cls]}_{overview}'] = (df[f'predict_proba_{self.cls[cls]}_{overview}_fold_{0}'] + df[f'predict_proba_{self.cls[cls]}_{overview}_fold_{1}'])
                    
        return df
    
    def predict_best(self, input_data):
        pass
    
    def predict_proba_best(self, input_data):
        pass
    
    @staticmethod
    def predict_gaia(input_data):
        df = input_data.copy()
        if not (('gaiaedr3_parallax' in input_data) and ('gaiaedr3_parallax_error' in input_data) and ('gaiaedr3_pmra' in input_data) and ('gaiaedr3_pmra_error' in input_data) and ('gaiaedr3_pmdec' in input_data) and ('gaiaedr3_pmdec_error' in input_data)):
            raise KeyError(f'Gaia data not in input data. Need {gaia_features}')
        
        df['gaia_star'] = 0
        df.loc[((df['gaiaedr3_parallax'] / df['gaiaedr3_parallax_error'])>4.) | (np.abs(df['gaiaedr3_pmra']/df['gaiaedr3_pmra_error'])>4.) | (np.abs(df['gaiaedr3_pmdec']/df['gaiaedr3_pmdec_error'])>4.), ['gaia_star']] = 1


        df['gaia_class'] = 0
        df.loc[(df['class'] != 1), ['gaia_class']] = df.loc[(df['class'] != 1), ['class']].values
        df.loc[(df['gaia_star'] > 0), ['gaia_class']] = 1
        return df
    
    
    
    def test_table(self, input_data, overview='sdssdr16+psdr2+all_decals8tr'):
        from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
        
        def get_confM(y_true, y_pred, weight=[1390, 5714, 1395]):
            matrix = confusion_matrix(y_true, y_pred)
            for i, row in enumerate(matrix):
                matrix[i] = row * (weight[i] / np.sum(row))
            return matrix
            
        def write_confM(df, parameters):
            for flax in parameters:
                print(flax)
                class_name = parameters[flax]['class_name']
                data = df.loc[parameters[flax]['data']][[class_name, f'predict_{overview}']].dropna()
                parameters[flax]['confM'] = get_confM(data[class_name], data[f'predict_{overview}'])
                
        def accuracy(confM):
            return np.sum([confM[i][i] for i in range(3)]) / np.sum(confM)
        
        def precision(confM, cls):
            M = confM.T
            return M[cls-1][cls-1] / np.sum(M[cls-1])
        
        def recall(confM, cls):
            return confM[cls-1][cls-1] / np.sum(confM[cls-1])

        def recall_add(pred, threshold=0.5):
            a = 1 *(np.max(pred, axis=1) > threshold)
            assert len(a) > 0, 'Ampty array'
            return sum(a) / len(a)

        def not_true(pred, threshold=0.5):
            a = 1 *(pred < threshold)
            assert len(a) > 0, 'Ampty array'
            return sum(a) / len(a)
        
        def add_for_data(rec, cl, df, parameters, columns, full_class=False):
            cls = self.cls[cl]
            for flax in parameters:
                tmp0 = df.loc[parameters[flax]['data']]
                confM = parameters[flax]['confM']
                class_name = parameters[flax]['class_name']
                a = [flax,
                     precision(confM, cl),
                     recall(confM, cl), 
                     roc_auc_score(1*(tmp0[class_name]==cl), tmp0[f'predict_proba_{cls}_{overview}']), 
                     np.sum(tmp0[class_name]==cl)]
                rec[cls] = rec[cls].append(pd.DataFrame([a], columns=columns), ignore_index=True)
                
        def add_for_full_data(df, parameters, columns):
            rec = []
            for flax in parameters:
                tmp0 = df.loc[parameters[flax]['data']]
                confM = parameters[flax]['confM']
                class_name = parameters[flax]['class_name']
#                 a = [flax,
#                      accuracy(confM), 
#                      roc_auc_score(tmp0[class_name], 
#                                    tmp0[[f'predict_proba_S_{overview}',
#                                          f'predict_proba_Q_{overview}',
#                                          f'predict_proba_G_{overview}']], multi_class='ovo'), 
#                      len(tmp0[f'predict_proba_S_{overview}'].dropna())]
                rec.append(accuracy(confM))
            return rec

       
        feature = 'decals8tr_z'
        
        if self.folds[0] == 'full':
            raise Exceptions("Can't")
        df = self.predict_gaia(self.predict_proba(self.predict(input_data, train=True, overviews=overview),
                                                  train=True, overviews=overview))
#         print(df)

        parameters = {
#             'GAIA': {
#                 'data': (df['gaia_class'] > 0),
#                 'class_name': 'gaia_class'
#             },
            'ALL': {
                'data': (df['class'] == df['class']),
                'class_name': 'class'
            },
#             f'{feature}<20': {
#                 'data': (df[feature] < 20.),
#                 'class_name': 'class'
#             },
            f'{feature}>20': {
                'data': (df[feature] > 20.),
                'class_name': 'class'
            }  
        }
        write_confM(df, parameters)
        rec = {}
        columns = ['data', 'precision', 'recall', 'roc-auc', 'count']
        for cl in self.cls:
            cls = self.cls[cl]
            rec[cls] = pd.DataFrame([], columns=columns)
            add_for_data(rec, cl, df, parameters, columns)
        

        for i, r in rec.items():
            rec[i] = r.set_index(columns[0])
            
        rec = pd.concat(rec, axis=1).sort_index()
        rec['ALL']=add_for_full_data(df, parameters, columns)
        
        return rec