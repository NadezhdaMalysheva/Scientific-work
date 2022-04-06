from classificationSGQ.predictionsZ.pzph1dot1 import Catalog, flux2mag, split_data, predict, assemble_and_analyze_results

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
mpl.rcParams.update({'font.size': 20})
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)
import plotly.tools as tls

import random, warnings, pickle, os, re, glob, scipy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from scipy.stats import kstest


from enum import Enum
     
class AugmType(Enum):
    """
    Type of implementetion augm models
    """
    KNN = 1
    RNN = 2 # not implemented
    MIX = 3

train_data_path = '/home/nmalysheva/task/S-G-Q_DESI+PanSTARRS+SDSS+WISE+J_UHS/classificationSGQ/train_data'
train_best = 'train_aug_data_best.x1.gz_pkl'
train_features = 'train_aug_data.features.gz_pkl'
train_predictions = 'train_aug_data_predictions.x1.gz_pkl'


def plt_version(version, x, y, ax=None, path_to_files=None, color=None, title=None, norm=None, xlim=[-0.5, 7], ylim=[10, 28.5], s=1, alpha=0.8, cmap='cool', colorbar=True):
    import inspect
    if ax is None:
        _, ax = plt.subplots()
    if isinstance(version, str):
        if path_to_files is None:
            path_to_files= f'/home/nmalysheva/task/S-G-Q_DESI+PanSTARRS+SDSS+WISE+J_UHS/pzph/all_from_local/output_augm/preds_{version}/end'
        
        if len(glob.glob(os.path.join(path_to_files, f'*{version}*.gz_pkl'))) > 1:
            df = pd.concat([pd.read_pickle(file, compression='gzip') for file in glob.glob(os.path.join(path_to_files, f'*{version}*.gz_pkl'))], axis=1)
        else:
            df = pd.read_pickle(list(glob.glob(os.path.join(path_to_files, f'*{version}*.gz_pkl')))[0], compression='gzip')
        ax.set_title(title or version)
    else:
        ax.set_title(title or 'Distribution')
        df = version
        
    x = df.loc[:, x] if not inspect.isfunction(x) else x(df)
    y = df.loc[:, y] if not inspect.isfunction(y) else y(df)
    if color is None:
        color = df.loc[:, 'zoo_best-x1_z_maxConf' if ('zoo_best-x1_z_maxConf' in df) else 'zoo_best_z_maxConf']
        del df
        norm = colors.Normalize(vmin=np.min(color), vmax=np.max(color), clip=True) if norm is None else norm
    elif isinstance(color, list) or isinstance(color, type(np.array([]))) or isinstance(color, type(pd.Series([]))):
        norm = colors.Normalize(vmin=np.min(color), vmax=np.max(color)) if norm is None else norm
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    gr = ax.scatter(x, y, c=color, norm=norm, s=s, cmap=cmap, alpha=alpha)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if colorbar:
        plt.colorbar(gr, ax=ax)
    return norm

def IOU(train_data, input_data):
    import math
    from scipy import interpolate, integrate
    warnings.simplefilter("ignore")
    
    def get_function(data):
        count, values = np.histogram(data.dropna().values, 50)
        count = count / np.max(count)
        f = interpolate.interp1d((values[1:] + values[:-1]) / 2, count, fill_value="extrapolate")
        return f
    
    def iou_function(f1, f2, x1, x2):
        return integrate.quad(lambda x: min(f1(x), f2(x)), x1, x2)[0] / \
               integrate.quad(lambda x: max(f1(x), f2(x)), x1, x2)[0]
    f1 = get_function(train_data)
    f2 = get_function(input_data)
    res = iou_function(f1, f2, min(np.min(train_data), np.min(input_data)), max(np.max(train_data), np.max(input_data)))
    warnings.simplefilter("default")
    return res

def KStest(target_data, input_data):
    D, p = kstest(target_data, input_data, N=100, alternative='two-sided')
    return (round(D, 3), round(p, 3))

keys = ['nrow', 'objID', 'ra', 'dec', 'zspec', 'class', 'fold']
f = ['sdss_psfFlux_u', 'sdss_psfFlux_g', 'sdss_psfFlux_r', 'sdss_psfFlux_i', 'sdss_psfFlux_z', 'sdss_cModelFlux_u', 'sdss_cModelFlux_g', 'sdss_cModelFlux_r', 'sdss_cModelFlux_i', 'sdss_cModelFlux_z', 'ps_gKronFlux', 'ps_rKronFlux', 'ps_iKronFlux', 'ps_zKronFlux', 'ps_yKronFlux', 'ps_gPSFFlux', 'ps_rPSFFlux', 'ps_iPSFFlux', 'ps_zPSFFlux', 'ps_yPSFFlux', 'ls_flux_g', 'ls_flux_r', 'ls_flux_z', 'ls_flux_w1', 'ls_flux_w2', 'ls_flux_w3', 'ls_flux_w4', 'sdss_psfFluxIvar_u', 'sdss_psfFluxIvar_g', 'sdss_psfFluxIvar_r', 'sdss_psfFluxIvar_i', 'sdss_psfFluxIvar_z', 'sdss_cModelFluxIvar_u', 'sdss_cModelFluxIvar_g', 'sdss_cModelFluxIvar_r', 'sdss_cModelFluxIvar_i', 'sdss_cModelFluxIvar_z', 'ps_gKronFluxErr', 'ps_rKronFluxErr', 'ps_iKronFluxErr', 'ps_zKronFluxErr', 'ps_yKronFluxErr', 'ps_gPSFFluxErr', 'ps_rPSFFluxErr', 'ps_iPSFFluxErr', 'ps_zPSFFluxErr', 'ps_yPSFFluxErr', 'ls_flux_ivar_g', 'ls_flux_ivar_r', 'ls_flux_ivar_z', 'ls_flux_ivar_w1', 'ls_flux_ivar_w2', 'ls_flux_ivar_w3', 'ls_flux_ivar_w4']
    
mag = ['sdssdr16_u_psf', 'sdssdr16_g_psf', 'sdssdr16_r_psf', 'sdssdr16_i_psf', 'sdssdr16_z_psf', 'sdssdr16_u_cmodel', 'sdssdr16_g_cmodel', 'sdssdr16_r_cmodel', 'sdssdr16_i_cmodel', 'sdssdr16_z_cmodel', 'psdr2_g_psf', 'psdr2_r_psf', 'psdr2_i_psf', 'psdr2_z_psf', 'psdr2_y_psf', 'psdr2_g_kron', 'psdr2_r_kron', 'psdr2_i_kron', 'psdr2_z_kron', 'psdr2_y_kron', 'decals8tr_g', 'decals8tr_r', 'decals8tr_z', 'decals8tr_Lw1', 'decals8tr_Lw2', 'decals8tr_Lw3', 'decals8tr_Lw4']

mag_agr = ['sdssdr16_u-g_psf', 'sdssdr16_u-r_psf', 'sdssdr16_u-i_psf', 'sdssdr16_u-z_psf', 'sdssdr16_u_psf-cmodel', 'sdssdr16_g-i_psf', 'sdssdr16_g_psf-cmodel', 'sdssdr16_r-i_psf', 'sdssdr16_i-z_psf', 'sdssdr16_i_psf-cmodel', 'decals8tr_Lw1-Lw2', 'sdssdr16_u_cmodel-decals8tr_Lw1', 'sdssdr16_u_cmodel-decals8tr_Lw2', 'sdssdr16_g_cmodel-decals8tr_Lw1', 'sdssdr16_g_cmodel-decals8tr_Lw2', 'sdssdr16_r_cmodel-decals8tr_Lw1', 'sdssdr16_r_cmodel-decals8tr_Lw2', 'sdssdr16_i_cmodel-decals8tr_Lw1', 'sdssdr16_i_cmodel-decals8tr_Lw2', 'sdssdr16_z_cmodel-decals8tr_Lw1', 'sdssdr16_z_cmodel-decals8tr_Lw2']

preds = ['zoo_best-x1_z_max', 'zoo_best-x1_z_maxConf']

gaia_features = [
    'gaiaedr3_parallax',
    'gaiaedr3_parallax_error',
    'gaiaedr3_pmra',
    'gaiaedr3_pmra_error',
    'gaiaedr3_pmdec',
    'gaiaedr3_pmdec_error',
    'gaiaedr3_phot_g_mean_flux',
    'gaiaedr3_phot_bp_mean_flux',
    'gaiaedr3_phot_rp_mean_flux',
    'gaiaedr3_phot_g_mean_flux_error',
    'gaiaedr3_phot_bp_mean_flux_error',
    'gaiaedr3_phot_rp_mean_flux_error',
#     'gaiaedr3_pseudocolour', 
#     'gaiaedr3_pseudocolour_error',
#     'gaiaedr3_dr2_radial_velocity',
#     'gaiaedr3_dr2_radial_velocity_error'
]

def change_coord(df_before):
    change = {
        "objID_sdssdr16": 'sdss_objID',
        "objID_psdr2" : "ps_objID",
        "mw_transmission_g": "ls_mw_transmission_g",
        'mw_transmission_r': 'ls_mw_transmission_r',
        'mw_transmission_z': 'ls_mw_transmission_z',
        'mw_transmission_w1': 'ls_mw_transmission_w1',
        'mw_transmission_w2': 'ls_mw_transmission_w2',
        'mw_transmission_w3': 'ls_mw_transmission_w3',
        'mw_transmission_w4': 'ls_mw_transmission_w4',
        "w1_nanomaggies" : 'ps_w1flux',
        "w1_nanomaggies_ivar": 'ps_dw1flux',
        "w2_nanomaggies": 'ps_w2flux',
        "w2_nanomaggies_ivar":  'ps_dw2flux',
        "objid": "ls_objid",
        "ebv": "ls_ebv"
    }
    
    
    
    df_bef = df_before.loc[:, keys + ['sdss_objID', 'ps_objID', 'ls_objid', 'ps_w1flux', 'ps_w2flux','ps_dw1flux', 'ps_dw2flux', 'ls_ebv', 'ls_mw_transmission_g', 'ls_mw_transmission_r', 'ls_mw_transmission_z', 'ls_mw_transmission_w1', 'ls_mw_transmission_w2', 'ls_mw_transmission_w3', 'ls_mw_transmission_w4'] + f + mag + mag_agr + gaia_features]
    
    for i in f:
        if 'sdss' in i:
            before = i[5:]
        else:
            before = i[3:] + '_min_error'
        print(f"{before} --> {i}")
        df_bef.loc[df_bef['class']!=1, i] = df_before.loc[df_before['class']!=1, before]
#         print(df_bef.loc[df_bef['class']!=1, i])
    for i, j in change.items():
        print(f"{i} --> {j}")
        df_bef.loc[df_bef['class']!=1, j] = df_before.loc[df_before['class']!=1, i]
#         print(df_bef.loc[df_bef['class']!=1, j])
    return df_bef


class Augmentation:
    import warnings
    
    def __init__(self, type_model=AugmType.KNN, n_neighbors=5, path_to_models=None, debug=False):
        self.path_to_data = None
        self.type = type_model
        self.n_neighbors = n_neighbors
        self.debug = debug
        if path_to_models is None:
            warnings.warn('path_to_models is not define. Will be used ./AugModel')
            if not os.path.exists('./AugModel'):
                os.mkdir('./AugModel')
            self.path_to_model = './AugModel'
        elif not os.path.exists(path_to_models):
            warnings.warn(f'{path_to_models} didn\'t exist! Try to create...')
            try:
                os.mkdir(path_to_models)
            except OSError:
                print (f"Failed to create directory {path_to_models}! Will be used ./AugModel")
                if not os.path.exists('./AugModel'):
                    os.mkdir('./AugModel')
                self.path_to_model = './AugModel'
            else:
                print ("The directory has been successfully created")
                self.path_to_model = path_to_models
        else:
            self.path_to_model = path_to_models
        
        
        self.used_flux = set()
        self.flux_err = {
            'sdss_psfFlux_u': 'sdss_psfFluxIvar_u',
            'sdss_psfFlux_g': 'sdss_psfFluxIvar_g',
            'sdss_psfFlux_r': 'sdss_psfFluxIvar_r',
            'sdss_psfFlux_i': 'sdss_psfFluxIvar_i',
            'sdss_psfFlux_z': 'sdss_psfFluxIvar_z',
            'sdss_cModelFlux_u': 'sdss_cModelFluxIvar_u',
            'sdss_cModelFlux_g': 'sdss_cModelFluxIvar_g',
            'sdss_cModelFlux_r': 'sdss_cModelFluxIvar_r',
            'sdss_cModelFlux_i': 'sdss_cModelFluxIvar_i',
            'sdss_cModelFlux_z': 'sdss_cModelFluxIvar_z',
            'ps_gKronFlux': 'ps_gKronFluxErr',
            'ps_rKronFlux': 'ps_rKronFluxErr',
            'ps_iKronFlux': 'ps_iKronFluxErr',
            'ps_zKronFlux': 'ps_zKronFluxErr',
            'ps_yKronFlux': 'ps_yKronFluxErr',
            'ps_gPSFFlux': 'ps_gPSFFluxErr',
            'ps_rPSFFlux': 'ps_rPSFFluxErr',
            'ps_iPSFFlux': 'ps_iPSFFluxErr',
            'ps_zPSFFlux': 'ps_zPSFFluxErr',
            'ps_yPSFFlux': 'ps_yPSFFluxErr',
            'ls_flux_g': 'ls_flux_ivar_g',
            'ls_flux_r': 'ls_flux_ivar_r',
            'ls_flux_z': 'ls_flux_ivar_z',
            'ls_flux_w1': 'ls_flux_ivar_w1',
            'ls_flux_w2': 'ls_flux_ivar_w2',
            'ls_flux_w3': 'ls_flux_ivar_w3',
            'ls_flux_w4': 'ls_flux_ivar_w4'
        }
        self.flux_mag = {
            'sdss_psfFlux_u': 'sdssdr16_u_psf',
            'sdss_psfFlux_g': 'sdssdr16_g_psf',
            'sdss_psfFlux_r': 'sdssdr16_r_psf',
            'sdss_psfFlux_i': 'sdssdr16_i_psf',
            'sdss_psfFlux_z': 'sdssdr16_z_psf',
            'sdss_cModelFlux_u': 'sdssdr16_u_cmodel',
            'sdss_cModelFlux_g': 'sdssdr16_g_cmodel',
            'sdss_cModelFlux_r': 'sdssdr16_r_cmodel',
            'sdss_cModelFlux_i': 'sdssdr16_i_cmodel',
            'sdss_cModelFlux_z': 'sdssdr16_z_cmodel',
            'ps_gPSFFlux': 'psdr2_g_psf',
            'ps_rPSFFlux': 'psdr2_r_psf',
            'ps_iPSFFlux': 'psdr2_i_psf',
            'ps_zPSFFlux': 'psdr2_z_psf',
            'ps_yPSFFlux': 'psdr2_y_psf',
            'ps_gKronFlux': 'psdr2_g_kron',
            'ps_rKronFlux': 'psdr2_r_kron',
            'ps_iKronFlux': 'psdr2_i_kron',
            'ps_zKronFlux': 'psdr2_z_kron',
            'ps_yKronFlux': 'psdr2_y_kron',
            'ls_flux_g': 'decals8tr_g',
            'ls_flux_r': 'decals8tr_r',
            'ls_flux_z': 'decals8tr_z',
            'ls_flux_w1': 'decals8tr_Lw1',
            'ls_flux_w2': 'decals8tr_Lw2',
            'ls_flux_w3': 'decals8tr_Lw3',
            'ls_flux_w4': 'decals8tr_Lw4'
        }
    
    @staticmethod
    def reverse_dictionary(input_dict):
        return {j:i for i, j in input_dict.items()}
        
    def train(self, path_to_data):
        self.path_to_data = path_to_data
        self.train_data = Catalog.read_table(self.path_to_data)
        if self.debug:
            print('Catalog was been reading. Shape: ', self.train_data.shape)
        for column in self.flux_err:
            if column in self.train_data:
                if self.debug:
                    print(f'Column {column}')
                self.used_flux.add(column)
                self._fit_for_flux(column)
        del self.train_data
        self.save(self)
                   
    def _fit_for_flux(self, flux_name):
        err_name = self.flux_err[flux_name]
        if self.debug:
            print(f'In _fit {flux_name}, {err_name}')
            
        assert err_name in self.train_data, f'Beeeeedaaaaa! {err_name} not in train table'
        flux, err = self._preprocessing(self.train_data[flux_name], self.train_data[err_name])
        
        model = {}
        if self.type.value % 2:
            if self.debug:
                print('Fit KNN')
                print(flux, flux.shape, err, err.shape)
            model['knn'] = KNeighborsRegressor(self.n_neighbors).fit(flux, err) 
        if self.type.value > 1:
            if self.debug:
                print('Fir RNR')
            model['rnn'] = RadiusNeighborsRegressor().fit(flux, err)
            model['normalize'] = 0.2 * np.max(flux) / 100
        model['y'] = err
        self._save_model(model, flux_name)
        return self
    
    
    def parse_cases(self, data, dm=0, gauss_augm=True, use_def_statistic=False, count_iter=1, type_model='shift'):
        tm = {'shift': 0, 'one_gauss': 1, 'add_gauss': 2, 'optimization': 3}
        type_model = tm.get(type_model, 0)
        
        # Если нужно использование статистики
        print('type_model', type_model)
        print('use_def_statistic --', use_def_statistic)
        if isinstance(use_def_statistic, str):
            if type_model == 0:
                type_model = 1
            self.return_data = Catalog.read_table(data).copy()
            self.train_data = Catalog.read_table(self.path_to_data)
            assert use_def_statistic in self.train_data, "features for obtaining statistics is not used in the train data"
            assert use_def_statistic in self.return_data, "features for obtaining statistics is not used in the predict data"
            if type_model < 4:
                dm, itr, fp = self.function_for_sampling(self.train_data[use_def_statistic],
                                                         self.return_data[use_def_statistic],
                                                         type_model)
            else:
                print('self.flux_mag[use_def_statistic]', self.flux_mag[use_def_statistic])
                dm, itr, fp = self.function_for_sampling(self.train_data[use_def_statistic],
                                                         self.return_data[use_def_statistic],
                                                         type_model,
                                                         self.train_data[self.flux_mag[use_def_statistic]],
                                                         self.return_data[self.flux_mag[use_def_statistic]])
                
            count_iter = count_iter or itr
            del self.train_data
        else:
            type_model = 0
        
        # Тест, что dm - приемлима
        if isinstance(dm, float) or isinstance(dm, int):
            dm_tmp = float(dm)
            dm = lambda x : [dm_tmp] * x
            type_model = 0
        elif not hasattr(dm, '__call__'):
            raise TypeError('dm is of the wrong type')
            
        # Нужно ли нам сохранять оригинал? Или об этом будет заботиться пользователь?
        if type_model < 2:
            self.return_data = pd.DataFrame()
            
        preprocessing_before_aug = not (use_def_statistic in self.used_flux) and (type_model > 2)
        
        get_leave_indexes = lambda def_data, real_data:  fp.get_probability(real_data)(def_data) > np.random.random(len(def_data))
        
        return preprocessing_before_aug, dm, get_leave_indexes
                
        

    def predict(self, data, dm=0, gauss_augm=True, use_def_statistic=False, count_iter=1, type_model='shift'):
        preprocessing_before_aug, dm, get_leave_indexes = self.parse_cases(data, dm, gauss_augm, use_def_statistic, count_iter, type_model)
        flux_statistic = self.reverse_dictionary(self.flux_mag)[use_def_statistic] if preprocessing_before_aug else ''
        print('count iter =', count_iter)
        for i in range(count_iter):
            print(f'i = {i}')
                
            df = Catalog.read_table(data).copy()
            
            df.loc[:, 'dm'] = dm(len(df))
            if preprocessing_before_aug:
                self._predict_for_flux(flux_statistic, df, gauss_augm)
#                 print('before flux2mag', df[use_def_statistic])
                df[use_def_statistic] = flux2mag(df[[flux_statistic]], df[[self.flux_err[flux_statistic]]], {flux_statistic: use_def_statistic})
#                 print('after flux2mag', df[use_def_statistic])
                df = df.loc[get_leave_indexes(df[use_def_statistic], self.return_data[use_def_statistic])]

           
            for flux_name in self.used_flux - set([flux_statistic]):
                print(flux_name)
                self._predict_for_flux(flux_name, df, gauss_augm) 
            
            del df['dm']
            
            # Заканчиваем преобразования
            self.return_data = pd.concat([self.return_data, df], axis=0, ignore_index=True)
            print('self.return_data.shape', self.return_data.shape)
            del df

                
        return self.return_data
            
        
    def _predict_for_flux(self, flux_name, df, gauss_augm):
        if flux_name not in df:
            return
        
        err_name = self.flux_err[flux_name]

        assert err_name in df, f'Beeeeedaaaaa! {err_name} not in predict table'
        if self.debug:
            print('before', df.loc[:, flux_name], np.sum(df.loc[:, flux_name]))

        flux, err = self._preprocessing(df[flux_name], df[err_name])
        local_dm = df.loc[self.tmp_index, 'dm'].values.reshape(-1, 1)

        if self.debug:
            print('preprocess', flux, np.sum(flux))
            self.plotik(flux, err, flux_name + 'before')

        flux *= np.power(10, 0.4 * local_dm)
        model = self._read_model(flux_name)
        err = self.neighbors(model, flux, sigma=err)

        if self.debug:
            self.plotik(flux, err, flux_name + 'after_tmp')

        if gauss_augm:
            flux = np.array(list(map(self.gauss_flux, zip(flux, err))))
            if self.debug:
                self.plotik(flux, err, flux_name + 'after')

        if self.debug:
            print('after', flux, np.sum(flux)) 

        df.loc[:, flux_name], df.loc[:, err_name] = self._postrocessing(flux, err)

        if self.debug:
            print('postprocess', df[flux_name], np.sum(df[flux_name])) 
    
    def neighbors(self, model, x_input, n_neighbors=None, radius=None, sigma=1e-9): #TO
        if self.debug:
            print(f'Hi! I\'m neighbors of {x_input}')
        x = x_input
        if isinstance(x_input, float) or isinstance(x_input, int):
            x = np.array([x_input])
        x = np.array(x, dtype=float)
        good_rows = ~(np.isnan(x) + np.isinf(x))
        res = np.zeros(x.shape)
        if not(n_neighbors is None):
            self.n_neighbors = n_neighbors
 
        if self.type == AugmType.KNN:
            assert self.n_neighbors > 0
            knb = model['knn'].kneighbors(x, n_neighbors=self.n_neighbors, return_distance=False)
            if self.debug:
                print('knn')
            res_tmp = []
            for x_loc, n, i in zip(x, knb, range(len(knb))):
                res_tmp.append(np.random.choice(model['y'][n][:, 0])) ################
            res = np.array(res_tmp)
                
        elif self.type == AugmType.RNN:
            pass

        elif self.type == AugmType.MIX:
            assert self.n_neighbors > 0
            for x_loc, s, i in zip(x, sigma, range(len(x))): # самая долгая часть - причина, почему обычные соседи могут быть лучше
                if radius is None:
                    r = np.max([0.2*x_loc, 3*s]) / model['normalize']
                else:
                    r = radius
                rnb = model['rnn'].radius_neighbors(x_loc.reshape(-1, 1), radius=r, return_distance=False)[0]
                if len(rnb) < self.n_neighbors:
                    knb = model['knn'].kneighbors(x_loc.reshape(-1, 1), n_neighbors=self.n_neighbors, return_distance=False)[0]
                    res[i] = np.random.choice(model['y'][knb][:, 0])
                else:
                    res[i] = np.random.choice(model['y'][rnb][:, 0])
        return res

    
    def _preprocessing(self, flux, err):
        if self.debug:
            print('In _preprocessing')
        if 'ivar' in err.name or 'Ivar' in err.name:
            err = np.power(err, -0.5)
            self.tmp_case = 1
        elif re.findall('^ps_dw[0-9]flux_ab$', err.name):
            flux, err = flux.replace(-999, np.NaN), err.replace(-999, np.NaN)
            self.tmp_case = 2
        else:
            flux, err = flux.replace(-999, np.NaN) / 3621e-9, err.replace(-999, np.NaN) / 3621e-9
            self.tmp_case = 3

        flux, err = np.array(flux.values, dtype=float), np.array(err.values, dtype=float)
        self.tmp_index = np.isfinite(flux) & np.isfinite(err)
            
        return flux[self.tmp_index].reshape(-1, 1), err[self.tmp_index].reshape(-1, 1)
        
        
    def _postrocessing(self, flux, err): #TO
        if self.debug:
            print('In _postprocessing')
        if self.tmp_case == 1:
            err = np.power(err, -2)
            
        elif self.tmp_case == 3:
            flux, err = flux * 3621e-9, err * 3621e-9
            
                
        if self.debug:
            print(self.tmp_index, self.tmp_index.shape)
            
        flux_res, err_res = np.full((len(self.tmp_index), 1), np.nan), np.full((len(self.tmp_index), 1), np.nan)
        flux_res[self.tmp_index], err_res[self.tmp_index] = flux.reshape(-1, 1), err.reshape(-1, 1)
        return flux_res, err_res
        
    @staticmethod
    def gauss_flux(inputs):
        mu, err = inputs
        sigma = np.sqrt(err ** 2 + (mu * 0.03) ** 2)
        return np.random.normal(mu, sigma)
    
    
    def plotik(self, x, y, name=''):
        import seaborn as sns
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(nrows = 1, ncols=1, figsize=(12, 8), sharex=True, sharey = True)
        ax.set_title(name) 
        ax.set_xlim([-1000, 30000])
        ax.set_ylim([-10, 400])
        ax.scatter(x, y, c='r')
        if not os.path.exists(os.path.join(self.path_to_model, 'plt')):
                os.mkdir(os.path.join(self.path_to_model, 'plt'))
        f.savefig(os.path.join(self.path_to_model, 'plt', self.type.name+'_'+name+'.png'))
    
    def _save_model(self, model, flux_name):
        if self.debug:
            print(f'Self mimi model')
        with open(os.path.join(self.path_to_model, self.type.name+'_'+flux_name), 'wb') as file:
            pickle.dump(model, file) 
            
    def _read_model(self, flux_name):
        if self.debug:
            print(f'Read mimi model')
        with open(os.path.join(self.path_to_model, self.type.name+'_'+flux_name), 'rb') as file:
            return pickle.load(file)
       
    def save(self, model, path=None):
        with open(os.path.join(path or self.path_to_model, self.type.name+'_CLASS'), 'wb') as file:
            pickle.dump(model, file) 
    
    @staticmethod
    def read(path, type_m=AugmType.KNN):
        with open(os.path.join(path, type_m.name+'_CLASS'), 'rb') as file:
            return pickle.load(file)
        
    @staticmethod
    def function_for_sampling(train_data, df_before, type_model, add_train=None, add_before=None):
        import math
        from scipy import interpolate
        
        def mode(series, bins=100):
            return pd.cut(series, bins).value_counts().sort_values().index[-1].mid
        
        class functions:
            def __init__(self, coeff, ftrain, faug):
                self.coeff = coeff
                self.ftrain = ftrain
                self.faug = faug
#                 print('init fp')
            
            def get_probability(self, data):
                def diff(f, f1, f2):
                    def norm_diff(x):
                        fx = f(x)
                        f1x = f1(x)
#                         print('norm_diff', x)
#                         print('f(x)', fx)
#                         print('f1(x)', f1x)
                        res = (fx - f1x)/ f1x
                        res[res < 0] = 0
                        res[res > 1] = 1
                        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
#                         print('res', np.sort(res))
                        return res
                    return norm_diff
                
                new = data.dropna().values
                count, values = np.histogram(new, 100)
                count = count * self.coeff
                func = interpolate.interp1d((values[1:] + values[:-1]) / 2, count, fill_value="extrapolate")
#                 print('get_probability')
                return diff(self.ftrain, func, self.faug)
        

        mu = mode(train_data) - mode(df_before)
        q = [0.1, 0.9]
        sigma = math.sqrt(max(
            train_data.loc[(train_data > train_data.quantile(q[0])) & (train_data < train_data.quantile(q[1]))].std()**2 - \
            df_before.loc[(df_before > df_before.quantile(q[0])) & (df_before < df_before.quantile(q[1]))].std()**2, 0))


        print(f'mu = {mu}, sigma = {sigma}')
        castom_dm = lambda x: - np.random.normal(mu, sigma, x)
        itr, fp = None, None
        
        if type_model > 2:
            train = train_data.dropna().values
            count, values = np.histogram(train, 100)
            count = count / np.max(count)
            

            before = df_before.dropna().values
            count1, values1 = np.histogram(before, 100)
            c1max= np.argmax(count1)
            coeff = count[np.argwhere((values <= values1[c1max+1]) & (values1[c1max - 1] <= values))][0][0] / np.max(count1)
            count1 = count1 * coeff

            itr = math.ceil(np.max(count) / np.max(count1))
            
            f = interpolate.interp1d((values[1:] + values[:-1]) / 2, count, fill_value="extrapolate")
            fp = functions(coeff, f, None)
        
        return castom_dm, itr, fp
    
def read_after_augm(path, version):
    df = pd.concat(
        [pd.read_pickle(file, compression='gzip') for file in glob.glob(
            os.path.join(path, f'df_aug_cls_*{version}*_3.features.gz_pkl'))], 
        axis=1)
    return df

def predict_SGQ(model, input_data, features='decals8tr_z', version='one_gauss', path_buf='./', plot=False, return_data=True):
    try:
        os.mkdir(path_buf)
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(path_buf, '__buf__'))
    except FileExistsError:
        print(f"{os.path.join(path_buf, '__buf__')} exists")
        
    path_buf = os.path.join(path_buf, '__buf__')
    train_data = Catalog.read_table(model.path_to_data)
    input_data = Catalog.read_table(input_data)
    for cls in [1, 2, 3]:
        print(f'CLASS = {cls}')
        df = input_data.loc[input_data['class']==cls]
        if plot:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (10, 10))
            lim=(11, 30)
            sns.histplot(df[features], ax = ax, bins = 100, color = 'g', binrange=lim, alpha=0.2, stat="probability")
            sns.histplot(train_data[features], ax = ax, bins = 100, color = 'r', binrange=lim, stat="probability", element="step", fill=False)
            ax.set_title(f"CLASS = {cls}, IOU = {round(IOU(train_data[features], df[features]), 3)}")
            plt.show()
        df_castom_dm = model.predict(df, use_def_statistic=features, type_model=version)
        df_castom_dm.to_pickle(os.path.join(path_buf, f'df_aug_cls_{cls}_{version}_1.gz_pkl'), compression='gzip')
        df_castom_dm.to_pickle(os.path.join(path_buf, f'df_aug_cls_{cls}_{version}_2.gz_pkl'), compression='gzip')


        catalog = Catalog(None, ('ra', 'dec'),
                      njobs=24, output_dir=path_buf,
                      assembled_dataset_path=os.path.join(path_buf, f'df_aug_cls_{cls}_{version}_2.gz_pkl'),
                      filename=f'df_augg_cls_{cls}_{version}_3')
        catalog.prepare_data(augmentation=False)
        
    if return_data:
        return read_after_augm(path_buf, version)
        
        
def plt_predict_SGQ(model, input_data, features='decals8tr_z', version='one_gauss', path_buf='__buf__', metric='KStest'):
    train_data = Catalog.read_table(model.path_to_data)
    input_data = Catalog.read_table(input_data)
    fig, axes = plt.subplots(ncols=3, figsize = (32, 7))
    lim=(11, 30)
    for cls in [1, 2, 3]:
        df = input_data.loc[input_data['class']==cls]
        ax = axes[cls-1]
        df_after = pd.read_pickle(os.path.join(path_buf, f'df_augg_cls_{cls}_{version}_3.features.gz_pkl'), compression='gzip')
        sns.histplot(df_after[features], ax = ax, bins = 100, color = 'b', binrange=lim, alpha=0.4, stat="probability")
        sns.histplot(df[features], ax = ax, bins = 100, color = 'g', binrange=lim, alpha=0.2, stat="probability")
        sns.histplot(train_data[features], ax = ax, bins = 100, color = 'r', binrange=lim, stat="probability", element="step", fill=False)
        if metric == 'IOU':
            ax.set_title(f"CLASS = {cls}, IOU = {round(IOU(train_data[features], df_after[features]), 3)}, count = {round(len(df_after) / 1000, 1)}K")
        elif metric == 'KStest':
            ax.set_title(f"CLASS = {cls}, KStest = {KStest(train_data[features], df_after[features])}, count = {round(len(df_after) / 1000, 1)}K")
        else:
            print('Use KStest metrics')
            ax.set_title(f"CLASS = {cls}, KStest = {KStest(train_data[features], df_after[features])}, count = {round(len(df_after) / 1000, 1)}K")
        

        
def read_after_pzph_predict(path, version, features=[]):
    for_clf = ['sdssdr16_u_psf', 'sdssdr16_g_psf', 'sdssdr16_r_psf', 'sdssdr16_i_psf', 'sdssdr16_z_psf', 'sdssdr16_u_cmodel', 'sdssdr16_i_cmodel', 'sdssdr16_u-g_psf', 'sdssdr16_u-r_psf', 'sdssdr16_u-i_psf', 'sdssdr16_u-z_psf', 'sdssdr16_u_psf-cmodel', 'sdssdr16_g-i_psf', 'sdssdr16_g_psf-cmodel', 'sdssdr16_r-i_psf', 'sdssdr16_i-z_psf', 'sdssdr16_i_psf-cmodel', 'psdr2_i_kron', 'psdr2_y_kron', 'psdr2_g_psf', 'psdr2_r_psf', 'psdr2_i_psf', 'psdr2_z_psf', 'psdr2_y_psf', 'psdr2_g-i_psf', 'psdr2_g-y_psf', 'psdr2_r-i_psf', 'psdr2_r-y_psf', 'psdr2_i-z_psf', 'psdr2_i-y_psf', 'psdr2_i_psf-kron', 'psdr2_z-y_psf', 'psdr2_y_psf-kron', 'decals8tr_Lw1-Lw2', 'decals8tr_Lw1', 'decals8tr_Lw2', 'decals8tr_g', 'decals8tr_r', 'decals8tr_z', 'decals8tr_g-r', 'decals8tr_g-z', 'decals8tr_r-z', 'psdr2_g_kron-decals8tr_Lw1', 'psdr2_g_kron-decals8tr_Lw2', 'psdr2_r_kron-decals8tr_Lw1', 'psdr2_r_kron-decals8tr_Lw2', 'psdr2_i_kron-decals8tr_Lw1', 'psdr2_i_kron-decals8tr_Lw2', 'psdr2_z_kron-decals8tr_Lw1', 'psdr2_z_kron-decals8tr_Lw2', 'psdr2_y_kron-decals8tr_Lw1', 'psdr2_y_kron-decals8tr_Lw2', 'sdssdr16_u_cmodel-decals8tr_Lw1', 'sdssdr16_u_cmodel-decals8tr_Lw2', 'sdssdr16_g_cmodel-decals8tr_Lw1', 'sdssdr16_g_cmodel-decals8tr_Lw2', 'sdssdr16_r_cmodel-decals8tr_Lw1', 'sdssdr16_r_cmodel-decals8tr_Lw2', 'sdssdr16_i_cmodel-decals8tr_Lw1', 'sdssdr16_i_cmodel-decals8tr_Lw2', 'sdssdr16_z_cmodel-decals8tr_Lw1', 'sdssdr16_z_cmodel-decals8tr_Lw2', 'sdssdr16_g_cmodel-decals8tr_g', 'sdssdr16_r_cmodel-decals8tr_r', 'sdssdr16_z_cmodel-decals8tr_z']
    df = pd.DataFrame()
    for file in glob.glob(os.path.join(path, f'df_augg_cls_*{version}*features.gz_pkl')):
        print(list(glob.glob(
                os.path.join(path, f"*{file.split('/')[-1].split('.')[-3]}*.gz_pkl"))))
        tmp = pd.concat(
            [pd.read_pickle(f, compression='gzip') for f in glob.glob(
                os.path.join(path, f"*{file.split('/')[-1].split('.')[-3]}*.gz_pkl"))], 
            axis=1)[keys + features + for_clf + preds]
        df = pd.concat([df, tmp], axis=0)
    return df

def pzph_predict_SGQ(version='one_gauss', path_buf='__buf__', 
                     modelsIds=[21, 35], modelsSeries='x1', njobs=24):
    try:
        os.mkdir(os.path.join(path_buf, f'preds_{version}'))
    except FileExistsError:
        pass
    print('split')
    for cls in [1, 2, 3]:
        df_tmp = pd.read_pickle(os.path.join(path_buf, f'df_augg_cls_{cls}_{version}_3.features.gz_pkl'),
                                compression='gzip')
        for i, chunk in enumerate(split_data(df_tmp)):
            fname = 'part-{:05d}'.format(i)
            print(fname)
            chunk['xray'].to_pickle(
                os.path.join(path_buf, 
                             f'preds_{version}', 
                             f'df_augg_cls_{cls}_{version}_3_{fname}.features.gz_pkl'), 
                compression='gzip')
    data_path = os.path.join(path_buf, f'preds_{version}')
    files = glob.glob(os.path.join(path_buf, 
                             f'df_augg_cls_*_{version}_3_*.features.gz_pkl'))
    print('predict')
    predict(datasets_files=files, modelsIds=modelsIds, modelsSeries=modelsSeries, njobs=njobs)
    
    try:
        os.mkdir(os.path.join(path_buf, 'end'))
    except FileExistsError:
        pass
    print('assemble_and_analyze_results')
    for cls in [1, 2, 3]:
        for file in glob.glob(os.path.join(path_buf, f'df_augg_cls_{cls}*{version}*features.gz_pkl')):
            assemble_and_analyze_results(path_buf, 
                                         os.path.join(path_buf, 'end'), 
                                         models_series=modelsSeries, 
                                         file_name=file.split('/')[-1].split('.')[0])
    
    return read_after_pzph_predict(os.path.join(path_buf, 'end'), version)  
  