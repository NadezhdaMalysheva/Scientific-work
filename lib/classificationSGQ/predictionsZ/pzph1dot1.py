"""
Small module containing functionality to make probabilistic photo-z predictions for eRosita.
"""
import argparse
import json
import functools
import glob
import importlib
import os
import multiprocessing
import pickle
import re
import shutil
import subprocess
import sys
import time
import gc
from collections import defaultdict

import astropy.table
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pandas as pd
import scipy.stats
import tqdm
import warnings

from classificationSGQ.predictionsZ.process_counterparts import process_counterparts
from classificationSGQ.predictionsZ.process_predictions import process_predictions
from classificationSGQ.predictionsZ.duplicates import clean_duplicates


# from panstarrscasjobs import panstarrs_casjobs


def _import_user_defined_features_transformation(
        path_to_code: str,
        transformation_name: str):
    path, module_name = os.path.split(os.path.splitext(path_to_code)[0])
    sys.path.append(path)
    user_module = importlib.import_module(module_name)
    return getattr(user_module, transformation_name)


def _columns_intersection(df1: pd.DataFrame, df2: pd.DataFrame):
    result = list()
    for col in df1.columns:
        if col in df2.columns:
            result.append(col)

    return result


def format_message(msg):
    return f'===== {msg} ====='


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size > 0


def parse_cli_args():
    def check_args(args):
        return True

    def _add_default_and_type(desc: str, arg_type=None, default_value=None,
                              isflag=False):
        """
        Helper function to form an argument description for argparse
        :param desc: helper description
        :param arg_type: argument type to insert to argument description
        :param default_value: default value of arg_type to insert to argument description
        :param isflag: if True, then adds info that the argument is a flag
        :return: argument description with type and default value information if such provided
        """
        assert arg_type is None or callable(arg_type)
        assert default_value is None or isinstance(default_value, arg_type)

        if isflag:
            default_value_msg = ''
            arg_type_msg = 'flag'
        else:
            arg_type_msg = ''
            if arg_type is not None:
                arg_type_msg = f'type: {type(arg_type()).__name__}'

            default_value_msg = ''
            if default_value is not None:
                if arg_type == str:
                    default_value_msg = f'default: "{default_value}"'
                else:
                    default_value_msg = f'default: {default_value}'

        if default_value_msg and arg_type_msg:
            return f'[{arg_type_msg}; {default_value_msg}] {desc}'

        if arg_type_msg:
            return f'[{arg_type_msg}] {desc}'

        if default_value_msg:
            return f'[{default_value_msg}] {desc}'

        return desc

    description = "Script to make photo-z estimations for eRosita. Note that all arguments are specified as" \
                  " optional because of several different use cases, so pay attention to them (see below)." \
                  "\n" \
                  "\n Use cases are:" \
                  "\n 0) import to your script (not recommended, it's pretty raw)" \
                  "\n 1) Make cross-correlation [and predictions] for X-ray catalog" \
                  " [in this case you should specify --xrayCatalog, --baseCatalog, --xrayObjIDColumn and" \
                  " other arguments (see below)]." \
                  "\n 2) Make cross-correlation [and predictions] for photometric catalog" \
                  " [in this case you should specify at least one of --sdss, --ps, --ls (based on your choice for" \
                  " --baseCatalog), also --sdssCounterpartIDColumn, --psCounterpartIDColumn and" \
                  " --lsCounterpartIDColumn respectively, and other arguments (see below). You can also specify]" \
                  "\n 3) [DO NOT USE] Process assembled dataset containing SDSS, PanSTARRS, DESI LIS and WISE data" \
                  " [specify only --assembledDataset and other arguments (see below)]" \
                  "\n 4) Apply model to features files. Then specify --predictOn, --modelsSeries and --modelsIds" \
                  "\n" \
                  "\n If you do not want to apply models, you should not specify --modelsIds" \
                  "\n" \
                  "\n List of \"OTHER ARGUMENTS\":" \
                  "\n --njobs, # TODO" \
                  "\n" \
                  "\n List of models" \
                  "\n " \
                  "\n   - gal0 - Models for galaxies. 19, 21, 22, 35 are available for now (see x1 description below)" \
                  "\n   - x0 - Old models for X-ray objects. 19, 21, 22, 35 are available (see x1 description below)" \
                  "\n   - x0pswf and gal0pswf are the same as described above, but predict using Rodion WISE for Pan-STARRS" \
                  "\n   - x1 and x1a - SRGz 1 Series models for X-ray objects" \
                  "\n   - with amd without errors perturbations respectively" \
                  "\n       - 18 - SDSS + WISE" \
                  "\n       - 19 - PanSTARRS + WISE" \
                  "\n       - 20 - SDSS + DESI LIS + WISE" \
                  "\n       - 21 - PanSTARRS + DESI LIS + WISE" \
                  "\n       - 22 - DESI LIS + WISE" \
                  "\n       - 34 - SDSS + PanSTARRS + WISE" \
                  "\n       - 35 - SDSS + PanSTARRS + DESI LIS + WISE"
    epilog = "Hint: arguments have long, inconvenient names (sorry for that), so I recommend to save your every" \
             " usage as a shell script" \
             "\n EXAMPLES:" + \
             """
             python3 pzph1.py --output results/ls --baseCatalog ls \
                --xrayCatalog xray_data.pkl --primaryRadius=30 \
                --xrayRaCol RA --xrayDecCol DEC \
                --baseRaCol ra --baseDecCol dec \
                --njobs $NJOBS --modelsSeries x1 --modelsIds 21 22 35 --xrayHealpixId healpix \
                --psFluxesManually
                # split data using 'healpix' columns, find counterparts in DESI LIS and apply 21, 22 and 35 x-ray models
             """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description, epilog=epilog)

    # use case 1 arguments
    argument_description = "Path to X-ray catalog with objects' unique identifiers, coordinates and, maybe, other" \
                           " columns. Supports fits and pickled pandas data frame."
    arg_type = str
    default_value = None
    parser.add_argument('--xrayCatalog', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Ra column in --xrayCatalog"
    arg_type = str
    default_value = 'RA_'
    parser.add_argument('--xrayRaCol', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Healpix number column in --xrayCatalog, that defines chunks."
    arg_type = str
    default_value = None
    parser.add_argument('--xrayHealpixId', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Base catalog to correlate X-ray data in 30\" radius with (or if already correlated)." \
                           " Possible values are:" \
                           "\n - 'ps' - use PanSTARRS as base catalog" \
                           "\n - 'ls' - use DESI LIS as base catalog" \
                           "\n - 'sdss' - use SDSS as base catalog (can only be chosen for use case 2)"
    arg_type = str
    default_value = "ls"
    parser.add_argument('--baseCatalog', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Ra column in base catalog"
    arg_type = str
    default_value = 'ra'
    parser.add_argument('--baseRaCol', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Dec column in base catalog"
    arg_type = str
    default_value = 'dec'
    parser.add_argument('--baseDecCol', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Dec column in --xrayCatalog"
    arg_type = str
    default_value = 'DEC_'
    parser.add_argument('--xrayDecCol', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    # argument_description = "Name of column(s) in --xrayCatalog that contain X-ray object's unique identifiers"
    # arg_type = str
    # default_value = None
    # parser.add_argument('--xrayObjIDColumn', type=arg_type, default=default_value,
    #                     help=_add_default_and_type(argument_description, arg_type, default_value))

    # use case 2 arguments
    argument_description = "Path to SDSS catalog with objects' identifiers and other columns." \
                           " Supports fits and pickled pandas dataframe. If not provided, cross-correlation " \
                           " in 1\" will be performed"
    arg_type = str
    default_value = None
    parser.add_argument('--sdss', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Path to PanSTARRS catalog with objects' identifiers and other columns." \
                           " Supports fits and pickled pandas dataframe. If not provided, cross-correlation " \
                           " in 1\" will be performed"
    arg_type = str
    default_value = None
    parser.add_argument('--ps', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Path to DESI LIS catalog with objects' identifiers and other columns." \
                           " Supports fits and pickled pandas dataframe. If not provided, cross-correlation " \
                           " in 1\" will be performed"
    arg_type = str
    default_value = None
    parser.add_argument('--ls', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Name of column(s) in --sdssCatalog that contain X-ray object's unique identifiers" \
                           "For base catalog you may specify '__workcid__' to make identifiers from 0 to len-1"
    arg_type = str
    default_value = None
    parser.add_argument('--sdssOn', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Name of column(s) in --psCatalog that contain X-ray object's unique identifiers" \
                           "For base catalog you may specify '__workcid__' to make identifiers from 0 to len-1"
    arg_type = str
    default_value = None
    parser.add_argument('--psOn', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Name of column(s) in --lsCatalog that contain X-ray object's unique identifiers" \
                           "For base catalog you may specify '__workcid__' to make identifiers from 0 to len-1"
    arg_type = str
    default_value = None
    parser.add_argument('--lsOn', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    # use case 3 arguments
    argument_description = "Path to assembled dataset. Supports fits and pickled pandas dataframe."
    arg_type = str
    default_value = None
    parser.add_argument('--assembledDataset', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    # predict_only
    argument_description = "Path to dir with features files to make predictions on."
    arg_type = str
    default_value = None
    parser.add_argument('--predictOn', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    # other arguments
    argument_description = "Path to output directory."
    arg_type = str
    default_value = None
    parser.add_argument('-o', '--outputDir', type=arg_type,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Radius (\" seconds) to cross-match with base catalog"
    arg_type = float
    default_value = 30.0
    parser.add_argument('--primaryRadius', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Radius (\" seconds) to cross-match base catalog with other catalogs"
    arg_type = float
    default_value = 1.0
    parser.add_argument('--secondaryRadius', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Do not check existing result files in directory."
    parser.add_argument('--coldStart', action='store_true',
                        help=_add_default_and_type(argument_description,
                                                   isflag=True))

    # argument_description = "Clean buffer directory before termination."
    # parser.add_argument('--cleanupBuffer', action='store_true',
    #                     help=_add_default_and_type(argument_description,
    #                                                isflag=True))

    # argument_description = "Base catalog to correlate X-ray data in 30\" radius with (or if already correlated)." \
    #                        " Possible values are:" \
    #                        "\n - 'default' - search for counterparts in DESI LIS first, if not found then search" \
    #                        " in PanSTARRS (can only be chosen for use case 1)" \
    #                        "\n - 'ps' - use PanSTARRS as base catalog" \
    #                        "\n - 'ls' - use DESI LIS as base catalog" \
    #                        "\n - 'both' - use both PanSTARRS and DESI LIS (can only be chosen for use case 1)" \
    #                        "\n - 'sdss' - use SDSS as base catalog (can only be chosen for use case 2)"

    argument_description = "Models series to apply. Possible values are 'x0', 'x1', 'gal0'" \
                           "'x1a', 'x0pswf', 'gal0pswf'."
    arg_type = str
    default_value = "x0pswf"
    parser.add_argument('--modelsSeries', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Specify list of models' ids to apply. If not specified, then will apply default" \
                           " set of models for specified series. See list of models above."
    arg_type = int
    default_value = None
    parser.add_argument('--modelsIds', type=arg_type, default=default_value,
                        nargs='+',
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "JSON with custom models"
    arg_type = str
    default_value = None
    parser.add_argument('--customModels', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "If used, then wise from Forced Photometry will be used, and otherwise friom DESI LIS"
    parser.add_argument('--useWiseForced', action='store_true',
                        help=_add_default_and_type(argument_description,
                                                   isflag=True))

    argument_description = "If used then all models will be loaded into memory at once" \
                           " in the beginning of predictions step. !!!CAUTION!!! ONE MODEL CONSUMES ABOUT 15 GB" \
                           " OF MEMORY"
    parser.add_argument('--keepModelsInMemory', action='store_true',
                        help=_add_default_and_type(argument_description,
                                                   isflag=True))

    argument_description = "Predictions chunk size. Ignored if --predictOn or --xrayHealpixId is specified"
    arg_type = int
    default_value = 100000
    parser.add_argument('--chunkSize', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Number of jobs for parallelism"
    arg_type = int
    default_value = 1
    parser.add_argument('--njobs', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Path to cross-correlation program"
    arg_type = str
    default_value = './getaroundr.py'
    parser.add_argument('--getaroundrPath', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))
    
    argument_description = "The value of magnitude shift." 
    arg_type = float
    default_value = 0.0
    parser.add_argument('--dm', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))
    #
    # argument_description = "Casjobs user ID (not username!)"
    # arg_type = str
    # default_value = None
    # parser.add_argument('--cjUserID', type=arg_type, default=default_value,
    #                     help=_add_default_and_type(argument_description, arg_type, default_value))
    #
    # argument_description = "Casjobs password"
    # arg_type = str
    # default_value = None
    # parser.add_argument('--cjPassword', type=arg_type, default=default_value,
    #                     help=_add_default_and_type(argument_description, arg_type, default_value))

    argument_description = "If specified, you'll need to download panstarrs fluxes from casjobs using generated" \
                           " csv file with objids"
    parser.add_argument('--psFluxesManually', action='store_true',
                        help=_add_default_and_type(argument_description,
                                                   isflag=True))

    argument_description = "Path to downloaded PanSTARRS fluxes. Do not specify if you did not specified" \
                           " --psFluxesManually or did not download fluxes yet."
    arg_type = str
    default_value = None
    parser.add_argument('--psFluxesPath', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Module to process \"features\", containing " \
                           "function with name definded in " \
                           "--featuresTransformName"
    arg_type = str
    default_value = None
    parser.add_argument('--featuresTransformModule',
                        type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Function to process \"features\" defined in " \
                           "--featuresTransformModule"
    arg_type = str
    default_value = None
    parser.add_argument('--featuresTransformName', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Pan-STARRS catalog to use in getaroundr.py " \
                           "Has to be 'ps2oldfluxradecbest' or 'ps2fluxbest'"
    arg_type = str
    default_value = 'ps2fluxbest'
    parser.add_argument('--psEdition', type=arg_type,
                        default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    return parser.parse_args()

# Args for Nadya
class Catalog:
    def __init__(self, xray_data_path=None, xray_radec_cols=('RA_', 'DEC_'),
                 base_catalog='ps',#'ls',
                 base_radec_cols=('raBest', 'decBest'),#('ra', 'dec'),
                 sdss_path=None, ps_path=None,
                 ls_path=None, sdss_on=None, ps_on=None,
                 ls_on=None, assembled_dataset_path=None, output_dir=None,
                 primary_radius=1.0, secondary_rasius=1,
                 getaroundr_path='./getaroundr.py',
                 njobs=1, cj_user_id=None,
                 cj_password=None, filename='features_gz.pkl',
                 ps_fluxes_manually=False, ps_fluxes=None,
                 user_defined_features_transformation=lambda x: x,
                 panstarrs_catalog_to_use_cause_my_bullshit_code_and_noone_to_download_the_entire_panstarrs_properly_once_and_forall='ps2fluxbest'):#'ps2oldfluxradecbest'):
        """
        Class to assemble catalog
        """

        def define_use_case():
            condition = self.assembled_dataset_path is not None
            if condition:
                return 4

            # condition = self.base_catalog == 'ls' and self.ls_data_path is not None
            # condition |= self.base_catalog == 'ps' and self.ps_data_path is not None
            # condition |= self.base_catalog == 'sdss' and self.sdss_data_path is not None

            condition = self.photo_data_paths['ls'] is not None
            condition &= self.photo_data_paths['ps'] is not None
            condition &= self.photo_data_paths['sdss'] is not None
            if condition:
                return 3

            condition = self.base_catalog == 'ls' and self.photo_data_paths[
                'ls'] is not None
            condition |= self.base_catalog == 'ps' and self.photo_data_paths[
                'ps'] is not None
            condition |= self.base_catalog == 'sdss' and self.photo_data_paths[
                'sdss'] is not None

            if condition:
                return 2

            condition = self.xray_data_path is not None
            if condition:
                return 1

            raise Exception("Unknown use case. Use pzphlib --help")

        self.xray_data_path = xray_data_path
        self.xray_catalog_radec_columns = xray_radec_cols

        self.base_catalog = base_catalog
        self.base_catalog_radec_columns = base_radec_cols
        print(self.base_catalog_radec_columns)

        # self.sdss_data_path = args.sdss
        # self.ps_data_path = args.ps
        # self.ls_data_path = args.ls
        self.filename = filename
        self.photo_data_paths = {
            'sdss': sdss_path,
            'ps': ps_path,
            'ls': ls_path,
            'gaiaedr3': None,
        }
        self.photo_on = {
            'sdss': sdss_on,
            'ps': ps_on,
            'ls': ls_on,
            'gaiaedr3': None,
        }

        self.assembled_dataset_path = assembled_dataset_path

        self.primary_radius = primary_radius
        self.secondary_radius = secondary_rasius

        self.output_dir = output_dir
        self.njobs = njobs

        # Other attributes
        self.use_case = define_use_case()
        self.work_xid_name = '__workxid__'
        self.work_cid_name = '__workcid__'

        self.ps_fluxes_manually = ps_fluxes_manually
        self.ps_fluxes = ps_fluxes
        self.ps_objids = None

        self.getaroundr_path = getaroundr_path
        self.getaroundr_suffix = '_input'
        self.catalogs_getaroundr_names = {
            'sdss': 'sdss_second',
            'ps': panstarrs_catalog_to_use_cause_my_bullshit_code_and_noone_to_download_the_entire_panstarrs_properly_once_and_forall,
            'ls': 'decals8tr',
            'gaiaedr3': 'gaiaedr3',
        }

        self.cj_user_id = cj_user_id
        self.cj_password = cj_password
        self.ps_casjobs_path = None

        self.xray_on = dict()

        self.buf_path = os.path.join(self.output_dir, '__buf__')
        os.makedirs(self.buf_path, exist_ok=True)
        self.xray_catalog_coords_path = os.path.join(self.buf_path,
                                                     'cross-match_x-ray_coords.fits')
        self.base_catalog_coords_path = os.path.join(self.buf_path,
                                                     'cross-match_base_catalog_coords.fits')

        # self.photo_catalog_data_paths = {
        #     'decals8tr': 'ls_fluxes.fits',
        #     'ps2': 'ps_fluxes.fits',
        #     'sdss_second': 'sdss_fluxes.fits',
        # }
        # self.photo_catalog_data_paths = {
        #     cat: os.path.join(self.buf_path, file) for cat, file in self.photo_catalog_data_paths.items()}

        self.sdss_additional_preparation_flag = False

        self.data = None
        self.assembled_dataset = None
        self.user_definded_features_transformation = user_defined_features_transformation

        print(format_message(f'Defined use case: {self.use_case}'))

    def getaroundr(self, input, output, ira, idec, radius, catalog,
                   stdout_fname='getaroundr_stdout',
                   stderr_fname='getaroundr_stderr'):
        """
        Method to apply cross-match program specified during initialisation of object
        :param input: path to input fits file. Must contain columns specified in ira and idec parameters
        :param output: path to output fits file
        :param ira: ra column in input
        :param idec: dec column in input
        :param radius: radius to perform cross-match in.
        :param catalog: catalog to perform cross-match with. See cross-match program help.
        :param stdout_path: file to write stdout.
        :param stderr_path: file to write stderr.
        """
        exist_msg = 'file {} does not exist'
        assert os.path.isfile(input), exist_msg.format(input)

        fits_msg = 'file {} must be fits'
        assert os.path.splitext(input)[1] == '.fits', fits_msg.format(input)
        assert os.path.splitext(output)[1] == '.fits', fits_msg.format(output)

        timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        stdout_path = os.path.join(self.buf_path,
                                   f'{stdout_fname}.{timestamp}.txt')
        stderr_path = os.path.join(self.buf_path,
                                   f'{stderr_fname}.{timestamp}.txt')

        print(format_message('====== CORRELATE ======'))
        command = f"{self.getaroundr_path} -i {input} -r {radius} -cat {catalog} -o {output} -asfx _input" \
                  f" -iRA {ira} -iDEC {idec} -iSEPNAME sep:sep1 -full"

        print(command)
        with open(stdout_path, 'w') as stdout, open(stderr_path,
                                                    'w') as stderr:
            try:
                subprocess.run(command, stdout=stdout, stderr=stderr,
                               shell=True, check=True)
            except subprocess.CalledProcessError as cpe:
                error_msg_prolog = f"Something went wrong during cross-match step:"
                error_msg_epilog = f"See {stdout_path} and {stderr_path} for more info"
                print(error_msg_prolog, cpe, error_msg_epilog)
                raise Exception("Cross-match failed.")

    def prepare_catalog_from_xray(self):
        """
        Step one in data preparation. Assume you have a X-ray objects list and base catalog is known. Then this function
        will produce dataset of X-ray objects and theirs counterparts in 30 seconds radius
        """
        # print(format_message(f'Correlate X-ray data with base catalog ({self.base_catalog})'))
        dst_path = os.path.join(self.buf_path,
                                f'{self.filename}.{self.base_catalog}_fluxes.fits')
        self.photo_data_paths[self.base_catalog] = dst_path

        self.xray_on['xray'] = self.work_xid_name
        self.xray_on[
            self.base_catalog] = self.work_xid_name + self.getaroundr_suffix
        self.photo_on[self.base_catalog] = self.work_cid_name

        if not file_exists(dst_path):
            xray_catalog = Catalog.read_table(self.xray_data_path)
            xray_ra_col, xray_dec_col = self.xray_catalog_radec_columns
            if self.work_xid_name not in xray_catalog.columns:
                xray_catalog[self.work_xid_name] = np.arange(0,
                                                             len(xray_catalog))

            xray_catalog = astropy.table.Table.from_pandas(
                xray_catalog[[self.work_xid_name, xray_ra_col, xray_dec_col]]
            )

            mask = True
            for col in [xray_ra_col, xray_dec_col]:
                print(xray_catalog.dtype)
                mask &= ~np.isnan(xray_catalog[col])

            xray_catalog = xray_catalog[mask]

            xray_catalog.write(self.xray_catalog_coords_path, overwrite=True)

            stdout_fname = os.path.basename(
                os.path.splitext(dst_path)[0]) + '.stdout'
            stderr_fname = os.path.basename(
                os.path.splitext(dst_path)[0]) + '.stderr'

            self.getaroundr(self.xray_catalog_coords_path, dst_path,
                            xray_ra_col, xray_dec_col, self.primary_radius,
                            self.catalogs_getaroundr_names[self.base_catalog],
                            stdout_fname=stdout_fname,
                            stderr_fname=stderr_fname)

            if self.base_catalog == 'sdss':
                Catalog._prepare_sdss(dst_path)

            if self.base_catalog == 'ps':
                Catalog._prepare_panstarrs(dst_path)
                if self.ps_fluxes_manually:
                    self.ps_objids = astropy.table.Table.read(dst_path)[
                        ['objID']]

        if not len(astropy.table.Table.read(dst_path)):
            raise Exception("Found nothing in base catalog")
            # ps_fluxes = panstarrs_casjobs(objids, self.cj_user_id, self.cj_password)
            # fname, ext = os.path.splitext(dst_path)
            # self.ps_casjobs_path = f'{fname}_casjobs{ext}'
            # ps_fluxes.write(self.ps_casjobs_path, overwrite=True)

        # print(format_message('Done'))

    def prepare_catalog_from_base_data(self):
        """
        Step two in data preparation. Assume you have base catalog data and, maybe, some other catalog data. Then you
        need to gain data from remaining catalogs needed and join dataset.
        """
        # catalogs = list(self.base_catalogs_technical_names.values())

        # sdss_path = self.photo_catalog_data_paths['sdss_second']
        # if sdss_path and os.path.isfile(sdss_path):
        #     catalogs.remove('sdss_second')
        #
        # ps_path = self.photo_catalog_data_paths['ps2']
        # if ps_path and os.path.isfile(ps_path):
        #     catalogs.remove('ps2')
        #
        # ls_path = self.photo_catalog_data_paths['decals8tr']
        # if ls_path and os.path.isfile(ls_path):
        #     catalogs.remove('decals8tr')

        # if self.base_catalog in catalogs:
        #     catalogs.remove(self.base_catalog)
        ra_col, dec_col = self.base_catalog_radec_columns
        need2correlate = False
        for cat, path in self.photo_data_paths.items():
            if path is None:
                dst_path = os.path.join(self.buf_path,
                                        f'{self.filename}.{cat}_fluxes.fits')
                if not file_exists(dst_path):
                    need2correlate = True
                    break

        if need2correlate:
            data = Catalog.read_table(self.photo_data_paths[self.base_catalog])
            if self.photo_on[self.base_catalog] == self.work_cid_name:
                data[self.work_cid_name] = np.arange(len(data))

            # try:
            #     data2print = data[
            #         [self.work_cid_name, 'raBest', 'decBest']]
            #     print(data2print)
            # except:
            #     pass

            coords = astropy.table.Table.from_pandas(
                data[[self.photo_on[self.base_catalog], ra_col, dec_col]])
            mask = True
            for col in [ra_col, dec_col]:
                mask &= ~np.isnan(coords[col])

            coords = coords[mask]
            coords.write(self.base_catalog_coords_path, overwrite=True)

        for cat, path in self.photo_data_paths.items():
            if path is None:
                print(format_message(f'Correlate with {cat}'))
                self.photo_data_paths[cat] = dst_path = os.path.join(
                    self.buf_path,
                    f'{self.filename}.{cat}_fluxes.fits')

                stdout_fname = os.path.basename(
                    os.path.splitext(dst_path)[0]) + '.stdout'
                stderr_fname = os.path.basename(
                    os.path.splitext(dst_path)[0]) + '.stderr'
                self.photo_on[cat] = self.photo_on[
                                         self.base_catalog] + self.getaroundr_suffix
                if not file_exists(dst_path):
                    self.getaroundr(self.base_catalog_coords_path, dst_path,
                                    ra_col, dec_col, self.secondary_radius,
                                    self.catalogs_getaroundr_names[cat],
                                    stdout_fname=stdout_fname,
                                    stderr_fname=stderr_fname)
                    if cat == 'sdss':
                        Catalog._prepare_sdss(self.photo_data_paths['sdss'])

                    if cat == 'ps':
                        Catalog._prepare_panstarrs(self.photo_data_paths['ps'])
                        if self.ps_fluxes_manually:
                            self.ps_objids = \
                            astropy.table.Table.read(dst_path)[['objID']]
                        # ps_fluxes = panstarrs_casjobs(objids, self.cj_user_id, self.cj_password)
                        # fname, ext = os.path.splitext(dst_path)
                        # self.ps_casjobs_path = f'{fname}_casjobs{ext}'
                        # ps_fluxes.write(self.ps_casjobs_path, overwrite=True)

                print(format_message('Done'))

    def prepare_catalog_from_correlated_data(self):
        """
        Assume you have all catalogs data and you know how to merge it. Assemble the dataset.
        """
        # print(format_message('Merge catalogs data'))
        data = {
            catalog: Catalog.read_table(path)
            for catalog, path in self.photo_data_paths.items()
        }
        # data['ps'] = data['ps'][data['ps']['primaryDetection'] == 1]
        # data['ps'] = Catalog.panstarrs_primary_detection(data['ps'])
        data['sdss'] = data['sdss'][data['sdss']['MODE'] == 1]

        if self.ps_fluxes_manually:
            if self.ps_fluxes is not None:
                print('merging panstarrs')
                left = data['ps']
                right = self.ps_fluxes[
                    os.path.basename(self.photo_data_paths['ps'])]
                drop_cols = _columns_intersection(left, right)
                if drop_cols:
                    for elem in ['objID', 'objid']:
                        if elem in drop_cols:
                            drop_cols.remove(elem)

                    left = left.drop(columns=drop_cols)

                data['ps'] = pd.merge(left, right, how='left', left_on='objID',
                                      right_on='objid')
            else:
                raise Exception(
                    f"You need to load PanSTARRS fluxes from casjobs.")

        data['ps'] = data['ps'].astype({'objID': str})

        merged = list()
        additional_columns_to_drop = ['ix_input']

        if self.work_cid_name not in data[self.base_catalog].columns:
            data[self.base_catalog][self.work_cid_name] = np.arange(
                len(data[self.base_catalog]))

            # try:
            #     data2print = data[self.base_catalog][[self.work_cid_name, 'raBest', 'decBest']]
            #     print(data2print)
            # except:
            #     pass

        if self.xray_data_path is not None and self.use_case == 1:
            # TODO only fits for DESI LIS. It is a stopgap
            # print('111111111111111111', self.base_catalog)
            if self.base_catalog == "ls":
                fluxes_columns = ['flux_g', 'flux_r', 'flux_z',
                                  'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4']
                errors_columns = ['flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                                  'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3',
                                  'flux_ivar_w4']
                errors_types = ['ivar', 'ivar', 'ivar', 'ivar', 'ivar', 'ivar',
                                'ivar']
                data[self.base_catalog] = clean_duplicates(
                    data[self.base_catalog], self.xray_on[self.base_catalog],
                    'ra', 'dec', fluxes_columns, errors_columns, errors_types,
                    self.secondary_radius, self.njobs)
            elif self.base_catalog == "ps":
                psdr2_flux_columns = [
                    'gKronFluxErr',
                    'gKronFlux', 'rKronFluxErr', 'rKronFlux',
                    'iKronFluxErr', 'iKronFlux',
                    'zKronFluxErr', 'zKronFlux',
                    'yKronFluxErr', 'yKronFlux',
                    'gPSFFluxErr', 'gPSFFlux', 'rPSFFluxErr',
                    'rPSFFlux', 'iPSFFluxErr', 'iPSFFlux',
                    'zPSFFluxErr', 'zPSFFlux', 'yPSFFluxErr',
                    'yPSFFlux'
                ]
                fluxes_columns = psdr2_flux_columns[1::2]
                errors_columns = psdr2_flux_columns[::2]
                errors_types = ["err"] * len(fluxes_columns)

                fluxes_columns += ["w1flux", "w2flux"]
                errors_columns += ["dw1flux", "dw2flux"]
                errors_types += ["err"] * 2

                print("===== DEBUG =====")
                print(f"= {self.filename}")
                print("===== END DEBUG =====")

                data[self.base_catalog] = clean_duplicates(
                    data[self.base_catalog], self.xray_on[self.base_catalog],
                    'raBest', 'decBest',
                    fluxes_columns, errors_columns, errors_types,
                    self.secondary_radius, self.njobs)
            # TODO end

            xray_coords_duplicated_columns = [col + self.getaroundr_suffix for
                                              col in
                                              self.xray_catalog_radec_columns]
            xray_coords_duplicated_columns += additional_columns_to_drop
            data[self.base_catalog] = data[self.base_catalog].drop(
                columns=xray_coords_duplicated_columns)
            self.xray_on[
                self.base_catalog] = f'{self.base_catalog}_{self.xray_on[self.base_catalog]}'

            data[self.base_catalog] = data[self.base_catalog].rename(
                columns=lambda x: f'{self.base_catalog}_{x}')
            self.photo_on[
                self.base_catalog] = f'{self.base_catalog}_{self.photo_on[self.base_catalog]}'

            buf = Catalog.read_table(self.xray_data_path)
            if self.work_xid_name not in buf.columns:
                buf[self.work_xid_name] = np.arange(len(buf))

            data['xray'] = buf

            assembled_dataset = data['xray'].copy()
            merged.append('xray')
            assembled_dataset = pd.merge(left=assembled_dataset,
                                         left_on=self.xray_on['xray'],
                                         right=data[self.base_catalog],
                                         right_on=self.xray_on[
                                             self.base_catalog],
                                         how='left')
            merged.append(self.base_catalog)
            assembled_dataset = assembled_dataset.drop(
                columns=self.xray_on[self.base_catalog])

        else:
            data[self.base_catalog] = data[self.base_catalog].rename(
                columns=lambda x: f'{self.base_catalog}_{x}')
            self.photo_on[
                self.base_catalog] = f'{self.base_catalog}_{self.photo_on[self.base_catalog]}'

            assembled_dataset = data[self.base_catalog]
            merged.append(self.base_catalog)

        print(self.photo_on)
        for cat, values in data.items():
            if cat not in merged:
                if self.use_case <= 2:
                    base_coords_duplicated_columns = [
                        col + self.getaroundr_suffix for col in
                        self.base_catalog_radec_columns]
                    base_coords_duplicated_columns += additional_columns_to_drop
                    values = values.drop(
                        columns=base_coords_duplicated_columns,
                        errors='ignore')

                if len(values):
                    values = Catalog._process_counterparts(values, cat,
                                                           self.photo_on[cat],
                                                           self.njobs)
                else:
                    values['__workcid__' + self.getaroundr_suffix] = []

                values = values.rename(columns=lambda x: f'{cat}_{x}')
                left_on = self.photo_on[self.base_catalog]
                right_on = self.photo_on[cat] = f'{cat}_{self.photo_on[cat]}'

                assembled_dataset = pd.merge(left=assembled_dataset,
                                             left_on=left_on,
                                             right=values, right_on=right_on,
                                             how='left')
                if right_on != left_on:
                    assembled_dataset = assembled_dataset.drop(
                        columns=right_on)

        self.assembled_dataset = assembled_dataset.reset_index(drop=True)
        self.assembled_dataset_path = os.path.join(self.output_dir,
                                                   'dataset.gz_pkl')
        assembled_dataset.to_pickle(self.assembled_dataset_path,
                                    compression='gzip', protocol=4)
        del self.data

        # print(format_message('Done'))

    def prepare_features(self):
        if self.assembled_dataset is None:
            self.assembled_dataset = pd.read_pickle(
                self.assembled_dataset_path, compression='gzip')

        # print(list(self.assembled_dataset.columns))
        wise_forced = True if 'PZPH1_FORCE_WISE_FORCED' in os.environ.keys() else False
        if wise_forced:
            print(format_message('PZPH1_FORCE_WISE_FORCED env var has been set'))

        self.assembled_dataset = calculate_features_on_full_catalog(
            self.assembled_dataset, ebv_accounting=False, wise_forced=wise_forced,
            user_defined_features_transformation=self.user_definded_features_transformation)

        # Raises type error if dataset contains predictions pdfs
        # self.assembled_dataset = self.assembled_dataset.replace(
        #     [-np.inf, np.inf], np.nan)
        for col in self.assembled_dataset.columns:
            if not re.findall("^zoo_.*_z_pdf$", col):
                self.assembled_dataset[col] = self.assembled_dataset[
                    col].replace([-np.inf, np.inf], np.nan)

#         flags_result = place_objid_flags(self.assembled_dataset['sdss_objID'])

#         for flag_name, flag_values in flags_result.items():
#             self.assembled_dataset[flag_name] = flag_values

        ra_col, dec_col = self.base_catalog_radec_columns
        ra_col = f'{self.base_catalog}_{ra_col}'
        dec_col = f'{self.base_catalog}_{dec_col}'
#         for flag_name, flag_values in place_coords_flags(
#                 self.assembled_dataset, ra_col, dec_col).items():
#             self.assembled_dataset[flag_name] = flag_values

        if not (self.assembled_dataset_path is None):
            os.remove(self.assembled_dataset_path)
        self.assembled_dataset_path = os.path.join(self.output_dir,
                                                   f'{self.filename}.features.gz_pkl')
        self.assembled_dataset.to_pickle(self.assembled_dataset_path,
                                         compression='gzip', protocol=4)

    def prepare_data(self, augmentation=False,
                     dms=[-1, 1, 2]):
        print('Use case', self.use_case)
        if self.use_case <= 0 or self.use_case is None:
            print('Use case 0 or None')
            return None

        if self.use_case <= 1:
            print('Use case prepare_catalog_from_xray')
            self.prepare_catalog_from_xray()

        if self.use_case <= 2:
            print('Use case prepare_catalog_from_base_data')
            self.prepare_catalog_from_base_data()

        if self.ps_fluxes_manually and self.ps_fluxes is None:
            print('Use case self.ps_fluxes_manually and self.ps_fluxes is None')
            if self.ps_objids is None:
                self.ps_objids = \
                astropy.table.Table.read(self.photo_data_paths['ps'])[
                    ['objID']]

            self.ps_objids = self.ps_objids.to_pandas()
            self.ps_objids['__file__'] = os.path.basename(
                self.photo_data_paths['ps'])
            return "ps_manual"

        if self.use_case <= 3:
            print('Use case prepare_catalog_from_correlated_data')
            self.prepare_catalog_from_correlated_data()

        if self.use_case <= 4:
            if augmentation:
                
                if isinstance(augmentation, str):
                    if os.path.exists(augmentation):
                        augmodel_path = augmentation
                        
                model = Augmentation.read(augmodel_path)
                model.debug=False
                filename = self.filename
                path = self.assembled_dataset_path
                
                # dm == 0
                import shutil
                shutil.copyfile(self.assembled_dataset_path,
                                os.path.join(self.output_dir,
                                             f'{filename}.augmentation_0.gz_pkl')
                               )
                self.filename = filename + '_0'
                self.prepare_features()
                
                # dm != 0
                for dm in dms:
                    if dm == 0:
                        continue
                    print(f'Use case augmentation dm == {dm}')
                    self.assembled_dataset = model.predict(os.path.join(self.output_dir, 
                                                                        f'{filename}.augmentation_0.gz_pkl'),
                                                           dm) #TODO
                    self.assembled_dataset.to_pickle(
                                         os.path.join(self.output_dir,f'{filename}.augmentation_{dm}.gz_pkl'),
                                         compression='gzip', protocol=4)
                    print(f'Use case prepare_features dm == {dm}')
                    self.filename = filename + f'_{dm}'
                    self.assembled_dataset_path = None
                    self.prepare_features()
                    
                return "Done"
            
            print('Use case prepare_features')
            self.prepare_features()

        return "Done"

    @staticmethod
    def _drop_multidims(table: astropy.table.Table):
        """
        drop multidimentional columns from astropy.Table so it can be converted to pandas.DataFrame
        """
        singledim_cols = list()
        multidim_cols = list()
        for col in table.colnames:
            if len(table[col].shape) == 1:
                singledim_cols.append(col)
            else:
                multidim_cols.append(col)

        return table[singledim_cols], multidim_cols

    @staticmethod
    def panstarrs_primary_detection(data: pd.DataFrame, group_columns=None,
                                    primary_column='primaryDetection'):
        if not len(data):
            return data

        if group_columns is None:
            group_columns = ['raBest', 'decBest']

        dst = []
        for key, group in data.groupby(by=group_columns):
            mask = group[primary_column] > 0
            if mask.any():
                dst.append(group.loc[mask])

            else:
                dst.append(group)

        return pd.concat(dst, sort=False)

    @staticmethod
    def read_table(table):
        if isinstance(table, str):
            _, ext = os.path.splitext(table)
            if ext == '.gz_pkl':
                try:
                    return pd.read_pickle(table, compression='gzip')
                except:
                    return pd.read_pickle(table)

            if ext == '.pkl':
                return pd.read_pickle(table)

            if ext == '.fits':
                table = astropy.table.Table.read(table)

        if isinstance(table, pd.DataFrame):
            return table

        if isinstance(table, astropy.table.Table):
            table, dropped_cols = Catalog._drop_multidims(table)
            if dropped_cols:
                warnings.warn(
                    "multidimentional columns are dropped from table : {}".format(
                        dropped_cols))

            return table.to_pandas()

        raise Exception('Unsupported format of table')

    @staticmethod
    def _prepare_sdss(path):
        data = astropy.table.Table.read(path)
        data.rename_column('RA', 'ra')
        data.rename_column('DEC', 'dec')
        data.rename_column('OBJID', 'objID')
        for i, pb in enumerate(['u', 'g', 'r', 'i', 'z']):
            data[f'psfFlux_{pb}'] = data['PSFFLUX'][:, i]
            data[f'psfFluxIvar_{pb}'] = data['PSFFLUX_IVAR'][:, i]
            data[f'cModelFlux_{pb}'] = data['CMODELFLUX'][:, i]
            data[f'cModelFluxIvar_{pb}'] = data['CMODELFLUX_IVAR'][:, i]

        data.write(path, overwrite=True)

    @staticmethod
    def _prepare_panstarrs(path):
        data = astropy.table.Table.read(path)
        data = astropy.table.Table.from_pandas(
            Catalog.panstarrs_primary_detection(data.to_pandas()))
        for col in ['raBest', 'decBest']:
            data[col][np.where(data[col] == -999.0)] = np.nan

        psdr2_flux_columns = [
            'gKronFluxErr',
            'gKronFlux', 'rKronFluxErr', 'rKronFlux',
            'iKronFluxErr', 'iKronFlux',
            'zKronFluxErr', 'zKronFlux',
            'yKronFluxErr', 'yKronFlux',
            'gPSFFluxErr', 'gPSFFlux', 'rPSFFluxErr',
            'rPSFFlux', 'iPSFFluxErr', 'iPSFFlux',
            'zPSFFluxErr', 'zPSFFlux', 'yPSFFluxErr',
            'yPSFFlux'
        ]

        for col in psdr2_flux_columns:
            mask = data[col] == -999.0
            data[col][mask] = np.nan

        for col in ['dw1flux', 'dw2flux']:
            mask = data[col] == -1.0
            data[col][mask] = np.nan

        data.write(path, overwrite=True)

    @staticmethod
    def _default2nan(data: pd.DataFrame):
        default_values = {
            # TODO
        }
        for col, val in default_values.items():
            if col in data.columns:
                data = data.reaplace(val, np.nan)

        return data

    @staticmethod
    def _process_counterparts(data: pd.DataFrame, cat: str, objdef: str,
                              njobs: int):
        if cat == 'sdss':
            conf = {
                "script_version": 0.1,
                "comments": [
                    "for sdss data"
                ],
                "input": "-",
                "output": "-",
                "interested_columns": [
                    [
                        "cModelFlux_(\\w)",
                        "cModelFluxIvar_(\\w)",
                        -999,
                        -999,
                        "flux",
                        'i'
                    ],
                    [
                        "psfFlux_(\\w)",
                        "psfFluxIvar_(\\w)",
                        -999,
                        -999,
                        "flux",
                        'i'
                    ]
                ],
                "object_definition": [
                    objdef
                ],
                "counterpart_definition": [
                    "ra",
                    "dec"
                ]
            }
        elif cat == 'ps':
            conf = {
                "script_version": 0.1,
                "comments": [
                    "for panstarrs data"
                ],
                "input": "-",
                "output": "-",
                "interested_columns": [
                    [
                        "(\\w)KronFlux",
                        "(\\w)KronFluxErr",
                        -999,
                        -999,
                        "flux"
                    ],
                    [
                        "(\\w)PSFFlux",
                        "(\\w)PSFFluxErr",
                        -999,
                        -999,
                        "flux"
                    ],
                    [
                        "w(\\w)flux",
                        "dw(\\w)flux",
                        -1.0,
                        0.0,
                        "flux"
                    ]
                ],
                "object_definition": [
                    objdef
                ],
                "counterpart_definition": [
                    "raBest",  # TODO radec best
                    "decBest"
                ]
            }
        elif cat == 'ls':
            conf = {
                "script_version": 0.1,
                "comments": [
                    "for desi lis data"
                ],
                "input": "-",
                "output": "-",
                "interested_columns": [
                    [
                        "flux_(\\w)",
                        "flux_ivar_(\\w)",
                        -999,
                        -999,
                        "flux",
                        'i'
                    ],
                    [
                        "flux_w(\\w)",
                        "flux_ivar_w(\\w)",
                        -999,
                        -999,
                        "flux",
                        'i'
                    ],
                ],
                "object_definition": [
                    objdef
                ],
                "counterpart_definition": [
                    "ra",
                    "dec"
                ]
            }
        elif cat == 'gaiaedr3':
            conf = {
                "script_version": 0.1,
                "comments": [
                    "for desi lis data"
                ],
                "input": "-",
                "output": "-",
                "interested_columns": [
                    [
                        "phot_(\\w)_mean_flux",
                        "phot_(\\w)_mean_flux_error",
                        -999,
                        -999,
                        "flux",
                    ],
                ],
                "object_definition": [
                    objdef
                ],
                "counterpart_definition": [
                    "ra",
                    "dec"
                ]
            }
        else:
            raise Exception("wrong catalog")

        return process_counterparts(conf, data, njobs)


def asinhmag_dm(flux, flux_err=None, flux_ivar=None, dm=0):
    """
    Calculate asinh mognitude with dm shift.
    ::flux      - flux in [nanomaggies]
    ::flux_ivar - inverse variance of flux in [1/nanomaggies**2]
    ::flux_err  - flux error in [nanomaggies]
    ::dm        - magnitude shift
    """
    assert (flux_err is not None) ^ (
                flux_ivar is not None), 'specify only flux_err or flux_ivar'
    f = flux / 1e9 * np.power(10, 0.4 * dm)
    if flux_ivar is not None:
        b = np.power(flux_ivar, -0.5) / 1e9 * np.power(10, 0.4 * dm)
    else:
        b = flux_err / 1e9 * np.power(10, 0.4 * dm)

    f, b = f.astype(np.float64), b.astype(
        np.float64)  # otherwise type error like
    # TypeError: loop of ufunc does not support argument 0 of type numpy.float64 which has no callable arcsinh method
    return (np.arcsinh(f / (2 * b)) + np.log(b)) * (-2.5 / np.log(10))


def flux2mag(flux, err, columns, dms=None):
    """
    ::flux - pandas.DataFrame with fluxes
    ::err - pandas.DataFrame with corresponding errors. Columns must be in the same order
        (e.g. flux_r, flux_u, flux_w2 => err_r, err_u, ivar_w2)
        if "ivar" or "Ivar" is substring of error column name, then numbers in this columns are interperted
        as inverse variance of flux in [1/nanomaggies**2], else as flux error in [nanomaggies]
    ::columns - dict where keys are columns names from flux param and values are result columns names
        (e.g. {"flux_z":"mag_z"})
    """
    result = pd.DataFrame()
    if dms is None:
        dms = np.zeros(len(columns))

    for f, i, dm in zip(flux.columns, err.columns, dms):
        if dm is None:
            dm = 0
        if 'ivar' in i or 'Ivar' in i:
            result[columns[f]] = asinhmag_dm(flux[f], flux_ivar=err[i], dm=dm)
        elif re.findall(r'^ps_dw\dflux_ab$', i):
            result[columns[f]] = asinhmag_dm(
                flux[f].replace(-999, np.NaN),
                flux_err=err[i].replace(-999, np.NaN),
                dm=dm
            )
        else:
            result[columns[f]] = asinhmag_dm(
                flux[f].replace(-999, np.NaN) / 3621e-9,
                flux_err=err[i].replace(-999, np.NaN) / 3621e-9,
                dm=dm
            )
    result.index = flux.index
    return result


def missing_kron_to_psf(data, mags_fmt='psdr2_{pb}_{typ}', passbands='grizy',
                        psf='psf', kron='kron'):
    """
    for panstarrs data only: replace nans in kron mags with psf mags
    """
    for pb in passbands:
        k = data[[mags_fmt.format(pb=pb, typ=kron)]].values
        p = data[[mags_fmt.format(pb=pb, typ=psf)]].values
        mask = np.isnan(k)
        k[mask] = p[mask]
        data[mags_fmt.format(pb=pb, typ=kron)] = k

    return data


def calculate_standard_features(df, passbands='ugriz',
                                mags_fmt='sdssdr16_{pb}_{typ}',
                                mag_types=('psf', 'cmodel'),
                                prefix='sdssdr16_'):
    for i, pb in enumerate(passbands):
        for pb2 in passbands[i + 1:]:
            df['{}{}-{}_{}'.format(prefix, pb, pb2, mag_types[0])] = \
                df[mags_fmt.format(pb=pb, typ=mag_types[0])] - df[
                    mags_fmt.format(pb=pb2, typ=mag_types[0])]

        df['{}{}_{}-{}'.format(prefix, pb, mag_types[0], mag_types[1])] = \
            df[mags_fmt.format(pb=pb, typ=mag_types[0])] - df[
                mags_fmt.format(pb=pb, typ=mag_types[1])]

    return df


def calculate_decals8tr_features(df):
    passbands = 'grz'
    wise_types = ['Lw1', 'Lw2']
    for i, pb in enumerate(passbands):

        for pb2 in passbands[i + 1:]:
            df['decals8tr_{}-{}'.format(pb, pb2)] = df[f'decals8tr_{pb}'] - df[
                f'decals8tr_{pb2}']

    for ds, typ in {'sdssdr16': 'cmodel', 'psdr2': 'kron'}.items():
        for i, pb in enumerate(passbands):
            df[f'{ds}_{pb}_{typ}-decals8tr_{pb}'] = df[f'{ds}_{pb}_{typ}'] - \
                                                    df[f'decals8tr_{pb}']

    for ds, (pbs, typ) in {'sdssdr16': ('ugriz', 'cmodel'),
                           'psdr2': ('grizy', 'kron')}.items():
        for pb in pbs:
            for wtyp in wise_types:
                df[f'{ds}_{pb}_{typ}-decals8tr_{wtyp}'] = \
                    df[f'{ds}_{pb}_{typ}'] - df[f'decals8tr_{wtyp}']

    for pb in 'grz':
        for wtyp in wise_types:
            df[f'decals8tr_{pb}-{wtyp}'] = \
                df[f'decals8tr_{pb}'] - df[f'decals8tr_{wtyp}']

    df['decals8tr_{}-{}'.format(wise_types[0], wise_types[1])] = \
        df[f'decals8tr_{wise_types[0]}'] - df[f'decals8tr_{wise_types[1]}']

    return df


def _calculate_features_on_full_catalog_helper(catalog: pd.DataFrame,
                                               wise_forced=False):
    sdss_flux_columns = [
        'psfFlux_u', 'psfFluxIvar_u', 'psfFlux_g',
        'psfFluxIvar_g', 'psfFlux_r', 'psfFluxIvar_r', 'psfFlux_i',
        'psfFluxIvar_i', 'psfFlux_z', 'psfFluxIvar_z', 'cModelFlux_u',
        'cModelFluxIvar_u', 'cModelFlux_g', 'cModelFluxIvar_g', 'cModelFlux_r',
        'cModelFluxIvar_r', 'cModelFlux_i', 'cModelFluxIvar_i', 'cModelFlux_z',
        'cModelFluxIvar_z'
    ]
    sdss_flux_columns = ['sdss_' + col for col in sdss_flux_columns]
    sdss_flux_columns, sdss_error_columns = sdss_flux_columns[
                                            ::2], sdss_flux_columns[1::2]
    sdss_mag_columns = {
        **{f'sdss_psfFlux_{pb}': f'sdssdr16_{pb}_psf' for pb in 'ugriz'},
        **{f'sdss_cModelFlux_{pb}': f'sdssdr16_{pb}_cmodel' for pb in 'ugriz'},
    }

    fluxes, errors = catalog[sdss_flux_columns], catalog[sdss_error_columns]
#     errors.where(errors < 0, np.nan, inplace=True)
    mags = flux2mag(fluxes, errors, sdss_mag_columns)
    catalog[mags.columns] = mags

    psdr2_flux_columns = [
        'gKronFluxErr',
        'gKronFlux', 'rKronFluxErr', 'rKronFlux',
        'iKronFluxErr', 'iKronFlux',
        'zKronFluxErr', 'zKronFlux',
        'yKronFluxErr', 'yKronFlux',
        'gPSFFluxErr', 'gPSFFlux', 'rPSFFluxErr',
        'rPSFFlux', 'iPSFFluxErr', 'iPSFFlux',
        'zPSFFluxErr', 'zPSFFlux', 'yPSFFluxErr',
        'yPSFFlux'
    ]
    psdr2_flux_columns = ['ps_' + col for col in psdr2_flux_columns]
    psdr2_flux_columns, psdr2_error_columns = psdr2_flux_columns[
                                              1::2], psdr2_flux_columns[::2]
    psdr2_mag_columns = {
        **{f'ps_{pb}PSFFlux': f'psdr2_{pb}_psf' for pb in 'grizy'},
        **{f'ps_{pb}KronFlux': f'psdr2_{pb}_kron' for pb in 'grizy'},
    }

    fluxes, errors = catalog[psdr2_flux_columns], catalog[psdr2_error_columns]
#     errors[errors <= 0] = np.nan
    mags = flux2mag(fluxes, errors, psdr2_mag_columns)
    mags = missing_kron_to_psf(mags)
    catalog[mags.columns] = mags

    flux_cols = [
        'flux_g_ebv', 'flux_r_ebv', 'flux_z_ebv',
        'flux_w1_ebv', 'flux_w2_ebv', 'flux_w3_ebv', 'flux_w4_ebv'
    ]
    flux_cols = ['ls_' + col for col in flux_cols]
    ivar_cols = [
        'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
        'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4'
    ]
    ivar_cols = ['ls_' + col for col in ivar_cols]
    mag_columns = {
        **{f'ls_flux_{pb}_ebv': f'decals8tr_{pb}' for pb in 'grz'},
        **{f'ls_flux_w{pb}_ebv': f'decals8tr_Lw{pb}' for pb in '1234'},
    }

    for pb in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:
        fcol = f'ls_flux_{pb}'
        fcol_new = f'ls_flux_{pb}_ebv'
        catalog[fcol_new] = catalog[
            fcol]  # WITHOUT EBV ACCOUNTING. IT IS JUST A NAIL.

    fluxes, errors = catalog[flux_cols].astype(float), catalog[
        ivar_cols].astype(float)
#     errors[errors <= 0] = np.nan
    mags = flux2mag(fluxes, errors, mag_columns)
    catalog[mags.columns] = mags

    if wise_forced:
        # print(catalog[['ps_w1flux', 'ps_dw1flux']])
        # catalog[['ps_w1flux', 'ps_dw1flux']] *= 10**(-2.699/2.5)
        # catalog[['ps_w2flux', 'ps_dw2flux']] *= 10**(-3.339/2.5)
        # print(catalog[['ps_w1flux', 'ps_dw1flux']])
        #
        # flux_cols = ['ps_w1flux', 'ps_w2flux']
        # err_cols = ['ps_dw1flux', 'ps_dw2flux']
        # mag_columns = {'ps_w1flux': 'decals8tr_Lw1',
        #                'ps_w2flux': 'decals8tr_Lw2'}
        #
        # fluxes, errors = catalog[flux_cols].astype(np.float), catalog[
        #     err_cols].astype(np.float)
        catalog[['ps_w1flux_ab', 'ps_dw1flux_ab']] = catalog[['ps_w1flux',
                                                              'ps_dw1flux']] * 10 ** (
                                                                 -2.699 / 2.5)
        catalog[['ps_w2flux_ab', 'ps_dw2flux_ab']] = catalog[['ps_w2flux',
                                                              'ps_dw2flux']] * 10 ** (
                                                                 -3.339 / 2.5)

        flux_cols = ['ps_w1flux_ab', 'ps_w2flux_ab']
        err_cols = ['ps_dw1flux_ab', 'ps_dw2flux_ab']
        mag_columns = {'ps_w1flux_ab': 'decals8tr_Lw1',
                       'ps_w2flux_ab': 'decals8tr_Lw2'}

        fluxes, errors = catalog[flux_cols].astype(float), catalog[
            err_cols].astype(float)

#         errors[errors <= 0] = np.nan
        mags = flux2mag(fluxes, errors, mag_columns)
        catalog[mags.columns] = mags

    catalog = calculate_standard_features(  # SDSS features
        catalog,
        passbands='ugriz',
        mag_types=['psf', 'cmodel'],
        mags_fmt='sdssdr16_{pb}_{typ}',
        prefix='sdssdr16_'
    )
    catalog = calculate_standard_features(  # PSDR2 features
        catalog,
        passbands='grizy',
        mag_types=['psf', 'kron'],
        mags_fmt='psdr2_{pb}_{typ}',
        prefix='psdr2_'
    )
    catalog = calculate_decals8tr_features(catalog)
    return catalog


def _calculate_features_on_full_catalog_ebv_helper(catalog: pd.DataFrame,
                                                   wise_forced=False):
    sdss_flux_columns = [
        'psfFlux_u', 'psfFluxIvar_u', 'psfFlux_g',
        'psfFluxIvar_g', 'psfFlux_r', 'psfFluxIvar_r', 'psfFlux_i',
        'psfFluxIvar_i', 'psfFlux_z', 'psfFluxIvar_z', 'cModelFlux_u',
        'cModelFluxIvar_u', 'cModelFlux_g', 'cModelFluxIvar_g', 'cModelFlux_r',
        'cModelFluxIvar_r', 'cModelFlux_i', 'cModelFluxIvar_i', 'cModelFlux_z',
        'cModelFluxIvar_z'
    ]
    sdss_flux_columns = ['sdss_' + col for col in sdss_flux_columns]
    sdss_flux_columns, sdss_error_columns = sdss_flux_columns[
                                            ::2], sdss_flux_columns[1::2]
    sdss_mag_columns = {
        **{f'sdss_psfFlux_{pb}': f'sdssdr16_{pb}_psf' for pb in 'ugriz'},
        **{f'sdss_cModelFlux_{pb}': f'sdssdr16_{pb}_cmodel' for pb in 'ugriz'},
    }
    ebv_column = 'ls_ebv'
    dms = [
        4.239 * catalog[ebv_column],
        3.303 * catalog[ebv_column],
        2.285 * catalog[ebv_column],
        1.698 * catalog[ebv_column],
        1.263 * catalog[ebv_column],

        4.239 * catalog[ebv_column],
        3.303 * catalog[ebv_column],
        2.285 * catalog[ebv_column],
        1.698 * catalog[ebv_column],
        1.263 * catalog[ebv_column],
    ]
    fluxes, errors = catalog[sdss_flux_columns], catalog[sdss_error_columns]
#     errors[errors <= 0] = np.nan
    mags = flux2mag(fluxes, errors, sdss_mag_columns, dms)
    catalog[mags.columns] = mags

    psdr2_flux_columns = [
        'gKronFluxErr',
        'gKronFlux', 'rKronFluxErr', 'rKronFlux',
        'iKronFluxErr', 'iKronFlux',
        'zKronFluxErr', 'zKronFlux',
        'yKronFluxErr', 'yKronFlux',
        'gPSFFluxErr', 'gPSFFlux', 'rPSFFluxErr',
        'rPSFFlux', 'iPSFFluxErr', 'iPSFFlux',
        'zPSFFluxErr', 'zPSFFlux', 'yPSFFluxErr',
        'yPSFFlux'
    ]
    psdr2_flux_columns = ['ps_' + col for col in psdr2_flux_columns]
    psdr2_flux_columns, psdr2_error_columns = psdr2_flux_columns[
                                              1::2], psdr2_flux_columns[::2]
    psdr2_mag_columns = {
        **{f'ps_{pb}PSFFlux': f'psdr2_{pb}_psf' for pb in 'grizy'},
        **{f'ps_{pb}KronFlux': f'psdr2_{pb}_kron' for pb in 'grizy'},
    }
    dms = [
        3.172 * catalog[ebv_column],
        2.271 * catalog[ebv_column],
        1.682 * catalog[ebv_column],
        1.322 * catalog[ebv_column],
        1.087 * catalog[ebv_column],

        3.172 * catalog[ebv_column],
        2.271 * catalog[ebv_column],
        1.682 * catalog[ebv_column],
        1.322 * catalog[ebv_column],
        1.087 * catalog[ebv_column],
    ]
    fluxes, errors = catalog[psdr2_flux_columns], catalog[psdr2_error_columns]
#     errors[errors <= 0] = np.nan
    mags = flux2mag(fluxes, errors, psdr2_mag_columns, dms)
    mags = missing_kron_to_psf(mags)
    catalog[mags.columns] = mags

    flux_cols = [
        'flux_g_ebv', 'flux_r_ebv', 'flux_z_ebv',
        'flux_w1_ebv', 'flux_w2_ebv', 'flux_w3_ebv', 'flux_w4_ebv'
    ]
    flux_cols = ['ls_' + col for col in flux_cols]
    ivar_cols = [
        'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
        'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4'
    ]
    ivar_cols = ['ls_' + col for col in ivar_cols]
    mag_columns = {
        **{f'ls_flux_{pb}_ebv': f'decals8tr_{pb}' for pb in 'grz'},
        **{f'ls_flux_w{pb}_ebv': f'decals8tr_Lw{pb}' for pb in '1234'},
    }

    for pb in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:
        fcol = f'ls_flux_{pb}'
        fcol_new = f'ls_flux_{pb}_ebv'
        mwcol = f'ls_mw_transmission_{pb}'
        catalog[fcol_new] = catalog[fcol] / catalog[mwcol]

    fluxes, errors = catalog[flux_cols].astype(float), catalog[
        ivar_cols].astype(float)
#     errors[errors <= 0] = np.nan
    mags = flux2mag(fluxes, errors, mag_columns)
    catalog[mags.columns] = mags

    if wise_forced:
        print("CALCULATING WISE FORCED")
        # catalog[['ps_w1flux', 'ps_dw1flux']] *= 10 ** (-2.699 / 2.5)
        # catalog[['ps_w2flux', 'ps_dw2flux']] *= 10 ** (-3.339 / 2.5)
        #
        # flux_cols = ['ps_w1flux', 'ps_w2flux']
        # err_cols = ['ps_dw1flux', 'ps_dw2flux']
        # mag_columns = {'ps_w1flux': 'decals8tr_Lw1',
        #                'ps_w2flux': 'decals8tr_Lw2'}

        catalog[['ps_w1flux_ab', 'ps_dw1flux_ab']] = catalog[['ps_w1flux',
                                                              'ps_dw1flux']] * 10 ** (
                                                                 -2.699 / 2.5)
        catalog[['ps_w2flux_ab', 'ps_dw2flux_ab']] = catalog[['ps_w2flux',
                                                              'ps_dw2flux']] * 10 ** (
                                                                 -3.339 / 2.5)

        flux_cols = ['ps_w1flux_ab', 'ps_w2flux_ab']
        err_cols = ['ps_dw1flux_ab', 'ps_dw2flux_ab']
        mag_columns = {'ps_w1flux_ab': 'decals8tr_Lw1',
                       'ps_w2flux_ab': 'decals8tr_Lw2'}

        fluxes, errors = catalog[flux_cols].astype(float), catalog[
            err_cols].astype(float)

        dms = [
            0.184 * catalog[ebv_column],
            0.113 * catalog[ebv_column],
        ]
        # print(dms)
        #     err_cols].astype(np.float)
#         errors[errors <= 0] = np.nan
        mags = flux2mag(fluxes, errors, mag_columns, dms)
        catalog[mags.columns] = mags

    print(catalog['decals8tr_Lw1'])

    # print(mags['decals8tr_Lw1'])

    catalog = calculate_standard_features(  # SDSS features
        catalog,
        passbands='ugriz',
        mag_types=['psf', 'cmodel'],
        mags_fmt='sdssdr16_{pb}_{typ}',
        prefix='sdssdr16_'
    )
    catalog = calculate_standard_features(  # PSDR2 features
        catalog,
        passbands='grizy',
        mag_types=['psf', 'kron'],
        mags_fmt='psdr2_{pb}_{typ}',
        prefix='psdr2_'
    )
    catalog = calculate_decals8tr_features(catalog)
    return catalog


def calculate_features_on_full_catalog(data: pd.DataFrame, ebv_accounting=True,
                                       wise_forced=False,
                                       user_defined_features_transformation=lambda x: x):

    data = user_defined_features_transformation(data)

    print(format_message(ebv_accounting))
    if ebv_accounting:
        dst = _calculate_features_on_full_catalog_ebv_helper(data,
                                                            wise_forced=wise_forced)
    else:
        dst = _calculate_features_on_full_catalog_helper(data,
                                                        wise_forced=wise_forced)

    return dst

def _load_obj(*args):
    with open(os.path.join(*args), 'rb') as fin:
        return pickle.load(fin)


def get_flags_data_path(check=False):
    # try:
    #     path = os.environ['PZPH1_FLAGS_DATA_PATH']
    #     if check:
    #         print(format_message(
    #             f'Found PZPH1_FLAGS_DATA_PATH env variable: {path}'))

    # except KeyError:
    #     path = '/data/SRGz/pzph1/objids'
    #     if check:
    #         print(format_message(
    #             f'Not found PZPH1_FLAGS_DATA_PATH env variable. Using default path: {path}'))

    try:
        path = os.environ['PZPH1_DATA_PATH']

    except KeyError:
        path = './'

    path = os.path.join(path, 'objids')
    assert os.path.isdir(path), f'There must be "objids" directory: {path}'

    if check:
        file_path = os.path.join(path, 'readme_n_files.csv')
        errors = False
        try:
            flags_info = pd.read_csv(file_path, index_col='flag')
        except:
            errors = True
            print(format_message(f"Something wrong with {file_path} file"))

        if not errors:
            for flag, file in flags_info.iterrows():
                file = os.path.join(path, file['sdss'])
                if not os.path.isfile(file):
                    print(format_message(
                        f"File for {flag} flag not found: {file}"))
                    errors = True

        if errors:
            raise Exception(
                "Flags data is broken. Check hea134:./ for correct flags data.")

    return path


def place_objid_flags(objids: pd.Series):
    path = get_flags_data_path()
    flags_info = pd.read_csv(os.path.join(path, 'readme_n_files.csv'),
                             index_col='flag')
    flags = {}
    for flag, file in flags_info.iterrows():
        file = os.path.join(path, file['sdss'])
        file = pd.read_csv(file, dtype={'objID': 'str'})['objID']
        flags[flag] = objids.isin(file)

    return flags


def place_coords_flags(d: pd.DataFrame, ra_col, dec_col):
    flags = {}

    s82x_mask = ((d[ra_col] > 298) | (d[ra_col] < 62)) & (
                d[dec_col] > -1.7) & (d[dec_col] < 1.7)
    xxln_mask = (d[ra_col] > 30) & (d[ra_col] < 39) & (d[dec_col] > -7.5) & (
                d[dec_col] < -2.5)
    lh_mask = (d[ra_col] > 154.5) & (d[ra_col] < 167.0) & (
                d[dec_col] > 54.0) & (d[dec_col] < 61)

    flag = s82x_mask.copy()
    flag[s82x_mask] = 'S82X'
    flag[xxln_mask] = 'XXLN'
    flag[lh_mask] = 'LH'
    flag[~(s82x_mask | xxln_mask | lh_mask)] = '-'
    flags['phot_test_field'] = flag

    return flags


def pertrub(data: pd.Series, flux_cols: list, err_cols: list, err_types: list,
            n: int,
            random_state=42, flag_col='__perturbed__', corrmode=1):
    """
    data : row from pandas dataframe in form of pandas series
    flux_cols : list of strings or tuples. e.g. "decals8rt_r" or ("sdss_i_psf", "sdss_i_model", pcorr)
    err_cols : corresponding flux errors. list of strings or tuples of length 2
    err_types : string "ivar" or something else
    """
    data = data[1]
    result = pd.DataFrame([data.copy() for i in range(n + 1)])
    result = result[['__tempid__']]
    result[flag_col] = np.arange(n + 1)

    for i, (mag_col, err_col, err_typ) in enumerate(
            zip(flux_cols, err_cols, err_types)):
        #         try:
        if not isinstance(mag_col, list):
            mag_value = data[mag_col]
            err_value = data[err_col]

            result[mag_col] = mag_value
            result[err_col] = err_value

            if err_typ == 'ivar':
                err_value = err_value ** (-1 / 2) if err_value else np.nan

            if not np.isnan(err_value):
                perturbs = scipy.stats.norm.rvs(
                    mag_value, scale=err_value, size=n,
                    random_state=random_state + i * corrmode
                )
                result.iloc[1:, result.columns.get_loc(mag_col)] = perturbs

        else:  # if tuple
            x, y = data[mag_col[:2]]
            xerr, yerr = data[err_col]
            xerrtyp, yerrtyp, *trash = err_typ

            result[mag_col[0]] = x
            result[mag_col[1]] = y
            result[err_col[0]] = xerr
            result[err_col[1]] = yerr

            pcorr = mag_col[2]
            if xerrtyp == 'ivar':
                xerr = xerr ** (-1 / 2) if xerr else np.nan

            if yerrtyp == 'ivar':
                yerr = yerr ** (-1 / 2) if yerr else np.nan

            if pd.isna(xerr):
                x = np.nan
                xerr = 0

            if pd.isna(yerr):
                y = np.nan
                yerr = 0

            mu = np.array([x, y])
            sigma = np.array([[xerr ** 2, pcorr * xerr * yerr],
                              [pcorr * xerr * yerr, yerr ** 2]])
            pertrubs = scipy.stats.multivariate_normal.rvs(mu, sigma, n,
                                                           random_state + i * corrmode)
            col_locs = [result.columns.get_loc(col) for col in mag_col[:2]]
            result.iloc[1:, col_locs] = pertrubs
    #         except:
    #             print('Some features are missing')

    return result


#
# def predict_photoz(regr, features, test_data, model_name, numbers_col, num, short=False, njobs=1,
#                    tempidcol='__tempid__'):
#     test_data = test_data.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
#     nan_mask = ~test_data[features].isna().sum(axis=1).astype(np.bool)
#     if not nan_mask.sum():
#         raise Exception('nans_in_features')
#
#     mid = model_name
#     mask = np.arange(len(regr.estimators_)) % (num + 1)
#
#     z_pdf = list(np.array(
#         [tree.predict(test_data.loc[test_data[numbers_col] == n, features])
#          for tree, n in zip(regr.estimators_, mask)]
#     ).T)
#
#     preds = pd.DataFrame()
#     preds[f'zoo_{mid}_z_pdf'] = z_pdf
#     tempids = test_data.loc[test_data[numbers_col] == 0, tempidcol]
#     preds.index = tempids
#
#     preds = process_predictions(preds, f'zoo_{mid}_z_pdf', prefix=f'zoo_{mid}_',  # TODO
#                                 progressbar=True, njobs=njobs,
#                                 short=short)
#
#     return preds


def powlaw(e, gamma):
    """ Compute flux between e[0]-e[1]
    Power law with slope gamma
    """
    if gamma == 2: gamma=1.9999
    f=(np.power(e[1], 2.-gamma) - np.power(e[0], 2.-gamma))/(2.-gamma)
    return f


def kcorr(fflux, e_obs, e_rf, z, verbose=False, **kwargs):
    """ Compute k-correction """
    k=fflux(e_rf/(1+z), **kwargs) / fflux(e_obs, **kwargs)
    if verbose: print(f'kcorr: e_obs={e_obs}, e_rf={e_rf}, z={z}, k-corr={k:.2e}')

    return k


def calculate_lx(features, predictions, zmax_col_prefix):
    xflux_col, xflux_err = 'ero_ml_flux_0', 'ero_ml_flux_err_0'
    # xflux_col = 'ps_gPSFFlux'
    # xflux_err = 'ps_gPSFFluxErr'
    xfluxes = features[[xflux_col, xflux_err]]
    data = pd.concat([xfluxes, predictions], axis=1)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    e_obs = np.array([0.3, 2.0])
    e_rf = np.array([2.0, 10.0])

    z_max = data[zmax_col_prefix + 'z_max'].astype(np.float64)

    for redshift_type in ['z_max', 'ci1a_68', 'ci1b_68', 'ci1a_90', 'ci1b_90', 'ci1a_95', 'ci1b_95']:
        redshift_col = zmax_col_prefix + redshift_type
        dlcm_col = redshift_col + '_DL_cm'

        redshift = data[redshift_col].astype(np.float64).copy()
        if redshift_type != 'z_max':
            redshift += z_max

        data[dlcm_col] = cosmo.luminosity_distance(redshift).to('cm').value
        kk = [kcorr(powlaw, e_obs=e_obs, e_rf=e_rf, z=z, gamma=1.8) for z in
              redshift]

        data[redshift_col + '_Lx'] = 4 * np.pi * data[dlcm_col] ** 2 * data[xflux_col] * kk
        data[redshift_col + '_Lx_err'] = 4 * np.pi * data[dlcm_col] ** 2 * data[xflux_err] * kk

    data = data.drop(columns=[xflux_col, xflux_err])
    data = data.dropna(axis=0, how='all')
    return data


def predict(datasets_files, modelsIds=None, modelsSeries="x1", customModels=None,
            useWiseForced=False, keep_in_memory=False, njobs=1,
            user_defined_features_transformation=lambda x: x):
#                     files2predict, args.modelsIds, args.customModels, args.useWiseForced,
#                     njobs=args.njobs, keep_in_memory=args.keepModelsInMemory,
#                     user_defined_features_transformation=user_defined_features_transformation
    print('kwargs in predict', datasets_files, modelsIds, modelsSeries, customModels,
            useWiseForced, keep_in_memory, njobs,
            user_defined_features_transformation)
    assert not (modelsIds is None), "Empty modelsIds!"
#     Добавляю основные преобразования над данными, подбором моделей сюда, чтобы разгрузить main,
#     сделать функцию более самостоятельной
    try:
        data_path = os.environ['PZPH1_DATA_PATH']
        print(format_message(f'Found PZPH1_DATA_PATH env variable: {data_path}'))

    except KeyError:
        data_path = '/data/SRGz/pzph1/'
        print(format_message(
            f'Not found PZPH1_DATA_PATH env variable. Using default path: {data_path}'))

    models_path = os.path.join(data_path, 'models')

    models_series = {
        'x0': {
            'path': os.path.join(models_path, 'x0'),
            'models': {
                # 15: 'sdssdr16_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',  # there is not sdss wise in getaroundr
                19: 'psdr2+wise_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                21: 'psdr2+all_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                22: 'deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                35: 'sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
            },
            'config': {
                'perturb': 0,
                'ebv_accounting': False,
            },
        },
        'x0pswf': {
            'path': os.path.join(models_path, 'x0'),
            'models': {
                19: 'psdr2+wise_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                21: 'psdr2+all_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                22: 'deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
                35: 'sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY-train_QSO_XbalancedGALAXY-sdss_unwise-wo_3XMM_XXLN_S82X_LH-w_VHzQs-v2-asinhmag_features',
            },
            'config': {
                'perturb': 0,
                'ebv_accounting': False,
                'use_wise_forced': True,
            },
        },
        "x1": {
            "path": os.path.join(models_path, 'x1'),
            "models": {
                "18": "sdssdr16+wise_deacls8tr_QSO+GALAXY_20201212141009",
                "19": "psdr2+wise_deacls8tr_QSO+GALAXY_20201212135046",
                "20": "sdssdr16+all_deacls8tr_QSO+GALAXY_20201212143658",
                "21": "psdr2+all_deacls8tr_QSO+GALAXY_20201212142333",
                "22": "deacls8tr_QSO+GALAXY_20201212135641",
                "34": "sdssdr16+psdr2+wise_deacls8tr_QSO+GALAXY_20201212131454",
                "35": "sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20201212133711"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        "x1_optimization": {
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/optimization'),
            "models": {
                "19": "ALL_psdr2+wise_deacls8tr_QSO+GALAXY_20220513114537",
                "21": "ALL_psdr2+all_deacls8tr_QSO+GALAXY_20220513124936",
                "22": "ALL_deacls8tr_QSO+GALAXY_20220513120708",
                "35": "ALL_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220512183540"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        "x1_optimization_fold_0":{
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/optimization'),
            "models": {
                "19": "0_psdr2+wise_deacls8tr_QSO+GALAXY_20220513095317",
                "21": "0_psdr2+all_deacls8tr_QSO+GALAXY_20220513104731",
                "22": "0_deacls8tr_QSO+GALAXY_20220513102029",
                "35": "0_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220511234245"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        "x1_optimization_fold_1":{
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/optimization'),
            "models": {
                "19": "1_psdr2+wise_deacls8tr_QSO+GALAXY_20220513101133",
                "21": "1_psdr2+all_deacls8tr_QSO+GALAXY_20220513110544",
                "22": "1_deacls8tr_QSO+GALAXY_20220513102929",
                "35": "1_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220512003941"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
#         "x1_optimization":{
#             "path": os.path.join('/home/nmalysheva/task/S-G-Q_DESI+PanSTARRS+SDSS+WISE+J_UHS/models/pzph', 'optimization'),
#             "models": {
#                 35: "ALL_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505092546",
#             },
#             "config": {
#                 "perturb": 8,
#                 "ebv_accounting": True
#             }
#         },
#         "x1_one_gauss_fold":{
#             "path": os.path.join('/home/nmalysheva/task/S-G-Q_DESI+PanSTARRS+SDSS+WISE+J_UHS/models/pzph', 'one_gauss'),
#             "models": {
#                 "35_0": "0_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505055058",
#                 "35_1": "1_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505055948"
#             },
#             "config": {
#                 "perturb": 8,
#                 "ebv_accounting": True
#             }
#         },
        "x1_one_gauss":{
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/one_gauss'),
            "models": {
                "19": "ALL_psdr2+wise_deacls8tr_QSO+GALAXY_20220513022224",
                "21": "ALL_psdr2+all_deacls8tr_QSO+GALAXY_20220513025549",
                "22": "ALL_deacls8tr_QSO+GALAXY_20220513023258",
                "35": "ALL_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505092546.pkl"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        "x1_one_gauss_fold_0":{
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/one_gauss'),
            "models": {
                "19": "0_psdr2+wise_deacls8tr_QSO+GALAXY_20220513012208",
                "21": "0_psdr2+all_deacls8tr_QSO+GALAXY_20220513015059",
                "22": "0_deacls8tr_QSO+GALAXY_20220513013641",
                "35": "0_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505055058"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        
        "x1_one_gauss_fold_1":{
            "path": os.path.join('/data/SRGz/srgz_v10/pzph1/models', 'aug/one_gauss'),
            "models": {
                "19": "1_psdr2+wise_deacls8tr_QSO+GALAXY_20220513013200",
                "21": "1_psdr2+all_deacls8tr_QSO+GALAXY_20220513020058",
                "22": "1_deacls8tr_QSO+GALAXY_20220513102929",
                "35": "1_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20220505055948"
            },
            "config": {
                "perturb": 8,
                "ebv_accounting": True
            }
        },
        "x1a": {
            "path": os.path.join(models_path, 'x1'),
            "models": {
                "18": "sdssdr16+wise_deacls8tr_QSO+GALAXY_20201212141009",
                "19": "psdr2+wise_deacls8tr_QSO+GALAXY_20201212135046",
                "20": "sdssdr16+all_deacls8tr_QSO+GALAXY_20201212143658",
                "21": "psdr2+all_deacls8tr_QSO+GALAXY_20201212142333",
                "22": "deacls8tr_QSO+GALAXY_20201212135641",
                "34": "sdssdr16+psdr2+wise_deacls8tr_QSO+GALAXY_20201212131454",
                "35": "sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20201212133711"
            },
            "config": {
                "perturb": 0,
                "ebv_accounting": True
            }
        },
        'gal0': {
            'path': os.path.join(models_path, 'gal0'),
            'models': {
                # 15: 'sdssdr16_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                19: 'psdr2+wise_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                21: 'psdr2+all_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                22: 'deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                # 34: 'sdssdr16+psdr2+wise_deacls8tr_QSO+GALAXY_20201004092833',
                35: 'sdssdr16+psdr2+all_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
            },
            'config': {
                'perturb': 7,
                'ebv_accounting': False,
            }
        },
        'gal0pswf': {
            'path': os.path.join(models_path, 'gal0'),
            'models': {
                # 15: 'sdssdr16_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                19: 'psdr2+wise_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                21: 'psdr2+all_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                22: 'deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
                # 34: 'sdssdr16+psdr2+wise_deacls8tr_QSO+GALAXY_20201004092833',
                35: 'sdssdr16+psdr2+all_deacls8tr_GALAXY-train_GALAXY_million-sdss_unwise-wo_XXLN_S82X_LH-asinhmag_features',
            },
            'config': {
                'perturb': 7,
                'ebv_accounting': False,
                'use_wise_forced': True,
            }
        }
    }
    
    if customModels is not None:
        with open(customModels, 'r') as fin:
            custom_models_series = json.load(fin)

        models_series = {**models_series, **custom_models_series}

    print(models_series[modelsSeries])

    models_path = models_series[modelsSeries]['path']
    models = {f'{modelsSeries}{mid}': model for mid, model in ############
              models_series[modelsSeries]['models'].items()
              if (int(mid) in modelsIds) or (mid in modelsIds)}
    config = models_series[modelsSeries]['config']
    
    try:
        use_wise_forced = config['use_wise_forced']
    except KeyError:
        use_wise_forced = False

    print(format_message("Use WISE forced = "), use_wise_forced, 'or', useWiseForced)
    wise_forced = use_wise_forced or useWiseForced
    print('HHHHHEEEEEEEYYYYYYY')

    models_data = defaultdict(dict)
    models_iterable = tqdm.tqdm(models.items(),
                                desc="Load models") if keep_in_memory else models.items()
    for mid, model in models_iterable:
        regr_path = os.path.join(models_path, f'model_{model}.pkl')
        features_path = os.path.join(models_path, f'features_{model}.pkl')
        models_data[mid]['regr'] = _load_obj(
            regr_path) if keep_in_memory else regr_path
        models_data[mid]['feats'] = _load_obj(
            features_path) if keep_in_memory else features_path
        
    if isinstance(datasets_files, str):
        datasets_files = [datasets_files]
        
    for ds_path in tqdm.tqdm(datasets_files, desc="Predictions"):

        fname, ext = os.path.splitext(ds_path)
        fname = os.path.splitext(fname)[0]

        need_to_pertrub = False
        for mid in models_data.keys():
            pdfs_dst_file = f'{fname}.pdfs.{mid}{ext}'
            if not file_exists(pdfs_dst_file):
                need_to_pertrub = True
                break
        print('try to read', ds_path)
        test_data = Catalog.read_table(os.path.join(ds_path))
        need_to_predict = 'sdss_objID' in test_data.columns
        print('sdss_objID', need_to_predict)
        need_to_predict &= 'ps_objID' in test_data.columns
        print('ps_objID', 'ps_objID' in test_data.columns)
        need_to_predict &= 'ls_objid' in test_data.columns
        print('ls_objid', 'ls_objid' in test_data.columns)
        if not need_to_predict:
            print('continue')
            continue

        test_data['__tempid__'] = test_data.index.copy()
        num = config['perturb']
        dp_dst_file = f'{fname}.dp.cf{ext}'
        if not file_exists(dp_dst_file):
            if need_to_pertrub and num:
                pcorr = 0.5
                corrmode = 1

                flux_cols = [[f'sdss_psfFlux_{pb}', f'sdss_cModelFlux_{pb}', pcorr]
                             for pb in
                             ['u', 'g', 'r', 'i', 'z']]
                err_cols = [[f'sdss_psfFluxIvar_{pb}', f'sdss_cModelFluxIvar_{pb}']
                            for pb in ['u', 'g', 'r', 'i', 'z']]
                err_types = [['ivar', 'ivar'] for _ in ['u', 'g', 'r', 'i', 'z']]

                flux_cols += [[f'ps_{pb}PSFFlux', f'ps_{pb}KronFlux', pcorr] for pb
                              in ['g', 'r', 'i', 'z', 'y']]
                err_cols += [[f'ps_{pb}PSFFluxErr', f'ps_{pb}KronFluxErr'] for pb
                             in ['g', 'r', 'i', 'z', 'y']]
                err_types += [['err', 'err'] for _ in ['g', 'r', 'i', 'z', 'y']]

                flux_cols += [f'ls_flux_{pb}' for pb in
                              ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']]
                err_cols += [f'ls_flux_ivar_{pb}' for pb in
                             ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']]
                err_types += ['ivar' for _ in
                              ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']]

                flux_cols += ['ps_w1flux', 'ps_w2flux']
                err_cols += ['ps_dw1flux', 'ps_dw2flux']
                err_types += ['err', 'err']

                gaia_fluxes = [f'gaiaedr3_phot_{pb}_mean_flux' for pb in ['g', 'bp', 'rp']]
                flux_cols += gaia_fluxes
                err_cols += [flux + '_error' for flux in gaia_fluxes]
                err_types += ['err'] * len(gaia_fluxes)
#                 print(test_data['decals8tr_Lw1'])
                

                with multiprocessing.Pool(min(len(test_data), njobs)) as p:
                    helper_func = functools.partial(
                        pertrub, flux_cols=flux_cols, err_cols=err_cols,
                        err_types=err_types, n=num, corrmode=corrmode,
                    )
                    data_perturbed = pd.concat(
                        tqdm.tqdm(p.imap(helper_func, test_data.copy().iterrows()),
                                  total=len(test_data),
                                  desc='Perturbations', leave=True))

                additional_ls_columns = ['__tempid__', 'ls_ebv'] + [
                    f'ls_mw_transmission_{pb}'
                    for pb in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']]
#                 print(test_data['decals8tr_Lw1'])
#                 print(data_perturbed)
                data_perturbed = pd.merge(data_perturbed,
                                          test_data[additional_ls_columns],
                                          right_on='__tempid__', left_index=True)
#                 print(data_perturbed['decals8tr_Lw1'])
                
                # print('HHHHHEEEEEEEYYYYYYY')
                # debug_path = '/data/victor/srgz_prod/pzphlib_testing/XXLNtest/test0001results/debug-00000.gz_pkl'
                # data_perturbed.to_pickle(debug_path, compression='gzip', protocol=4)
                # print("USER DEFINDE CHECK", 'HereMyColumn' in data_perturbed.columns)
                # print(data_perturbed.columns)
            else:
                data_perturbed = test_data
                data_perturbed['__perturbed__'] = 0
            
#             print(data_perturbed['decals8tr_Lw1'])
            data_perturbed = calculate_features_on_full_catalog(
                data_perturbed, ebv_accounting=config['ebv_accounting'],
                wise_forced=wise_forced,
                user_defined_features_transformation=user_defined_features_transformation)
#             print(data_perturbed['decals8tr_Lw1'])
            
#             print(data_perturbed)
            
            data_perturbed[data_perturbed.select_dtypes(include=['float', 'int']).columns] = data_perturbed[data_perturbed.select_dtypes(include=['float', 'int']).columns].replace([-np.inf, np.inf], np.nan)
            print('SAVE')
            data_perturbed.to_pickle(dp_dst_file, compression='gzip')#, protocol=4)
            
        print('HHHHHHHHHHHHHHHHHHHHEEEEEEEEEEEEEEEEEEEEEEYYYYYYYYYYYYYYYYYYYY@@@@@@@@@@@@@@@@@')
        
        for mid, model_data in tqdm.tqdm(models_data.items()):
            used_regr = False
            pdfs_dst_file = f'{fname}.pdfs.{mid}{ext}'
            if not file_exists(pdfs_dst_file):
                if keep_in_memory:
                    # regr = model_data['regr']
                    feats = model_data['feats']
                else:
                    print('feats')
                    print('tot_m = {}, used_m = {}, free_m = {}'.format(*map(int, os.popen('free -t -m').readlines()[-2].split()[1:])))
                    # regr = _load_obj(model_data['regr'])
                    feats = _load_obj(model_data['feats'])
                    print('tot_m = {}, used_m = {}, free_m = {}'.format(*map(int, os.popen('free -t -m').readlines()[-2].split()[1:])))
                
                data_perturbed = pd.read_pickle(dp_dst_file, compression='gzip')#, protocol=4)
                gc.collect()

                notna_mask = data_perturbed[feats].notna().all(axis=1)
                
                
                if not notna_mask.any():
                    print('continue')
                    continue
                    
                test_data_notna = data_perturbed.loc[notna_mask]
                # preds = predict_photoz(regr, feats, test_data_notna, mid, numbers_col='__perturbed__', njobs=njobs, num=8,
                #                        tempidcol='__tempid__')
                
                del data_perturbed
                
                if keep_in_memory:
                    regr = model_data['regr']
                    # feats = model_data['feats']
                else:
                    print('regr', model_data['regr'])
                    print('tot_m = {}, used_m = {}, free_m = {}'.format(*map(int, os.popen('free -t -m').readlines()[-2].split()[1:])))
                    regr = _load_obj(model_data['regr'])
                    print('tot_m = {}, used_m = {}, free_m = {}'.format(*map(int, os.popen('free -t -m').readlines()[-2].split()[1:])))
                    
                    # feats = _load_obj(model_data['feats'])

                mask = (np.arange(len(regr.estimators_))) % (num + 1)
                maxim = np.max([np.sum(test_data_notna['__perturbed__'] == n) for n in mask])
                z_pdf = [np.array(tree.predict(test_data_notna.loc[test_data_notna[
                                                          '__perturbed__'] == n, feats].values), dtype=float)
                     for tree, n in zip(regr.estimators_, mask)]
                for i, n, j in zip(z_pdf, mask, np.arange(len(mask))):
                    if i.shape[0] < maxim:
                        z_pdf[j] = np.array(list(i) + [0.0]*(maxim - i.shape[0]))
                z_pdf = list(np.array(z_pdf).T)

                preds = pd.DataFrame()
                preds[f'zoo_{mid}_z_pdf'] = z_pdf
                tempids = test_data_notna.loc[
                    test_data_notna['__perturbed__'] == 0, '__tempid__']
                preds.index = tempids

                if not keep_in_memory:
                    del regr

                preds.to_pickle(pdfs_dst_file, compression='gzip', protocol=4)
                used_regr = True

            preds_dst_file = f'{fname}.preds.{mid}{ext}'
            if not file_exists(preds_dst_file):
                pdf_col = f'zoo_{mid}_z_pdf'
                if not used_regr:
                    preds = pd.read_pickle(pdfs_dst_file, compression='gzip')

                short = False  # TODO
                preds = process_predictions(preds, f'zoo_{mid}_z_pdf',
                                            prefix=f'zoo_{mid}_',  # TODO
                                            progressbar=True, njobs=njobs,
                                            short=short, leave=True)

                preds = preds.rename(columns={
                    'ProcessPredictionsError': f'zoo_{mid}_ProcessPredictionsError'})
                preds = preds.drop(columns=pdf_col)

                # print(preds.shape)
                if 'ero_ml_flux_0' in test_data.columns:
                    print("calculate lx")
                    preds = calculate_lx(test_data, preds, f'zoo_{mid}_')
                # print(preds.shape)

                preds.to_pickle(preds_dst_file, compression='gzip', protocol=4)

            # dst_file = f'{fname}-pdfs.{mid}{ext}'
            # pdf_col = f'zoo_{mid}_z_pdf'
            # preds[[pdf_col]].to_pickle(dst_file, compression='gzip', protocol=4)
            # preds = preds.drop(columns=pdf_col)
            # dst_file = f'{fname}-preds.{mid}{ext}'
            # preds.to_pickle(dst_file, compression='gzip')


def read_predictions_apart(path: str):
    dataset_format = 'part-{}.features.gz_pkl'
    pdfs_format = 'part-{}.pdfs.{}.gz_pkl'
    preds_format = 'part-{}.preds.{}.gz_pkl'
    parts = []
    for file in glob.glob(os.path.join(path, dataset_format.format('*'))):
        dataset_part = Catalog.read_table(file)
        fname = os.path.basename(file)
        chunk_number = re.findall(dataset_format.format(r'(\d*)'), fname)
        if not chunk_number:
            continue

        chunk_number = chunk_number[0]
        prediction_files = glob.glob(
            os.path.join(path, pdfs_format.format(chunk_number, '*')))
        prediction_files += glob.glob(
            os.path.join(path, preds_format.format(chunk_number, '*')))
        for pred_file in prediction_files:
            preds = Catalog.read_table(pred_file)
            dataset_part = pd.concat([dataset_part, preds], axis=1)

        parts.append(dataset_part)

    return pd.concat(parts)


def split_data(xray=None, xray_hp_id_col=None, sdss=None, ps=None, ls=None,
               chunksize=100000, base_catalog='ls',
               sdss_on=None, ps_on=None, ls_on=None):
    if xray is not None:
        if xray_hp_id_col is not None:
            power = 1
            maxnum = xray[xray_hp_id_col].max()
            while maxnum // 10:
                power += 1
                maxnum //= 10

            fmt = '{{:0{}d}}'.format(power)
            for k, v in xray.groupby(by=xray_hp_id_col):
                yield dict(xray=v, fnum=fmt.format(k))
        else:
            for istart in range(0, len(xray), chunksize):
                iend = min(istart + chunksize, len(xray) + 1)
                yield dict(xray=xray.iloc[istart: iend])

    else:
        catalogs = dict(sdss=sdss, ps=ps, ls=ls)
        on_columns = dict(sdss=sdss_on, ps=ps_on, ls=ls_on)
        assert catalogs[base_catalog] is not None and on_columns[
            base_catalog] is not None
        catalogs[base_catalog] = catalogs[base_catalog].sort_values(
            by=on_columns[base_catalog])
        for istart in range(0, len(catalogs[base_catalog]), chunksize):
            iend = min(istart + chunksize, len(catalogs[base_catalog]) + 1)
            chunk = catalogs[base_catalog].iloc[istart:iend]
            result = {base_catalog: chunk}
            on_values = chunk[on_columns[base_catalog]]
            onstart, onend = on_values.min(), on_values.max()
            for cat_name, values in catalogs.items():
                if cat_name != base_catalog:
                    if values is None:
                        result[cat_name] = None
                    else:
                        on_values = values[on_columns[cat_name]]
                        mask = (onstart <= on_values) & (on_values <= onend)
                        result[cat_name] = values.loc[mask]

            yield result


def assemble_and_analyze_results(buf_path: str, dst_path: str, models_series, file_name=None, fold=''):
    start_nrow = 1
    models_goodness = [f'{models_series}{mid}{fold}' for mid in
                       [22, 19, 18, 15, 21, 20, 34, 35]]
    assert isinstance(file_name, str) or file_name is None
    for file in glob.glob(os.path.join(buf_path, '*.features.gz_pkl')) if file_name is None else [os.path.join(buf_path, file_name+'.features.gz_pkl')]:
        data = pd.read_pickle(file, compression='gzip')
        data['__nrow__'] = np.arange(start_nrow, start_nrow + len(data))
        # data.to_pickle(os.path.join(dst_path, os.path.basename(file)),
        #                compression='gzip', protocol=4)
        data.to_pickle(file, compression='gzip', protocol=4)
        copyfile_link(file, os.path.join(dst_path, os.path.basename(file)))

        start_nrow = start_nrow + len(data)

        # shutil.copy(file, dst_path)
        fname = re.findall(r'(.*)\.features\.gz_pkl$', os.path.basename(file))[
            0]

        preds_dst_file = os.path.join(dst_path,
                                      f'{fname}.predictions.{models_series}{fold}.gz_pkl')
        best_dst_file = os.path.join(dst_path,
                                     f'{fname}.best.{models_series}{fold}.gz_pkl')

        preds = []
        for preds_file in glob.glob(
                os.path.join(buf_path, fr'{fname}.p*.{models_series}*')):
            is_preds_file = re.findall(fr'\.{models_series}\d\d\_?\d?.', preds_file)
            if is_preds_file:
                preds.append(pd.read_pickle(preds_file, compression='gzip'))

        try:
            data = pd.concat(preds, axis=1)
        except ValueError as ve:
            if str(ve) == 'No objects to concatenate':
                continue
            else:
                raise ValueError(ve)

        data.to_pickle(preds_dst_file, compression='gzip', protocol=4)

        mid = f'best-{models_series}{fold}'
        mid_column_name = f'zoo_{mid}_model_id'
        best_columns = [f'zoo_{mid}_z_pdf', f'zoo_{mid}_z_max',
                        f'zoo_{mid}_z_maxConf', f'zoo_{mid}_z_max_DL_cm',
                        f'zoo_{mid}_z_max_Lx', f'zoo_{mid}_z_max_Lx_err',
                        f'zoo_{mid}_ci1a_68', f'zoo_{mid}_ci1b_68',
                        f'zoo_{mid}_ci1a_68_DL_cm', f'zoo_{mid}_ci1b_68_DL_cm',
                        f'zoo_{mid}_ci1a_68_Lx', f'zoo_{mid}_ci1b_68_Lx',
                        f'zoo_{mid}_ci1a_68_Lx_err', f'zoo_{mid}_ci1b_68_Lx_err',
                        f'zoo_{mid}_ci1a_90', f'zoo_{mid}_ci1b_90',
                        f'zoo_{mid}_ci1a_90_DL_cm', f'zoo_{mid}_ci1b_90_DL_cm',
                        f'zoo_{mid}_ci1a_90_Lx', f'zoo_{mid}_ci1b_90_Lx',
                        f'zoo_{mid}_ci1a_90_Lx_err', f'zoo_{mid}_ci1b_90_Lx_err',
                        f'zoo_{mid}_ci1a_95', f'zoo_{mid}_ci1b_95',
                        f'zoo_{mid}_ci1a_95_DL_cm', f'zoo_{mid}_ci1b_95_DL_cm',
                        f'zoo_{mid}_ci1a_95_Lx', f'zoo_{mid}_ci1b_95_Lx',
                        f'zoo_{mid}_ci1a_95_Lx_err', f'zoo_{mid}_ci1b_95_Lx_err']

        best = pd.DataFrame(index=data.index, columns=best_columns)

        for mid in models_goodness:
            model_columns = [f'zoo_{mid}_z_pdf', f'zoo_{mid}_z_max',
                        f'zoo_{mid}_z_maxConf', f'zoo_{mid}_z_max_DL_cm',
                        f'zoo_{mid}_z_max_Lx', f'zoo_{mid}_z_max_Lx_err',
                        f'zoo_{mid}_ci1a_68', f'zoo_{mid}_ci1b_68',
                        f'zoo_{mid}_ci1a_68_DL_cm', f'zoo_{mid}_ci1b_68_DL_cm',
                        f'zoo_{mid}_ci1a_68_Lx', f'zoo_{mid}_ci1b_68_Lx',
                        f'zoo_{mid}_ci1a_68_Lx_err', f'zoo_{mid}_ci1b_68_Lx_err',
                        f'zoo_{mid}_ci1a_90', f'zoo_{mid}_ci1b_90',
                        f'zoo_{mid}_ci1a_90_DL_cm', f'zoo_{mid}_ci1b_90_DL_cm',
                        f'zoo_{mid}_ci1a_90_Lx', f'zoo_{mid}_ci1b_90_Lx',
                        f'zoo_{mid}_ci1a_90_Lx_err', f'zoo_{mid}_ci1b_90_Lx_err',
                        f'zoo_{mid}_ci1a_95', f'zoo_{mid}_ci1b_95',
                        f'zoo_{mid}_ci1a_95_DL_cm', f'zoo_{mid}_ci1b_95_DL_cm',
                        f'zoo_{mid}_ci1a_95_Lx', f'zoo_{mid}_ci1b_95_Lx',
                        f'zoo_{mid}_ci1a_95_Lx_err', f'zoo_{mid}_ci1b_95_Lx_err']

            for col in model_columns:
                if col not in data.columns:
                    data[col] = np.nan

            zmax_col = f'zoo_{mid}_z_max'
            if zmax_col not in data.columns:
                continue

            mask = data[zmax_col].notna()
            best.loc[mask, best_columns] = data.loc[
                mask, model_columns].rename(
                columns=dict(zip(model_columns, best_columns))
            )
            best.loc[mask, mid_column_name] = mid

        best.to_pickle(best_dst_file, compression='gzip', protocol=4)


def copyfile_link(src, dst, symlink=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    link = os.symlink if symlink else os.link

    # print(src, dst)
    try:
        link(os.path.abspath(src), dst)
    except FileExistsError:
        pass  # TODO replace symlink if file is newer then link


def main():

    args = parse_cli_args()

    assert args.baseCatalog in ['ps', 'ls', 'sdss', 'gaiaedr3'], 'Other catalogs not implemented yet'
    assert args.psEdition in ['ps2oldfluxradecbest', 'ps2fluxbest']
    # assert not args.useWiseForced, 'Wise forced not implemented yet'

    if args.baseCatalog == "ls":
        args.baseRaCol = 'ra'
        args.baseDecCol = 'dec'
    elif args.baseCatalog == "ps":
        args.baseRaCol = "raBest"
        args.baseDecCol = "decBest"
    elif args.baseCatalog == "sdss":
        args.baseRaCol = 'ra'
        args.baseDecCol = 'dec'
    elif args.baseCatalog == "gaiaedr3":
        args.baseRaCol = 'ra'
        args.baseDecCol = 'dec'

    get_flags_data_path(check=True)
    print(args)

    if args.featuresTransformModule is not None and args.featuresTransformName is not None:
        user_defined_features_transformation = _import_user_defined_features_transformation(
            args.featuresTransformModule, args.featuresTransformName
        )
    else:
        user_defined_features_transformation = lambda x: x

    if args.coldStart:
        try:
            shutil.rmtree(os.path.join(args.outputDir))
        except FileNotFoundError:
            pass

    os.makedirs(args.outputDir, exist_ok=True)
    ps_objids = []
    buf_path = os.path.join(args.outputDir, 'buf')
    os.makedirs(buf_path, exist_ok=True)
    files2predict = []
    if args.predictOn is None:
        catalog_kws = dict(
            xray_data_path=args.xrayCatalog,
            xray_radec_cols=(args.xrayRaCol, args.xrayDecCol),
            base_catalog=args.baseCatalog,
            base_radec_cols=(args.baseRaCol, args.baseDecCol),
            sdss_path=args.sdss, ps_path=args.ps, ls_path=args.ls,
            sdss_on=args.sdssOn, ps_on=args.psOn, ls_on=args.lsOn,
            assembled_dataset_path=args.assembledDataset, output_dir=buf_path,
            primary_radius=args.primaryRadius,
            secondary_rasius=args.secondaryRadius, njobs=args.njobs,
            getaroundr_path=args.getaroundrPath,
            # cj_user_id=args.cjUserID, cj_password=args.cjPassword,
            ps_fluxes_manually=args.psFluxesManually, ps_fluxes=None,
            user_defined_features_transformation=user_defined_features_transformation,
            panstarrs_catalog_to_use_cause_my_bullshit_code_and_noone_to_download_the_entire_panstarrs_properly_once_and_forall=args.psEdition,
        )

        data_path = os.path.join(args.outputDir, 'data')
        data_written_file = os.path.join(data_path, "DATA_WRITTEN_FILE.txt")
        if not os.path.isfile(data_written_file):
            os.makedirs(data_path, exist_ok=True)

            if args.xrayCatalog is not None:
                if os.path.isdir(args.xrayCatalog):
                    iterator = [{'xray': file} for file in
                                glob.glob(os.path.join(args.xrayCatalog, '*'))]
                elif args.xrayHealpixId is not None:
                    iterator = list(
                        split_data(xray=Catalog.read_table(args.xrayCatalog),
                                   xray_hp_id_col=args.xrayHealpixId)
                    )
                else:
                    iterator = list(
                        split_data(xray=Catalog.read_table(args.xrayCatalog),
                                   chunksize=args.chunkSize))

            else:
                iterator = list(split_data(sdss=Catalog.read_table(
                    args.sdss) if args.sdss is not None else None,
                                           ps=Catalog.read_table(
                                               args.ps) if args.ps is not None else None,
                                           ls=Catalog.read_table(
                                               args.ls) if args.ls is not None else None,
                                           base_catalog=args.baseCatalog,
                                           sdss_on=args.sdssOn,
                                           ps_on=args.psOn, ls_on=args.lsOn,
                                           chunksize=args.chunkSize))

            for i, chunk in tqdm.tqdm(enumerate(iterator), total=len(iterator),
                                      desc='Preparing data'):
                if 'xray' in chunk and isinstance(chunk['xray'], str):
                    fname = os.path.basename(
                        os.path.splitext(chunk['xray'])[0])
                elif 'xray' in chunk and 'fnum' in chunk:
                    fname = 'part-{}'.format(chunk['fnum'])
                else:
                    fname = 'part-{:05d}'.format(i)

                for k in ['xray', 'sdss', 'ps', 'ls']:
                    try:
                        chunk_data = chunk[k]
                    except KeyError:
                        continue

                    chunk_dst_path = os.path.join(data_path,
                                                  f'{fname}.{k}.gz_pkl')
                    chunk_data.to_pickle(chunk_dst_path, compression='gzip',
                                         protocol=4)

            with open(data_written_file, 'w'):
                pass

        chunks_files = defaultdict(dict)
        for file in glob.glob(os.path.join(data_path, '*')):
            parsed_filename = re.findall(r'^(.*)\.(.*)\.gz_pkl$',
                                         os.path.basename(file))
            print(parsed_filename)
            if parsed_filename:
                fname, chunk_type = parsed_filename[0]
                chunks_files[fname][chunk_type] = file

        for i, (fname, chunk) in tqdm.tqdm(enumerate(chunks_files.items()),
                                           total=len(chunks_files)):
            print(fname)
            # chunk_number = re.findall("^part-(\d*)$", fname)
            # print(chunk_number)
            # if not len(chunk_number):
            #     raise Exception("Wrong file name: {}".format(fname))
            # else:
            #     chunk_number = int(chunk_number[0])
            #     if chunk_number in [1]:
            #         continue

            dst_path = os.path.join(buf_path,
                                    f'{fname}.features.gz_pkl')  # TODO nice names format

            if file_exists(dst_path):
                files2predict.append(dst_path)
                continue

            catalog_kws_to_chunk_types = {
                'xray_data_path': 'xray', 'sdss_path': 'sdss', 'ps_path': 'ps',
                'ls_path': 'ls'
            }
            for kw, chunk_type in catalog_kws_to_chunk_types.items():
                try:
                    catalog_kws[kw] = chunk[chunk_type]
                except KeyError:
                    catalog_kws[kw] = None

            if args.psFluxesPath:
                ps_fluxes = pd.read_csv(args.psFluxesPath,
                                        dtype={'objID': int})
                ps_fluxes = {k: v for k, v in ps_fluxes.groupby(by='__file__')}
                catalog_kws['ps_fluxes'] = ps_fluxes

            catalog_kws['filename'] = fname
            catalog = Catalog(**catalog_kws)
            try:
                status = catalog.prepare_data()
            except Exception as e:
                if str(e) == 'Found nothing in base catalog':
                    print(dst_path, fname)
                    shutil.copy(
                        os.path.join(data_path, f'{fname}.xray.gz_pkl'),
                        dst_path)
                    status = None
                else:
                    print(e)
                    raise Exception(e)

            if status == "ps_manual":
                ps_objids.append(catalog.ps_objids)
            # shutil.move(catalog.assembled_dataset_path, dst_path)
            else:
                files2predict.append(dst_path)
    else:
        for file in glob.glob(
                os.path.join(args.predictOn, '*.features.gz_pkl')):
            ### !!! Copying files
            # shutil.copy(file, buf_path)
            copyfile_link(file, buf_path)
            files2predict.append(
                os.path.join(buf_path, os.path.basename(file)))

    if ps_objids:
        objids_csv_path = os.path.join(args.outputDir, 'ps_objids.csv')
        pd.concat(ps_objids).to_csv(objids_csv_path, index=False)
        print("""
        Now you are to download PanSTARRS fluxes from casjobs.
        Upload generated csv and execute query with PanSTARRS_DR2 context:
            select t.__file__, m.objid,
            m.gPSFFlux, m.gPSFFluxErr, m.gKronFlux, m.gKronFluxErr,
            m.rPSFFlux, m.rPSFFluxErr, m.rKronFlux, m.rKronFluxErr,
            m.iPSFFlux, m.iPSFFluxErr, m.iKronFlux, m.iKronFluxErr,
            m.zPSFFlux, m.zPSFFluxErr, m.zKronFlux, m.zKronFluxErr,
            m.yPSFFlux, m.yPSFFluxErr, m.yKronFlux, m.yKronFluxErr

            into MyDB.<destination table>
            from MyDB.<table you created from csv> t
            left join StackObjectAttributes m on m.objid=t.objid

        Generated csv: {}
        """.format(objids_csv_path))
    else:
        if args.modelsIds is not None:
            files2predict = sorted(files2predict)
            predict(files2predict, args.modelsIds, args.modelsSeries, args.customModels, args.useWiseForced,
                    njobs=args.njobs, keep_in_memory=args.keepModelsInMemory,
                    user_defined_features_transformation=user_defined_features_transformation)

        assemble_and_analyze_results(buf_path, args.outputDir,
                                     models_series=args.modelsSeries)
        # if args.cleanupBuffer:
        #     shutil.rmtree(buf_path)

if __name__ == '__main__':
    main()
