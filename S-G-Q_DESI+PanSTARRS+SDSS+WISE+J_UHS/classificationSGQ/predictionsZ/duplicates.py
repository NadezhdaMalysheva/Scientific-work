import functools
import multiprocessing
import os
import subprocess
import warnings

import astropy.coordinates
import healpy as hp
import numpy as np
import pandas as pd
from tqdm import tqdm

from classificationSGQ.predictionsZ.process_counterparts import process_counterparts


def _check_file_exists(file, file_exists_error):
    """
    A helper function to check that file exists given the action in file_exists_error
    :param file: path to target file
    :param file_exists_error: "raise", "warn" or None
    """
    try:
        assert not os.path.isfile(file)
    except AssertionError:
        msg = 'file {} exists!'.format(file)
        if file_exists_error == 'raise':
            raise Exception(msg)

        if file_exists_error == 'warn':
            warnings.warn(msg)


def _deg2rad(x):
    """
    Degrees to radians
    :param x: scalar or array containing angles in degrees
    :return: scalar or array, depending on the input, containing angles in radians
    """
    return x * np.pi / 180


def _arrays2series(index, *args):
    """
    Convert arrays to pandas.Series given index
    :param index: array (index of pandas.DataFrame)
    :param args: list of arrays that len(array from list) == len(index)
    :return: list of arrays in form of pd.Series with given index
    """
    return (pd.Series(v, index=index) for v in args)


def _radec2euclid(ra_x: float, dec_x: float, ra_y: pd.Series, dec_y: pd.Series) -> pd.Series:
    """
    Calculates euclid distance on a sphere of unit radius between a single point X and every point in Y
        in form of 2 pandas.Series
    :param ra_x: float, right ascension in degrees of point X
    :param dec_x: float, declination in degrees of point X
    :param ra_y: pandas.Series, right ascension in degrees of list of points
    :param dec_y: pandas.Series, declination in degrees of list of points
    :return: pandas.Series containing corresponding distances
    """
    x1, y1, z1 = astropy.coordinates.spherical_to_cartesian(1, _deg2rad(dec_x), _deg2rad(ra_x))
    x2, y2, z2 = astropy.coordinates.spherical_to_cartesian(1, _deg2rad(dec_y), _deg2rad(ra_y))
    x2, y2, z2 = _arrays2series(ra_y.index, x2, y2, z2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** (1 / 2)


def _index_counterparts_helper(data, ra_target, dec_target, one_second_hoard, healpix_ids_col, nside) -> pd.Series:
    """
    A helper function used in _index_counterparts for multiprocessing.Pool
    :param data: tiple (_, pandas.DataFrame). The second contains data on counterparts of single source
    :param ra_target: str, name of column containing right ascension of counterparts in given data
    :param dec_target: str, name of column containing declination of counterparts in given data
    :param one_second_hoard: float, distance threshold between counterparts to consider them as duplicates
    :param healpix_ids_col: str, column containing healpix index
    :param nside: int, nside for healpix
    :return: pandas.Series containing corresponding duplicates ids for given data
    """
    _, counterparts = data
    healpix_ids = counterparts[healpix_ids_col]
    duplicate_ids = pd.Series(-np.ones(len(counterparts), dtype=np.int64), index=counterparts.index)
    next_id = 0
    while (duplicate_ids == -1).any():
        not_processed_mask = duplicate_ids == -1
        hpid0 = healpix_ids.loc[not_processed_mask].iloc[0]
        ra0, dec0 = counterparts.loc[not_processed_mask].iloc[0][[ra_target, dec_target]]

        target_hpids = np.append(hpid0, hp.pixelfunc.get_all_neighbours(nside, hpid0, nest=True))
        healpix_mask = healpix_ids.isin(target_hpids) & not_processed_mask

        radec = counterparts.loc[healpix_mask][[ra_target, dec_target]]

        distances = _radec2euclid(ra0, dec0, radec[ra_target], radec[dec_target])
        distances_mask = distances < one_second_hoard
        distances_mask = distances_mask.loc[distances_mask].index
        duplicate_ids.loc[distances_mask] = next_id
        next_id += 1

    return duplicate_ids


def _index_counterparts(df: pd.DataFrame, orig_objid, ra_target, dec_target, radius=1.0, njobs=1) -> pd.DataFrame:
    """
    A function to index duplicates in circles of 1-second radius. The index is local for each id in orig_objid,
        so a group of duplicates is is uniquely determined by the pair (the id, the index).
    :param df: pandas.DataFrame containing columns specified in orig_objid, ra_target and dec_target parameters
    :param orig_objid: str, name of column in df with unique identifier of original objects
    :param ra_target: str, name of column in df with right ascension of counterparts
    :param dec_target: str, name of column in df with declination of counterparts
    :param njobs: number of jobs for multiprocessing
    :return: pandas.DataFrame with duplicates indices
    """
    log2nside = 1
    while hp.nside2resol(2 ** log2nside, arcmin=True) / 60 * 60 * 60 > radius:
        # hp.nside2resol(NSIDE, arcmin=True) / 60 is in degrees according to healpy tutorial
        log2nside += 1

    log2nside -= 1

    nside = 2 ** log2nside
    second = _deg2rad(radius / 60 / 60)
    one_second_hoard = np.sin(second) / np.sin((np.pi - second) / 2)
    healpix_ids_col = f'healpix_id_log2nside{log2nside}'

    if healpix_ids_col not in df:
        print("===== DEBUG =====")
        print(f"= ra_target -- {ra_target} has shape {df[ra_target].shape}")
        print(f"= dec_target -- {dec_target} has shape {df[dec_target].shape}")
        print("===== END DEBUG =====")
        df[healpix_ids_col] = hp.ang2pix(
            nside,
            _deg2rad(-df[dec_target] + 90).astype(np.float).values,
            _deg2rad(df[ra_target]).astype(np.float).values,
            nest=True,
        )

    kws = dict(
        ra_target=ra_target,
        dec_target=dec_target,
        one_second_hoard=one_second_hoard,
        healpix_ids_col=healpix_ids_col,
        nside=nside
    )
    helper = functools.partial(_index_counterparts_helper, **kws)

    data = df.groupby(by=orig_objid)
    with multiprocessing.Pool(min(njobs, len(data))) as p:
        result = pd.concat(list(
            tqdm(
                p.imap(helper, data),
                total=len(data),
                desc="Indexing Duplicates",
                leave=False
            )
        ), sort=False)

    df['duplicate_id'] = result
    return df


def clean_duplicates(df: pd.DataFrame, orig_objid: str, ra_target: str, dec_target: str,
                     fluxes: list, errors: list, errors_types: list, radius=1.0, njobs=1) -> pd.DataFrame:
    """
    A function to process duplicates.
    :param df: pandas.DataFrame containing columns specified in orig_objid, ra_target and dec_target,
        fluxes and errors parameters
    :param orig_objid: str, name of column in df with unique identifier of original objects
    :param ra_target: str, name of column in df with right ascension of counterparts
    :param dec_target: str, name of column in df with declination of counterparts
    :param fluxes: list of columns, containing fluxes to process
        (choose fluxes and errors with the highest signal-to-noise)
    :param errors: list of columns, containing corresponding flux errors to process
    :param errors_types: list containing "err" or "ivar" values, specifying type of corresponding error
    :param njobs: number of jobs for multiprocessing
    :return: pd.DataFrame with processed duplicates.

    Example:
        fluxes_columns = ['flux_g', 'flux_r', 'flux_z',
                          'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4']
        errors_columns = ['flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                          'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
        errors_types = ['ivar', 'ivar', 'ivar', 'ivar', 'ivar', 'ivar', 'ivar']
        df = clean_duplicates(df, 'ID_SRC', 'ra', 'dec', fluxes_columns, errors_columns, errors_types, njobs=4)
    """
    assert 'duplicate_id' not in df.columns
    assert orig_objid in df.columns
    assert ra_target in df.columns
    assert dec_target in df.columns

    assert len(errors) == len(fluxes), "Fluxes list length and errors list length must be queal"
    assert len(errors_types) == len(fluxes), "Fluxes list length and errors types list length must be queal"

    wrong_err_types_values = [et for et in errors_types if et not in ["err", "ivar"]]
    assert not wrong_err_types_values, f"Incorrect values in errors_types : {wrong_err_types_values}." \
                                       f" Allowed values are 'err' and 'ivar'"

    not_in_df_columns = [col for col in fluxes + errors if col not in df.columns]
    assert not not_in_df_columns, f"Some of columns specified in fluxes and errors not found if dataframe:" \
                                  f" {not_in_df_columns}"

    df = _index_counterparts(df, orig_objid, ra_target, dec_target, radius, njobs)

    interested_columns = [
        [f, e, np.nan, np.nan, "flux", 'i' if et == 'ivar' else 's']
        for f, e, et in zip(fluxes, errors, errors_types)
    ]

    conf = {
        "script_version": 0.1,
        "comments": [''],
        "input": "-",
        "output": "-",
        "interested_columns": interested_columns,
        "object_definition": [orig_objid, 'duplicate_id'],
        "counterpart_definition": [ra_target, dec_target],
    }

    result = process_counterparts(conf, df, njobs)

    return result.drop(columns='duplicate_id')


def getaroundr(input_path, output_path, ira, idec, radius, catalog, dst_path, stdout=True, file_exists_error='warn'):
    """
    Method to apply cross-match program specified during initialisation of object
    :param input_path: path to input fits file. Must contain columns specified in ira and idec parameters
    :param output_path: path to output fits file
    :param ira: ra column in input
    :param idec: dec column in input
    :param radius: radius to perform cross-match in.
    :param catalog: catalog to perform cross-match with. See cross-match program help.
    :param dst_path: destination fits file
    :param stdout: if true, writes stdout and stderr into files <filename>.getaroundr_std<err/out>.txt in the same
        directory as
    :param file_exists_error: "raise", "warn" or None. If none, says nothing
    """
    exist_msg = 'file {} does not exist'
    assert os.path.isfile(input_path), exist_msg.format(input_path)

    fits_msg = 'file {} must be fits'
    assert os.path.splitext(input_path)[1] == '.fits', fits_msg.format(input_path)
    assert os.path.splitext(output_path)[1] == '.fits', fits_msg.format(output_path)

    fname, _ = os.path.splitext(dst_path)
    stdout_path = f'{fname}.getaroundr_stdout.txt'
    stderr_path = f'{fname}.getaroundr_stderr.txt'

    _check_file_exists(dst_path, file_exists_error)
    if stdout:
        _check_file_exists(stdout_path, file_exists_error)
        _check_file_exists(stdout_path, file_exists_error)

    getaroundr_path = '/home/horungev/Catalogs/SRG/crossmatch/getaroundr.py'

    command = f"{getaroundr_path} -i {input_path} -r {radius} -cat {catalog} -o {output_path} -asfx _input" \
              f" -iRA {ira} -iDEC {idec} -iSEPNAME sep:sep1 -full"

    print(command)
    with open(stdout_path, 'w') as stdout, open(stderr_path, 'w') as stderr:
        try:
            subprocess.run(command, stdout=stdout, stderr=stderr, shell=True, check=True)
        except subprocess.CalledProcessError as cpe:
            error_msg_prolog = f"Something went wrong during cross-match step:"
            error_msg_epilog = f"See {stdout_path} and {stderr_path} for more info"
            print(error_msg_prolog, cpe, error_msg_epilog)
            raise Exception("Cross-match failed.")
