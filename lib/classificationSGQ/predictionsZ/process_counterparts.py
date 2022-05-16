import argparse
import functools
import json
import os
import re
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pprint import pprint

import astropy.table
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


def program_template(help_lang=''):
    """
    Just generates a "program" template
    :return: dict
    """
    # TODO
    desc = {
        'ENG': """
    
        """,
        'RUS': """
        
        """
    }
    template = {
        'script_version': 0.1,
        'comments': [
            'template for process_counterparts.py',
            'leave any comments you want here',
            'they will be ignored',
            'TODO write help',
        ],
        'input': 'file.gz_pkl',
        'output': 'output.gz_pkl',
        'interested_columns': [
            ['iKronMag', 'iKronMagErr', -999, -999, 'mag'],
            ['rPSFMag', 'rPSFMagErr', -999, -999, 'mag', 's'],
            [r'(\w)APMag', r'(\w)APMagErr', -999, -999, 'mag'],
            [r'w(\d)mag', r'dw(\d)mag', -999, -999, 'mag'],
            ['flux_w1', 'flux_ivar_w1', -1, -1, 'flux', 'i'],
        ],
        'object_definition': ['ra', 'dec'],
        'counterpart_definition': ['ra0', 'dec0'],
    }

    if help_lang:
        template['help'] = desc[help_lang]

    return template


def _write_table(data, path):
    _, ext = os.path.splitext(path)
    if ext == '.gz_pkl':
        pd.to_pickle(data, path, compression='gzip')
    else:
        raise Exception('Unsupported format of output{}'.format(path))


def _read_table(path):
    assert os.path.isfile(path), "File does not exist {}".format(path)
    _, ext = os.path.splitext(path)
    if ext == '.gz_pkl':
        return pd.read_pickle(path, compression='gzip'), []

    elif ext == '.fits':
        data = astropy.table.Table.read(path)
        single_cols = [name for name in data.colnames if len(data[name].shape) <= 1]
        multi_cols = [name for name in data.colnames if len(data[name].shape) > 1]
        data = data[single_cols].to_pandas()
        warnings.warn('!!! There are multidimentioanl columns that were dropped {} !!!'.format(multi_cols))
        return data, multi_cols

    elif ext == '.csv':
        return pd.read_csv(path), []

    else:
        raise Exception("unsupported format of input {}. Only csv, gz_pkl and fits are supported".format(path))


def _check_program(prog):
    def _check_re(expr):
        if '\\' in expr:
            res_l = len(re.findall(r'\(', expr))
            res_r = len(re.findall(r'\)', expr))
            if res_l == 1 and res_r == 1:
                return True

        else:
            return True

        return False

    error_msg = ""
    if 'comments' not in prog.keys():
        prog['comments'] = ""

    if 'input' not in prog.keys():
        error_msg += "Error : you must specify input.\n"

    if 'output' not in prog.keys():
        error_msg += "Error : you must specify output.\n"

    if 'interested_columns' not in prog.keys():
        error_msg += "Error : you must specify interested columns.\n"
    else:
        for template in prog['interested_columns']:
            if not(_check_re(template[0]) and _check_re(template[1])):
                error_msg += "Defective expression in {}.\n".format(template)

            if len(template) < 5 and len(template) > 6:
                error_msg += "Defective template {}.\n".format(template)

    if 'object_definition' not in prog.keys():
        error_msg += "Error : you must specify object definition.\n"

    if 'counterpart_definition' not in prog.keys():
        error_msg += "Error : you must specify counterpart definition.\n"

    assert not error_msg, error_msg+"See help.\n"


def _parse_program(prog, cols):
    _check_program(prog)
    interested_columns = []
    missing_values = defaultdict(list)
    for template in prog['interested_columns']:
        error_type = 's'
        if len(template) == 5:
            mag_expr, error_expr, mag_missing, error_missing, mag_type = template
        else:
            mag_expr, error_expr, mag_missing, error_missing, mag_type, error_type = template

        cols_mag = {}
        cols_error = {}
        for col in cols:
            key_mag = re.findall(r'^('+mag_expr+r')$', col)
            key_error = re.findall(r'^('+error_expr+r')$', col)

            if len(key_mag):
                key_mag = key_mag[0]
                if isinstance(key_mag, str):
                    cols_mag[0] = key_mag
                elif isinstance(key_mag, tuple):
                    col, key = key_mag
                    cols_mag[key] = col

                missing_values[mag_missing].append(col)

            if len(key_error):
                key_error = key_error[0]
                if isinstance(key_error, str):
                    cols_error[0] = key_error
                elif isinstance(key_error, tuple):
                    col, key = key_error
                    cols_error[key] = col

                missing_values[error_missing].append(col)

        assert len(cols_error), "error: no columns found for template {} {}".format(template, cols)
        assert len(cols_error) == len(cols_mag), "error: len(cols_error) != len(cols_mag) {}".format(template)

        for key in cols_mag.keys():
            interested_columns.append(
                [cols_mag[key], cols_error[key], mag_missing, error_missing, mag_type, error_type]
            )

    prog['missing_values'] = dict(missing_values)
    prog['interested_columns'] = interested_columns
    return prog


def _missing2nan(data, columns, missing_value, strict=False):
    """
    Place NaN instead default values in panstarrs data

    :param data: pandas DataFrame
    :param columns: list of columns names to seek and change default values in
    :param missing_value: value that will be changed on NaN in columns specified in columns parameter
    :param strict: if True then raises exception if there is colunm in columns that abcents in dataframe

    :returns: pandas.DataFrame
    """
    df = data.copy()
    cols = [col for col in columns if col in df.columns]
    if strict and len(cols) != len(columns):
        raise Exception("These columns are not in data:", [col for col in columns if col not in data.columns])

    if len(cols) != 0:
        mask = df[cols] == missing_value
        df[mask] = np.NaN

    return df


# =======================
# COUNTERPARTS PROCESSING
# =======================
def _process_counterparts_min_error(src, interested_columns):
    data = src.copy().reset_index(drop=True)
    dst = dict()
    for col in interested_columns:
        mag_col = col[0]
        err_col = col[1]
        err_typ = col[-1]

        if err_typ == 'i':
            idx = data[err_col].idxmax()
        else:
            idx = data[err_col].idxmin()

        if np.isnan(idx):
            dst[err_col] = [data[err_col][0]]
            dst[mag_col] = [data[mag_col][0]]
        else:
            dst[err_col] = [data[err_col][idx]]
            dst[mag_col] = [data[mag_col][idx]]

    data = pd.DataFrame.from_dict(dst, orient='columns')
    # return data.rename(columns={col: col+'_min_error' for col in data.columns}).iloc[0]
    return data.iloc[0]


def _process_counterparts_mean(data):
    data = pd.DataFrame(data.mean(axis=0)).T.rename(columns={col: col+'_mean' for col in data.columns}).iloc[0]
    return data


def _process_counterparts_max_s2n(src, interested_columns):
    data = src.copy().reset_index(drop=True)
    dst = dict()
    for col in interested_columns:
        mag_col = col[0]
        err_col = col[1]
        err_typ = col[-1]

        err = data[err_col]**(-0.5) if err_typ == 'i' else data[err_col]
        s2n = data[mag_col]/err
        idx = s2n.idxmax()

        if np.isnan(idx):
            dst[err_col] = [data[err_col][0]]
            dst[mag_col] = [data[mag_col][0]]
        else:
            dst[err_col] = [data[err_col][idx]]
            dst[mag_col] = [data[mag_col][idx]]

    data = pd.DataFrame.from_dict(dst, orient='columns')
    data.index = src.index[0:1]
    # return data.rename(columns={col: col+'_min_error' for col in data.columns}).iloc[0]
    return data.iloc[0:1]


def _process_counterparts_helper(data, cols, other_cols, counterpart_definition, interested_columns):
    _, data = data  # data is tuple(key, pandas.DataFrame) from pandas.DataFrame.groupby
    ra_col, dec_col = counterpart_definition
    if pd.isna(data[ra_col]).all():
        result = data[other_cols].iloc[0:1]
        counterparts_number = 0
        single_counterpart = False
        counterpart_type = 'NotFound'

    elif len(data) == 1:
        counterparts_number = 1
        single_counterpart = True
        counterpart_type = 'Single'
        result = data[other_cols].iloc[0:1]
        result[cols] = _process_counterparts_max_s2n(data[cols], interested_columns)[cols]

    elif (data[ra_col] == data.iloc[0][ra_col]).all() and (data[dec_col] == data.iloc[0][dec_col]).all():
        counterparts_number = len(data)
        single_counterpart = True
        counterpart_type = 'Duplicate'
        result = data[other_cols].iloc[0:1]
        result[cols] = _process_counterparts_max_s2n(data[cols], interested_columns)[cols]

    else:
        counterparts_number = len(data)
        single_counterpart = False
        counterpart_type = 'Several'
        result = data[other_cols].iloc[0:1]
        result[cols] = _process_counterparts_max_s2n(data[cols], interested_columns)[cols]

    result['counterparts_number'] = counterparts_number
    result['single_counterpart'] = single_counterpart
    result['counterparts_type'] = counterpart_type


    return result


def process_counterparts(prog, data=None, njobs=1):
    # TODO missing 2 nan
    if prog['input'] != '-':
        data, dropped_cols = _read_table(prog['input'])
    else:
        dropped_cols = []
        
    prog['dropped_cols'] = dropped_cols
    prog = _parse_program(prog, data.columns)
    if prog['input'] != '-' and prog['output'] != '-':
        pprint(prog)

    for missing_value, cols in prog['missing_values'].items():
        data = _missing2nan(data, cols, missing_value)

    cols = list()
    for col in prog['interested_columns']:
        cols += [col[0], col[1]]

    other_cols = [col for col in data.columns if col not in cols]

    with Pool(min(njobs, len(data))) as p:
        helper = functools.partial(
            _process_counterparts_helper,
            cols=cols,
            other_cols=other_cols,
            counterpart_definition=prog['counterpart_definition'],
            interested_columns=prog['interested_columns']
        )
        data = data.groupby(by=prog['object_definition'])
        data = list(
            tqdm(
                p.imap(helper, data),
                total=len(data),
                desc="Process Counterparts",
                leave=False
            )
        )
        try:
            data = pd.concat(data, sort=False)
        except:
            for idx, row in data.iterrows():
                print(len(row), row['counterparts_type'])
            raise Exception()
    
    if prog['output'] != '-':
        _write_table(data, prog['output'])
    else:
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument(
        '-p', '--prog', type=str,
        help='json file with "program". See above'
    )
    parser.add_argument(
        '-i', '--input', type=str, default="",
        help='json file with "program". See above'
    )
    parser.add_argument(
        '-o', '--output', type=str, default="",
        help='json file with "program". See above'
    )
    parser.add_argument(
        '-g', '--generate_template', type=str, default='',
        help='Specify path to save template of "program" in. Must be a directory'
    )
    parser.add_argument(
        '-n', '--njobs', type=int, default=1,
        help='Number of jobs'
    )

    args = parser.parse_args()

    if args.generate_template:
        assert os.path.isdir(args.generate_template), "{} is not a directory".format(args.generate_template)

        path = os.path.join(args.generate_template, 'template.json')
        with open(path, 'w') as fout:
            json.dump(program_template(), fout, indent=4, sort_keys=False)
            print('template saved to {}'.format(path))

    else:
        with open(args.prog) as fin:
            prog = json.load(fin)

        if args.input:
            prog['input'] = args.input

        if args.output:
            prog['output'] = args.output

        process_counterparts(prog, args.njobs)

