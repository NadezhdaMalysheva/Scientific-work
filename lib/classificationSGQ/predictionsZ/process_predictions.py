"""
Script for calculating things for photo-z predictions (see docs for process_predictions)

=========
CHANGELOG
=========

=== 20Feb2020 ===
- Added arguments bw_method and z_conf_neighbourhood
=== 25Feb2020 ===
- Evenly spaced grid -> evenly spaced grid + samples
- In z_conf, shift neighborhood of point if it cant fit in interval [min(grid), max(grid)]
- Plot distributions using plotly #TODO
"""
import functools
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm.autonotebook import tqdm


def conf_int_type1(pdf, grid, confidence_level=0.68, eps=None):
    """
    Calculates confidence intervals as interval that has confidence_level square under pdf in it
    :param pdf: numpy.ndarray, shape (grid length), with pdf values calculated in grid points
    :param grid: numpy.ndarray, shape (grid length), grid itself
    :param confidence_level: float, confidence level
    :param eps: float, controls precision. Precision may not be achieved if length of grid is too small (not used here)
    :return: numpy.ndarray, shape(1, 2), where each row contains left and right border (as index of grid element) of
        each confidence interval respectively
    """
    idx_mp = np.argmax(pdf)

    pdf_left = pdf[:idx_mp + 1].copy()
    grid_left = grid[:idx_mp + 1]
    if idx_mp > 0:
        pdf_left /= np.trapz(pdf_left, x=grid_left)

    pdf_right = pdf[idx_mp:].copy()
    grid_right = grid[idx_mp:]
    if idx_mp < len(pdf) - 1:
        pdf_right /= np.trapz(pdf_right, x=grid_right)

    cia_idx = []

    for step, start, border, p, x in zip(
            [-1, 1], [idx_mp, 0], [0, len(pdf_right) - 1], [pdf_left, pdf_right], [grid_left, grid_right]
    ):
        integral = 0
        idx = start
        while integral < confidence_level:
            idx += step
            if step * idx > border:
                break

            integral += p[idx - int(step == 1)] * (x[idx] - x[idx - 1])

        cia_idx.append(idx + idx_mp - start - step)

    return np.array(cia_idx).reshape(1, -1)


def conf_int_type2(pdf, grid, confidence_level=0.68, eps=0.0001):
    """
    Calculates confidence intervals as interval predictions
    :param pdf: numpy.ndarray, shape (grid length), with pdf values calculated in grid points
    :param grid: numpy.ndarray, shape (grid length), grid itself
    :param confidence_level: float, confidence level
    :param eps: float, controls precision. Precision may not be achieved if length of grid is too small
    :return: numpy.ndarray, shape(number of confidence intervals found, 2), where each row contains left and right
        border (as index of grid element) of each confidence interval respectively
    """

    def find_border(arr):
        borders = np.where(arr[1:] != arr[:-1])[0] + 1
        if arr[0]:
            borders = np.insert(borders, 0, 0)

        if arr[-1]:
            borders = np.append(borders, arr.shape[0])

        return borders.reshape(-1, 2)

    a1 = pdf.min()
    a2 = pdf.max()
    conf_ints_idxs = None
    a1_last = a1
    a2_last = a2
    for step in range(int(1 / eps)):
        alfa = (a1 + a2) / 2
        conf_ints_idxs = find_border(pdf > alfa)
        integral = 0
        for ii1, ii2 in zip(conf_ints_idxs[:, 0], conf_ints_idxs[:, 1]):
            integral += np.trapz(pdf[ii1:ii2], x=grid[ii1:ii2])

        if abs(integral - confidence_level) < eps:
            conf_ints_idxs[:, 1] -= 1
            return conf_ints_idxs

        if integral > confidence_level:
            a1 = alfa
        if integral < confidence_level:
            a2 = alfa

        if a1 == a1_last and a2 == a2_last:
            conf_ints_idxs[:, 1] -= 1
            return conf_ints_idxs
        else:
            a1_last, a2_last = a1, a2

    conf_ints_idxs[:, 1] -= 1
    return conf_ints_idxs


def z_conf(pdf, grid, start, dx=0.06):
    """
    Calculates zConf for point x given probability dinsity as pair (grid and pdf), where pdf is array
        of values of probability density functions in ponits of grid
    :param pdf: np.array
    :param grid: np.array
    :param start: int, index of grid point to calculate zConf for
    :param dx: scalar, parameter of zConf, defines neighborhood of point x to calculate zConf
    :return: scalar, value of zConf
    """
    x = grid[start]
    idx0 = start
    idx1 = start
    dx_norm = dx * (1+x)

    # if x - grid.min() < dx_norm:
    #     x_start = grid[np.argmin(np.abs(grid - grid.min() - dx_norm))]
    # elif grid.max() - x < dx_norm:
    #     x_start = grid[np.argmin(np.abs(grid.max() - grid - dx_norm))]
    # else:
    #     x_start = x
    x_start = x

    while idx0 >= 0 and abs(x_start - grid[idx0]) <= dx_norm:
        idx0 -= 1

    while idx1 < len(grid) and abs(x_start - grid[idx1]) <= dx_norm:
        idx1 += 1

    return np.trapz(pdf[idx0:idx1], x=grid[idx0:idx1])


def distribution_features(pdf, grid):
    """
    Calculates mean, std, skewness, kurtosis and entropy for given distribution
    :param pdf: np.array, values of probability density functions in grid points (see param grid)
    :param grid: np.array
    :return: dict
    """
    mean = np.trapz(grid * pdf, x=grid)
    f = grid - mean
    std = np.trapz(f ** 2 * pdf, x=grid)
    skewness = np.trapz(f ** 3 * pdf, x=grid) / std ** (3 / 2)
    kurtosis = np.trapz(f ** 4 * pdf, x=grid) / std ** 2 - 3

    mask = pdf > 0
    entropy = np.trapz(pdf[mask] * np.log(pdf[mask]), x=grid[mask])

    return {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis, 'entropy': entropy}


# def _plot(idx, row, pdf, grid, prefix, path, name_columns):
#     fig = plotly.graph_objects.Figure()

#     fig.add_trace(plotly.graph_objects.Scatter(x=grid, y=pdf))

#     z_max_col = prefix+'z_max'
#     fig.add_trace(
#         plotly.graph_objects.Scatter(
#             x=[0, np.where(grid == row[z_max_col])], y=[row[z_max_col], row[z_max_col]],
#             mode='lines+markers', name=z_max_col
#         )
#     )

#     filename = str(idx)
#     if name_columns:
#         filename += ' ' + str(row[name_columns])

#     filename += '.html'

#     plotly.offline.plot(fig, filename=os.path.join(path, filename), auto_open=False)


def _process_predictions_helper(data, pred_column, bw_method, z_conf_neighbourhood, prefix,
                                plot, plot_path, name_columns):
    """
    !!! You should not use this yourself !!!
    Helper function for process_counterparts
    :param data: tuple of (int, pandas.Series) given by pandas.DataFrame.iterrows
    :param pred_column: str, column that contains predictions
    :param bw_method: float, bandwidth for kernel density estimator
    :param z_conf_neighbourhood: float, neighborhood of point to calculate zConf
    :param prefix: str, prefix for new columns names
    :return: pd.Series
    """
    idx, row = data
    try:
        confidence_levels = [68, 90, 95]

        if not isinstance(row[pred_column], np.ndarray) or len(row[pred_column]) <= 1:
            return pd.DataFrame(row).T

        sample = row[pred_column]
        pred_max = sample.max()
        pred_min = sample.min()
        r = max(pred_max - pred_min, 0.0001)
        z_grid = np.linspace(pred_min - r, pred_max + r, 1001)
        z_grid = np.sort(np.concatenate((z_grid, np.unique(sample))))

        kde = gaussian_kde(sample, bw_method=bw_method)
        z_pdf = kde(z_grid)

        z_pdf /= np.trapz(z_pdf, x=z_grid)

        zmax_idx = z_pdf.argmax()
        row[prefix + 'z_max'] = z_grid[zmax_idx]
        row[prefix + 'z_max_proba'] = z_pdf[zmax_idx]
        row[prefix + 'z_maxConf'] = z_conf(z_pdf, z_grid, zmax_idx, dx=z_conf_neighbourhood)
        mask3 = z_grid >= 2.7
        row[prefix + 'proba_z_ge_3'] = np.trapz(z_pdf[mask3], x=z_grid[mask3])

        for cl in confidence_levels:
            ci1 = conf_int_type1(z_pdf, z_grid, cl / 100)[0]
            # row[prefix + 'ci1_' + str(cl)] = tuple(z_grid[ci1])
            row[prefix + 'ci1a_' + str(cl)] = z_grid[ci1][0] - row[
                prefix + 'z_max']
            row[prefix + 'ci1b_' + str(cl)] = z_grid[ci1][1] - row[
                prefix + 'z_max']

            ci2 = conf_int_type2(z_pdf, z_grid, cl / 100)

            peaks_idx = [ci[0] + z_pdf[ci[0]:ci[1]].argmax() for ci in ci2 if ci[0] != ci[1]]
            peaks = z_grid[peaks_idx]
            peaks_proba = z_pdf[peaks_idx]
            peaks_z_conf = [z_conf(z_pdf, z_grid, idx, dx=z_conf_neighbourhood) for idx in peaks_idx]

            try:
                ci2_begin_end = [ci2[0, 0], ci2[-1, -1]]
            except:
                print(ci2)
                raise Exception()

            row[prefix + 'ci2_{}_short'.format(cl)] = tuple(z_grid[ci2_begin_end])
            row[prefix + 'ci2_{}'.format(cl)] = tuple(z_grid[ci2])
            row[prefix + 'ci2_{}_peaks'.format(cl)] = tuple(peaks)
            row[prefix + 'ci2_{}_peaks_proba'.format(cl)] = tuple(peaks_proba)
            row[prefix + 'ci2_{}_peaks_zConf'.format(cl)] = tuple(peaks_z_conf)

        feats = distribution_features(z_pdf, z_grid)
        for feat, value in feats.items():
            row[prefix + feat] = value

    #     if plot:
    #         _plot(idx, row, z_pdf, z_grid, prefix, plot_path, name_columns)

    except:
        row['ProcessPredictionsError'] = True
    else:
        row['ProcessPredictionsError'] = False

    return pd.DataFrame(row).T


def _process_predictions_helper_short(data, pred_column, bw_method, z_conf_neighbourhood, prefix,
                                plot, plot_path, name_columns):
    """
    !!! You should not use this yourself !!!
    Helper function for process_counterparts
    :param data: tuple of (int, pandas.Series) given by pandas.DataFrame.iterrows
    :param pred_column: str, column that contains predictions
    :param bw_method: float, bandwidth for kernel density estimator
    :param z_conf_neighbourhood: float, neighborhood of point to calculate zConf
    :param prefix: str, prefix for new columns names
    :return: pd.Series
    """
    idx, row = data
    try:
        if not isinstance(row[pred_column], np.ndarray) or len(row[pred_column]) <= 1:
            return pd.DataFrame(row).T

        sample = row[pred_column]
        pred_max = sample.max()
        pred_min = sample.min()
        r = max(pred_max - pred_min, 0.0001)
        z_grid = np.linspace(pred_min - r, pred_max + r, 1001)
        z_grid = np.sort(np.concatenate((z_grid, np.unique(sample))))

        kde = gaussian_kde(sample, bw_method=bw_method)
        z_pdf = kde(z_grid)

        z_pdf /= np.trapz(z_pdf, x=z_grid)

        zmax_idx = z_pdf.argmax()
        row[prefix + 'z_max'] = z_grid[zmax_idx]
        # row[prefix + 'z_max_proba'] = z_pdf[zmax_idx]
        row[prefix + 'z_maxConf'] = z_conf(z_pdf, z_grid, zmax_idx, dx=z_conf_neighbourhood)

        confidence_levels = [68, 90, 95]
        for cl in confidence_levels:
            ci1 = conf_int_type1(z_pdf, z_grid, cl / 100)[0]
            row[prefix + 'ci1a_' + str(cl)] = z_grid[ci1][0] - row[prefix + 'z_max']
            row[prefix + 'ci1b_' + str(cl)] = z_grid[ci1][1] - row[prefix + 'z_max']

        # feats = distribution_features(z_pdf, z_grid)
        # for feat, value in feats.items():
        #     row[prefix + feat] = value
    except:
        row['ProcessPredictionsError'] = True
    else:
        row['ProcessPredictionsError'] = False

    return pd.DataFrame(row).T


def process_predictions(
        data, pred_column,
        bw_method=0.1, z_conf_neighbourhood=0.06,
        prefix='_', njobs=1, progressbar=False,
        plot=False, plot_path='', name_columns=None,
        short=False, leave=False):
    """
    Calculates for every prediction in data[pred_column]:

        - z_max (float) - the most probable value of redshift for object
        - z_max_proba (float) - value of probability density function in z_max
        - z_maxConf (float) - zConf for z_max

        - ci1_68 (tuple of 2 floats) - confidence interval around z_max with confidence 0.68
        - ci1_90, ci1_95 - similarly to ci1_68

        - ci2_68 (tuple of 2 floats) - beginning of the first interval and end of the last interval in
            Highest Probability Confidence Interval (HPDI) with confidence 0.68
        - ci2_68_peaks (tuple of floats) - the most probable value of redshift in each subinterval of HPDI 0.68
        - ci2_68_peaks_proba (tuple of floats) - value of probability density function for each point in ci2_68_peaks
        - ci2_68_peaks_zConf (tuple of floats) - zConf for each point in ci2_68_peaks
        - ci2_90, ci2_90_peaks, ci2_90_peaks_proba, ci2_90_peaks_zConf -
            similarly to ci2_68, ci2_68_peaks, ci2_68_peaks_proba, ci2_68_peaks_zConf
        - ci2_95, ci2_95_peaks, ci2_95_peaks_proba, ci2_95_peaks_zConf -
            similarly to ci2_68, ci2_68_peaks, ci2_68_peaks_proba, ci2_68_peaks_zConf

        - mean (float)
        - std (float) - standard deviation
        - skewness (float)
        - kurtosis (float)
        - entropy (float) - differential entropy

    :param data: pandas.DataFrame
    :param pred_column: str, name of column which contains predictions in form of numpy.ndarray in every row
    :param bw_method: float, bandwidth for kernel density estimator
    :param z_conf_neighbourhood: float, neighborhood of point to calculate zConf
    :param prefix: str, prefix for new columns names (i.e. <with default value of parameter> z_max will be "_z_max"
        ci2_68_peaks will be "_ci2_68_peaks", ci2_95_peaks_zConf will be "_ci2_95_peaks_zConf" and etc.)
    :param njobs: int, number of jobs
    :param progressbar: bool, displays progressbar, if True
    :param plot:
    :param plot_path:
    :param name_columns:

    :return: pandas.DataFrame with both old data and new calculated columns
    """
    assert isinstance(data, pd.DataFrame), "data must be pandas.DataFrame"
    assert pred_column in data.columns, "specified pred_column '{}' absents in data".format(pred_column)
    assert isinstance(prefix, str), "prefix must be string"
    assert isinstance(njobs, int), "njobs must be integer"
    assert isinstance(plot_path, str), "plot_path must be string"
    assert (plot and plot_path) or not plot, "you must specify plot_path to plot graphs"
    
    if short:
        helper = _process_predictions_helper_short
    else:
        helper = _process_predictions_helper
    
    helper = functools.partial(
        helper,
        pred_column=pred_column,
        prefix=prefix, bw_method=bw_method, z_conf_neighbourhood=z_conf_neighbourhood,
        plot=plot, plot_path=plot_path, name_columns=name_columns
    )

#     if plot:
#         os.makedirs(plot_path, exist_ok=True)

    with Pool(min(njobs, len(data))) as p:
        if progressbar:
            result = list(
                tqdm(
                    p.imap(helper, data.iterrows()),
                    total=len(data),
                    leave=leave
                )
            )
        else:
            result = list(
                p.imap(helper, data.iterrows())
            )

    result = pd.concat(result, sort=False)
    return result

