"""Functions for post-processing material test data. (Stress-strain)"""
import copy
from functools import wraps
from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

from src.paramat_test.data_chief import DataItem


def process_data(dataitem: DataItem, cfg: Dict):
    """ Apply processing functions to a datafile object. """
    processing_operations = [
        store_initial_indices,
        trim_using_sampling_rate,
        trim_using_max_force,
        trim_initial_cluster,
        # shift_using_initial_points,
        # shift_using_yield_stress,
        # correct_for_compliance,
        # correct_for_friction,
        # correct_for_ringing,
        # correct_for_thermal_expansion
    ]

    for proc_op in processing_operations:
        if proc_op.__name__ in cfg['operations']:
            print(f'{".": <10}Running {proc_op.__name__}.')
            dataitem = proc_op(dataitem)
    return dataitem


def processing_function(function: Callable[[DataItem], DataItem]):
    """ Applies function to dataitem then returns it. Just returns dataitem if any exception raised. """

    @wraps(function)
    def wrapper(dataitem: DataItem):
        try:
            return function(dataitem)
        except TypeError as e:
            print(e)
            return dataitem

    return wrapper


@processing_function
def store_initial_indices(dataitem):
    df = dataitem.data
    dataitem.info_row['raw data indices'] = (df['Time(sec)'].idxmin(), df['Time(sec)'].idxmax())
    return dataitem


@processing_function
def trim_using_sampling_rate(dataitem):
    df = dataitem.data
    mindex = df['Time(sec)'].diff().diff().idxmin() + 1
    maxdex = df['Time(sec)'].diff().diff().idxmax() - 1
    if df[mindex:maxdex].empty:
        dataitem.data.reset_index()
        return dataitem
    dataitem.info_row['sampling rate trim indices'] = (mindex, maxdex)
    dataitem.data = df[mindex:maxdex].reset_index(drop=False)
    return dataitem


@processing_function
def trim_using_max_force(dataitem):
    df = dataitem.data
    maxdex = df['Force(kN)'].idxmax()
    dataitem.data = df[:maxdex].reset_index(drop=True)
    try:
        dataitem.info_row['max force trim indices'] = (df['index'][0], df['index'][maxdex])
    except KeyError:
        df = dataitem.data
        dataitem.info_row['max force trim indices'] = (df['Time(sec)'].idxmin(), df['Time(sec)'].idxmax())
    return dataitem


@processing_function
def trim_initial_cluster(dataitem):
    # fig, ax = plt.subplots(1,1)
    # test_df = copy.deepcopy(dataitem.data)
    # test_df.plot(ax=ax, x='Strain', y='Stress(MPa)')
    ####
    eps, min_samples = 3, 8
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df = copy.deepcopy(dataitem.data)
    strain = df['Strain'].values.reshape(-1, 1)
    stress = df['Stress(MPa)'].values.reshape(-1, 1)
    X = np.hstack([strain, stress])
    yhat = model.fit_predict(X)
    clusters = np.unique(yhat)
    initial_cluster = clusters[1]
    row_ix = np.where(yhat == initial_cluster)
    clusters = np.unique(yhat)
    row_ixs = list([np.where(yhat == cluster) for cluster in clusters])
    min_cluster_idx = pd.Series([np.average(rix) for rix in row_ixs]).idxmin()
    remove_row_ixs = row_ixs[min_cluster_idx]
    mindex = max(remove_row_ixs)[-1]
    dataitem.data = df[mindex:].reset_index(drop=True)
    try:
        dataitem.info_row['cluster trim indices'] = (df['index'][mindex], df['index'].iloc[-1])
    except KeyError:
        df = dataitem.data
        dataitem.info_row['cluster trim indices'] = (df['Time(sec)'].idxmin(), df['Time(sec)'].idxmax())
    return dataitem


@processing_function
def calculate_rate_and_smoothed_rate(dataitem):
    df = dataitem.data
    info = dataitem.info_row
    df['rate'] = np.hstack([[0], np.diff(df['Strain']) / np.diff(df['Time(sec)'])])
    df['smoothed rate'] = savgol_filter(df['rate'], 20, 2)
    dataitem.data = df
    return dataitem


@processing_function
def shift_using_initial_points(dataitem):
    return dataitem


@processing_function
def find_yield_stress(dataitem) -> DataItem:
    # # todo: fit log function to data
    # # todo: find average slope
    # # todo: find where slope first drops below average slope
    # # todo: find nearest neighbour
    #
    # start_point = (strain[0], stress[0])
    # end_point = (strain[-1], stress[-1])
    #
    # average_slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    #
    # log_func = lambda x, a, c: a * np.log(x) + c
    #
    # # todo: make predicted data
    # # todo: find gradient of log_func-predicted data
    # # todo: compare gradients, find yield stress
    # # todo: find value and index of nearest neighbour
    #
    # print(op.curve_fit(log_func, strain, stress))
    # popt, pcov = op.curve_fit(log_func, strain, stress)
    # processing_params['finding yield stress fitted log function params'] = popt
    return dataitem


@processing_function
def shift_using_yield_stress(dataitem) -> DataItem:
    # processing_params = find_yield_stress(processing_params, data['Strain'], data['Stress(MPa)'])
    # todo: get yieldstress from dataitem
    # # todo: fit line between first point and yield stress
    # # todo: find x-intercept of fitted line
    # # todo:
    # return processing_params, data
    return dataitem


@processing_function
def separate_elastic_and_inelastic_data(dataitem) -> DataItem:
    # processing_params = find_yield_stress(processing_params, data['Strain'], data['Stress(MPa)'])
    # return processing_params, data
    return dataitem


@processing_function
def correct_for_compliance(dataitem) -> DataItem:
    # # todo: get stress as function of load
    # # todo: get compliance as function of load
    # # todo: compliance-corrected stress as function of load
    # # todo: store
    return dataitem


@processing_function
def correct_for_friction(dataitem) -> DataItem:
    return dataitem


@processing_function
def correct_for_ringing(dataitem) -> DataItem:
    return dataitem


@processing_function
def correct_for_thermal_expansion(dataitem) -> DataItem:
    return dataitem
