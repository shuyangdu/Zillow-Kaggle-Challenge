from __future__ import print_function

import pandas as pd


def get_hyper_params(file_path, n=1, fixed_params={}):
    """
    Get best hyper parameters from search result csv
    :param file_path: file name of the search result
    :param n: number of best hyper parameter sets 
    :param fixed_params: fixed parameter to be added
    :return: list of dictionaries for hyper parameters
    """
    search_results = pd.read_csv(file_path, index_col=0)
    params_lst = []
    for i in range(n):
        params = search_results.iloc[i, :-2].to_dict()  # the last 2 element are loss and status
        params.update(fixed_params)
        params_lst.append(params)
    return params_lst
