import itertools
from typing import Tuple, Dict
from collections.abc import Iterable

from tqdm.auto import tqdm
import numpy as np
import time
import pandas as pd
# import cupy as cp
import skdim
from scipy.spatial.distance import cdist

import yaml

from PHdim import PHD


def read_config(path: str) -> Dict:
    with open(path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return data

def create_data(
        n_points: int = 1000, 
        space_dim: int = 10,
        sphere_dim: int = 5,
        random_state: int = 42,
        sphere_radius: int = 1,
        method: str = 'classic',
):  
    if method == 'cupy':
        X = cp.zeros((n_points, space_dim))
    else:
        X = np.zeros((n_points, space_dim))

    X[:,:sphere_dim] = skdim.datasets.hyperBall(
        n = n_points, 
        d = sphere_dim, 
        radius = sphere_radius, 
        random_state=random_state
    )
    return X

def get_param_grid(params_list):
    list_list_params = []
    for el in params_list:
        if not isinstance(el, Iterable) or isinstance(el, str) :
            list_list_params.append([el])
        else:
            list_list_params.append(el)
    
    print(list_list_params)
    grid = itertools.product(*list_list_params)
    return grid

def multirun(n_runs, func, params):
    times = np.zeros(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        func(*params)
        end = time.perf_counter()
        times[i] = end - start
    return np.mean(times), np.std(times)

def run_experiment(
        n_rerun_time,
        n_space_points, 
        space_dim, 
        n_subsample_points, 
        n_reruns_algo,
        method,
    ):
    time = 0

    X = create_data(n_space_points, space_dim, method=method)
    phd = PHD(n_reruns=n_reruns_algo, n_points=n_subsample_points, mst_method_name=method)
    time_mean, time_std = multirun(n_rerun_time, phd.fit_transform, [X])
    return time_mean, time_std 

def main():
    cfg =  read_config(path='config.yaml')

    params_list = [cfg['n_space_points'], cfg['space_dim'], cfg['n_subsample_points'], cfg['n_reruns_algo'], cfg['method']]
    param_grid = get_param_grid(params_list)

    res = list() 
    col_names = ['n_space_points', 'space_dim', 'n_subsample_points', 'n_reruns_algo', 'method', 'n_rerun_time', 'time_mean', 'time_std']
    for params in tqdm(param_grid):
        print(params)
        time_mean, time_std = run_experiment(cfg['n_rerun_time'], *params)
        res.append([*params, cfg['n_rerun_time'], time_mean, time_std])


    df_res = pd.DataFrame(res, columns=col_names)
    df_res.to_csv(f"results/{cfg['experiment_name']}.csv", index=False)


if __name__ == '__main__':
    main()
