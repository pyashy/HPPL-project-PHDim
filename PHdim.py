from scipy.spatial.distance import cdist
import numpy as np
import cupy as cp
from numba import jit, njit
import multiprocessing as mp
import time

from joblib import Parallel, delayed

import threading
from queue import Queue
from itertools import repeat

import warnings
warnings.filterwarnings("ignore")

class PHD():
    
    def __init__(
        self, 
        alpha=1.0, 
        dist='cdist',
        metric='euclidean', 
        n_reruns=5, 
        n_points=15, 
        mst_method_name: str = 'classic',
        n_workers: int = 2,
        verbose: int = 0
    ):
        '''
        Initializes the instance of PH-dim estimator
        Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen
        lower than the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations 
        4) n_points --- Number of subsamples to be drawn at each subsample
        '''

        if dist not in ['cdist', 'numpy', 'numba', 'cupy']:
            raise ValueError("dist should be cdist, cupy, numpy or numba")

        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.dist = dist
        self.metric = metric
        self.is_fitted_ = False
        self.mst_method_name = mst_method_name
        self.verbose = verbose

        method = {
            'classic': self.get_mst_value,
            'multiprocessing': self.get_mp_mst_value,
            'numba': self.get_nb_mst_value,
            'cupy': self.get_cp_mst_value,
            'joblib': self.get_jl_mst_value,
            'threading': self.get_thread_mst_value
        }

        if mst_method_name not in method.keys():
            raise ValueError(f'Method {mst_method_name} is not implemented')
        self.mst_method = method[mst_method_name]
        self.n_workers = n_workers

    def _generate_samples(self, dist_mat, min_points):
        n = dist_mat.shape[0]
        test_n = np.linspace(min_points, n * 0.9, self.n_points).astype(int)
        random_indices = []
        for i in np.repeat(test_n, self.n_reruns):
            random_indices.append(np.random.choice(n, size=i, replace=False))
        return random_indices, test_n
    
    def _prim_tree(self, adj_matrix, ids, alpha=1.0):
    
        adj_matrix = adj_matrix[np.ix_(ids,ids)]

        infty = np.max(adj_matrix) + 10

        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty

            v = np.argmin(dst)
            s += (adj_matrix[v][ancestor[v]] ** alpha)

        return s.item()
    
    @jit
    def _nb_prim_tree(self, adj_matrix, ids, alpha=1.0):
    
        adj_matrix = adj_matrix[np.ix_(ids,ids)]

        infty = np.max(adj_matrix) + 10

        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty

            v = np.argmin(dst)
            s += (adj_matrix[v][ancestor[v]] ** alpha)

        return s
    
    def _cp_prim_tree(self, adj_matrix, ids, alpha=1.0):
    
        adj_matrix = adj_matrix[cp.ix_(ids,ids)]

        infty = cp.max(adj_matrix) + 10

        dst = cp.ones(adj_matrix.shape[0]) * infty
        visited = cp.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -cp.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = cp.minimum(dst, adj_matrix[v])
            dst[visited] = infty

            v = cp.argmin(dst)
            s += (adj_matrix[v][ancestor[v]] ** alpha)

        return s.item()
    
    
    def _mp_prim_tree(self, adj_matrix, ids, alpha=1.0):
    
        adj_matrix = adj_matrix[np.ix_(ids,ids)]

        infty = np.max(adj_matrix) + 10

        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty

            v = np.argmin(dst)
            s += (adj_matrix[v][ancestor[v]] ** alpha)
        return_dict[l] = s.item()
    
    def _thread_prim_tree(self, q, alpha=1.0):
        while not q.empty():
            adj_matrix, ids, return_dict, l = q.get()
            adj_matrix = adj_matrix[np.ix_(ids,ids)]
            infty = np.max(adj_matrix) + 10
            dst = np.ones(adj_matrix.shape[0]) * infty
            visited = np.zeros(adj_matrix.shape[0], dtype=bool)
            ancestor = -np.ones(adj_matrix.shape[0], dtype=int)
            v, s = 0, 0.0
            for i in range(adj_matrix.shape[0] - 1):
                visited[v] = 1
                ancestor[dst > adj_matrix[v]] = v
                dst = np.minimum(dst, adj_matrix[v])
                dst[visited] = infty

                v = np.argmin(dst)
                s += (adj_matrix[v][ancestor[v]] ** alpha)
            return_dict[l] = s.item()
            q.task_done()
    
    def get_mst_value(self, random_indices, dist_mat):
        mst_values = np.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._prim_tree(dist_mat, ids)
        return mst_values
    
    def get_mp_mst_value(self, random_indices, dist_mat):
        with mp.Pool(self.n_workers) as pool:
            mst_values = pool.starmap(self._prim_tree, zip(repeat(dist_mat), random_indices))
        mst_values = np.zeros(len(random_indices))
        return np.array(mst_values)

    def get_cp_mst_value(self, random_indices, dist_mat):
        mst_values = cp.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._cp_prim_tree(dist_mat, ids)
        return mst_values.get()
    
    def pairwise_distance_matrix(self, points):
        squared_distances = np.sum(points ** 2, axis=1, keepdims=True) 
        squared_distances = squared_distances + np.sum(points ** 2, axis=1) 
        squared_distances = squared_distances - 2 * np.dot(points, points.T)
        distance_matrix = np.sqrt(np.maximum(squared_distances, 0))
        return distance_matrix

    def pairwise_distance_matrix_cp(self, points):
        squared_distances = cp.sum(points ** 2, axis=1, keepdims=True) 
        squared_distances = squared_distances + cp.sum(points ** 2, axis=1) 
        squared_distances = squared_distances - 2 * cp.dot(points, points.T)
        distance_matrix = cp.sqrt(cp.maximum(squared_distances, 0))
        return distance_matrix

    @jit
    def pairwise_distance_matrix_nb(self, points):
        squared_distances = np.sum(points ** 2, axis=1, keepdims=True) 
        squared_distances = squared_distances + np.sum(points ** 2, axis=1) 
        squared_distances = squared_distances - 2 * np.dot(points, points.T)
        distance_matrix = np.sqrt(np.maximum(squared_distances, 0))
        return distance_matrix
    
    @jit
    def get_nb_mst_value(self, random_indices, dist_mat):
        mst_values = np.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._nb_prim_tree(dist_mat, ids)
        return mst_values
    
    def get_jl_mst_value(self, random_indices, dist_mat):
        def compute_mst(ids):
            return self._prim_tree(dist_mat, ids)

        mst_values = Parallel(n_jobs=self.n_workers)(delayed(compute_mst)(ids) for ids in random_indices)
        return np.array(mst_values)
    
    def get_thread_mst_value(self, random_indices, dist_mat):
        mst_values = np.zeros(len(random_indices))
        jobs = Queue()
        for i, ids in enumerate(random_indices):
            jobs.put((dist_mat, ids, mst_values, i))
        for i in range(self.n_workers):
            worker = threading.Thread(target=self._thread_prim_tree, args=(jobs,))
            worker.start()
        jobs.join()
        return np.array(mst_values)

    def fit_transform(self, X, y=None, min_points = 50):
        '''
        Computing the PH-dim 
        Parameters:
        1) X --- point cloud of shape (n_points, n_features), 
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        '''

        start_time_total = time.perf_counter()

        if self.dist == 'numpy':
            if self.metric == 'euclidean':
                dist_mat = self.pairwise_distance_matrix(X)
            else: ValueError(f'metric {self.metric} not implemented')
        elif self.dist == 'numba':
            if self.metric == 'euclidean':
                dist_mat = self.pairwise_distance_matrix_nb(X)
            else: ValueError(f'metric {self.metric} not implemented')
        elif self.dist == 'cupy':
            if self.metric == 'euclidean':
                dist_mat = self.pairwise_distance_matrix_cp(cp.asarray(X)).get()
                cp.cuda.Device(0).synchronize() # this is required for correct time measurement
            else: ValueError(f'metric {self.metric} not implemented')
        else:
            dist_mat = cdist(X, X, metric=self.metric)

        elapsed_time_cdist = time.perf_counter() - start_time_total
        if self.verbose == 1:
            print(f"Time taken by cdist: {elapsed_time_cdist} seconds")

        random_indices, x = self._generate_samples(dist_mat, min_points)
        
        # Measure the time for the loop
        start_time_loop = time.perf_counter()
        ##### HERE IS THE ONLY LOOP WE NEED TO SPEED UP #####
        mst_values = self.mst_method(random_indices, dist_mat)

        elapsed_time_loop = time.perf_counter() - start_time_loop
        if self.verbose == 1:
            print(f"Time taken by loop: {elapsed_time_loop} seconds")
            
        y = mst_values.reshape(-1, self.n_reruns).mean(axis = 1)
        
        x = np.log(x)
        y = np.log(y)
        N = self.n_points
        
        m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

        # Record the end time for the entire algorithm
        end_time_total = time.perf_counter()
        total_time = end_time_total - start_time_total

        # Calculate the percentage of time spent in cdist relative to the total time
        percentage_time_cdist = (elapsed_time_cdist / total_time) * 100

        if self.verbose == 1:
            print(f"Total time for the algorithm: {total_time} seconds")
            print(f"Percentage time spent in cdist: {percentage_time_cdist:.2f}%")
            print(f"Percentage time spent in the loop: {(elapsed_time_loop / total_time) * 100:.2f}%")

        return 1 / (1 - m)
    
