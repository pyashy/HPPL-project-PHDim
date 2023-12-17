from scipy.spatial.distance import cdist
import numpy as np
import cupy as cp
from numba import jit
import multiprocessing as mp

from joblib import Parallel, delayed


class PHD():
    
    def __init__(
        self, 
        alpha=1.0, 
        metric='euclidean', 
        n_reruns=5, 
        n_points=15, 
        mst_method_name: str = 'classic'):
        '''
        Initializes the instance of PH-dim estimator
        Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen
        lower than the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations 
        4) n_points --- Number of subsamples to be drawn at each subsample
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.metric = metric
        self.is_fitted_ = False

        method = {
            'classic': self.get_mst_value,
            'multiprocessing': self.get_mp_mst_value,
            'numba': self.get_nb_mst_value,
            'cupy': self.get_cp_mst_value,
            'joblib': self.get_jl_mst_value
        }

        if mst_method_name not in method.keys():
            raise ValueError(f'Method {mst_method_name} is not implemented')
        self.mst_method = method[mst_method_name]

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
    
    
    def _mp_prim_tree(self, adj_matrix, ids, return_dict, l, alpha=1.0):
    
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
        return_dict[l] = s.item()
    
    def get_mst_value(self, random_indices, dist_mat):
        mst_values = np.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._prim_tree(dist_mat, ids)
        return mst_values
    
    def get_mp_mst_value(self, random_indices, dist_mat):
        
        mst_values = np.zeros(len(random_indices))
        # num_workers = mp.cpu_count()  
        manager = mp.Manager()
        return_dict = manager.dict()
        pool = mp.Pool(2)
        for i, ids in enumerate(random_indices):
            pool.apply_async(self._mp_prim_tree, args = (dist_mat, ids, return_dict, i))
        pool.close()
        pool.join()
        
        for k in return_dict.keys():
            mst_values[k] = return_dict[k]
        
        return mst_values

    def get_cp_mst_value(self, random_indices, dist_mat):
        mst_values = cp.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._cp_prim_tree(dist_mat, ids)
        return mst_values
    
    @jit
    def get_nb_mst_value(self, random_indices, dist_mat):
        mst_values = np.zeros(len(random_indices))
        for i, ids in enumerate(random_indices):
            mst_values[i] = self._prim_tree(dist_mat, ids)
        return mst_values
    
    def get_jl_mst_value(self, random_indices, dist_mat, n_jobs=-1):
        def compute_mst(ids):
            return self._prim_tree(dist_mat, ids)

        mst_values = Parallel(n_jobs=n_jobs)(delayed(compute_mst)(ids) for ids in random_indices)
        return np.array(mst_values)

    def fit_transform(self, X, y=None, min_points = 50):
        '''
        Computing the PH-dim 
        Parameters:
        1) X --- point cloud of shape (n_points, n_features), 
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        '''
        dist_mat = cdist(X, X, metric=self.metric)
        random_indices, x = self._generate_samples(dist_mat, min_points)
        
        
        ##### HERE IS THE ONLY LOOP WE NEED TO SPEED UP #####
        mst_values = self.mst_method(random_indices, dist_mat)
            
        y = mst_values.reshape(-1, self.n_reruns).mean(axis = 1)
        
        x = np.log(x)
        y = np.log(y)
        N = self.n_points
        
        m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
        return 1 / (1 - m)