# PHD optimization final project

Repo for final project of High Perfomance Python Lab course in Skoltech '23

## Idea

The focus of the project is Persistence Homology Dimension (PHdim) algorithm described in [1]. The goal of the project is to analyse differrent methods (joblib, multiprocessing, threading, Numba, CuPy) to optimize and speedup the algorithm.



## Repository contents

| File or Folder | Content |
| --- | --- |
| results | the folder contains results of the project|
| config.yaml | config file with parameters of the experiments |
| pipeline.py |  python-file with scripts of running all experiments |
| plots.ipynb | jupyter notebook for drawing plots |
| requirements.txt | file with all necessary packages |

## Results

The cupy method for pair distance optimization and joblib package for prim tree paralleling perform the best.

## Contacts

| **Name** | **Telegram** |
|----:|:----------:|
| Petr Sokerin | @Petr_Sokerin |
| Kristian Kuznetsov | @pyashy |
| Alexander Yugay | @AleksandrY99 |
| Irena Gureeva | @thdktgdk |



[1] Tulchinskii, E., Kuznetsov, K., Laida, K., Cherniavskii, D., Nikolenko, S., Burnaev, E., Barannikov, S., & Piontkovskaya, I. (2023). Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts. Thirty-Seventh Conference on Neural Information Processing Systems.
