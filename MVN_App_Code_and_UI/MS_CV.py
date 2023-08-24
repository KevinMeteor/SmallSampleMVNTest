# %%
# from MS_function_0314 import MS_3D_array_gpu
from MS_function import MS_3D_array_gpu, MS_3D_array_cpu
import numpy as np
import cupy as cp
import time


"""
20230601
************************************
計算目標 (一次算完) : 
MS_CV_gpu : Emipical distribution, Emipical critical value 
MS_CV_cpu : Emipical distribution, Emipical critical value

************************************
若維度與樣本數夠小, 可以一次計算完成時, 不用迴圈一次算完
若維度與樣本數太大, 不可以一次計算完成時, 用迴圈分批跑完

在所設的 dims = [2,3,4,5,7,10,20] 中只要 sample_size>90 GPU 就會 OutOfMemory
所以設只要 sample_size > 80, 就用迴圈把 N=10萬 分割為 2 等分

"""


__all__ = ['MS_CV_gpu', 'MS_CV_cpu']

# %%


def MS_CV_gpu(N, N_num_interval, sample_size, dim, alpha, H0='MVN'):
    """
    The function compute MS test's critical values under set level alpha.
    The test generates the empirical critival value by taking right tail
    probability of empirical distribotion of MS test statistic.

    It is recommended that N be set N large or equal than 50,000.

    Limit : If you are using NVidia RTX 3080 12GB vision GPU as a computational 
            tool, the dimension of the input data is restricted to between 2 and 20,
            because of the availability of the empirical critical values. 
            The sample size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Inputs : 
    N : int
        times of Monte-Carlo simulation, 50000 is suggested to be the minimum.
    N_num_interval : int
        how many are intervals the N been segmented for for loop to compute.
    sample_size : int
    dim : int
        dimenstion of H0's distribution
    alpha : float
        significant level of the test, suggested and defaulted as 0.05
    H0 : str
        the type of H0 data put, defaulting two types, i.e.
        'MVN' : standard MVN -- N_p(0, I), where p denotes p dimensional distrition.
        'D_MVN' : 老師所用的 H0 : MVN with mean vector 0, and
            covariance matrix with 1's diagonal and 0.9's off-diagonal. 
    ---------------------------------------------
    Returns :
    MS_CV : np.array 
        shape -- (N,)
        the MS test statistic's empirical distribution. 
    MS_cv : np.array
        scalar
        empirical critical value 要取右尾, 即從右邊數來第 N * alpha 個
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    N_per_for_loop = int(N/N_num_interval)

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=cp.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=0.9*cp.ones((dim, dim))+0.1*cp.eye(dim), size=(N, sample_size))

    MS_CV = np.zeros(N)

    if dim <= 20 and sample_size <= 80:  # 一次算完 20230601
        MS_CV = MS_3D_array_gpu(X=X_gpu).get()

    else:  # 模擬資料過大, 用迴圈
        for k in range(N_num_interval):
            MS_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = MS_3D_array_gpu(
                X=X_gpu[k*N_per_for_loop:(k+1)*N_per_for_loop]).get()

    del X_gpu
    X_gpu = None

    MS_CV = np.sort(MS_CV)
    MS_cv = MS_CV[-np.ceil(N*alpha).astype('int')]  # 取右尾 cv

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return MS_CV, MS_cv


def MS_CV_cpu(N, N_num_interval, sample_size, dim, alpha, H0='MVN'):
    """
    The function compute MS test's critical values under set level alpha.
    The test generates the empirical critival value by taking right tail
    probability of empirical distribotion of MS test statistic.

    It is recommended that N be set N large or equal than 50,000.

    Limit : If you are using NVidia RTX 3080 12GB vision GPU as a computational 
            tool, the dimension of the input data is restricted to between 2 and 20,
            because of the availability of the empirical critical values. 
            The sample size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Inputs : 
    N : int
        times of Monte-Carlo simulation, 50000 is suggested to be the minimum.
    N_num_interval : int
        how many are intervals the N been segmented for for loop to compute.
    sample_size : int
    dim : int
        dimenstion of H0's distribution
    alpha : float
        significant level of the test, suggested and defaulted as 0.05
    H0 : str
        the type of H0 data put, defaulting two types, i.e.
        'MVN' : standard MVN -- N_p(0, I), where p denotes p dimensional distrition.
        'D_MVN' : 老師所用的 H0 : MVN with mean vector 0, and
            covariance matrix with 1's diagonal and 0.9's off-diagonal. 
    ---------------------------------------------
    Returns :
    MS_CV : np.array 
        shape -- (N,)
        the MS test statistic's empirical distribution. 
    MS_cv :  np.array
        scalar
        empirical critical value 要取右尾, 即從右邊數來第 N*alpha 個
    """
    N_per_for_loop = int(N/N_num_interval)

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=0.9*np.ones((dim, dim))+0.1*np.eye(dim), size=(N, sample_size))

    MS_CV = np.zeros(N)

    if dim <= 20 and sample_size <= 80:  # 一次算完 20230601
        MS_CV = MS_3D_array_cpu(X=X_cpu)

    else:
        for k in range(N_num_interval):
            MS_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = MS_3D_array_cpu(
                X=X_cpu[k*N_per_for_loop:(k+1)*N_per_for_loop])

    del X_cpu
    X_cpu = None

    MS_CV = np.sort(MS_CV)
    MS_cv = MS_CV[-np.ceil(N*alpha).astype('int')]  # 取右尾 cv

    return MS_CV, MS_cv


# %%
if __name__ == '__main__':
    # paras setting
    paras = {
        'N': 10**5,
        'N_num_interval': 2,  # 不要動 20230601
        'sample_size': 100,
        'dim': 20,   #
        'alpha': 0.05,
        'H0': 'MVN'
    }

    # compute MS CV

    MS_cv = MS_CV_gpu(**paras)

    dim = paras['dim']  # [::-1]
    sample_size = paras['sample_size']  # [::-1]

    # compute SK CV
    time_start = time.time()
    MS_CV, MS_cv = MS_CV_gpu(**paras)
    print('MS gpu time : {}'.format(time.time() - time_start))
    # print(MS_cv)
