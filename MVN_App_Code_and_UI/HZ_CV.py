# %%
from HZ_function import HZ_3D_array_gpu, HZ_3D_array_cpu
import numpy as np
import cupy as cp
import time


"""
20230525
************************************
計算目標 (一次算完) : 
HZ_CV_gpu : Emipical distribution, Emipical critical value 
HZ_CV_cpu : Emipical distribution, Emipical critical value

************************************
若維度與樣本數夠小, 可以一次計算完成時, 不用迴圈一次算完
若維度與樣本數太大, 不可以一次計算完成時, 用迴圈分批跑完

在所設的 dims = [2,3,4,5,7,10,20] 中只要 sample_size>50 就會 OutOfMemory
所以設只要 sample_size > 50, 就用迴圈把 N=10萬 分割為 5 等分
"""

__all__ = ['HZ_CV_gpu', 'HZ_CV_cpu']


# %%
def HZ_CV_gpu(N, N_num_interval, sample_size, dim, alpha, H0='MVN'):
    """
    The function compute HZ test's empirical critical value under set level alpha.
    The test gets critival value by taking right tail probability 
    of HZ test statistic's empirical distribition.

    It is recommended that N be set N large or equal than 10,000.

    Limit : If you are using NVidia RTX 3080 12GB vision GPU as a computational 
            tool, the dimension of the input data is restricted to between 2 and 20,
            because of the availability of the empirical critical values. 
            The sample size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on May 25, 2023.
    ---------------------------------------------
    Inputs : 
    N : int
        times of Monte-Carlo simulation, 10000 is suggested to be the minimum.
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
    HZ_CV : np.array 
        shape -- (N, )
        the HZ test statistic's empirical distribution. 
    HZ_cv :  np.array
        scalar
        empirical critical value 取右尾, 即從右邊數來第 N*alpha 個
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    N_per_for_loop = int(N/N_num_interval)
    HZ_CV = np.zeros((N))

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=cp.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=0.9*cp.ones((dim, dim))+0.1*cp.eye(dim), size=(N, sample_size))

    HZ_stat = np.zeros(N)

    if sample_size > 50:  # 模擬資料過大, 用迴圈ㄊㄧ
        # <up-- 特化過 20230526 # 不要更動 ***
        for k in range(N_num_interval):
            HZ_stat[k*N_per_for_loop:(k+1)*N_per_for_loop] = HZ_3D_array_gpu(
                X=X_gpu[k*N_per_for_loop:(k+1)*N_per_for_loop]).get()

    else:  # 一次算完
        HZ_stat = HZ_3D_array_gpu(X=X_gpu).get()

    del X_gpu
    X_gpu = None

    HZ_CV = np.sort(HZ_stat)
    HZ_cv = HZ_CV[-np.ceil(N*alpha).astype('int')]

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return HZ_CV, HZ_cv


def HZ_CV_cpu(N, N_num_interval, sample_size, dim, alpha, H0='MVN'):
    """
    The function compute HZ test critical values under set level alpha.
    The test generates critival values by taking two equil tail probability 
    of level alpha of ECDF, i.e. under each tail contains alpha/2 probability.

    It is recommended that N be set N large or equal than 10,000.

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
        times of Monte-Carlo simulation, 10000 is suggested to be the minimum.
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
    HZ_CV : np.array 
        shape -- (N, len(sample_sizes), len(dims))
        the HZ test statistic's empirical distribution. 
    HZ_cv :  np.array
        scalar
        empirical critical value 取右尾, 即從右邊數來第N*alpha個
    """
    N_per_for_loop = int(N/N_num_interval)
    HZ_CV = np.zeros((N))

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=0.9*np.ones((dim, dim))+0.1*np.eye(dim), size=(N, sample_size))

    HZ_stat = np.zeros(N)

    if sample_size > 50:  # 模擬資料過大, 用迴圈
        for k in range(N_num_interval):
            HZ_stat[k*N_per_for_loop:(k+1)*N_per_for_loop] = HZ_3D_array_cpu(
                X=X_cpu[k*N_per_for_loop:(k+1)*N_per_for_loop])

    else:  # 一次算完
        HZ_stat = HZ_3D_array_cpu(X=X_cpu)

    del X_cpu
    X_cpu = None

    HZ_CV = np.sort(HZ_stat)
    HZ_cv = HZ_CV[-np.ceil(N*alpha).astype('int')]

    return HZ_CV, HZ_cv


# %%
if __name__ == '__main__':
    # paras setting
    paras = {
        'N': 10**5,
        'N_num_interval': 5,  # <-- 特化過 20230526 # 不要更動 ***
        # [10,  20 ,  30 ,  50, 100, 130,  150, 160][::-1],  #
        'sample_size': 100,
        'dim': 20,  # [ 2 , 3 , 4, 5, 10, 15, 20][::-1],   #
        'alpha': 0.05,
        'H0': 'MVN'
    }

    dims = paras['dim']  # [::-1]
    sample_sizes = paras['sample_size']  # [::-1]

    # compute HZ CV
    time_start = time.time()
    HZ_CV, HZ_cv = HZ_CV_gpu(**paras)
    print(time.time() - time_start)

    print(HZ_CV[-int(len(HZ_CV)*paras['alpha'])])
    print(HZ_cv)
