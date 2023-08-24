# %%
from MK_function import MK_3D_array_gpu, MK_3D_array_cpu
import numpy as np
import cupy as cp


""" 
20230524 
************************************
計算目標 (一次算完) : 
MK_CV_gpu : Emipical distribution, Emipical critical value 
MK_CV_cpu : Emipical distribution, Emipical critical value

************************************

若維度與樣本數夠小, 可以一次計算完成時, 不用迴圈一次算完
若維度與樣本數太大, 不可以一次計算完成時, 用迴圈分批跑完

# 模擬資料過大
在所設的 (dim>20) or (dim>15 and sample_size>90)就會 OutOfMemory: 
所以設只要 (dim>20) or (dim>15 and sample_size>90), 就用迴圈把 N=10萬 分割為 2 等分
"""


__all__ = ['MK_CV_gpu', 'MK_CV_cpu']


# %%
def MK_CV_gpu(N, N_num_interval, sample_size, dim, alpha=0.05, H0='MVN'):
    """
    The function compute MK test critical values under set level alpha.
    The test generates critival values by taking two equil tail probability 
    of level alpha of ECDF, i.e. under each tail contains alpha/2 probability.

    It is recommended that N be set N large or equal than 10,000.

    Limit : If you are using NVidia RTX 3080 12GB vision GPU as a computational 
            tool, the dimension of the input data is restricted to between 2 and 20,
            because of the availability of the empirical critical values. 
            The sample size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on May 24, 2023.
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
    MK_CV : np.array  # critical value 取雙尾各 alhha / 2
        shape -- (N,)
        the MK test's empirical distribution. 
    MK_cv_alpha_lower : np.array
        scalar
        the MK test lower empirical critical value under level alpha
    MK_cv_alpha_upper : np.array
        scalar
        the MK test upper empirical critical value under level alpha

    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    N_per_for_loop = int(N/N_num_interval)
    MK_CV = np.zeros((N))

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=cp.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_gpu = cp.random.multivariate_normal(
            mean=cp.zeros(dim), cov=0.9*cp.ones((dim, dim))+0.1*cp.eye(dim), size=(N, sample_size))

    MK_CV = np.zeros(N)

    if dim <= 20 and (dim <= 15 or sample_size <= 90):  # 一次算完
        MK_CV = MK_3D_array_gpu(X=X_gpu).get()

    else:  # 模擬資料過大, 用迴圈
        for k in range(N_num_interval):
            MK_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = MK_3D_array_gpu(
                X=X_gpu[k*N_per_for_loop:(k+1)*N_per_for_loop]).get()

    # 跟上面同理, 指示優先判斷適合一次計算可一次計算的參數範圍 20230524
    # if (dim>20) or (dim>15 and sample_size>90): # 模擬資料過大, 用迴圈
    #     for k in range(N_num_interval):
    #         MK_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = MK_3D_array_gpu(
    #             X=X_gpu[k*N_per_for_loop:(k+1)*N_per_for_loop]).get()

    # else: # 一次算完
    #     MK_CV = MK_3D_array_gpu(
    #         X=X_gpu).get()

    del X_gpu
    X_gpu = None

    MK_CV = np.sort(MK_CV)

    # 雙尾 CV
    # 兩側各取 alpha / 2
    MK_cv_alpha_lower = np.sort(MK_CV)[np.ceil(N*alpha/2).astype('int')]
    MK_cv_alpha_upper = np.sort(MK_CV)[np.ceil(N - N*alpha/2).astype('int')]

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return MK_CV, MK_cv_alpha_lower, MK_cv_alpha_upper


def MK_CV_cpu(N, N_num_interval, sample_size, dim, alpha=0.05, H0='MVN'):
    """
    The function compute MK test critical values under set level alpha.
    The test generates critival values by taking two equil tail probability 
    of level alpha of ECDF, i.e. under each tail contains alpha/2 probability.

    It is recommended that N be set N large or equal than 10,000.

    Limit : If you are using NVidia RTX 3080 12GB vision GPU as a computational 
            tool, the dimension of the input data is restricted to between 2 and 20,
            because of the availability of the empirical critical values. 
            The sample size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on May 24, 2023.
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
    MK_CV : np.array  # critical value 取雙尾各 alhha / 2
        shape -- (N,)
        the MK test's empirical distribution. 
    MK_cv_alpha_lower : np.array
        scalar
        the MK test lower empirical critical value under level alpha
    MK_cv_alpha_upper : np.array
        scalar
        the MK test upper empirical critical value under level alpha

    """
    N_per_for_loop = int(N/N_num_interval)
    MK_CV = np.zeros((N))

    if H0 == 'MVN':
        # 我用的 H0 : standard MVN
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.eye(dim), size=(N, sample_size))
    elif H0 == 'D_MVN':
        # 老師所用的 H0 : MVN with mean vector 0, cov 1's diagonal and 0.9's off-diagonal
        X_cpu = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=0.9*np.ones((dim, dim))+0.1*np.eye(dim), size=(N, sample_size))

    MK_CV = np.zeros(N)

    if dim <= 20 and (dim <= 15 or sample_size <= 90):
        MK_CV = MK_3D_array_cpu(X=X_cpu)

    else:  # 模擬資料過大, 用迴圈
        for k in range(N_num_interval):
            MK_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = MK_3D_array_cpu(
                X=X_cpu[k*N_per_for_loop:(k+1)*N_per_for_loop])

    MK_CV = np.sort(MK_CV)

    # 雙尾 CV
    # 兩側各取 alpha / 2
    MK_cv_alpha_lower = np.sort(MK_CV)[np.ceil(N*alpha/2).astype('int')]
    MK_cv_alpha_upper = np.sort(MK_CV)[np.ceil(N - N*alpha/2).astype('int')]

    return MK_CV, MK_cv_alpha_lower, MK_cv_alpha_upper


# %%
if __name__ == '__main__':
    # paras setting
    import time
    paras = {
        'N': 10**5,
        'N_num_interval': 1,
        'sample_size': 40,  # [10,  20 ,  30 ,  50, 100, 130,  150, 160][::-1],
        'dim': 3,  # [ 2 , 3 , 4, 5, 10, 15, 20][::-1],
        'alpha': 0.05,
        'H0': 'MVN'
    }

    dim = paras['dim']  # [::-1]
    sample_size = paras['sample_size']  # [::-1]

    # compute MK CV
    time_start = time.time()
    MK_CV, cv_lower, cv_upper = MK_CV_gpu(**paras)
    print('MK gpu time : {}'.format(time.time() - time_start))
    # print(MK_CV)
    print(cv_lower)
    print(cv_upper)
