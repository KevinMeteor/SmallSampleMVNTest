# %%
from Shapiro_Wilks_Roy_GPU_N import W_stat_1992_gpu, W_stat_1992_gpu_3D
from Shapiro_Wilks_Roy_CPU_N import W_stat_1992_cpu, W_stat_1992_cpu_3D  # return W
import numpy as np
import cupy as cp
# from cupyx.profiler import benchmark
# from mv_dist import multi_PearsonII


"""
20230605 : 
************************************
計算目標 (計算基本單元) : 
Wmin_m_3D_array_gpu : Emipical distribution
Wmin_m_3D_array_cpu : Emipical distribution
Wmin_m_2D_array_gpu : Sample Statisric, p-value
Wmin_m_2D_array_cpu : Sample Statisric, p-value

************************************
為了降低資料在 device 與 host 間傳輸之時間
刪去 ~3D_array_gpu & ~3D_array_gpu 中設輸入為 cp.array 或 np.array 的指令 
(~2D_array 沒有更動)

假設原始輸入的資料就符合檔案要求
GPU 時 輸出與輸入的為 cp.array
CPU 時 輸出與輸入的為 np.array

************************************
為 RTX 3080 12GB 特化

if n * m * N < 5*10**8 and N <= 10**3:  # <--20230605 特化過 不要改動
        R = W_stat_1992_gpu_3D(YL_T)

    else:
        R = cp.zeros((N, m))
        for i in range(N):
            R[i, :] = W_stat_1992_gpu_3D(YL_T[i, :, :].reshape(1, n, m))[0]

"""


__all__ = ['Wmin_m_3D_array_gpu',
           'Wmin_m_3D_array_cpu',
           'Wmin_m_2D_array_gpu',
           'Wmin_m_2D_array_cpu']


# %%


def cov_3D_array_gpu_M(X_gpu, unbias_cov=False):
    """
    ---------------------------------------------
    Inputs :
    X_gpu : cp.array  
        shape -- (N, sample_size, P)
    unbias_cov : boolean
        Whether covariance matrix is unbias or not
        Dafault: True
        If unbias = False (value: 0), use a bias estimator.
        If unbias = True  (value: 1), use an unbias estimator.
    unbias_cov : boolean
        Whether covariance matrix is unbias or not
        Dafault: True
        If unbias = False (value: 0), use a bias estimator.
        If unbias = True  (value: 1), use an unbias estimator.

    ---------------------------------------------
    Return :
    covariance matrix : cp.array
        shape -- (N, p, p)
    """
    n = X_gpu.shape[1]
    m1 = X_gpu - cp.sum(X_gpu, 1, keepdims=True) / n  # (N, n, p)

    return cp.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)


def cov_3D_array_cpu_M(X_cpu, unbias_cov=False):
    """
    ---------------------------------------------
    Inputs:
    X_cpu : np.array  
        shape -- (N, sample_size, P)
    unbias_cov : boolean
        Whether covariance matrix is unbias or not
        Dafault: True
        If unbias = False (value: 0), use a bias estimator.
        If unbias = True  (value: 1), use an unbias estimator.
    unbias_cov : boolean
        Whether covariance matrix is unbias or not
        Dafault: True
        If unbias = False (value: 0), use a bias estimator.
        If unbias = True  (value: 1), use an unbias estimator.

    ---------------------------------------------
    Return:
    covariance matrix : np.array
        shape -- (N, p, p)
    """
    n = X_cpu.shape[1]
    m1 = X_cpu - np.sum(X_cpu, 1, keepdims=True) / n  # (N, n, p)

    return np.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)


# %%alpha
def Wmin_m_3D_array_gpu(X, m, q):
    """
    This function generates the empirical q*100 th percentile of the Wmin_m(q)
    statistic from a set of W-tests, which contains m statistics W(X*c_i),
    i=1,2,...,m. c_i are m uniformly scattered points in p-dimensional uint
    sphere.
    limit: the dimension of the input data is restricted to between 2 and 10
    because of the availability of the empirical critical values. The sample
    size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on June 5, 2023.
    ---------------------------------------------
    Inputs : 
    X : cp.array
        shape -- (N, n, p), i.e.  (trails, sample size, dimension of distrition)
    m : float
        how many uniformly scattered points, 10000 is suggested to be the minimum.
    q : float
        the percentile e.g. q=0.05 for the 5th percentile. q can be a vector.
    alpha : float
        p-value of the test
    ---------------------------------------------
    Returns :
    wmin_m_CV : cp.array
        shape -- (N, )
        the wmin_m(q) test statistic from m W-tests. 
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    S = cov_3D_array_gpu_M(X, unbias_cov=False)

    # Z, shape = (N, n, p)
    # rank(S) = p 的情況較常出現，所以放前面先算
    # 是否可以把 inv 改為 pinv？ (本是 inv)
    Z = cp.transpose(cp.linalg.inv(cp.linalg.cholesky(S)) @
                     cp.transpose(X, (0, 2, 1)), (0, 2, 1))

    # if cp.sum(cp.linalg.matrix_rank(S) < p) > 0:
    #     Z[cp.linalg.matrix_rank(S) < p] = X[cp.linalg.matrix_rank(S) < p]

    not_full_rank_S_position = cp.linalg.matrix_rank(S) < p

    if cp.sum(not_full_rank_S_position < p) > 0:
        Z[not_full_rank_S_position] = X[not_full_rank_S_position]

    del not_full_rank_S_position
    del X
    del S
    not_full_rank_S_position = None
    X = None
    S = None

    # Y, shape = (N, m, p)
    Y = cp.random.multivariate_normal(
        mean=cp.zeros(p), cov=cp.eye(p), size=(N, m))

    # del mean
    # del Cov
    # mean = None
    # Cov = None

    # YL, shape = (N, m, p) @ (N, p, n) = (N, m, n)
    YL = (Y / cp.tile(cp.sqrt(cp.sum(Y**2, axis=2)).reshape(N, -1, 1),
          (1, p))) @ cp.transpose(Z, (0, 2, 1))  # linear combinations

    del Z
    del Y
    Z = None
    Y = None

    YL_T = cp.transpose(YL, (0, 2, 1))

    del YL
    YL = None

    if n * m * N < 5*10**8 and N <= 10**3:  # <--20230605 為 RTX 3080 12 GB 特化過 不要改動
        R = W_stat_1992_gpu_3D(YL_T)

    else:
        R = cp.zeros((N, m))
        for i in range(N):
            R[i, :] = W_stat_1992_gpu_3D(YL_T[i, :, :].reshape(1, n, m))[0]

    del YL_T
    YL_T = None

    # shpae = (N, m)
    R = cp.sort(R, axis=1)

    # shape = (N,)
    wmin_m_CV = cp.sort(R[:, cp.ceil(m * q - 1).astype('int')])

    del R
    R = None

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return wmin_m_CV

# %%


def Wmin_m_2D_array_gpu(X, m=10000, q=0.05):
    '''
    This function generates the empirical q*100 th percentile of the Wmin_m(q)
    statistic from a set of W-tests, which contains m statistics W(X*c_i),
    i=1,2,...,m. c_i are m uniformly scattered points in p-dimensional uint
    sphere.
    limit: the dimension of the input data is restricted to between 2 and 10
    because of the availability of the empirical critical values. The sample
    size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on March 28, 2023.
    ---------------------------------------------
    Inputs :
        X: multivariate data with dimension in [2 10].
        m: how many uniformly scattered points, 10000 is suggested to be the minimum.
        q: the percentile e.g. q=0.05 for the 5th percentile. q can be a vector.
    ---------------------------------------------
    Return :
        wmin_m : the wmin_m(q) sample statistic from m W-tests. 
    '''
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    n, p = X.shape
    X = cp.asarray(X)
    # Standardize input data X into a matrix with identity covariance matrix
    S = cp.cov(X, rowvar=False, bias=False)

    if cp.linalg.matrix_rank(S) == p:
        Z = (cp.linalg.inv(cp.linalg.cholesky(S)) @ X.T).T
    else:
        Z = X  # ill-conditioned, no standardization

    del S
    del X
    S = None
    X = None

    Y = cp.random.multivariate_normal(mean=cp.zeros(p), cov=cp.eye(p), size=m)

    # linear combinations
    YL = (Y / cp.tile(cp.sqrt(cp.sum(Y**2, axis=1)).reshape(-1, 1), (1, p))) @ Z.T

    del Y
    del Z
    Y = None
    Z = None

    # for the sake of insufficient memory,use loop.
    if n * m < 1e08:  # 多加上 N
        wmin_m = W_stat_1992_gpu(YL.T)  # (1 x m) W test stats
    else:
        wmin_m = cp.zeros(1, m)
        for i in range(m):
            wmin_m[i] = W_stat_1992_gpu(YL[i, :])

    del YL
    YL = None

    wmin_m = cp.sort(wmin_m)[cp.ceil(m * q - 1).astype('int')]
    wmin_m = cp.asnumpy(wmin_m)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return wmin_m


def Wmin_m_3D_array_cpu(X, m, q):
    """
    This function generates the empirical q*100 th percentile of the Wmin_m(q)
    statistic from a set of W-tests, which contains m statistics W(X*c_i),
    i=1,2,...,m. c_i are m uniformly scattered points in p-dimensional uint
    sphere.
    limit: the dimension of the input data is restricted to between 2 and 10
    because of the availability of the empirical critical values. The sample
    size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on June 5, 2023.
    ---------------------------------------------
    Inpust : 
    X : np.array
        shape -- (N, n, p), i.e.  (trails, sample size, dimension of distrition)
    m : float
        how many uniformly scattered points, 10000 is suggested to be the minimum.
    q : float
        the percentile e.g. q=0.05 for the 5th percentile. q can be a vector.
    ---------------------------------------------
    Return :
    wmin_m : np.array
        shape -- (N, )
        the wmin_m(q) test statistic from m W-tests. 

    """
    X = np.asarray(X)
    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    S = cov_3D_array_cpu_M(X, unbias_cov=False)

    if n > p:
        pass
    else:
        # 20230615 增修
        # np.linalg.cholesky(S) : 需要 S > 0 , 但若 n<=p 則會有 S 的 eigenvalue < 0,
        # i.e., S 非正定會使 np.linalg.cholesky(S) 算不了, 網路上的解決方法如下 :
        # (cp.linalg.cholesky(S) 還是可以算, 不受  n<=p  影響)
        # But since the matrix may be positive SEMI-definite due to rank deficiency
        # we must regularize.
        # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy

        S = S + np.eye((p)) * 1e-14

    # Z, shape = (N, n, p)
    # rank(S) = p 的情況較常出現，所以放前面先算
    # 是否可以把 inv 改為 pinv？ (本是 inv)
    Z = np.transpose(np.linalg.inv(np.linalg.cholesky(S)) @
                     np.transpose(X, (0, 2, 1)), (0, 2, 1))

    not_full_rank_S_position = np.linalg.matrix_rank(S) < p
    # if np.sum(np.linalg.matrix_rank(S) < p) > 0:
    #     Z[np.linalg.matrix_rank(S) < p] = X[np.linalg.matrix_rank(S) < p]

    if np.sum(not_full_rank_S_position < p) > 0:
        Z[not_full_rank_S_position] = X[not_full_rank_S_position]

    del not_full_rank_S_position
    del X
    del S
    X = None
    S = None
    not_full_rank_S_position = None

    # Y, shape = (N, m, p)
    Y = np.random.multivariate_normal(
        mean=np.zeros(p), cov=np.eye(p), size=(N, m))

    # YL, shape = (N, m, p) @ (N, p, n) = (N, m, n)
    YL = (Y / np.tile(np.sqrt(np.sum(Y**2, axis=2)).reshape(N, -1, 1),
                      (1, p))) @ np.transpose(Z, (0, 2, 1))  # linear combinations

    del Z
    del Y
    Z = None
    Y = None

    YL_T = np.transpose(YL, (0, 2, 1))

    del YL
    YL = None

    if n * m * N < 5*10**8 and N <= 10**3:  # <--20230605 特化過 不要改動
        # R shape = (N, m)
        R = W_stat_1992_cpu_3D(YL_T)

    else:
        R = np.zeros((N, m))
        for i in range(N):
            R[i, :] = W_stat_1992_cpu_3D(YL_T[i, :, :].reshape(1, n, m))[0]

    del YL_T
    YL_T = None

    # shape = (N,)
    wmin_m_CV = np.sort(R, axis=1)[:, np.ceil(m * q - 1).astype('int')]

    del R
    R = None

    return wmin_m_CV


def Wmin_m_2D_array_cpu(X, m=10000, q=0.05):
    '''
    This function generates the empirical q*100 th percentile of the Wmin_m(q)
    statistic from a set of W-tests, which contains m statistics W(X*c_i),
    i=1,2,...,m. c_i are m uniformly scattered points in p-dimensional uint
    sphere.
    limit: the dimension of the input data is restricted to between 2 and 10
    because of the availability of the empirical critical values. The sample
    size is suggested to be bewteen [10 200]. 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on March 28, 2023.
    ---------------------------------------------
    Inputs :
        X: multivariate data with dimension in [2 10].
        m: how many uniformly scattered points, 10000 is suggested to be the minimum.
        q: the percentile e.g. q=0.05 for the 5th percentile. q can be a vector.
    ---------------------------------------------
    Return :
        wmin_m : the wmin_m(q) sample statistic from m W-tests. 
    '''
    n, p = X.shape
    X = np.asarray(X)
    # Standardize input data X into a matrix with identity covariance matrix
    S = np.cov(X, rowvar=False, bias=False)

    if np.linalg.matrix_rank(S) == p:
        Z = (np.linalg.inv(np.linalg.cholesky(S)) @ X.T).T
    else:
        Z = X  # ill-conditioned, no standardization

    del S
    del X
    S = None
    X = None

    Y = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=m)

    # linear combinations
    YL = (Y / np.tile(np.sqrt(np.sum(Y**2, axis=1)).reshape(-1, 1), (1, p))) @ Z.T

    del Y
    del Z
    Y = None
    Z = None

    # for the sake of insufficient memory,use loop.
    if n * m < 1e08:  # 多加上 N
        wmin_m = W_stat_1992_cpu(YL.T)  # (1 x m) W test stats
    else:
        wmin_m = np.zeros(1, m)
        for i in range(m):
            wmin_m[i] = W_stat_1992_cpu(YL[i, :])

    del YL
    YL = None

    wmin_m = np.sort(wmin_m)[np.ceil(m * q - 1).astype('int')]

    return wmin_m


# %%

if __name__ == '__main__':
    import scipy
    import numpy as np
    # # 2D ---------------------------------------------------------
    # X = np.array([[1,2], [3, 4], [-1, 1], [0, 4], [5, 6], [-2,2]])
    # W_stat = Wmin_m_2D_array_gpu(X)
    # print('Answer is: ')
    # print(W_stat) # 答案為:0.7974759107738356 / 0.798107090252974/ etc.

    # 3D --------------------------------------------------------
    N = 100
    m = 10000
    q = 0.05
    alpha = 0.05

    n = 20
    p = 20

    N_per_for_loop = 100

    N_divived = int(N/N_per_for_loop)

    # 我用的 H0
    X_cpu = np.random.multivariate_normal(
        mean=np.zeros(p), cov=np.eye(p), size=(N, n))

    # 老師所用的 H0
    # X_cpu = np.random.multivariate_normal(
    #     mean=np.zeros(p), cov=0.9*np.ones((p,p))+0.1*np.eye(p), size=(N, n))

    import time
    time_start = time.time()
    Wmin_m_3D_array_cpu(np.array(X_cpu), m, q)
    print('time cost = {}'.format(time.time()-time_start))

    # Wmin_CV = np.zeros(N)

    # for k in range(N_divived):
    #     Wmin_CV[k*N_per_for_loop: (k+1)*N_per_for_loop] = Wmin_m_3D_array_gpu(
    #         X=cp.array(X_cpu)[k*N_per_for_loop: (k+1)*N_per_for_loop], m=m, q=q).get()

    # Wmin_cv = np.sort(Wmin_CV)[np.ceil(N*alpha - 1).astype('int')]

    # print(Wmin_CV)
    # print('Cretical value undel level alpha: ')
    # print(Wmin_cv) # 0.8987334255458839 / 0.898822545253657 / etc.

# %%
