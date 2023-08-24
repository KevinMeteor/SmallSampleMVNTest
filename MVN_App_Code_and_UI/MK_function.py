# %%
from scipy.special import ndtr
import numpy as np
import cupy as cp
# import cupyx

"""
20230524 
************************************
計算目標 (計算基本單元) : 
MK_3D_array_gpu : Emipical distribution
MK_3D_array_cpu : Emipical distribution
MK_2D_array_gpu : Sample Statisric, p-value
MK_2D_array_cpu : Sample Statisric, p-value

************************************
為了降低資料在 device 與 host 間傳輸之時間
刪去 ~3D_array_gpu & ~3D_array_gpu 中設輸入為 cp.array 或 np.array 的指令 
(~2D_array 沒有更動)

假設原始輸入的資料就符合檔案要求
GPU 時 輸出與輸入的為 cp.array
CPU 時 輸出與輸入的為 np.array
"""


__all__ = ['cov_3D_array_gpu_M', 'cov_3D_array_cpu_M',
           'MK_3D_array_gpu', 'MK_3D_array_cpu',
           'MK_2D_array_gpu', 'MK_2D_array_cpu']


def cov_3D_array_gpu_M(X_gpu, unbias_cov=False):
    """
    ---------------------------------------------
    Input:
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
    Return:
    covariance matrix : cp.array
        shape -- (N, p, p)
    """
    n = X_gpu.shape[1]
    # print(cp._environment._get_preload_logs())
    m1 = X_gpu - cp.sum(X_gpu, 1, keepdims=True) / n  # (N, n, p)

    Cov = cp.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)
    # del n
    # n = None
    del m1
    m1 = None

    return Cov


def cov_3D_array_cpu_M(X_cpu, unbias_cov=False):
    """
    ---------------------------------------------
    Input:
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

    # del n
    # del m1
    # n = None
    # m1 = None
    return np.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)


# %%
def MK_3D_array_gpu(X):
    """
    Mardia's Test for kurtosis simulation N times
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on March 28, 2023.
    ---------------------------------------------
    Input : 
    X : cp.array
        shape -- (N, n, p), i.e.  (sample_size, trails, dimension of distrition)
    alpha : float
        Significant level
    ---------------------------------------------
    Returns :
    b_2p : np.array
        MK test statistic
        shape -- (N,)
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # X = cp.asarray(X)  # 20230524 拿掉
    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    # S is the biased here
    S = cov_3D_array_gpu_M(X, unbias_cov=False)

    # pinv of S, return shape = (N, p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (N, n, p)
    difT = X - cp.sum(X, axis=1, keepdims=True) / n

    del X
    X = None

    # shape = (N, n, n) --> (N, n)
    Dj = cp.diagonal(
        difT  @  (S_inv @ cp.transpose(difT, (0, 2, 1))), axis1=1, axis2=2)

    del difT
    del S_inv
    difT = None
    S_inv = None

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MK
    # (N, n) --> (N, )
    b_2p_cv = cp.sort(cp.sum(Dj**2, axis=1) / n)

    del Dj
    Dj = None

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return b_2p_cv


def MK_3D_array_cpu(X):
    #
    """
    Mardia's Test for kurtosis simulation N times
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on March 28, 2023.
    ---------------------------------------------
    Input : 
    X : np.array
        shape -- (N, n, p), i.e.  (trails, sample_size, dimension of distrition)
    Returns
    ---------------------------------------------
    b_2p : np.array
        MK test statistic
        shape -- (N,)
    """
    # X = np.asarray(X)   # 20230524 拿掉
    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    # S is the biased here
    S = cov_3D_array_cpu_M(X, unbias_cov=False)

    # pinv of S, return shape = (N, p, p)
    S_inv = np.linalg.pinv(S).astype(X.dtype)

    # (N, n, p)
    difT = X - np.sum(X, axis=1, keepdims=True) / n

    # shape = (N, n, n) --> (N, n)
    Dj = np.diagonal(difT  @  S_inv @ np.transpose(difT,
                     (0, 2, 1)), axis1=1, axis2=2)

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MK
    # (N, n) --> (N, )
    b_2p_cv = np.sort(np.sum(Dj**2, axis=1) / n)

    return b_2p_cv


def MK_2D_array_gpu(X):
    """
    Mardia's Test for kurtosis simulation N times
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 10, 2023.
    ---------------------------------------------
    Input : 
    X : cp.array
        shape -- (n, p), i.e.  
        (sample_size, dimension of distrition)
    ---------------------------------------------
    Returns :
    b_2p : np.array
        MK test statistic
        shape -- (1,)
    pval_MK : np.array
        p-values
        shape -- (1,)
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    X = cp.asarray(X, dtype='float64')
    n, p = X.shape

    # Sample Covaince matrix, return shape = (p, p)
    # S is the biased here
    S = cp.cov(X, rowvar=False, bias=True)

    # pinv of S, return shape = (p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (n, p)
    difT = X - cp.mean(X, axis=0, keepdims=True)

    del X
    X = None

    # shape = (n, n) --> (n)
    Dj = cp.diag(difT  @  S_inv @ difT.T)
    b_2p = cp.average(Dj**2)

    del difT
    del S_inv
    difT = None
    S_inv = None

    del Dj
    Dj = None

    # p-value
    # 由數學計算得到(2023/03/12)：
    # for MK
    # (n,) --> (1, )
    b_2p = cp.asnumpy(b_2p)
    b_2p_asy = (b_2p - p * (p+2) * (n-1) / (n+1)) / \
        np.sqrt(8 * p * (p+2) / n)

    # 取雙尾 p-value
    pval_MK = (1 - ndtr(np.absolute(b_2p_asy))) * 2

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return b_2p, pval_MK


def MK_2D_array_cpu(X):
    """
    Mardia's Test for kurtosis
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 24, 2023.
    ---------------------------------------------
    Input : 
    X : np.array
        shape -- (n, p), i.e.  
        (sample_size, dimension of distrition)
    ---------------------------------------------
    Returns :
    b_2p : np.array
        MK test statistic
        shape -- (1,)
    pval_MK : np.array
        p-values
        shape -- (1,)
    """
    X = np.asarray(X, dtype='float64')
    n, p = X.shape

    # Sample Covaince matrix, return shape = (p, p)
    # S is the biased here
    S = np.cov(X, rowvar=False, bias=True)

    # pinv of S, return shape = (p, p)
    S_inv = np.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (n, p)
    difT = X - np.mean(X, axis=0, keepdims=True)

    del X
    X = None

    # shape = (n, n) --> (n)
    Dj = np.diag(difT  @  S_inv @ difT.T)
    b_2p = np.average(Dj**2)

    del difT
    del S_inv
    difT = None
    S_inv = None

    del Dj
    Dj = None

    # p-value
    # 由數學計算得到(2023/03/12)：
    # for MK
    # (n,) --> (1, )
    b_2p = np.asarray(b_2p)
    b_2p_asy = (b_2p - p * (p+2) * (n-1) / (n+1)) / \
        np.sqrt(8 * p * (p+2) / n)

    # 取雙尾 p-value
    pval_MK = (1 - ndtr(np.absolute(b_2p_asy))) * 2

    return b_2p, pval_MK


# %%
if __name__ == '__main__':
    # 測試 function
    p = 3
    sample_size = 20
    N = 10000

    # new X_gpu.shape = (N, n, p)
    X_cpu = np.random.multivariate_normal(
        mean=np.zeros(p), cov=np.eye(p), size=(N, sample_size))

    # uniform 測不出來， power==0
    # X_cpu = np.random.standard_normal(size=(N, sample_size, p))

    # X_cpu = np.random.chisquare(df=10, size=(N, sample_size, p))
    # X_cpu = np.random.f(dfnum=5, dfden=14, size=(N, sample_size, p))

    b_2p_cv = MK_3D_array_cpu(X=X_cpu)

    # b_2p, pval_MK, power_MK = MK_3D_array_cpu(X=X_cpu,  alpha=0.05)

    print()
    print('MK:')
    print(b_2p_cv[::10])
    # print(CV_MK_alpha_lower)
    # print(CV_MK_alpha_upper)

    # 　CDF of chi_square
    import scipy
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    from scipy.stats import norm

    # sample_sizes = [10, 20, 30, 50, 100, 150]
    sample_sizes = [100, 200, 300]

    b_2p = np.zeros((N, len(sample_sizes)))  # (N, n)
    Y = np.arange(1, N+1) / N

    # ECDF -----------------------------------------------
    df = p * (p+1) * (p+2) / 6

    for i, n in enumerate(sample_sizes):

        sample_size = n

        X_gpu = cp.random.multivariate_normal(
            mean=np.zeros(p), cov=np.eye(p), size=(N, sample_size))
        # X_gpu = cp.random.standard_normal(size=(N, sample_size, p))
        # X_gpu = cp.random.chisquare(df=10, size=(N, sample_size, p))
        # X_gpu = cp.random.f(dfnum=5, dfden=14, size=(N, sample_size, p))

        print(i)
        b_2p_tmp = MK_3D_array_gpu(X=X_gpu)
        b_2p[:, i] = np.sort(b_2p_tmp.get() - p * (p+2) *
                             (n-1) / (n+1)) / np.sqrt(8 * p * (p+2) / n)

        color = cm.rainbow(np.linspace(0, 1, n))
        plt.plot(b_2p[:, i], Y, label='ECDF, n = {}'.format(n))

    plt.plot(b_2p[:, len(sample_sizes)-1],
             norm.cdf(x=b_2p[:, len(sample_sizes)-1], ),
             label='CDF'.format(int(df)), color='black',
             lw=3, linestyle='--', alpha=0.5)

    plt.legend(loc='lower right')
    plt.title('MK Test for testing $N_{}(0, 1)$'.format(p))
    plt.show()

    print(cp.show_config())
    print(cp._environment._get_preload_logs())
    MK_2D_array_gpu(X=np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=(sample_size))
                    )
    p = 3
    n = 20
    N = 10000
    MK = cp.zeros(N)
    mean = cp.zeros(p)
    Cov = cp.eye(p)
    for i in range(N):
        X = cp.random.multivariate_normal(mean, Cov, size=n)
        MK[i], _ = MK_2D_array_gpu(X)

    plt.hist(MK.get())
    plt.show()

    X = cp.array([[1, 2], [3, 4], [-1, 1], [0, 4], [5, 6], [-2, 2]])
    MK, _ = MK_2D_array_gpu(X)
    print(MK)  # 答案為:4.504779470909761

# %%
