# %%
from scipy.special import chdtr
import numpy as np
import cupy as cp


__all__ = ['cov_3D_array_gpu_M', 'cov_3D_array_cpu_M',
           'MS_3D_array_gpu', 'MS_2D_array_gpu',
           'MS_3D_array_cpu', 'MS_2D_array_cpu']

"""
20230413
************************************
計算目標 (計算基本單元) : 
MS_3D_array_gpu : Emipical distribution
MS_3D_array_cpu : Emipical distribution
MS_2D_array_gpu : Sample Statisric, p-value
MS_2D_array_cpu : Sample Statisric, p-value

************************************
為了降低資料在 device 與 host 間傳輸之時間
刪去 ~3D_array_gpu & ~3D_array_gpu 中設輸入為 cp.array 或 np.array 的指令 
(~2D_array 沒有更動)

假設原始輸入的資料就符合檔案要求
GPU 時 輸出與輸入的為 cp.array
CPU 時 輸出與輸入的為 np.array

************************************
以下所有 MS 函數 return 的 CV 或 cv 都是 (n/6) * b_{1, p} 的型態，
而不是 b_{1, p}的型態。

"""


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

    # del n
    # del m1
    # n = None
    # m1 = None
    return cp.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)


def cov_3D_array_cpu_M(X_cpu, unbias_cov=False):
    """
    ---------------------------------------------
    Inputs :
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
    Returns :
    covariance matrix : np.array
        shape -- (N, p, p)
    """
    n = X_cpu.shape[1]

    m1 = X_cpu - cp.sum(X_cpu, 1, keepdims=True) / n  # (N, n, p)

    # del n
    # del m1
    # n = None
    # m1 = None
    return np.transpose(m1, (0, 2, 1))  @  m1 / (n - unbias_cov)


# %%

def MS_3D_array_gpu(X):
    """
    Mardia's Test for skewness simulation N times.
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Input :
    X : cp.array
        shape -- (N, n, p), i.e.  (sample_size, trails, dimension of distrition)
    ---------------------------------------------
    Returns : 
    b_1p * n / 6 : cp.array
        MS test statistic
        shape -- (N,)
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # print(mempool.used_bytes())              # 8478401536
    # print(mempool.total_bytes())             # 11897600512
    # print(pinned_mempool.n_free_blocks())    # 1

    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    S = cov_3D_array_gpu_M(X, unbias_cov=False)

    # pinv of S, return shape = (N, p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (N, n, p)
    difT = X - cp.sum(X, axis=1, keepdims=True) / n

    del X
    X = None

    # shape = (N, n, n) 應改為 (N, )
    Gij = difT  @  S_inv @ cp.transpose(difT, (0, 2, 1))

    del S_inv
    del difT

    S_inv = None
    difT = None

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MS
    # b_1p / 6 .~ chiq(df)
    # p-value of skewness to the right
    # (N, n, n) --> (N, )
    b_1p = cp.sum(Gij ** 3, axis=(1, 2)) / n**2

    del Gij
    Gij = None

    b_1p = cp.sort(b_1p)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return b_1p * n / 6


def MS_2D_array_gpu(X, unbias_cov=False):
    """
    Mardia's Test for skewness 
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Input : 
    X : cp.array
        shape -- (n, p), 
        i.e.  (sample size, dimension of distrition)
    ---------------------------------------------
    Returns :
    b_1p * n / 6 : np.array
        MS test statistic
        scalar
    pval_MS : np.array
        p-values
        scalar
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    X = cp.asarray(X)
    n, p = X.shape

    S = cp.cov(X, rowvar=False, bias=not unbias_cov)

    # pinv of S, return shape = (p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (n, p)
    difT = X - cp.mean(X, axis=0, keepdims=True)

    del X
    X = None

    # shape = (n, n)
    Gij = difT  @  S_inv @ difT.T

    del S_inv
    del difT

    S_inv = None
    difT = None

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MS
    # b_1p / 6 .~ chiq(df)
    # p-value of skewness to the right
    # (n, n) --> (1,)
    b_1p = cp.sum(Gij ** 3) / n**2

    del Gij
    Gij = None

    b_1p = cp.asnumpy(b_1p)

    if n > 50:
        g = (n*b_1p) / 6  # MS test statistics ~ chi2(v)
    else:
        K = ((p+1)*(n+1)*(n+3))/(n*(((n+1)*(p+1))-6))  # correction for n>=50
        g = (n*b_1p*K) / 6  # for small sample

    v = p * (p+1) * (p+2) / 6  # degree of freedom
    # 我修正後的 : 注意 chdtr() 變數放的位置
    pval_MS = 1 - chdtr(v, g)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return b_1p * n / 6, pval_MS


def MS_3D_array_cpu(X):
    """
    Mardia's Test for skewness simulation N times
    Parameters
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Input : 
    X : np.array
        shape -- (N, n, p), i.e.  (trails, sample sizes, dimension of distrition)
    # alpha : float
    #     Significant level
    ---------------------------------------------
    Return :
    b_1p * n / 6: np.array
        MS test statistic
        shape -- (N,)
    """
    X = np.array(X)
    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    S = cov_3D_array_cpu_M(X, unbias_cov=False)

    # pinv of S, return shape = (N, p, p)
    S_inv = np.linalg.pinv(S).astype(X.dtype)

    # del S
    # S = None

    # (N, n, p)
    difT = X - np.sum(X, axis=1, keepdims=True) / n

    # del X
    # X = None

    # shape = (N, n, n) 應改為 (N, )
    Gij = difT  @  S_inv @ np.transpose(difT, (0, 2, 1))

    # del S_inv
    # del difT

    # S_inv = None
    # difT = None

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MS
    # b_1p / 6 .~ chiq(df)
    # p-value of skewness to the right
    # (N, n, n) --> (N, )
    b_1p = np.sort(np.sum(Gij ** 3, axis=(1, 2)) / n**2)

    return b_1p * n / 6


def MS_2D_array_cpu(X, unbias_cov=False):  # , alpha=0.05
    """
    Mardia's Test for skewness
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 13, 2023.
    ---------------------------------------------
    Input :
    X : np.array
        shape -- (n, p), 
        i.e.  (sample size, dimension of distrition)
    ---------------------------------------------
    Returns :
    b_1p * n / 6: np.array
        MS test statistic
        scalar
    pval_MS : np.array
        p-values
        scalar
    """
    X = np.asarray(X)
    n, p = X.shape

    S = np.cov(X, rowvar=False, bias=not unbias_cov)

    # pinv of S, return shape = (p, p)
    S_inv = np.linalg.pinv(S).astype(X.dtype)

    del S
    S = None

    # (n, p)
    difT = X - np.mean(X, axis=0, keepdims=True)

    del X
    X = None

    # shape = (n, n)
    Gij = difT  @  S_inv @ difT.T

    del S_inv
    del difT

    S_inv = None
    difT = None

    # P-value
    # 由數學計算得到(林楷崙 2023/03/12)：
    # for MS
    # b_1p / 6 .~ chiq(df)
    # p-value of skewness to the right
    # (n, n) --> (1,)
    b_1p = np.sum(Gij ** 3) / n**2

    del Gij
    Gij = None

    if n > 50:
        g = (n*b_1p) / 6  # MS test statistics ~ chi2(v)
    else:
        K = ((p+1)*(n+1)*(n+3))/(n*(((n+1)*(p+1))-6))  # correction for n>=50
        g = (n*b_1p*K) / 6  # for small sample

    # 我修正後的 : 注意 chdtr() 變數放的位置
    v = p*(p+1)*(p+2)/6  # degree of freedom
    pval_MS = 1 - chdtr(v, g)

    return b_1p * n / 6, pval_MS


# %%

if __name__ == '__main__':
    # 　CDF of chi_square
    # import scipy
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    from scipy.stats import chi2

    # function 測試 -------------------------------------------------
    p = 3
    sample_size = 20
    N = 10000

    # new X_gpu.shape = (N, n, p)
    X_gpu = cp.random.multivariate_normal(
        mean=cp.zeros(p), cov=cp.eye(p), size=(N, sample_size))

    # uniform 測不出來， power==0
    # sample_size 小，分配資料放normal，power 都不好

    # X_gpu = np.random.standard_normal(size=(N, sample_size, p))
    # X_gpu = np.random.chisquare(df=10, size=(N, sample_size, p))
    # X_gpu = np.random.f(dfnum=5, dfden=14, size=(N, sample_size, p))

    # b_1p, pval_MS, power_MS = MS_3D_array_cpu(X=X_gpu, alpha=0.05)
    b_1p = MS_3D_array_gpu(X=X_gpu)
    print('cv = {}'.format(np.sort(b_1p)[int(N*0.05)]))

    print('MS:')
    print(b_1p.shape)
    # print(pval_MS[:10])
    # print(power_MS)
    print()

    # ECDF -------------------------------------------------
    sample_sizes = [10, 20, 30, 50, 100, 150, 200]

    b_1p = np.zeros((N, len(sample_sizes)))  # (N, n)
    Y = np.arange(1, N+1) / N

    p = 3
    df = p * (p+1) * (p+2) / 6
    # chi_cdf = lambda x: scipy.stats.chi2.cdf(x, df=df)

    for i, n in enumerate(sample_sizes):

        sample_size = n

        X_gpu = np.random.multivariate_normal(
            mean=np.zeros(p), cov=np.eye(p), size=(N, sample_size))
        # X_gpu = np.random.standard_normal(size=(N, sample_size, p))
        # X_gpu = np.random.chisquare(df=10, size=(N, sample_size, p))
        # X_gpu = np.random.f(dfnum=5, dfden=14, size=(N, sample_size, p))
        X_gpu = cp.array(X_gpu)
        print(i)
        b_1p_tmp = MS_3D_array_gpu(X=X_gpu)
        b_1p[:, i] = np.sort(cp.asnumpy(b_1p_tmp))  # * n / 6

        color = cm.rainbow(np.linspace(0, 1, n))
        plt.plot(b_1p[:, i], Y, label='ECDF, n = {}'.format(n))

    plt.plot(b_1p[:, len(sample_sizes)-1],
             chi2.cdf(x=b_1p[:, len(sample_sizes)-1], df=df),
             label='CDF'.format(int(df)), color='black',
             lw=3, linestyle='--', alpha=0.5)

    plt.legend(loc='lower right')
    plt.title('MS Test for testing $N_{}(0, 1)$'.format(p))
    plt.show()

    # MS 統計量的分布
    p = 3
    n = 20
    N = 10000
    MS = cp.zeros(N)
    mean = cp.zeros(p)
    Cov = cp.eye(p)
    for i in range(N):
        X = cp.random.multivariate_normal(mean, Cov, size=n)
        MS[i] = MS_2D_array_gpu(X)[0]

    plt.hist(MS.get())
    plt.show()

    X = cp.array([[1, 2], [3, 4], [-1, 1], [0, 4], [5, 6], [-2, 2]])
    MS, _ = MS_2D_array_gpu(X)
    print(MS)  # 答案為:0.784739014261637

# %%
