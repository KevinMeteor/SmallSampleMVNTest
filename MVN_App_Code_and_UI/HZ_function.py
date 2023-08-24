# %%
import numpy as np
import cupy as cp
import cupyx
import scipy

"""
20230525 
************************************
計算目標 (計算基本單元) : 
MS_3D_array_gpu : Emipical distribution
MS_3D_array_cpu : Emipical distribution
MS_2D_array_gpu : Sample Statisric, p-value
MS_2D_array_cpu : Sample Statisric, p-value

************************************
相較於 HZ_function.py,
為了降低資料在 device 與 host 間傳輸之時間
刪去 ~3D_array_gpu & ~3D_array_gpu 中設輸入為 cp.array 或 np.array 的指令 
(~2D_array 沒有更動)

假設原始輸入的資料就符合檔案要求
GPU 時 輸出與輸入的為 cp.array
CPU 時 輸出與輸入的為 np.array

"""


__all__ = ['cov_3D_array_gpu_new', 'cov_3D_array_cpu_new',
           'HZ_3D_array_gpu', 'HZ_3D_array_cpu',
           'HZ_2D_array_gpu', 'HZ_2D_array_cpu']


# %%
def cov_3D_array_gpu_new(X_gpu):
    """
    ---------------------------------------------
    Input:
    X_gpu.shape: (N, n, p)
    ---------------------------------------------
    Return:
    cov with bias(divided by N), shape (N, p, p)
    """

    n = X_gpu.shape[1]
    m1 = X_gpu - cp.sum(X_gpu, 1, keepdims=True) / n  # (N, n, p)

    # del n
    # del m1
    # n = None
    # m1 = None

    return cp.transpose(m1, (0, 2, 1))  @  m1 / n


def cov_3D_array_cpu_new(X_cpu):
    """
    ---------------------------------------------
    Input:
    X_cpu.shape: (N, n, P)
    ---------------------------------------------
    Return:
    cov with bias(divided by N), shape (N, p, p)
    """
    n = X_cpu.shape[1]
    m1 = X_cpu - np.sum(X_cpu, 1, keepdims=True) / n  # (N, n, p)

    return np.transpose(m1, (0, 2, 1))  @  m1 / n


def HZ_3D_array_gpu(X):
    """
    Henze-Zirkler Test for simulation N times
    Parameters
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on February 26, 2023.
    ---------------------------------------------
    Input : 
    X : cp.array
        shape -- (N, n, p), i.e.  (trails, sample_size, dimension of distrition)
    ---------------------------------------------
    Returns : 
    hz : cp.array
        HZ test statistic
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
    S = cov_3D_array_gpu_new(X)

    # pinv of S, return shape = (N, p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    # (N, n, p)
    difT = X - cp.sum(X, axis=1, keepdims=True) / n

    # shape = (N, n, n) --> (N, n)  chech 過沒問題 20240422
    Dj = cp.diagonal(difT  @  S_inv @ cp.transpose(difT,
                     (0, 2, 1)), axis1=1, axis2=2)

    del difT
    difT = None

    # shape = (N, n, n), Y^T = Y
    Y = X @ S_inv @ cp.transpose(X, (0, 2, 1))

    del X
    del S_inv
    X = None
    S_inv = None

    # shape (N, n, n) chech 過沒問題 20240422
    Djk = -2 * Y + cp.repeat(
        cp.diagonal(Y, axis1=1, axis2=2),
        repeats=n, axis=1).reshape(N, n, -1) \
        + cp.transpose(cp.tile(cp.diagonal(Y, axis1=1, axis2=2),
                       (n, 1, 1)), (1, 0, 2))

    del Y
    Y = None

    # Smoothing parameter
    # b is a scalar
    b = 1 / (cp.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))

    # check if S matrix full-rank (columns are linear indepenent).
    # shape (N,)  chech 過結果與用 HZ_2D_array_gpu 相同 20240422
    hz = n * (
        1 / (n**2) * cp.sum(cp.sum(cp.exp(-(b**2) / 2 * Djk), axis=1), axis=1)
        - 2

        * ((1 + (b**2)) ** (-p / 2))
        * (1 / n)
        * (cp.sum(cp.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj), axis=1))
        + ((1 + (2 * (b ** 2))) ** (-p / 2))
    )

    del Dj
    del Djk
    Dj = None
    Djk = None

    # shape (N,)
    hz[cp.linalg.matrix_rank(S) != p] = 4 * n

    del S
    S = None

    hz = cp.sort(hz)

    b = None

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return hz


def HZ_2D_array_gpu(X):
    """
    Henze-Zirkler Test for simulation N times
    Parameters
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 20, 2023.
    ---------------------------------------------
    Input : 
    X : cp.array
        shape -- (n, p), i.e.  (sample_size, dimension of distrition)
    ---------------------------------------------
    Returns :
    hz : np.array
        HZ test statistic
        shape -- (1,)
    pval : np.array
        p-values
        shape -- (1,)
    ---------------------------------------------
    The Henze-Zirkler test [1]_ has a good overall power against alternatives
    to normality and works for any dimension and sample size.
    Adapted to Python from 
    https://github.com/raphaelvallat/pingouin/blob/master/pingouin/multivariate.py
    References
    ---------------------------------------------
    .. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
       tests for multivariate normality. Communications in Statistics-Theory
       and Methods, 19(10), 3595-3617.
    """
    # release GPU cuda
    # https://docs.cupy.dev/en/stable/user_guide/memory.html
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    X = cp.asarray(X)
    n, p = X.shape

    # Sample Covaince matrix, return shape = (p, p)
    S = cp.cov(X, rowvar=False, bias=True)

    # pinv of S, return shape = (p, p)
    S_inv = cp.linalg.pinv(S).astype(X.dtype)

    # (n, p)
    difT = X - cp.mean(X, axis=0, keepdims=True)

    # shape = (n, n) --> (n,)
    Dj = cp.diag(difT  @  S_inv @ difT.T)

    del difT
    difT = None

    # shape = (n, n), Y^T = Y
    Y = X @ S_inv @ X.T

    del X
    del S_inv
    X = None
    S_inv = None

    # 以下 3 種 Djk 都可以, 選用不需要矩陣相乘的方式 2023/04/20
    # Djk = -2 * Y.T + cp.repeat(cp.diag(Y.T), n).reshape(n, -1) + cp.tile(cp.diag(Y.T), (n, 1))
    # Djk = -2 * Y.T + cp.ones((n, 1)) @ cp.diag(Y).reshape(1, -1) +cp.diag(Y).reshape(-1, 1) @ cp.ones((1, n))
    Djk = -2 * Y.T + np.repeat(np.diag(Y), n).reshape(n, -1) + \
        np.tile(np.diag(Y), (n, 1))

    del Y
    Y = None

    # Smoothing parameter
    # b is a scalar
    b = 1 / (cp.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))

    if cp.linalg.matrix_rank(S) == p:
        hz = n * (
            1 / (n**2) * cp.sum(cp.sum(cp.exp(-(b**2) / 2 * Djk)))
            - 2
            * ((1 + (b**2)) ** (-p / 2))
            * (1 / n)
            * (cp.sum(cp.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
            + ((1 + (2 * (b**2))) ** (-p / 2))
        )
    else:
        hz = n * 4

    del Dj
    del Djk
    Dj = None
    Djk = None

    del S
    S = None

    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2

    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a +
                              (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    del a
    del b
    del wb
    a = None
    b = None
    wb = None

    # Lognormal mean and variance
    pmu = cp.log(cp.sqrt(mu**4 / (si2 + mu**2)))
    psi = cp.sqrt(cp.log1p(si2 / mu**2))
    # psi = cp.sqrt(cp.log((si2+mu^2) / mu**2)) 20230422 老師的

    del mu
    del si2
    mu = None
    si2 = None

    # P-value
    # 由數學計算得到(林楷崙 2022/02/26)： 以下 3 種計算 p-value of HZ 都可以
    # lognorm cdf, F(t|s, loc=0, scale) = Phi(1/s * (log(t) - log(scale)))，Phi 為 N(0,1) cdf ---------
    # pval = lognorm.sf(hz, psi, scale=cp.exp(pmu))
    # pval = 1 - ndtr((cp.log(hz) - pmu) / psi) # log(scale) = log(exp(pmu)) = pmu
    # log(scale) = log(exp(pmu)) = pmu
    pval = 1 - cupyx.scipy.special.ndtr((cp.log(hz) - pmu) / psi)

    del pmu
    del psi
    pmu = None
    psi = None

    hz = cp.asnumpy(hz)
    pval = cp.asnumpy(pval)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return hz, pval


def HZ_3D_array_cpu(X):
    """
    Henze-Zirkler Test for simulation N times
    Parameters
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on February 26, 2023.
    ---------------------------------------------
    Input : 
    X : np.array
        shape -- (N, n, p), i.e.  (sample_size, trails, dimension of distrition)
    # alpha : float
    #     Significant level
    ---------------------------------------------
    Return :
    hz : np.array
        HZ test statistic
        shape -- (N,)
    """

    # X = np.asarray(X)
    N, n, p = X.shape

    # Sample Covaince matrix, return shape = (N, p, p)
    S = cov_3D_array_cpu_new(X)

    # pinv of S, return shape = (N, p, p)
    S_inv = np.linalg.pinv(S, hermitian=True).astype(X.dtype)

    # (N, n, p)
    difT = X - np.sum(X, axis=1, keepdims=True) / n

    # shape = (N, n, n) --> (N, n)
    Dj = np.diagonal(difT  @  S_inv @ np.transpose(difT,
                     (0, 2, 1)), axis1=1, axis2=2)

    # shape = (N, n, n), Y^T = Y
    Y = X @ S_inv @ np.transpose(X, (0, 2, 1))

    Djk = -2 * Y + np.repeat(
        np.diagonal(Y, axis1=1, axis2=2),
        repeats=n, axis=1).reshape(N, n, -1) \
        + np.transpose(np.tile(np.diagonal(Y, axis1=1, axis2=2),
                       (n, 1, 1)), (1, 0, 2))

    # Smoothing parameter
    # b is a scalar
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))

    # Is matrix full-rank (columns are linear indepenent)?
    # shape (N,)
    hz = n * (
        1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk), axis=1), axis=1)
        - 2
        * ((1 + (b**2)) ** (-p / 2))
        * (1 / n)
        * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj), axis=1))
        + ((1 + (2 * (b ** 2))) ** (-p / 2))
    )
    # else:
    # hz_singular = n * 4

    # shape (N,)
    hz[np.linalg.matrix_rank(S) != p] = (4 * n)

    hz = np.sort(hz)

    return hz


def HZ_2D_array_cpu(X):
    """
    Henze-Zirkler Test for simulation N times
    Parameters
    ---------------------------------------------
    Python Code is written by 林楷崙 (LIN, KAI LUN) at National Taipei University
    on April 24, 2023.
    ---------------------------------------------
    Input : 
    X : np.array
        shape -- (n, p), i.e.  (sample_size, dimension of distrition)
    ---------------------------------------------
    Returns :
    hz : np.array
        HZ test statistic
        shape -- (1,)
    pval : np.array
        p-values
        shape -- (1,)
    ---------------------------------------------
    The Henze-Zirkler test [1]_ has a good overall power against alternatives
    to normality and works for any dimension and sample size.
    Adapted to Python from 
    https://github.com/raphaelvallat/pingouin/blob/master/pingouin/multivariate.py
    References
    ---------------------------------------------
    .. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
       tests for multivariate normality. Communications in Statistics-Theory
       and Methods, 19(10), 3595-3617.
    """
    X = np.asarray(X)
    n, p = X.shape

    # Sample Covaince matrix, return shape = (p, p)
    S = np.cov(X, rowvar=False, bias=True)

    # pinv of S, return shape = (p, p)
    S_inv = np.linalg.pinv(S).astype(X.dtype)

    # (n, p)
    difT = X - np.mean(X, axis=0, keepdims=True)

    # shape = (n, n) --> (n,)
    Dj = np.diag(difT  @  S_inv @ difT.T)

    del difT
    difT = None

    # shape = (n, n), Y^T = Y
    Y = X @ S_inv @ X.T

    del X
    del S_inv
    X = None
    S_inv = None

    # 以下 3 種 Djk 都可以, 選用不需要矩陣相乘的方式 2023/04/20
    # Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + np.tile(np.diag(Y.T), (n, 1))
    Djk = -2 * Y.T + np.repeat(np.diag(Y), n).reshape(n, -1) + \
        np.tile(np.diag(Y), (n, 1))
    # Djk = -2 * Y.T + np.ones((n, 1)) @ np.diag(Y).reshape(1, -1) +np.diag(Y).reshape(-1, 1) @ np.ones((1, n))

    del Y
    Y = None

    # Smoothing parameter
    # b is a scalar
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))

    if np.linalg.matrix_rank(S) == p:
        hz = n * (
            1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk)))
            - 2
            * ((1 + (b**2)) ** (-p / 2))
            * (1 / n)
            * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
            + ((1 + (2 * (b**2))) ** (-p / 2))
        )
    else:
        hz = n * 4

    del Dj
    del Djk
    Dj = None
    Djk = None

    del S
    S = None

    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2

    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a +
                              (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    del a
    del b
    del wb
    a = None
    b = None
    wb = None

    # Lognormal mean and variance
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log1p(si2 / mu**2))

    del mu
    del si2
    mu = None
    si2 = None

    # P-value
    # 由數學計算得到(林楷崙 2022/02/26)： 以下 3 種計算 pvalue of HZ 都可以
    # lognorm cdf, F(t|s, loc=0, scale) = Phi(1/s * (log(t) - log(scale)))，Phi 為 N(0,1) cdf ---------
    # pval = lognorm.sf(hz, psi, scale=np.exp(pmu))
    # pval = 1 - ndtr((np.log(hz) - pmu) / psi) # log(scale) = log(exp(pmu)) = pmu
    pval = 1 - scipy.special.ndtr((np.log(hz) - pmu) / psi)

    del pmu
    del psi
    pmu = None
    psi = None

    return hz, pval


# %%

if __name__ == '__main__':
    import scipy
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    from scipy.stats import chi2
    p = 3
    sample_size = 20
    N = 10000

    # new X_gpu.shape = (N, n, p)
    X_gpu = cp.random.multivariate_normal(
        mean=np.zeros(p), cov=np.eye(p), size=(N, sample_size))

    # hz_nonsingular, pval, power = HZ_3D_array_cpu(X_gpu)
    hz_nonsingular = HZ_3D_array_gpu(X_gpu)
    # print(power)
    print(hz_nonsingular)

    # ECDF ------------------------------------
    sample_sizes = [10, 20, 30, 50, 100, 150]

    hz = np.zeros((N, len(sample_sizes)))  # (N, n)
    Y = np.arange(1, N+1) / N

    p = 3
    df = p * (p+1) * (p+2) / 6
    # chi_cdf = lambda x: scipy.stats.chi2.cdf(x, df=df)

    for i, n in enumerate(sample_sizes):

        sample_size = n

        X_gpu = cp.random.multivariate_normal(
            mean=np.zeros(p), cov=np.eye(p), size=(N, sample_size))
        # X_gpu = np.random.standard_normal(size=(N, sample_size, p))
        # X_gpu = np.random.chisquare(df=10, size=(N, sample_size, p))
        # X_gpu = np.random.f(dfnum=5, dfden=14, size=(N, sample_size, p))

        print(i)
        hz_nonsingular = HZ_3D_array_gpu(X=X_gpu)
        hz[:, i] = np.sort(cp.asnumpy(hz_nonsingular))  # * n / 6

        color = cm.rainbow(np.linspace(0, 1, n))
        plt.plot(hz[:, i], Y, label='ECDF, n = {}'.format(n))

    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))
    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2

    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a +
                              (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    # 老師matlab的 pmu psi
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2+mu**2) / mu**2))

    plt.plot(hz[:, len(sample_sizes)-1], \
             # chi2.cdf(x=hz[:, len(sample_sizes)-1], df=df),  \
             scipy.stats.lognorm.cdf(
        # x=hz[:, len(sample_sizes)-1]+1, s=np.sqrt(si2), loc=mu, scale=1
        # x=hz[:, len(sample_sizes)-1], s=psi, loc=pmu, scale=1
        x=hz[:, len(sample_sizes)-1], s=psi,  scale=np.exp(pmu)
        # pval = lognorm.sf(hz, psi, scale=cp.exp(pmu))
    ), \
        label='CDF'.format(int(df)), color='black', \
        lw=3, linestyle='--', alpha=0.5)

    plt.legend(loc='lower right')
    plt.title('HZ Test for testing $N_{}(0, 1)$'.format(p))
    plt.show()

    # HZ 統計量的分布
    p = 3
    n = 20
    N = 1000
    HZ = cp.zeros(N)
    mean = cp.zeros(p)
    Cov = cp.eye(p)
    for i in range(N):
        X = cp.random.multivariate_normal(mean, Cov, size=n)
        HZ[i] = HZ_2D_array_gpu(X)[0]

    plt.hist(HZ.get(), density=True, bins=10)

    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 /
                                                 (p + 4)) * (n ** (1 / (p + 4)))
    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2

    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a +
                              (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    # 老師matlab的 pmu psi
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2+mu**2) / mu**2))

    x = np.linspace(0, 2, 1000)
    plt.plot(x, scipy.stats.lognorm.pdf(
        # x=x, s=psi, loc=pmu, scale=1 # 錯的 應該不放 loc 並改 scale
        x=x, s=psi, scale=np.exp(pmu)
    ))

    plt.show()

    X = cp.array([[1, 2], [3, 4], [-1, 1], [0, 4], [5, 6], [-2, 2]])
    HZ, _ = HZ_2D_array_gpu(X)
    print(HZ)  # 答案為:0.7934019947824305


# %%
