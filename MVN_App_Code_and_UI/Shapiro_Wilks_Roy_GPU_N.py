import numpy as np
import cupyx as cpx
import cupy as cp


__all__ = ['wtest1992_gpu', 'W_stat_1992_gpu', 'My_shapiro_gpu']


def W_stat_1992_gpu_3D(y):
    """
    ---------------------------------------------
    Input:
    X_gpu : cp.array  
        shape -- (N, sample_size, m)
    ---------------------------------------------
    Return:
    W : cp.array
        W test statistic
        shape -- (N, m)
    """
    N, n, m = y.shape

    # shape = (N, n, m)
    y = cp.sort(y, axis=1)  # sorting columnwise

    # shape = (n,)
    i = cp.arange(1, n + 1)
    m = cpx.scipy.special.ndtri(
        (i - 3 / 8) / (n + 1 / 4))  # inverse normal CDF
    x = 1 / cp.sqrt(n)
    c = 1 / cp.linalg.norm(m) * m  # c=1/cp.sqrt(m*m')*m

    # --- compute coefficients a1,a2,...,an
    a = cp.zeros(n)
    a[-1] = c[-1] + 0.221157 * x - 0.147981 * x ** 2 - \
        2.071190 * x ** 3 + 4.434685 * x ** 4 - 2.706056 * x ** 5
    a[-2] = c[-2] + 0.042981 * x - 0.293762 * x ** 2 - \
        1.752461 * x ** 3 + 5.682633 * x ** 4 - 3.582663 * x ** 5

    if n <= 5:
        phi = (cp.square(cp.linalg.norm(m)) - 2 *
               m[-1] ** 2) / (1 - 2 * a[-1] ** 2)
        a[1:-1] = m[1:-1] / cp.sqrt(phi)
        a[0] = -a[-1]
    else:  # n > 5
        phi = (cp.square(cp.linalg.norm(m)) - 2 *
               m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * a[-1] ** 2 - 2 * a[-2] ** 2)
        a[2:-2] = m[2:-2] / cp.sqrt(phi)
        a[0:2] = cp.array([-a[-1], -a[-2]])

    # -- In Royston' paper, n=3 is excluded.
    if n == 3:  # set theoretical value for a at n=3
        a = cp.array([-0.707107, 0, 0.707107])  # 2**(-1/2) = 0.707107

    # -- compute W-statistic
    W = cp.dot(a, y) ** 2.0 / ((n - 1) *
                               cp.var(y, axis=(1,), ddof=1))  # W-statistic
    return W


def W_stat_1992_gpu(y):
    """
    This function computes W statistic for data of sample size from  3 to 1000 based on
    Royston (1992).

    syntax : W=W_stat_1992(y)

    inputs :
       y: data, can be a vector or a matrix (column-wide) # 每一個 column 都是一筆要檢定的資料 

    outputs :
       W: W test statistic

    created by    Chun-Chao Wang
                   Dept. of Statistics
                   National Taipei University
                   Taipei, Taiwan

    References:
            Royston JP (1992). "Approximating the Shapiro-Wilk W test for non-normality."
            Stat. Comput. 2,117-119.
    """

    n = y.shape[0]  # sample size
    y = cp.sort(y, axis=0)  # sorting columnwise
    # print(len(y))
    i = cp.arange(1, n + 1)
    m = cpx.scipy.special.ndtri(
        (i - 3 / 8) / (n + 1 / 4))  # inverse normal CDF
    c = 1 / cp.linalg.norm(m) * m  # c=1/cp.sqrt(m*m')*m
    x = 1 / cp.sqrt(n)
    # --- compute coefficients a1,a2,...,an
    a = cp.zeros(n)
    a[-1] = c[-1] + 0.221157 * x - 0.147981 * x ** 2 - \
        2.071190 * x ** 3 + 4.434685 * x ** 4 - 2.706056 * x ** 5
    a[-2] = c[-2] + 0.042981 * x - 0.293762 * x ** 2 - \
        1.752461 * x ** 3 + 5.682633 * x ** 4 - 3.582663 * x ** 5

    if n <= 5:
        phi = (cp.square(cp.linalg.norm(m)) - 2 *
               m[-1] ** 2) / (1 - 2 * a[-1] ** 2)
        a[1:-1] = m[1:-1] / cp.sqrt(phi)
        a[0] = -a[-1]
    else:  # n > 5
        phi = (cp.square(cp.linalg.norm(m)) - 2 *
               m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * a[-1] ** 2 - 2 * a[-2] ** 2)
        a[2:-2] = m[2:-2] / cp.sqrt(phi)
        a[0:2] = cp.array([-a[-1], -a[-2]])

    # -- In Royston' paper, n=3 is excluded.
    if n == 3:  # set theoretical value for a at n=3
        a = cp.array([-0.707107, 0, 0.707107])  # 2**(-1/2) = 0.707107

    # -- compute W-statistic
    W = cp.dot(a, y) ** 2.0 / ((n - 1) *
                               cp.var(y, axis=0, ddof=1))  # W-statistic
    return W


if __name__ == '__main__':
    p = 4
    m = p  # p
    n = 100
    N = 1000

    X_cpu = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p),
                                          size=(N, n))

    W1 = cp.zeros((N, m))

    for i in range(N):
        W1[i] = W_stat_1992_gpu(cp.array(X_cpu[i]))

    print("W = {}".format(W1))

    print()

    W2 = W_stat_1992_gpu_3D(cp.array(X_cpu))

    print("增加1-dimension後的W是否與原本的相符 = {}".format(W2))

    print(cp.allclose(W1, W2))
