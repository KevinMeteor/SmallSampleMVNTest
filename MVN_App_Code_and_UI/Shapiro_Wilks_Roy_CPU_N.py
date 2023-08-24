import numpy as np
import scipy

__all__ = ['W_stat_1992_cpu_3D', 'W_stat_1992_cpu']


def W_stat_1992_cpu_3D(y):
    """
    ---------------------------------------------
    Input:
    X_cpu : np.array  
        shape -- (N, sample_size, m)
    ---------------------------------------------
    Return:
    W : np.array
        W test statistic
        shape -- (N, m)
    """
    N, n, m = y.shape

    # shape = (N, n, m)
    y = np.sort(y, axis=1)  # sorting columnwise

    # shape = (n,)
    i = np.arange(1, n + 1)
    m = scipy.special.ndtri((i - 3 / 8) / (n + 1 / 4))  # inverse normal CDF
    x = 1 / np.sqrt(n)
    c = 1 / np.linalg.norm(m) * m  # c=1/np.sqrt(m*m')*m

    # --- compute coefficients a1,a2,...,an
    a = np.zeros(n)
    a[-1] = c[-1] + 0.221157 * x - 0.147981 * x ** 2 - \
        2.071190 * x ** 3 + 4.434685 * x ** 4 - 2.706056 * x ** 5
    a[-2] = c[-2] + 0.042981 * x - 0.293762 * x ** 2 - \
        1.752461 * x ** 3 + 5.682633 * x ** 4 - 3.582663 * x ** 5

    if n <= 5:
        phi = (np.square(np.linalg.norm(m)) - 2 *
               m[-1] ** 2) / (1 - 2 * a[-1] ** 2)
        a[1:-1] = m[1:-1] / np.sqrt(phi)
        a[0] = -a[-1]
    else:  # n > 5
        phi = (np.square(np.linalg.norm(m)) - 2 *
               m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * a[-1] ** 2 - 2 * a[-2] ** 2)
        a[2:-2] = m[2:-2] / np.sqrt(phi)
        a[0:2] = np.array([-a[-1], -a[-2]])

    # -- In Royston' paper, n=3 is excluded.
    if n == 3:  # set theoretical value for a at n=3
        a = np.array([-0.707107, 0, 0.707107])  # 2**(-1/2) = 0.707107

    # -- compute W-statistic
    W = np.dot(a, y) ** 2.0 / ((n - 1) *
                               np.var(y, axis=(1,), ddof=1))  # W-statistic
    return W


def W_stat_1992_cpu(y):
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
    y = np.sort(y, axis=0)  # sorting columnwise
    # print(len(y))
    i = np.arange(1, n + 1)
    m = scipy.special.ndtri((i - 3 / 8) / (n + 1 / 4))  # inverse normal CDF
    c = 1 / np.linalg.norm(m) * m  # c=1/np.sqrt(m*m')*m
    x = 1 / np.sqrt(n)
    # --- compute coefficients a1,a2,...,an
    a = np.zeros(n)
    a[-1] = c[-1] + 0.221157 * x - 0.147981 * x ** 2 - \
        2.071190 * x ** 3 + 4.434685 * x ** 4 - 2.706056 * x ** 5
    a[-2] = c[-2] + 0.042981 * x - 0.293762 * x ** 2 - \
        1.752461 * x ** 3 + 5.682633 * x ** 4 - 3.582663 * x ** 5

    if n <= 5:
        phi = (np.square(np.linalg.norm(m)) - 2 *
               m[-1] ** 2) / (1 - 2 * a[-1] ** 2)
        a[1:-1] = m[1:-1] / np.sqrt(phi)
        a[0] = -a[-1]
    else:  # n > 5
        phi = (np.square(np.linalg.norm(m)) - 2 *
               m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * a[-1] ** 2 - 2 * a[-2] ** 2)
        a[2:-2] = m[2:-2] / np.sqrt(phi)
        a[0:2] = np.array([-a[-1], -a[-2]])

    # -- In Royston' paper, n=3 is excluded.
    if n == 3:  # set theoretical value for a at n=3
        a = np.array([-0.707107, 0, 0.707107])  # 2**(-1/2) = 0.707107

    # -- compute W-statistic
    W = np.dot(a, y) ** 2.0 / ((n - 1) *
                               np.var(y, axis=0, ddof=1))  # W-statistic
    return W


if __name__ == '__main__':
    p = 4
    m = p  # p
    n = 100
    N = 1000

    X_cpu = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p),
                                          size=(N, n))

    W1 = np.zeros((N, m))

    for i in range(N):
        W1[i] = W_stat_1992_cpu(np.array(X_cpu[i]))

    print("W = {}".format(W1))

    print()

    W2 = W_stat_1992_cpu_3D(np.array(X_cpu))

    print("增加1-dimension後的W是否與原本的相符 = {}".format(W2))

    print(np.allclose(W1, W2))
