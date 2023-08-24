# %%
from Wmin5_function import Wmin_m_3D_array_gpu, Wmin_m_3D_array_cpu
# from mv_dist import multi_PearsonII
import numpy as np
import cupy as cp

'''
20230605 : 
************************************
計算目標 (迴圈分次計算) : 
Wmin_CV_divideN_gpu : Emipical distribution, Emipical critical value 
Wmin_CV_divideN_cpu : Emipical distribution, Emipical critical value

************************************
Wmin(5) 檢定的經驗臨界值模擬計算量龐大, 即使下調 N 到 1萬, 
RTX3080 12GB GPU 仍無法一次算完, 需以迴圈分次計算模擬.
所以對 Wmin(5) 檢定沒有一次算玩的方法,
也沒有寫 Wmin5_CV Python 程式.

************************************


'''

__all__ = ['Wmin_CV_divideN_gpu', 'Wmin_CV_divideN_cpu']


# %%
def Wmin_CV_divideN_gpu(N, N_num_interval, sample_size, dim, m=10**4, q=0.05, alpha=0.05, H0='MVN'):
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
    N : int
        times of Monte-Carlo simulation, 10000 is suggested to be the minimum.
    N_num_interval : int
        how many are intervals the N been segmented.
    sample_size : int
    dim : int
    m : int
        how many uniformly scattered points, 10000 is suggested to be the minimum.
    q : float 
        the percentile e.g. q=0.05 for the 5th percentile. q can be a vector,
        default 0.05.
    alpha : float
        alpha of the test, default 0.05
    H0 : str
        the type of H0 data put, defaulting two types, i.e.
        'MVN' : standard MVN -- N_p(0, I), where p denotes p dimensional distrition.
        'D_MVN' : 老師所用的 H0 : MVN with mean vector 0, and
            covariance matrix with 1's diagonal and 0.9's off-diagonal. 
    ---------------------------------------------
    Returns : 
    Wmin_CV : np.array  # left-tail test
        shape -- (N,)
        the wmin_m(q) test statistic from m W-tests 
        (empirical distribution of wmin_m(q)). 
    Wmin_cv_alpha : np.array
        scalar
        the empirical critical value from m W-tests under the alpha
        and percentile q. 
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

    Wmin_CV = np.zeros(N)

    for k in range(N_num_interval):
        Wmin_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = Wmin_m_3D_array_gpu(
            X=X_gpu[k*N_per_for_loop:(k+1)*N_per_for_loop], m=m, q=q).get()  # return Wmun(q%) non sort

    del X_gpu
    X_gpu = None

    Wmin_CV = np.sort(Wmin_CV)
    Wmin_cv_alpha = Wmin_CV[np.ceil(N*alpha-1).astype('int')]

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return Wmin_CV, Wmin_cv_alpha


def Wmin_CV_divideN_cpu(N, N_num_interval, sample_size, dim, m=10**4, q=0.05, alpha=0.05, H0='MVN'):
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
    N : int
        times of Monte-Carlo simulation, 10000 is suggested to be the minimum.
    N_num_interval : int
        how many are intervals the N been segmented.
    sample_size : int
    dim : int
    m : int
        how many uniformly scattered points, 10000 is suggested to be the minimum.
    q : float 
        the percentile e.g. q=0.05 for the 5th percentile. q can be a vector,
        default 0.05.
    alpha : float
        alpha of the test, default 0.05
    H0 : str
        the type of H0 data put, defaulting two types, i.e.
        'MVN' : standard MVN -- N_p(0, I), where p denotes p dimensional distrition.
        'D_MVN' : 老師所用的 H0 : MVN with mean vector 0, and
            covariance matrix with 1's diagonal and 0.9's off-diagonal. 
    ---------------------------------------------
    Returns : 
    Wmin_CV : np.array
        shape -- (N,)
        the wmin_m(q) test statistic from m W-tests. 
        (empirical distribution of wmin_m(q)). 
    Wmin_cv_alpha : np.array  # left-tail test
        scalar 
        the empirical critical value from m W-tests under the alpha
        and percentile q. 
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

    Wmin_CV = np.zeros(N)

    for k in range(N_num_interval):
        Wmin_CV[k*N_per_for_loop:(k+1)*N_per_for_loop] = Wmin_m_3D_array_cpu(
            X=X_cpu[k*N_per_for_loop:(k+1)*N_per_for_loop], m=m, q=q)  # return Wmun(q%) non sort

    del X_cpu
    X_cpu = None

    Wmin_CV = np.sort(Wmin_CV)
    Wmin_cv_alpha = Wmin_CV[np.ceil(N*alpha-1).astype('int')]

    return Wmin_CV, Wmin_cv_alpha


if __name__ == '__main__':
    # %%
    import time
    # W test  paras
    # 'sample_sizes' and 'dims' are set for large to small
    W_paras = {'N': 100000,
               'N_num_interval': 200,
               # [10,  20 , 30,  50, 100, 130, 150, 160][::-1],
               'sample_size': 100,
               'dim':  2,  # [ 2 , 3 , 4, 5, 10, 20][::-1],
               'm': 10000,
               'q': 0.05,
               'alpha': 0.05,
               'H0': 'MVN'
               }

    # %%
    # testing
    time_start = time.time()
    Wmin_CV, Wmin_cv_alpha = Wmin_CV_divideN_gpu(**W_paras)
    print(Wmin_CV[:10])
    print(Wmin_cv_alpha)
    print('time cost = {}'.format(time.time() - time_start))
    # print(Wmin_pval(Wmin_cv_alpha, Wmin_CV))

    # %%
    # # transform Wmin_CV_005 to pandas.DataFrame and save as csv

    # # build Wmin_CV_005 panas.DataFrame
    # df_Wmin_CV_005 = pd.DataFrame(Wmin_cv_alpha_005_x, columns=W_paras['dims'], index=W_paras['sample_sizes'])

    # # reverse columns' and rows' order.
    # df_Wmin_CV_005 = df_Wmin_CV_005.iloc[::-1, ::-1]

    # # reverse columns' and rows' order and reshape.
    # Wmin_cv = Wmin_cv_x[:, ::-1, ::-1]
    # # Wmin_cv = Wmin_cv_x.iloc[:, ::-1, ::-1].reshape((W_paras['N'], -1))
    # # Wmin_cv = np.reshape(Wmin_cv_x.iloc[:, ::-1, ::-1])

    # # build df column name after reverse columns' and rows'
    # p_ = np.repeat(W_paras['dims'][::-1], len(W_paras['sample_sizes']))
    # n_ = np.tile(W_paras['sample_sizes'][::-1], len(W_paras['dims']))
    # CV_columns = 'p' + np.char.array(p_.astype(str)) + '_' +  'n' + np.char.array(n_.astype(str))

    # # transform CV data from shape (N, n, p) to shape (p * n, N)
    # Wmin_cv_to_df  = np.zeros((len(W_paras['dims']) * len(W_paras['sample_sizes']), W_paras['N']))

    # for i, p in enumerate(W_paras['dims'][::-1]):
    #     for j, n in enumerate(W_paras['sample_sizes'][::-1]):
    #         Wmin_cv_to_df[len(W_paras['sample_sizes']) * i + j, :] = Wmin_cv[:, j, i]

    #     #    Wmin_cv_to_df[len(W_paras['dims']) * i + j] = Wmin_cv[:, len(W_paras['dims']) * i + j]

    # # build Wmin_CV panas.DataFrame
    # df_Wmin_CV = pd.DataFrame(Wmin_cv_to_df.T, columns=CV_columns)

    # %%

    # save to csv
    # check過，順序正確
    # df_Wmin_CV_005.to_csv('Wmin_CV_005.csv')
    # df_Wmin_CV.to_csv('Wmin_CV.csv')


# %%
