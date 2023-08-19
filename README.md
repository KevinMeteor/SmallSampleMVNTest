# SmallSampleMVNTest

本應用程式專注利用 GPU 加速小樣本下多變量常態檢定的經驗臨界值之模擬。

應用程式可選擇只利用 CPU 計算，或加入 GPU 加速計算。
論文實驗所使用的 GPU 為 RTX 3080 12GB，電腦安裝 CUDA 12.2.79，Python 程式執行環境如 requirements.txt 所示。
應用程式執行時，可能會因會 GPU 專屬記憶體不足而無法執行，可以在應用程式中調高分次計算次數。
若是調高分次計算次數之後，使用 Wmin(5) 檢定仍遇到 GPU 專屬記憶體不足，可於 W_function.py 中：


    if n * m * N < 1e08:  # 多加上 N
        # R shape = (N, m)
        R = W_stat_1992_gpu_3D(YL_T)
    else:
        R = cp.zeros((N, m))
        for i in range(N):
            R[i, :] = W_stat_1992_gpu_3D(YL_T[i, :, :].reshape(1, n, m))[0]

  
修改 1e08 到更小的數值，直到所使用的計算設備可以承受的計算資料大小。




Icons in the APP are from https://www.freepik.com .


