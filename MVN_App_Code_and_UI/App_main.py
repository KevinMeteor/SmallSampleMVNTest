# %%
from MS_function import MS_2D_array_gpu, MS_2D_array_cpu
from MS_CV import MS_CV_gpu, MS_CV_cpu  # 一次算完
from MS_CV_divideN import MS_CV_divideN_gpu, MS_CV_divideN_cpu  # 迴圈計算

from MK_function import MK_2D_array_gpu, MK_2D_array_cpu
from MK_CV import MK_CV_gpu, MK_CV_cpu  # 一次算完
from MK_CV_divideN import MK_CV_divideN_gpu, MK_CV_divideN_cpu  # 迴圈計算

from HZ_function import HZ_2D_array_gpu, HZ_2D_array_cpu
from HZ_CV import HZ_CV_gpu, HZ_CV_cpu  # 一次算完
from HZ_CV_divideN import HZ_CV_divideN_gpu, HZ_CV_divideN_cpu  # 迴圈計算

from Wmin5_function import Wmin_m_2D_array_gpu, Wmin_m_2D_array_cpu
from Wmin5_CV_divideN import Wmin_CV_divideN_gpu, Wmin_CV_divideN_cpu  # 迴圈計算

from PyQt6.QtWidgets import QMessageBox, QWidget 
from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6 import QtWidgets, uic, QtGui, QtCore
import pyqtgraph as pg
import pandas as pd
import numpy as np
import scipy
import sys


class NumpyTableModel(QAbstractTableModel):

    def __init__(self, data):

        super(NumpyTableModel, self).__init__()
        self._data = np.around(data, 4)  # 四捨五入到小數點後第4位

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            # pandas's iloc method
            value = self._data[index.row(), index.column()]
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignVCenter + Qt.AlignmentFlag.AlignHCenter

        if role == Qt.ItemDataRole.BackgroundRole and (index.row() % 2 == 0):
            return QtGui.QColor('#FFEECA')  # d8ffdb 淺綠

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    # # Add Row and Column header
    # def headerData(self, section, orientation, role):
    #     # section is the index of the column/row.
    #     if role == Qt.ItemDataRole.DisplayRole: # more roles
    #         if orientation == Qt.Orientation.Horizontal:
    #             return str(self._data.columns[section])

    #         if orientation == Qt.Orientation.Vertical:
    #             return str(self._data.index[section])


class AnotherWindowHistogram(QWidget):
    # create a customized signal
    submitted = QtCore.pyqtSignal(str)  # "submitted" is like a component name

    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        uic.loadUi('./MVN_GUI/PyQt_MVN_histogram_widget.ui', self)
        self.setWindowTitle('MVN Marginal Histogram')

        # Signal
        self.pushButton_HistogramToMain.clicked.connect(self.on_submit)
        self.spinBox_histSag.valueChanged.connect(
            lambda: self.plotMarginal(self.Data, self.paras))
        self.comboBox_histDensity.currentIndexChanged.connect(
            lambda: self.plotMarginal(self.Data, self.paras))
        self.radioButton_marginalHist.toggled.connect(
            self.histogramQQplot_clicked)
        self.radioButton_QQplot.toggled.connect(self.histogramQQplot_clicked)

    # Slot

    def passHistogramInfo(self, Data, paras):
        # print('subHistogramWindow receive paras: ')
        # print(paras)
        self.paras = paras
        # print()
        # print('subHistogramWindow receive Data: ')
        # print(Data)
        self.Data = Data
        self.plotMarginal(self.Data, self.paras)

    def histogramQQplot_clicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            # print(radioBtn.text())
            if radioBtn.text() == '邊際直方圖':
                self.label_marginal_plot.setText('邊際直方圖')
                self.plotMarginal(self.Data, self.paras)

            elif radioBtn.text() == 'Q-Q 圖':
                self.label_marginal_plot.setText('Q-Q 圖')
                self.QQplot(self.Data)

    def plotMarginal(self, Data, paras):
        self.radioButton_marginalHist.setChecked(True)
        self.graphicsView_Histogram.clear()
        self.setWindowTitle('MVN Diagnosis')
        win = self.graphicsView_Histogram

        # data
        n, p = Data.shape

        cols = 4
        rows = int(p/cols) + (0 if p % cols == 0 else 1)

        try:
            n_bins = int(self.spinBox_histSag.text())
        except ValueError:
            display_message('直方圖分割數須為正整數')

        # n_bins = int(self.spinBox_histSag.text())

        if n_bins <= 0:
            display_message('直方圖分割數須為正整數')
        else:
            pass

        # print(f's = {s}')
        # his 1
        for i in range(p):
            self.plt = win.addPlot()

            if self.comboBox_histDensity.currentText() == '是':  # 有Density
                y, x = np.histogram(Data[:, i], bins=n_bins, density=True)
                self.plt.plot(x, y, stepMode="center", fillLevel=0,
                              fillOutline=True, brush=(0, 0, 255, 150))
                self.plt.setXRange(
                    np.min(Data[:, i]), np.max(Data[:, i]), padding=0)

                if max(y) <= 1:
                    self.plt.setYRange(0, min(1, max(y) + 0.1))
                else:
                    self.plt.setYRange(0, max(1, np.ceil(max(y))))

            else:  # 沒有Density
                y, x = np.histogram(Data[:, i], bins=n_bins, density=False)
                self.plt.plot(x, y, stepMode="center", fillLevel=0,
                              fillOutline=True, brush=(0, 0, 255, 150))
                self.plt.setXRange(
                    np.min(Data[:, i]), np.max(Data[:, i]), padding=0)

                self.plt.setYRange(0, np.max(y)+1, padding=0)

            if (i+1) % cols == 0 and i > 0:
                win.nextRow()
            else:
                pass

    def QQplot(self, Data):
        self.graphicsView_Histogram.clear()
        self.setWindowTitle('MVN Diagnosis')
        win = self.graphicsView_Histogram

        # data
        n, p = Data.shape

        cols = 4
        rows = int(p/cols) + (0 if p % cols == 0 else 1)

        # Data
        # standardized data
        Z = (Data - np.mean(Data, axis=0)) / np.std(Data, axis=0)

        for i in range(p):
            a, b = scipy.stats.probplot(Z[:, i], dist='norm')
            x = np.linspace(min(a[0]-0.5), max(a[0]+0.5), 2)
            q = scipy.stats.norm.cdf(x=x, loc=0, scale=1)
            # inverse of cdf, q=cdf \in (0,1)
            cdf_inv = scipy.stats.norm.ppf(q=q, loc=0, scale=1)

            self.plt = win.addPlot()

            self.plt.plot(x, cdf_inv, pen='red')
            pen = pg.mkPen(width=5, color='blue')
            scatter = pg.ScatterPlotItem(pen=pen, symbol='o', size=2)
            self.plt.addItem(scatter)
            scatter.setData(a[0], a[1])
            self.plt.setXRange(np.floor(min(a[0])), np.ceil(max(a[0])))
            self.plt.setYRange(np.floor(min(a[1])), np.ceil(max(a[1])))
            self.plt.showGrid(x=True, y=True)

            if (i+1) % cols == 0 and i > 0:

                win.nextRow()
            else:
                pass

    def on_submit(self):
        # emit a signal and pass data along
        # print('back to main')
        self.close()


class AnotherWindowTestResult(QWidget):
    # create a customized signal
    submitted = QtCore.pyqtSignal(str)  # "submitted" is like a component name
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        uic.loadUi('./MVN_GUI/PyQt_MVN_Test_Result_widget.ui', self)
        self.setWindowTitle('MVN Testing Results')

        # Signal
        self.pushButton_to_main.clicked.connect(self.close_to_main)
        self.pushButton_closeAll.clicked.connect(close_all)

    def passInfo(self, compute_method, chosenTests,  testResults):
        # print('subwindow receive paras: ')
        # print(testResults)
        self.reportTestResults(chosenTests, testResults)

    def reportTestResults(self, chosenTests, testResults):
        # MS -------------------------------------------------
        if chosenTests['MS'] == True:
            MS_ts, MS_pval, MS_CV, MS_cv = \
                testResults['MS_ts'], testResults['MS_pval'], testResults['MS_CV'], \
                testResults['MS_cv']
            # Sample Statistic
            self.label_MS_TS.setText(
                np.format_float_positional(MS_ts, precision=4, trim='k', min_digits=4))
            # p-value
            self.label_MS_pval.setText(
                np.format_float_positional(MS_pval, precision=4, trim='k', min_digits=4))
            # Empirical CV
            self.label_MS_CV.setText(
                np.format_float_positional(MS_cv, precision=4, trim='k', min_digits=4))
            # Empirical p-value
            self.label_MS_pval_fromCV.setText(
                np.format_float_positional(
                    np.mean(MS_ts <= MS_CV), precision=4, trim='k', min_digits=4)
            )

            # concludsion from CV and test statistic 4/13
            if MS_ts < MS_cv:
                MS_testing_CV_conculsion = 'Not Reject'
            else:
                MS_testing_CV_conculsion = 'Reject'
            self.label_MS_result.setText(MS_testing_CV_conculsion)

        else:
            self.label_MS_TS.setText('None')
            self.label_MS_pval.setText('None')
            self.label_MS_pval_fromCV.setText('None')
            self.label_MS_CV.setText('None')
            self.label_MS_result.setText('None')

        # MK -------------------------------------------------
        # print('MK {}'.format(chosenTests))
        if chosenTests['MK'] == True:

            MK_ts, MK_pval, MK_CV, MK_cv_lower, MK_cv_upper = \
                testResults['MK_ts'], testResults['MK_pval'], testResults['MK_CV'], \
                testResults['MK_cv_lower'], testResults['MK_cv_upper']
            # Sample Statistic
            self.label_MK_TS.setText(
                np.format_float_positional(MK_ts, precision=4, trim='k', min_digits=4))
            # p-value
            self.label_MK_pval.setText(
                np.format_float_positional(MK_pval, precision=4, trim='k', min_digits=4))
            # Empirical p-value
            self.label_MK_CV.setText(
                '(' +
                np.format_float_positional(MK_cv_lower, precision=4, trim='k', min_digits=4) +
                ', ' +
                np.format_float_positional(
                    MK_cv_upper, precision=4, trim='k', min_digits=4)
                + ')'
            )
            if MK_ts > np.median(MK_CV):
                self.label_MK_pval_fromCV.setText(  # two-tail
                    np.format_float_positional(
                        2 * np.mean(MK_ts <= MK_CV), precision=4, trim='k', min_digits=4)
                )
            else:
                # print('MK lower than median')
                self.label_MK_pval_fromCV.setText(  # two-tail
                    np.format_float_positional(
                        2 * np.mean(MK_ts >= MK_CV), precision=4, trim='k', min_digits=4)
                )

            # concludsion from CV and test statistic 4/13
            if MK_ts > MK_cv_lower and MK_ts < MK_cv_upper:
                MK_testing_CV_conculsion = 'Not Reject'
            else:
                MK_testing_CV_conculsion = 'Reject'
            self.label_MK_result.setText(MK_testing_CV_conculsion)

        else:
            self.label_MK_TS.setText('None')
            self.label_MK_pval.setText('None')
            self.label_MK_pval_fromCV.setText('None')
            self.label_MK_CV.setText('None')
            self.label_MK_result.setText('None')

        # HZ -------------------------------------------------
        if chosenTests['HZ'] == True:
            HZ_ts, HZ_pval, HZ_CV, HZ_cv = \
                testResults['HZ_ts'], testResults['HZ_pval'], testResults['HZ_CV'], \
                testResults['HZ_cv']
            # Sample Satistic
            self.label_HZ_TS.setText(
                np.format_float_positional(HZ_ts, precision=4, trim='k', min_digits=4))
            # p-value
            self.label_HZ_pval.setText(
                np.format_float_positional(HZ_pval, precision=4, trim='k', min_digits=4))
            # Empirical CV
            self.label_HZ_CV.setText(
                np.format_float_positional(HZ_cv, precision=4, trim='k', min_digits=4))
            # Empirical p-value
            self.label_HZ_pval_fromCV.setText(np.format_float_positional(
                np.mean(HZ_ts <= HZ_CV), precision=4, trim='k', min_digits=4))

            # concludsion from CV and test statistic 4/13
            if HZ_ts < HZ_cv:
                HZ_testing_CV_conculsion = 'Not Reject'
            else:
                HZ_testing_CV_conculsion = 'Reject'
            self.label_HZ_result.setText(HZ_testing_CV_conculsion)

        else:
            self.label_HZ_TS.setText('None')
            self.label_HZ_pval.setText('None')
            self.label_HZ_pval_fromCV.setText('None')
            self.label_HZ_CV.setText('None')
            self.label_HZ_result.setText('None')

        # Wmin(5) -------------------------------------------------
        if chosenTests['Wmin5'] == True:
            Wmin5_ts, Wmin5_pval, Wmin5_CV, Wmin5_cv = \
                testResults['Wmin5_ts'], testResults['Wmin5_pval'], testResults['Wmin5_CV'], \
                testResults['Wmin5_cv']

            # print(Wmin5_cv)
            # print(type(Wmin5_cv))
            # Sample Statistic
            self.label_Wmin5_TS.setText(
                np.format_float_positional(Wmin5_ts, precision=4, trim='k', min_digits=4))
            # Empirical CV
            self.label_Wmin5_CV.setText(
                np.format_float_positional(Wmin5_cv, precision=4, trim='k', min_digits=4))
            # Empirical p-value
            self.label_Wmin5_pval_fromCV.setText(np.format_float_positional(
                np.mean(Wmin5_ts > Wmin5_CV), precision=4, trim='k', min_digits=4))

            if Wmin5_ts > Wmin5_cv:
                Wmin5_testing_CV_conculsion = 'Not Reject'
            else:
                Wmin5_testing_CV_conculsion = 'Reject'

            self.label_Wmin5_result.setText(Wmin5_testing_CV_conculsion)

        else:
            self.label_Wmin5_TS.setText('None')
            self.label_Wmin5_pval_fromCV.setText('None')
            self.label_Wmin5_CV.setText('None')
            self.label_Wmin5_result.setText('None')

    def close_to_main(self):
        # emit a signal and pass data along
        self.close()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page by PyQt6
        # <Proiblem> Icon Image Path 20230822
        # .ui 檔用從 "編輯樣本表" 中所插入的 icon, 在 Qt Designer 預覽看得到, 但實際執行 .py
        # 檔案 icon 沒有顯示出來
        # <Solution> icon 的相對路徑以 .ui 檔所在資料夾, icon 在 Qt Designer 預覽看得到,
        # 但實際執行看不到. icon 相對路徑應以 .py 檔 (本檔) 所在資料夾為主, 即
        # 例如原為 : comboBox_ChooseDist::down-arrow{image: url(down_7344919.png);
        # 改為     : comboBox_ChooseDist::down-arrow{mage: url(./MVN_GUI/down_7344919.png);

        uic.loadUi('./MVN_GUI/PyQt_MVN_Test_main.ui', self)
        self.setWindowTitle('MVN Test')

        # warn up gpu code before using the App
        gpu_warn_up()

        # default paras
        self.Data = None
        self.compute_method = 'GPU'
        self.paras = {
            'N': 100000,
            'N_num_interval': 100,
            'sample_size': int(self.lineEdit_SampleSize.text()),  #
            'dim': int(self.lineEdit_VariableDim.text()),   #
            'alpha': 0.05,
            'H0': 'MVN'
        }

        # Signals --------------------------------------------------------
        # 輸入資料 radioButtom # 20230425
        self.radioButton_ImportData.toggled.connect(self.onChickedData)
        self.radioButton_GenerateData.toggled.connect(self.onChickedData)

        # 從本機讀取資料 pushbuttom
        self.pushButton_OpenFileData.clicked.connect(self.importFileData)
        self.lineEdit_FromDataFile.returnPressed.connect(self.importFileData)

        # 使用模擬資料
        self.lineEdit_VariableDim.returnPressed.connect(self.generateData)
        self.lineEdit_SampleSize.returnPressed.connect(self.generateData)
        self.comboBox_ChooseDist.currentIndexChanged.connect(self.generateData)

        # 預檢資料
        self.pushButton_Plot.clicked.connect(self.call_subHistogram)

        # 顯著水準 checkbox--Test
        self.comboBox_alpha.currentIndexChanged.connect(self.alphaChange)

        # 經驗關鍵值模擬分次計算次數 slider bar 4/20
        self.hSlider_N_per_for_loop.valueChanged.connect(self.sliderMove)

        # 檢定計算 (執行檢定)
        self.pushButton_Compute.clicked.connect(self.MVNComputeMethod)

        # 跳出視窗
        self.pBut_exit.clicked.connect(self.close)


# Slots


    def onChickedData(self):
        # 20230818 發現 :
        # setCurrentIndex 會觸發
        # self.comboBox_ChooseDist.currentIndexChanged.connect(self.generateData)
        # 而導致 self.generateData 誤發, 尤其是其中的 display_message 被誤發
        # 20230818 已修正
        self.comboBox_ChooseDist.setCurrentIndex(0)

        # TableView 清空視窗 20230817
        # TableView 無法直接清空視窗 --> solution : set self.Data = np.zeros((1,1))

        self.Data = np.zeros(shape=(1, 1))
        self.model = NumpyTableModel(np.array(self.Data))

        self.tableView_DataView.setModel(self.model)

        radioButton = self.sender()
        if radioButton.isChecked():
            # print(radioButton.text())

            if radioButton.text() == '從本機讀取資料':
                self.lineEdit_VariableDim.setText(' ')
                self.lineEdit_SampleSize.setText(' ')
                self.label_PreviewVarDim.setText(' ')
                self.label_PreviewSampleSize.setText(' ')

            elif radioButton.text() == '使用模擬資料':
                self.lineEdit_FromDataFile.setText(' ')
                self.label_PreviewVarDim.setText(' ')
                self.label_PreviewSampleSize.setText(' ')

    def call_subWin(self):
        # create a sub-window
        self.anotherwindow = AnotherWindowTestResult()
        # pass information to sub-window
        if self.compute_method == 'GPU':
            finalTestResults = self.TestResults_gpu
        elif self.compute_method == 'CPU':
            finalTestResults = self.TestResults_cpu
        else:
            # 查表
            # 如果外來要加入查表的功能, 可從此處加入
            # print('查表')
            display_message('請查表')
            return

        self.anotherwindow.passInfo(
            self.compute_method, self.chosenTests, finalTestResults)
        self.anotherwindow.show()

        # 視窗置頂 20230817
        # https://blog.csdn.net/weixin_35754676/article/details/129070597?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-129070597-blog-87365818.235%5Ev38%5Epc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-129070597-blog-87365818.235%5Ev38%5Epc_relevant_sort_base2&utm_relevant_index=7
        self.anotherwindow.activateWindow()

    def call_subHistogram(self):

        if (self.Data is None) or np.all((self.Data == np.zeros(shape=(1, 1)))):  # list
            display_message('沒有檢定資料')
            return
        else:
            pass

        self.subHistogramWindow = AnotherWindowHistogram()
        self.subHistogramWindow.passHistogramInfo(self.Data, self.paras)
        # print('從 main 傳送資料與參數到 subHistogramWindow')
        self.subHistogramWindow.show()
        # 視窗置頂 20230817
        # https://blog.csdn.net/weixin_35754676/article/details/129070597?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-129070597-blog-87365818.235%5Ev38%5Epc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-129070597-blog-87365818.235%5Ev38%5Epc_relevant_sort_base2&utm_relevant_index=7
        self.subHistogramWindow.activateWindow()

    def MVNComputeMethod(self):
        self.compute_method = self.comboBox_computeMethod.currentText()
        # print('檢定的計算藉由{}'.format(self.compute_method))

        if self.paras['dim'] == 1 or self.paras['sample_size'] == 1:
            display_message('變數個數與樣本數皆須大於 1 電腦才可計算檢定')

        if self.compute_method == 'GPU':
            self.testingMVN_gpu()
            # print('GPU')
        elif self.compute_method == 'CPU':
            self.testingMVN_cpu()
            # print('CPU')
        else:
            # print('查表')
            pass

    def alphaChange(self):
        self.paras['alpha'] = float(self.comboBox_alpha.currentText())
        # print('alpha combo box change')
        # print(self.paras)

    def importFileData(self):
        if self.radioButton_ImportData.isChecked():
            pass
        else:
            return
        # getOpenFileName : 選擇單一檔案
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                      "", "EXCEL files (*.xlsx *.xls);;Text files (*.txt);;csv(*.csv)")

        # ;;All Files (*)
        # print(fname)
        # print(fname[0])
        if fname[0]:
            try:
                if fname[1] == 'EXCEL files (*.xlsx *.xls)':
                    # openpyxl
                    # header = 0 : column names are inferred from the first line of the file
                    # header = None : data don't have column name
                    # convert from pandas to numpy array
                    self.Data = pd.read_excel(
                        fname[0], index_col=None, header=None).to_numpy()
                elif fname[1] == 'Text files (*.txt)':
                    self.Data = pd.read_csv(
                        fname[0], sep='\t', index_col=None, header=None).to_numpy()

                elif fname[1] == 'csv(*.csv)':
                    self.Data = pd.read_csv(
                        fname[0], sep=',', index_col=None, header=None).to_numpy()

                else:
                    pass

                # 若有任一格資料為空值
                if np.any(np.isnan(self.Data)):
                    display_message('資料具有缺失值')
                    self.Data = np.zeros(shape=(1, 1))
                else:
                    # print('成功導入資料')
                    pass
            except:
                # print('no data import or wrong data')
                display_message('資料錯誤或沒有引入資料')
                self.Data = np.zeros(shape=(1, 1))

            try:
                self.model = NumpyTableModel(np.array(self.Data))
            except:
                # print('no data import or wrong data')
                display_message('資料錯誤或沒有引入資料')
                return

            self.tableView_DataView.setModel(self.model)
            self.lineEdit_FromDataFile.setText(fname[0])

            self.label_PreviewSampleSize.setText(str(self.Data.shape[0]))
            self.label_PreviewVarDim.setText(str(self.Data.shape[1]))

            self.paras['dim'] = int(self.Data.shape[1])
            self.paras['sample_size'] = int(self.Data.shape[0])

    def generateData(self):
        #  Changes the state of radio button
        if self.radioButton_GenerateData.isChecked():
            pass
        else:
            return

        try:
            dim = int(self.lineEdit_VariableDim.text())
            sample = int(self.lineEdit_SampleSize.text())
            self.paras['dim'] = dim
            self.paras['sample_size'] = sample

            # print(f'dim = {dim}')
            # print(f'sample = {sample}')
            if dim * sample <= 0:
                # print('變數個數或樣本數輸入錯誤')
                display_message('變數個數或樣本數輸入錯誤')
                return
        except:
            # print('no dim or no sample')
            display_message('變數個數或樣本數輸入錯誤或沒有輸入')
            return

        # print(self.comboBox_ChooseDist.currentText())
        if self.comboBox_ChooseDist.currentText() == '標準多變量常態分配':
            self.Data = np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye((dim)), size=(sample))
        elif self.comboBox_ChooseDist.currentText() == '多變量 t(3) 分配':
            if sample != 1 and dim != 1:
                self.Data = scipy.stats.multivariate_t.rvs(
                    df=3, size=(sample, dim))

            else:
                self.Data = np.reshape(scipy.stats.multivariate_t.rvs(
                    df=3, size=(sample, dim)), (sample, dim))

        elif self.comboBox_ChooseDist.currentText() == '多變量 t(10) 分配':
            if sample != 1 and dim != 1:
                self.Data = scipy.stats.multivariate_t.rvs(
                    df=10, size=(sample, dim))

            else:
                self.Data = np.reshape(scipy.stats.multivariate_t.rvs(
                    df=10, size=(sample, dim)), (sample, dim))

        else:
            # print('no distribution been chosen.')
            display_message('請選擇分配')
            self.Data = np.zeros((1, 1))

        # print(self.Data[0])
        self.model = NumpyTableModel(self.Data)
        self.tableView_DataView.setModel(self.model)

        self.label_PreviewSampleSize.setText(str(sample))
        self.label_PreviewVarDim.setText(str(dim))
        self.label_computeStatus.setText('計算重置')

    def testingMVN_gpu(self):
        self.TestResults_gpu = {
            # MS
            'MS_ts': 'None',
            'MS_pval': 'None',
            'MS_CV': 'None',
            'MS_cv': 'None',
            # MK
            'MK_ts': 'None',
            'MK_pval': 'None',
            'MK_CV': 'None',
            'MK_cv_lower': 'None',
            'MK_cv_upper': 'None',
            # HZ
            'HZ_CV': 'None',
            'HZ_cv': 'None',
            'HZ_ts': 'None',
            'HZ_pval': 'None',
            # Wmin5
            'Wmin5_ts': 'None',
            'Wmin5_CV': 'None',
            'Wmin5_cv': 'None',
            'Wmin5_pval': 'No asy. distibution'
        }

        if (self.Data is None) or np.all((self.Data == np.zeros(shape=(1, 1)))):  # list
            display_message('沒有檢定資料')
        else:
            # print('testing')
            self.label_computeStatus.setText('計算中...')

            # print(self.paras)

            self.chosenTests = dict(MS=False, MK=False, HZ=False, Wmin5=False)

            count_test_denominator = 0

            if self.checkBox_MS_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['MS'] = True
            else:
                self.chosenTests['MS'] = False
            if self.checkBox_MK_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['MK'] = True
            else:
                self.chosenTests['MK'] = False
            if self.checkBox_HZ_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['HZ'] = True
            else:
                self.chosenTests['HZ'] = False
            if self.checkBox_Wmin5_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['Wmin5'] = True
            else:
                self.chosenTests['Wmin5'] = False

            # 使用者沒有選擇檢定方法
            if count_test_denominator == 0:
                display_message('請至少選擇一種檢定方法')
                return
            else:
                pass

            N_num_interval = self.paras['N_num_interval']

            count_test_numerator = 0
            self.progressBar_Test.setValue(
                int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_MS_Test.isChecked():
                MS_ts, MS_pval = \
                    MS_2D_array_gpu(X=self.Data)

                if N_num_interval == 1:  # 一次算完
                    try:
                        MS_CV, MS_cv = MS_CV_gpu(**self.paras)
                    except:
                        display_message('計算 MS Test 時 VRAM 不足 請調高分割迴圈數')
                        return
                else:  # 分作迴圈
                    try:
                        MS_CV, MS_cv = MS_CV_divideN_gpu(**self.paras)
                    except:
                        display_message('計算 MS Test 時 VRAM 不足 請調高分割迴圈數')
                        return

                self.TestResults_gpu['MS_ts'] = MS_ts
                self.TestResults_gpu['MS_pval'] = MS_pval
                self.TestResults_gpu['MS_CV'] = MS_CV
                self.TestResults_gpu['MS_cv'] = MS_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_MK_Test.isChecked():
                MK_ts, MK_pval = \
                    MK_2D_array_gpu(X=self.Data)

                # CV
                if N_num_interval == 1:  # 一次算完
                    try:
                        MK_CV, MK_cv_lower, MK_cv_upper = MK_CV_gpu(
                            **self.paras)  # 已經取完 alpha
                    except:
                        display_message('計算 MK Test 時 VRAM 不足 請調高分割迴圈數')
                        return
                else:  # 分作迴圈
                    try:
                        MK_CV, MK_cv_lower, MK_cv_upper = MK_CV_divideN_gpu(
                            **self.paras)  # 已經取完 alpha
                    except:
                        display_message('計算 MK Test 時 VRAM 不足 請調高分割迴圈數')
                        return

                self.TestResults_gpu['MK_ts'] = MK_ts
                self.TestResults_gpu['MK_pval'] = MK_pval
                self.TestResults_gpu['MK_CV'] = MK_CV
                self.TestResults_gpu['MK_cv_lower'] = MK_cv_lower
                self.TestResults_gpu['MK_cv_upper'] = MK_cv_upper

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_HZ_Test.isChecked():
                HZ_ts, HZ_pval = HZ_2D_array_gpu(X=self.Data)

                # CV
                if N_num_interval == 1:
                    try:
                        HZ_CV, HZ_cv = HZ_CV_gpu(**self.paras)
                    except:
                        display_message('計算 HZ Test 時 VRAM 不足 請調高分割迴圈數')
                        return
                else:
                    try:
                        HZ_CV, HZ_cv = HZ_CV_divideN_gpu(**self.paras)
                    except:
                        display_message('計算 HZ Test 時 VRAM 不足 請調高分割迴圈數')
                        return

                self.TestResults_gpu['HZ_ts'] = HZ_ts
                self.TestResults_gpu['HZ_pval'] = HZ_pval
                self.TestResults_gpu['HZ_CV'] = HZ_CV
                self.TestResults_gpu['HZ_cv'] = HZ_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_Wmin5_Test.isChecked():
                Wmin5_ts = Wmin_m_2D_array_gpu(X=self.Data, m=10000, q=0.05)

                # CV
                self.paras['N'] = 10**4
                try:  # Wmin(5) 只模擬 10**4 次
                    Wmin5_CV, Wmin5_cv = Wmin_CV_divideN_gpu(
                        **self.paras, m=10**4, q=0.05)
                    self.paras['N'] = 10**5
                except:
                    # 暫時削減 N
                    display_message('計算 Wmin(5) Test 時 VRAM 不足 請調高分割迴圈數')
                    self.paras['N'] = 10**5
                    return

                self.TestResults_gpu['Wmin5_ts'] = Wmin5_ts
                self.TestResults_gpu['Wmin5_pval'] = 'No asy p-value'
                self.TestResults_gpu['Wmin5_CV'] = Wmin5_CV
                self.TestResults_gpu['Wmin5_cv'] = Wmin5_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            # print(self.TestResults_gpu)
            self.label_computeStatus.setText('GPU計算完成')
            # print('gpu compute done')
            self.call_subWin()

    def testingMVN_cpu(self):
        self.TestResults_cpu = {
            # MS
            'MS_ts': 'None',
            'MS_pval': 'None',
            'MS_CV': 'None',
            'MS_cv': 'None',
            # MK
            'MK_ts': 'None',
            'MK_pval': 'None',
            'MK_CV': 'None',
            'MK_cv_lower': 'None',
            'MK_cv_upper': 'None',
            # HZ
            'HZ_CV': 'None',
            'HZ_cv': 'None',
            'HZ_ts': 'None',
            'HZ_pval': 'None',
            # Wmin5
            'Wmin5_ts': 'None',
            'Wmin5_CV': 'None',
            'Wmin5_cv': 'None',
            'Wmin5_pval': 'No asy. distibution'
        }

        if (self.Data is None) or np.all((self.Data == np.zeros(shape=(1, 1)))):  # list
            display_message('沒有檢定資料')
        else:
            # print('testing')
            self.label_computeStatus.setText('計算中...')

            # print(self.paras)

            self.chosenTests = dict(MK=False, MS=False, HZ=False, Wmin5=False)

            count_test_denominator = 0

            if self.checkBox_MS_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['MS'] = True
            else:
                self.chosenTests['MS'] = False
            if self.checkBox_MK_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['MK'] = True
            else:
                self.chosenTests['MK'] = False
            if self.checkBox_HZ_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['HZ'] = True
            else:
                self.chosenTests['HZ'] = False
            if self.checkBox_Wmin5_Test.isChecked():
                count_test_denominator += 1
                self.chosenTests['Wmin5'] = True
            else:
                self.chosenTests['Wmin5'] = False

            # 使用者沒有選擇檢定方法
            if count_test_denominator == 0:
                display_message('請至少選擇一種檢定方法')
                return
            else:
                pass

            N_num_interval = self.paras['N_num_interval']

            count_test_numerator = 0
            self.progressBar_Test.setValue(
                int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_MS_Test.isChecked():
                MS_ts, MS_pval = \
                    MS_2D_array_cpu(X=self.Data)

                if N_num_interval == 1:  # 一次算完
                    try:
                        MS_CV, MS_cv = MS_CV_cpu(**self.paras)
                    except:
                        display_message('計算 MS Test 時 RAM 不足 請調高分割迴圈數')
                        return
                else:  # 分作迴圈
                    try:
                        MS_CV, MS_cv = MS_CV_divideN_cpu(**self.paras)
                    except:
                        display_message('計算 MS Test 時 RAM 不足 請調高分割迴圈數')
                        return

                self.TestResults_cpu['MS_ts'] = MS_ts
                self.TestResults_cpu['MS_pval'] = MS_pval
                self.TestResults_cpu['MS_CV'] = MS_CV
                self.TestResults_cpu['MS_cv'] = MS_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_MK_Test.isChecked():
                MK_ts, MK_pval = \
                    MK_2D_array_cpu(X=self.Data)

                # CV
                if N_num_interval == 1:  # 一次算完
                    try:
                        MK_CV, MK_cv_lower, MK_cv_upper = MK_CV_cpu(
                            **self.paras)  # 已經取完 alpha
                    except:
                        display_message('計算 MK Test 時 RAM 不足 請調高分割迴圈數')
                        return
                else:  # 分作迴圈
                    try:
                        MK_CV, MK_cv_lower, MK_cv_upper = MK_CV_divideN_cpu(
                            **self.paras)  # 已經取完 alpha
                    except:
                        display_message('計算 MK Test 時 RAM 不足 請調高分割迴圈數')
                        return

                self.TestResults_cpu['MK_ts'] = MK_ts
                self.TestResults_cpu['MK_pval'] = MK_pval
                self.TestResults_cpu['MK_CV'] = MK_CV
                self.TestResults_cpu['MK_cv_lower'] = MK_cv_lower
                self.TestResults_cpu['MK_cv_upper'] = MK_cv_upper

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_HZ_Test.isChecked():
                HZ_ts, HZ_pval = HZ_2D_array_cpu(X=self.Data)

                # CV
                if N_num_interval == 1:
                    try:
                        HZ_CV, HZ_cv = HZ_CV_cpu(**self.paras)
                    except:
                        display_message('計算 HZ Test 時 RAM 不足 請調高分割迴圈數')
                        return
                else:
                    try:
                        HZ_CV, HZ_cv = HZ_CV_divideN_cpu(**self.paras)
                    except:
                        display_message('計算 HZ Test 時 RAM不足 請調高分割迴圈數')
                        return

                self.TestResults_cpu['HZ_ts'] = HZ_ts
                self.TestResults_cpu['HZ_pval'] = HZ_pval
                self.TestResults_cpu['HZ_CV'] = HZ_CV
                self.TestResults_cpu['HZ_cv'] = HZ_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            if self.checkBox_Wmin5_Test.isChecked():
                Wmin5_ts = Wmin_m_2D_array_cpu(X=self.Data, m=10000, q=0.05)

                self.paras['N'] = 10**4  # Wmin(5) 只模擬 10**4 次

                try:
                    Wmin5_CV, Wmin5_cv = Wmin_CV_divideN_cpu(
                        **self.paras, m=10**4, q=0.05)
                    self.paras['N'] = 10**5  # 條整回 N=10**5

                except:
                    # 暫時削減 N
                    display_message('計算 Wmin(5) Test 時 RAM 不足 請調高分割迴圈數')
                    self.paras['N'] = 10**5  # 條整回 N=10**5
                    return

                self.TestResults_cpu['Wmin5_ts'] = Wmin5_ts
                self.TestResults_cpu['Wmin5_pval'] = 'None'
                self.TestResults_cpu['Wmin5_CV'] = Wmin5_CV
                self.TestResults_cpu['Wmin5_cv'] = Wmin5_cv

                count_test_numerator += 1
                self.progressBar_Test.setValue(
                    int(count_test_numerator / count_test_denominator * 100))

            # print(self.TestResults_cpu)
            self.label_computeStatus.setText('CPU計算完成')
            # print('cpu compute done')
            self.call_subWin()

    def sliderMove(self):
        if self.hSlider_N_per_for_loop.value() == 1:
            N_num_interval = 1
        elif self.hSlider_N_per_for_loop.value() == 2:
            N_num_interval = 10
        elif self.hSlider_N_per_for_loop.value() == 3:
            N_num_interval = 100
        elif self.hSlider_N_per_for_loop.value() == 4:
            N_num_interval = 1000
        elif self.hSlider_N_per_for_loop.value() == 5:
            N_num_interval = 10000
        else:  # 例外處理
            display_message('迴圈分次計算次數錯誤')
            # N_num_interval = 10000
            return

        # print('vales = {}'.format(self.hSlider_N_per_for_loop.value()))

        self.paras["N_num_interval"] = N_num_interval
        # print(self.paras)


def display_message(message):
    dlg = QMessageBox()
    dlg.setWindowTitle("Error Information")

    if len(message) < 25:  # 調整錯誤視窗的最小之大小
        dlg.setText(message.ljust(25 - len(message)))
    else:
        dlg.setText(message)

    dlg.setStandardButtons(QMessageBox.StandardButton.Yes)
    buttonY = dlg.button(QMessageBox.StandardButton.Yes)
    buttonY.setText('OK')
    dlg.setIcon(QMessageBox.Icon.Information)
    dlg.exec()


def gpu_warn_up():
    paras_ = {
        'N': 100,
        'N_num_interval': 1,
        'sample_size': 10,  #
        'dim': 2,   #
        'alpha': 0.05,
        'H0': 'MVN'
    }

    alpha_ = paras_['alpha']
    dim_ = paras_['dim']
    N_ = paras_['N']
    sample_ = paras_['sample_size']

    Data_ = np.random.multivariate_normal(
        mean=np.zeros(dim_), cov=np.eye((dim_)), size=(sample_))

    # MS
    MS_2D_array_gpu(X=Data_)
    MS_CV_gpu(**paras_)  # 已經取完 alpha

    # MK
    MK_2D_array_gpu(X=Data_)
    MK_CV_gpu(**paras_)  # 已經取完 alpha

    # HZ
    HZ_2D_array_gpu(X=Data_)
    HZ_CV_gpu(**paras_)  # 已經取完 alpha

    # Wmin5
    Wmin_m_2D_array_gpu(X=Data_, m=10000, q=0.05)
    Wmin_CV_divideN_gpu(**paras_, m=10**4, q=0.05)  # 已經取完 alpha

    # print('Warm up gpu code done ! \n')


def close_all():
    # https://forum.qt.io/topic/137968/closing-all-window-instances-when-mainwindow-is-closed/3
    QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


# %%
