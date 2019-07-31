import csv
import os
from PyQt4 import QtGui, uic, QtCore

import mfm
import numpy as np
import pandas as pd
from mfm import Genealogy
from mfm.experiments import Setup


def save_xy(filename, x, y, verbose=False, fmt="%.3f\t%.3f", header=None):
    """
    Saves data x, y to file in format (csv). x and y
    should have the same lenght.

    :param filename: string
        Target filename
    :param x: array
    :param y: array
    :param verbose: bool
    :param fmt:
    """
    if verbose:
        print("Writing histogram to file: %s" % filename)
    fp = open(filename, 'w')
    if header is not None:
        fp.write(header)
    for p in zip(x, y):
        fp.write(fmt % (p[0], p[1]))
    fp.close()


class Csv(object):

    """
    Csv is a class to handle coma separated value files.

    :param kwargs:

    Examples
    --------
    Two-column data

    >>> import mfm.io.txt_csv
    >>> csv = mfm.io.txt_csv.Csv()
    >>> filename = './sample_data/ibh/Decay_577D.txt'
    >>> csv.load(filename)
    >>> csv.data_x
    array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
     4.09400000e+03,   4.09500000e+03,   4.09600000e+03])
     >>> csv.data_y
     array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
     >>> max(csv.data_y)
     50010.0

    One-column Jordi data

    >>> csv = mfm.io.txt_csv.Csv()
    >>> filename = './sample_data/Jordi/02_18-577+7.5uM(577)UP_8ps.dat'
    >>> csv.load(filename)
    >>> csv.data_x
    array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
     4.09400000e+03,   4.09500000e+03,   4.09600000e+03])
     >>> csv.data_y
     array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
     >>> max(csv.data_y)
     50010.0

    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.use_header = kwargs.get('use_header', None)
        self._x = kwargs.get('x', None)
        self._y = kwargs.get('y', None)
        self._ex = kwargs.get('ex', None)
        self._ey = kwargs.get('ey', None)
        self.x_on = kwargs.get('x_on', True)
        self.error_y_on = kwargs.get('y_on', False)
        self.col_x = kwargs.get('col_x', 0)
        self.col_y = kwargs.get('col_y', 1)
        self.col_ex = kwargs.get('col_ex', 2)
        self.col_ey = kwargs.get('col_ex', 3)
        self.error_x_on = kwargs.get('error_x_on', False)
        self.directory = kwargs.get('directory', '.')
        self.skiprows = kwargs.get('skiprows', 9)
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.mode = kwargs.get('mode', 'csv')
        self._filename = kwargs.get('filename', None)
        self.colspecs = kwargs.get('colspecs', '(16,33), (34,51)')
        self.data = kwargs.get('data', pd.DataFrame())

    @property
    def filename(self):
        """
        The currently open filename (after setting this parameter the file is opened)
        """
        return self._filename

    @filename.setter
    def filename(self, v):
        self._filename = v
        self.load(v)

    def load(self, filename, skiprows=None, **kwargs):
        """
        This method loads a filename to the `Csv` object
        :param filename: string specifying the file
        :param skiprows: number of rows to skip in the file. By default the value of the instance is used
        :param verbose: The method is verbose if verbose is set to True of the verbose attribute of the instance is
        True.
        """
        verbose = kwargs.get('verbose', self.verbose)
        use_header = kwargs.get('use_header', self.use_header)
        skiprows = kwargs.get('skiprows', self.skiprows)

        if os.path.isfile(filename):
            self.directory = os.path.dirname(filename)
            self._filename = filename
            colspecs = self.colspecs

            if self.mode == 'csv':
                if verbose:
                    print("Using CSV-mode:")
                    print("Use header: %s", use_header)
                    print("Skip rows: %s", skiprows)
                if self.x_on:
                    sniffer = csv.Sniffer()
                    txt = open(filename, 'rbU').readlines()
                    dialect = sniffer.sniff(txt[-2])
                    if verbose:
                        print("Dialect: %s", dialect)
                    print("Dialect: %s", dialect)
                    df = pd.read_csv(filename, skiprows=skiprows, dialect=dialect, header=use_header)
                else:
                    if verbose:
                        print("Using no x-axis (Jordi-Format)")
                    df = pd.read_csv(filename, header=self.use_header)
            else:
                if verbose:
                    print("Using fixed width file-format")
                df = pd.read_fwf(filename, colspecs=colspecs, skiprows=skiprows, header=use_header)
            self.data = df

    def save(self, data, filename, delimiter='\t', mode='txt'):
        if self.verbose:
            print "Saving"
            print "------"
            print "filename: %s" % filename
            print "mode: %s" % mode
            print "delimiter: %s" % delimiter
            print "Object-type: %s" % type(data)
        if isinstance(data, mfm.Curve):
            d = np.array(data[:])
        elif isinstance(data, np.ndarray):
            d = data
        else:
            d = np.array(data)
        if mode == 'txt':
            np.savetxt(filename, d.T, delimiter=delimiter)
        if mode == 'npy':
            np.save(filename, d.T)

    @property
    def n_cols(self):
        """
        The number of columns
        """
        return self._data.shape[1]

    @property
    def n_rows(self):
        """
        The number of rows
        """
        return self._data.shape[0]

    @property
    def data(self):
        """
        Numpy array of the data
        """
        return np.array(self._data, dtype=np.float64).T

    @data.setter
    def data(self, df):
        self._data = df

    @property
    def header(self):
        """
        A list of the column headers
        """
        if self.use_header is not None:
            header = list(self._data.columns)
        else:
            header = range(self._data.shape[1])
        return [str(i) for i in header]

    def reload_csv(self):
        """
        Reloads the csv as specified by :py:attribute:`.CSV.filename`
        """
        self.load(self.filename)

    @property
    def data_x(self):
        """
        The x-values of the loaded file as numpy.array
        """
        if self.x_on:
            try:
                return self.data[self.col_x]
            except IndexError:
                return np.arange(self.data_y.shape[0], dtype=np.float64)
        else:
            return np.arange(self.data_y.shape[0], dtype=np.float64)

    @property
    def error_x(self):
        """
        The errors of the x-values of the loaded file as numpy.array
        """
        if self.error_x_on:
            return self.data[self.col_ex]
        else:
            return self._ex

    @error_x.setter
    def error_x(self, prop):
        self.error_x_on = False
        self._ex = prop

    @property
    def error_y(self):
        """
        The errors of the y-values of the loaded file as numpy.array
        """
        if self.error_y_on:
            return self.data[self.col_y]
        else:
            if self._ey is None:
                return np.ones_like(self.data_y)
            else:
                return self._ey

    @error_y.setter
    def error_y(self, prop):
        self.error_y_on = False
        self._ey = prop

    @property
    def data_y(self):
        """
        The y-values of the loaded file as numpy.array
        """
        if self.data is not None:
            return self.data[self.col_y]
        else:
            return None

    @property
    def n_points(self):
        """
        The number of data points corresponds to the number of rows :py:attribute`.CSV.n_rows`
        """
        return self.n_rows


class CsvWidget(QtGui.QWidget, Csv):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/csvInput.ui', self)
        Csv.__init__(self, **kwargs)
        self.connect(self.spinBox, QtCore.SIGNAL("valueChanged(int)"), self.reload_csv)
        self.verbose = kwargs.get('verbose', mfm.verbose)

    @property
    def error_y_on(self):
        return self.checkBox_4.isChecked()

    @error_y_on.setter
    def error_y_on(self, v):
        self.checkBox_4.setChecked(bool(v))

    @property
    def col_ex(self):
        return self.comboBox_3.currentIndex()

    @col_ex.setter
    def col_ex(self, v):
        pass

    @property
    def error_x_on(self):
        return self.checkBox_3.isChecked()

    @error_x_on.setter
    def error_x_on(self, v):
        self.checkBox_3.setChecked(bool(v))

    @property
    def x_on(self):
        return self.checkBox.isChecked()

    @x_on.setter
    def x_on(self, v):
        self.checkBox.setChecked(bool(v))

    @property
    def col_x(self):
        return self.comboBox.currentIndex()

    @col_x.setter
    def col_x(self, v):
        pass

    @property
    def col_y(self):
        return self.comboBox_2.currentIndex()

    @col_y.setter
    def col_y(self, v):
        pass

    @property
    def data(self):
        return Csv.data.fget(self)

    @data.setter
    def data(self, v):
        Csv.data.fset(self, v)
        self.lineEdit_9.setText("%d" % v.shape[1])
        bx = [self.comboBox, self.comboBox_2, self.comboBox_3, self.comboBox_4]
        if self.n_rows > 0:
            for i, b in enumerate(bx):
                b.clear()
                b.addItems(self.header)
                b.setCurrentIndex(i % self.n_rows)

    @property
    def skiprows(self):
        return int(self.spinBox.value())

    @skiprows.setter
    def skiprows(self, v):
        self.spinBox.setValue(v)

    @property
    def use_header(self):
        if self.checkBox_2.isChecked():
            return True
        else:
            return None

    @use_header.setter
    def use_header(self, v):
        if v is None:
            self.checkBox_2.setChecked(False)
        else:
            self.checkBox_2.setChecked(True)

    @property
    def colspecs(self):
        return eval(str(self.lineEdit.text()))

    @colspecs.setter
    def colspecs(self, v):
        self.lineEdit.setText(v)

    @property
    def filename(self):
        return str(self.lineEdit_8.text())

    @filename.setter
    def filename(self, v):
        Csv.filename.fset(self, v)
        self.lineEdit_8.setText(v)

    @property
    def mode(self):
        mode = 'csv' if self.radioButton_2.isChecked() else 'fwf'
        return mode

    @mode.setter
    def mode(self, v):
        pass

    def load(self, filename=None, **kwargs):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open File', self.directory))
            print filename
        Csv.load(self, filename, **kwargs)
        self.lineEdit_8.setText(filename)


class CSVFileWidget(QtGui.QWidget, Genealogy, Setup):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        Genealogy.__init__(self)
        self.parent = kwargs.get('parent', None)
        self.name = kwargs.get('name', 'CSV-File')
        self.weight_calculation = kwargs.get('weight_calculation', None)

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.csvWidget = CsvWidget(**kwargs)
        self.layout.addWidget(self.csvWidget)

    def load_data(self, filename=None):
        """
        Loads csv-data into a Curve-object
        :param filename:
        :return: Curve-object
        """
        d = mfm.DataCurve()
        if filename is not None:
            self.csvWidget.load(filename)
            d.filename = filename
        else:
            self.csvWidget.load()
            d.filename = self.csvWidget.filename

        d.x, d.y = self.csvWidget.data_x, self.csvWidget.data_y
        if self.weight_calculation is None:
            d.set_weights(self.csvWidget.error_y)
        else:
            d.set_weights(self.weight_calculation(d.y))
        return d