import tempfile
import os

from PyQt4 import QtGui, QtCore, uic
import numpy as np
import tables

from . import tttrlib


filetypes = {
    "hdf": {
        'name': "High density file",
        'ending': '.h5'
    },
    "bh132": {
        'name': "Becker-Hickl-132",
        'ending': '.spc',
        'nTAC': 4095,
        'nROUT': 255,
        'read': tttrlib.beckerMerged
    },
    "bh630_x48": {
        'name': "Becker-Hickl-630",
        'ending': '.spc'
    },
    "ht3": {
        'name': "PicoQuant-ht3",
        'ending': '.ht3',
        'nTAC': 65535,
        'nROUT': 255,
        'read': tttrlib.ht3
    },
    "iss": {
        'name': "ISS-FCS",
        'ending': '.fcs',
        'nTAC': 1,
        'nROUT': 2,
        'read': tttrlib.iss
    }
}


class Photon(tables.IsDescription):
    ROUT = tables.UInt8Col()
    TAC = tables.UInt32Col()
    MT = tables.UInt64Col()
    FileID = tables.UInt16Col()


class Header(tables.IsDescription):
    DINV = tables.UInt16Col() # DataInvalid,
    NROUT = tables.UInt16Col() # Number of routing channels
    MTCLK = tables.Float32Col() # Macro Time clock
    nTAC = tables.UInt16Col()
    FileID = tables.UInt16Col()
    Filename = tables.StringCol(120)
    routine = tables.StringCol(10)


def read_header(binary, routine_name):
    """
    Reads the header-information of binary TTTR-files. The TTTR-files have to be passed as
    numpy array of type numpy.uint8
    :param binary: numpy-array (dtype=numpy.uint8)
        A numpy array continaing the data of the SPC or HT3 files
    :param routine_name: string
        either 'bh132' or 'ht3'
    :return:
        An dictionary containing the most important header data.
    Example
    -------
    >>> import glob
    >>> import lib
    >>> directory = "./sample_data/tttr/spc132/hGBP1_18D"
    >>> spc_files = glob.glob(directory+'/*.spc')
    >>> b = np.fromfile(spc_files[0], dtype=np.uint8)
    >>> header = lib.io.photons.read_header(b, 'bh132')
    >>> print header
    {'MTCLK': 13.6, 'DINV': 0, 'nEvents': 1200000}
    """
    b = binary
    if routine_name == 'bh132':
        bHeader = np.unpackbits(b[0:4])
        conv8le = np.array([128, 64, 32, 16, 8, 4, 2, 1])
        conv24be = np.array([1, 256, 65536])
        bMTclock = bHeader[0:24]
        b0 = np.dot(bMTclock[0:8], conv8le)
        b1 = np.dot(bMTclock[8:16], conv8le)
        b2 = np.dot(bMTclock[16:24], conv8le)
        MTclock = np.dot(np.array([b0, b1, b2]), conv24be) / 10.
        DataInvalid = int(bHeader[31])
    elif routine_name == 'iss':
        # acquisition frequency in Hz
        frequency = b[2:6].view(dtype=np.uint32)[0]
        MTclock = 1./float(frequency) * 1.e9
        DataInvalid = 0
    elif routine_name == 'ht3':
        # TODO doesnt read header properly!!!!!
        frequency = b[2:6].view(dtype=np.uint32)[0]
        MTclock = 1./float(frequency) * 1.e9
        DataInvalid = 0
    dHeader = {'DINV': DataInvalid, 'MTCLK': MTclock, 'nEvents': b.shape[0]}
    return dHeader


def spc2hdf(spc_files, routine_name="bh132", verbose=False, **kwargs):
    """
    Converts BH-SPC files into hdf file format
    :param spc_files: list
        A list of spc-files
    :param routine_name:
        Name of the used reading routine by default "bh132" alternatively "bh630_x48"
    :param verbose: bool
        By default False
    :param kwargs:
        If the parameter 'filename' is not provided only a temporary hdf (.h5) file is created
        If the parameter 'title' is provided the data is stored in the hdf-group provided by the parameter 'title.
        Otherwise the default 'title' spc is used to store the data within the HDF-File.

    :return: tables.file.File

    Example
    -------
    If the HDF-File doesn't exist it will be created
    >>> import lib
    >>> import glob
    >>> directory = "./sample_data/tttr/spc132/hGBP1_18D"
    >>> spc_files = glob.glob(directory+'/*.spc')
    >>> h5 = lib.io.photons.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D')

    To an existing HDF-File simply a new group with the title will be created
    >>> h5 = lib.io.photons.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D_2')
    After finished work with the HDF-File it should be closed.
    >>> h5.close()
    """
    title = kwargs.get('title', "spc")
    filename = kwargs.get('filename', tempfile.mktemp(".h5"))
    if isinstance(spc_files, str):
        spc_files = [spc_files]
    read = filetypes[routine_name]['read']
    name = filetypes[routine_name]['name']
    nTAC = filetypes[routine_name]['nTAC']
    nROUT = filetypes[routine_name]['nROUT']
    if verbose:
        print("===========================================")
        print(" Reading routine - %s" % name)
    spcs = []
    for i, spc_file in enumerate(spc_files):
        b = np.fromfile(spc_file, dtype=np.uint8)
        header = read_header(b, routine_name)
        nPh, aMT, aTAC, aROUT = read(b)
        spc = {'filename': spc_file, 'header': header,
               'photon': {
                   'ROUT': aROUT[:nPh],
                   'MT': aMT[:nPh],
                   'TAC': aTAC[:nPh]
               }
        }
        spc['photon']['MT'] += max(spcs[-1]['photon']['MT']) if i > 0 else 0
        spcs.append(spc)
        if verbose:
            print("%s: reading photons..." % spc_file)
            print("-------------------------------------------")
            print(" Filename: %s" % filename)
            print(" Macro time clock        : %s" % (header['MTCLK']))
            print(" Number of events        : %i" % ((header['nEvents']) / 4))
    h5 = tables.openFile(filename, mode="a", title=title)
    if verbose:
        print("-------------------------------------------")
        print(" Total number of files: %i " % (len(spc_files)))
        print("===========================================")
        print("HDF-file: %s" % filename)
    filters = tables.Filters(complib='blosc', complevel=9)
    h5.createGroup("/", title, 'Name of measurement: %s' % title)
    headertable = h5.createTable('/' + title, 'header',  description=Header, filters=filters)
    header = headertable.row
    photontable = h5.createTable('/' + title, 'photons', Photon, filters=filters)
    for fileID, spc in enumerate(spcs):
        # Add Header
        header['DINV'] = spc['header']['DINV']
        header['NROUT'] = nROUT
        header['Filename'] = spc['filename']
        header['MTCLK'] = spc['header']['MTCLK']
        header['FileID'] = fileID
        header['routine'] = routine_name
        header['nTAC'] = nTAC
        header.append()
        # Add Photons
        fileID = np.zeros(spc['photon']['MT'].shape, np.uint16) + fileID
        photonA = np.rec.array((fileID, spc['photon']['MT'], spc['photon']['ROUT'], spc['photon']['TAC']))
        photontable.append(photonA)

    photontable.cols.ROUT.createIndex()
    h5.flush()
    if verbose:
        print("Reading done!")
    return h5


def read_BIDs(filenames, stack_files=True):
    """
    Reads Seidel-BID files and returns a list of numpy arrays. Each numpy array contains the indexes of the photons of
    the burst. These indexes can be used to slice a 'photon'-stream.

    Seidel BID-files only contain the first and the last photon of the Burst and not all photons of
    the burst. Thus, the Seidel BID-files have to be converted to array-type objects containing all
    photons of the burst to be able to use standard Python slicing syntax to select photons.
    :param filenames: filename pointing to a Seidel BID-file
    :param stack: bool
        If stack is True the returned list is stacked and the numbering of the bursts is made continuous.
        This is the default behavior.
    :return:

    Example
    -------

    >>> import lib
    >>> import glob
    >>> directory = "./sample_data/tttr/spc132/hGBP1_18D/burstwise_All 0.1200#30\BID"
    >>> files = glob.glob(directory+'/*.bst')
    >>> bids = lib.io.photons.read_BIDs(files)
    >>> bids[1]
    array([20384, 20385, 20386, 20387, 20388, 20389, 20390, 20391, 20392,
       20393, 20394, 20395, 20396, 20397, 20398, 20399, 20400, 20401,
       20402, 20403, 20404, 20405, 20406, 20407, 20408, 20409, 20410,
       20411, 20412, 20413, 20414, 20415, 20416, 20417, 20418, 20419,
       20420, 20421, 20422, 20423, 20424, 20425, 20426, 20427, 20428,
       20429, 20430, 20431, 20432, 20433, 20434, 20435, 20436, 20437,
       20438, 20439, 20440, 20441, 20442, 20443, 20444, 20445, 20446,
       20447, 20448, 20449, 20450, 20451, 20452, 20453, 20454, 20455,
       20456, 20457, 20458, 20459, 20460, 20461, 20462, 20463, 20464,
       20465, 20466, 20467, 20468, 20469, 20470, 20471, 20472, 20473,
       20474, 20475, 20476, 20477, 20478, 20479, 20480, 20481, 20482,
       20483, 20484, 20485, 20486, 20487, 20488, 20489, 20490, 20491,
       20492, 20493, 20494, 20495, 20496, 20497, 20498, 20499, 20500,
       20501, 20502, 20503, 20504, 20505, 20506, 20507, 20508, 20509,
       20510, 20511, 20512, 20513, 20514, 20515, 20516, 20517, 20518,
       20519, 20520, 20521, 20522, 20523, 20524, 20525, 20526, 20527,
       20528, 20529, 20530, 20531, 20532, 20533, 20534, 20535, 20536,
       20537, 20538, 20539, 20540, 20541, 20542, 20543, 20544, 20545,
       20546, 20547, 20548, 20549, 20550, 20551, 20552, 20553, 20554,
       20555, 20556, 20557, 20558, 20559, 20560, 20561, 20562, 20563,
       20564, 20565, 20566, 20567, 20568, 20569, 20570, 20571, 20572,
       20573, 20574, 20575, 20576, 20577, 20578, 20579, 20580, 20581,
       20582, 20583, 20584, 20585, 20586, 20587, 20588, 20589, 20590,
       20591, 20592, 20593, 20594, 20595, 20596, 20597, 20598, 20599,
       20600, 20601, 20602, 20603, 20604, 20605, 20606, 20607, 20608,
       20609, 20610, 20611, 20612, 20613, 20614, 20615, 20616, 20617,
       20618, 20619, 20620, 20621, 20622, 20623, 20624, 20625, 20626,
       20627, 20628, 20629, 20630, 20631, 20632, 20633, 20634, 20635,
       20636, 20637, 20638, 20639, 20640, 20641, 20642, 20643, 20644,
       20645, 20646, 20647, 20648, 20649, 20650, 20651, 20652, 20653,
       20654, 20655], dtype=uint64)
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    re = dict()
    for file in filenames:
        bids = np.loadtxt(file, dtype=np.int32)
        re[file] = [np.arange(bid[0], bid[1], dtype=np.int32) for bid in bids]
    if not stack_files:
        return re
    else:
        b = re[filenames[0]]
        if len(filenames) > 1:
            for i, fn in enumerate(filenames[1:]):
                offset = re[filenames[i]][-1][-1]
                for j in range(len(re[fn])):
                    b.append(re[fn][j] + offset)
        return b


class Photons(object):

    def __init__(self, p_object, file_type=None, **kwargs):
        """

        :param p_object:
            Is either a list of filenames or a single string containing the path to one file. If the first argument
            is n
        :param file_type:
            The file type of the files passed using the first argument (p_object) is specified using the
            'file_type' parameter. This string is either 'hdf' or 'bh132', 'bh630_x48', 'ht3', 'iss'. If
            the file type is not an hdf file the files are temporarily converted to hdf-files to guarantee
            a consistent interface.
        :param kwargs:
        :return:

        Examples
        --------
        >>> import glob
        >>> import lib
        >>> directory = "./sample_data/tttr/spc132/hGBP1_18D"
        >>> spc_files = glob.glob(directory+'/*.spc')
        >>> photons = lib.io.photons.Photons(spc_files, file_type="bh132")
        >>> print photons
        -------------------------------
        File-type: bh132
        Filename(s):
                ./sample_data/tttr/spc132/hGBP1_18D\m000.spc
                ./sample_data/tttr/spc132/hGBP1_18D\m001.spc
                ./sample_data/tttr/spc132/hGBP1_18D\m002.spc
                ./sample_data/tttr/spc132/hGBP1_18D\m003.spc
        nTAC:   4095
        nROUT:  255
        MTCLK [ms]:     1.36000003815e-05

        -------------------------------
        >>> print photons[:10]
        -------------------------------
        File-type: None
        Filename(s):    None
        nTAC:   4095
        nROUT:  255
        MTCLK [ms]:     1.36000003815e-05

        -------------------------------
        """

        if p_object is not None or isinstance(p_object, tables.file.File):
            if isinstance(p_object, str):
                p_object = [p_object]
            if isinstance(p_object, list):
                p_object.sort()
            self._filenames = p_object

            if file_type == 'hdf':
                self._h5 = tables.openFile(p_object[0], mode='r')
            elif file_type in ['bh132', 'bh630_x48', 'ht3', 'iss']:
                self._h5 = spc2hdf(self.filenames, file_type)
        else:
            self._h5 = p_object
            self._filenames = []

        self.filetype = file_type
        self._sample_name = None
        self._cr_filter = None
        self._selection = None
        self._tac = None
        self._mt = None
        self._rout = None
        self._nTAC = None
        self._nROUT = None
        self._MTCLK = None

    @property
    def dt(self):
        return self.MTCLK / self.nTAC

    @property
    def filenames(self):
        return self._filenames

    @property
    def h5(self):
        return self._h5

    @property
    def measTime(self):
        return self.mt[-1] * self.MTCLK / 1000.0

    @property
    def cr_filter(self):
        if self._cr_filter is None:
            return np.ones(self.rout.shape, dtype=np.float32)
        else:
            return self._cr_filter

    @cr_filter.setter
    def cr_filter(self, v):
        self._cr_filter = v

    @property
    def shape(self):
        return self.rout.shape

    @property
    def nPh(self):
        return self.rout.shape[0]

    @property
    def rout(self):
        return self.sample.photons.col('ROUT') if self._rout is None else self._rout

    @rout.setter
    def rout(self, v):
        self._rout = v

    @property
    def tac(self):
        return self.sample.photons.col('TAC') if self._tac is None else self._tac

    @tac.setter
    def tac(self, v):
        self._tac = v

    @property
    def mt(self):
        return self.sample.photons.col('MT') if self._mt is None else self._mt

    @mt.setter
    def mt(self, v):
        self._mt = v

    @property
    def nTAC(self):
        return self.sample.header[0]['nTAC'] if self._nTAC is None else self._nTAC

    @nTAC.setter
    def nTAC(self, v):
        self._nTAC = v

    @property
    def nROUT(self):
        return self.sample.header[0]['NROUT'] if self._nROUT is None else self._nROUT

    @nROUT.setter
    def nROUT(self, v):
        self._nROUT = v

    @property
    def MTCLK(self):
        return self.sample.header[0]['MTCLK']*1e-6 if self._MTCLK is None else self._MTCLK

    @MTCLK.setter
    def MTCLK(self, v):
        self._MTCLK = v

    @property
    def sample(self):
        if isinstance(self.h5, tables.file.File):
            if isinstance(self._sample_name, str):
                return self.h5.getNode('/' + self._sample_name)
            else:
                sample_name = self.sample_names[0]
                return self.h5.getNode('/' + sample_name)
        else:
            return None

    @sample.setter
    def sample(self, v):
        if isinstance(v, str):
            self._sample_name = str

    @property
    def samples(self):
        return [s for s in self.h5.root]

    @property
    def sample_names(self):
        return [sample._v_name for sample in self.h5.root]

    def __str__(self):
        s = ""
        s += "\n-------------------------------\n"
        s += "File-type: %s\n" % self.filetype
        s += "Filename(s):\t"
        if len(self.filenames) > 0:
            s += "\n"
            for fn in self.filenames:
                s += "\t" + fn + "\n"
        else:
            s += "None\n"
        s += "nTAC:\t%d\n" % self.nTAC
        s += "nROUT:\t%d\n" % self.nROUT
        s += "MTCLK [ms]:\t%s\n" % self.MTCLK
        s += "\n-------------------------------\n"
        return s

    def __del__(self):
        if self.h5 is not None:
            self.h5.close()
            if self.filetype in ['bh132', 'bh630_x48']:
                os.unlink(self.readSPC.tempfile)

    def __len__(self):
        return self.nPh

    def __getitem__(self, key):
        re = Photons(None)

        if isinstance(key, int):
            key = np.array(key)
        elif isinstance(key, np.ndarray):
            key = key
        else:
            start = None if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = None if key.step is None else key.step
            key = np.arange(start, stop, step)

        re.tac = self.tac.take(key)
        re.mt = self.mt.take(key)
        re.rout = self.rout.take(key)
        re.cr_filter = self.cr_filter.take(key)

        re.MTCLK = self.MTCLK
        re.nROUT = self.nROUT
        re.nTAC = self.nTAC
        return re


class SpcFileWidget(QtGui.QWidget):
    def __init__(self, parent):
        QtGui.QWidget.__init__(self)
        uic.loadUi('lib/io/ui/spcSampleSelectWidget.ui', self)
        self.parent = parent
        self.filetypes = filetypes

        self.connect(self.actionSample_changed, QtCore.SIGNAL('triggered()'), self.onSampleChanged)
        self.connect(self.actionLoad_sample, QtCore.SIGNAL('triggered()'), self.onLoadSample)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onFileTypeChanged)

    @property
    def sampleName(self):
        try:
            return self.filenames[0] + "_" + self.comboBox.currentText()
        except AttributeError:
            return "--"

    @property
    def dt(self):
        return float(self.doubleSpinBox.value())

    @dt.setter
    def dt(self, v):
        self.doubleSpinBox.setValue(v)

    def onSampleChanged(self):
        index = self.comboBox.currentIndex()
        self._photons.sample = self.samples[index]
        self.dt = float(self._photons.MTCLK/self._photons.nTAC) * 1e6
        self.nTAC = self._photons.nTAC
        self.nROUT = self._photons.nROUT
        self.number_of_photons = self._photons.nPh
        self.measurement_time =  self._photons.measTime
        self.lineEdit_7.setText("%.2f" % self.count_rate)

    @property
    def measurement_time(self):
        return float(self._photons.measTime)

    @measurement_time.setter
    def measurement_time(self, v):
        self.lineEdit_6.setText("%.1f" % v)

    @property
    def number_of_photons(self):
        return int(self.lineEdit_5.value())

    @number_of_photons.setter
    def number_of_photons(self, v):
        self.lineEdit_5.setText(str(v))

    @property
    def rep_rate(self):
        return float(self.doubleSpinBox_2.value())

    @rep_rate.setter
    def rep_rate(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def nROUT(self):
        return int(self.lineEdit_3.text())

    @nROUT.setter
    def nROUT(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def nTAC(self):
        return int(self.lineEdit.text())

    @nTAC.setter
    def nTAC(self, v):
        self.lineEdit.setText(str(v))

    @property
    def filetypes(self):
        return self._file_types

    @filetypes.setter
    def filetypes(self, v):
        self._file_types = v
        self.comboBox_2.addItems(list(v.keys()))

    @property
    def count_rate(self):
        return self._photons.nPh / float(self._photons.measTime) / 1000.0

    def onFileTypeChanged(self):
        self._photons = None
        self.comboBox.clear()
        if self.fileType == "hdf":
            self.comboBox.setDisabled(False)
        else:
            self.comboBox.setDisabled(True)

    @property
    def fileType(self):
        return str(self.comboBox_2.currentText())

    def onLoadSample(self):
        if self.fileType in ("hdf"):
            filenames = [str(QtGui.QFileDialog.getOpenFileName(None, 'Open Photon-HDF', '', 'link file (*.h5)'))]
        else:
            directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
            filenames = [directory + '/' + s for s in os.listdir(directory)]
        self.filenames = filenames
        self._photons = Photons(filenames, self.fileType)
        self.samples = self._photons.samples
        self.comboBox.addItems(self._photons.sample_names)

    @property
    def photons(self):
        return self._photons

