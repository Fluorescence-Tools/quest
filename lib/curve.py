from copy import copy, deepcopy
import os.path
import numpy as np
from lib import Genealogy


class Data(Genealogy):

    def __init__(
            self,
            origin=Genealogy
    ):
        self._name = None
        Genealogy.__init__(self)
        self.origin = origin

    @property
    def name(self) -> str:
        if self._name is None:
            try:
                fn = copy(self.filename)
                return fn.replace('/', '/ ')
            except AttributeError:
                return "None"
        else:
            return self._name

    @name.setter
    def name(
            self,
            v: str
    ):
        self._name = v

    def __str__(self):
        s = "origin: " + str(self.origin) + "\n"
        return s


class Curve(object):

    def __init__(self, **kwargs):
        """
        The `Curve`-class takes x and y values of a curve and implements some useful magic-members
         useful when adding, shifting and multiplying curves.

        :param x: array like
        :param y: array like should be same length as x
        """
        self.x = kwargs.get('x', np.array([]))
        self.y = kwargs.get('y', np.array([]))
        self.yo = kwargs.get('y', np.array([]))
        if len(self.y) != len(self.x):
            raise ValueError("x and y should have the same length")

    def norm(self, tp="max", c=None):
        if not isinstance(c, Curve):
            if tp == "sum":
                self.y /= sum(self.y)
            elif tp == "max":
                self.y /= max(self.y)
        else:
            if tp == "sum":
                self.y = self.y / sum(self.y) * sum(c.y)
            elif tp == "max":
                if max(self.y) != 0:
                    self.y = self.y / max(self.y) * max(c.y)

    def reset(self):
        self.y = deepcopy(self.yo)

    def __add__(self, c):
        y = np.array(self.y, dtype=np.float64)
        y += np.array(c.y, dtype=np.float64)
        x = self.x
        return Curve(x, y)

    def __len__(self):
        return len(self.y)

    def __sub__(self, v):
        y = copy(np.array(self.y, dtype=np.float64))
        if isinstance(v, float):
            y -= v
        elif isinstance(v, Curve):
            y -= np.array(v.y, dtype=np.float64)
        c = copy(self)
        c.y = y
        return c

    def __mul__(self, b):
        c = deepcopy(self)
        c.y *= b
        return c

    def __div__(self, other):
        c = deepcopy(self)
        c.y = self.y.astype(dtype=np.float64)
        c.y /= other
        return c

    def __lshift__(self, tsn):
        if not np.isnan(tsn):
            ts = -tsn
            tsi = int(np.floor(ts))
            tsf = ts - tsi
            ysh = np.roll(self.y, tsi) * (1 - tsf) + np.roll(self.y, tsi + 1) * tsf
            if ts > 0:
                ysh[:tsi] = 0.0
            elif ts < 0:
                ysh[tsi:] = 0.0
            c = copy(self)
            c.y = ysh
            return c
        else:
            return self


class DataCurve(Data, Curve):

    def __init__(
            self,
            filename:str = "",
            skiprows: int = 9
    ):
        Curve.__init__(self)
        Data.__init__(self)
        self.filename = filename
        self.skiprows = skiprows
        self.x = []
        self.y = []
        self._ex = []
        self._ey = []
        self._weights = []
        if os.path.isfile(filename):
            self.loadData(filename)

    def __str__(self):
        if self.x is not None and self.y is not None:
            s = "filename: " + self.filename + "\n"
            s += "length  : %s\n" % len(self)
            for (x, y) in zip(self.x[:3], self.y[:3]):
                s += "{0:<12.3e}\t{0:<12.3e}\n".format(x, y)
            s += "....\n"
            for (x, y) in zip(self.x[-3:], self.y[-3:]):
                s += "{0:<12.3e}\t{0:<12.3e}\n".format(x, y)
        else:
            s = "No data\n"
        return s

    def setData(
            self,
            filename: str,
            x: np.array,
            y: np.array,
            weights: np.array
    ):
        self.filename = filename
        self._weights = weights
        self.x = x
        self.y = y
        self._len = len(x)

    def loadData(
            self,
            filename: str
    ):
        if os.path.isfile(filename):
            self.x, self.y = np.loadtxt(filename, unpack=True, skiprows=self.skiprows)
            self._len = len(self.y)
            self._weights = np.ones(self.y.shape)

    def set_weights(
            self,
            w: np.array
    ):
        self._weights = w

    def __getitem__(
            self,
            key: str
    ):
        xmin, xmax = 0, len(self.y)
        start = xmin if key.start is None else key.start
        stop = xmax if key.stop is None else key.stop
        step = 1 if key.step is None else key.step
        x, y, w = self.x[start:stop], self.y[start:stop], self.weights[start:stop:step]
        return x, y, w

    def __sub__(self, v):
        y = np.array(self.y, dtype=np.float64)
        if isinstance(v, Curve):
            y -= v.y
        else:
            y -= v
        y = np.array(y, dtype=np.float64)
        c = copy(self)
        c.y = y
        return c
    
    def __len__(self) -> int:
        try:
            return self._len
        except AttributeError:
            return len(self.y)

    @property
    def dt(self) -> float:
        if len(self.x) > 1:
            return self.x[1] - self.x[0]
        else:
            return 1.0

    @property
    def ex(self) -> np.array:
        return self._ex

    @ex.setter
    def ex(
            self,
            v: np.array
    ):
        self._ex = v

    @property
    def ey(self) -> np.array:
        return self._ey

    @ey.setter
    def ey(
            self,
            v: np.array
    ):
        self._ey = v

    @property
    def weights(self) -> np.array:
        if self._weights is None:
            return 1./self.ey
        else:
            return self._weights

    @weights.setter
    def weights(
            self,
            v: np.array
    ):
        self._weights = v

    def clean(self):
        new = DataCurve()
        new.ey = self.ey
        new.ex = self.ex
        new.x = self.x
        new.y = self.y
        new.weights = self.weights
        return new