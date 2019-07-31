from copy import copy, deepcopy
import os.path
import uuid
import numpy as np
import mfm


class Genealogy(object):
    """
    Directed tree.
    """

    @property
    def name(self):
        """
        The name of the dataset. The name does not have to be unique
        """
        if self._name is None:
            try:
                fn = copy(self.filename)
                return fn
            except AttributeError:
                return "None"
        else:
            return self._name

    @name.setter
    def name(self, v):
        self._name = v

    def __init__(self, parents=[], **kwargs):
        object.__init__(self)
        self._name = kwargs.get('name', None)
        self._parents = parents
        self._children = []
        self.id = uuid.uuid1()

    def name_dict(self):
        """
        returns a dict as -as_dict- however
        """

    def as_dict(self):
        dict = {self: self.get_children()}
        for child in self.get_children():
            dict.update(child.as_dict())
        return dict

    def from_dict(self, dict):
        for child in dict[self]:
            child.link(self)
            child.from_dict(dict)

    def is_root(self):
        return len(self._parents) == 0

    def get_parents(self):
        return self._parents

    def get_children(self):
        return list(self._children)

    def get_siblings(self):
        if self.is_root():
            return []
        siblings = [p.get_children() for p in self.get_parents()]
        siblings.remove(self)
        return siblings

    def get_ancestors(self, n_generations=1):
        if self.is_root() or n_generations < 1: return []
        parents = self.get_parents()
        ancestors = [parents]
        for p in parents:
            ancestors += p.get_ancestors(n_generations - 1)
        return ancestors

    def get_descendants(self, n_generations=None):
        descendants = [self]
        if n_generations is None:
            for child in self.get_children():
                descendants += child.get_descendants()
            return descendants
        else:
            if n_generations > 0:
                for child in self.get_children():
                    descendants += child.get_descendants(n_generations=n_generations - 1)
            return descendants

    def is_child(self, group):
        return group in self.get_children()

    def has_children(self):
        return len(self.get_children()) > 0

    def link(self, parents):
        self._p_changed = 1
        if type(parents) != list:
            parents = [parents]
        elif parents is None:
            return
        self._parents = parents
        for parent in parents:
            parent._children.append(self)

    def unlink(self, child=None):
        if child is None:
            self._p_changed = 1
            if not self.is_root():
                for parent in self._parents:
                    parent._children.remove(self)
        else:
            for i, c in enumerate(self._children):
                if c is child:
                    self._children.pop(i)

    def trace_back(self):
        trace = [self]
        generation = self
        while not generation.is_root():
            generation = generation.get_parents()
            trace.insert(0, generation)
        return trace

    def _new_id(self):
        self._p_changed = 1
        self.id = uuid.uuid1()

    def __iter__(self):
        # http://stackoverflow.com/questions/5434400/python-is-it-possible-to-make-a-class-iterable-using-the-standard-syntax
        for each in self.get_children():
            yield each
        #for each in self.__dict__.keys():
        #    yield self.__getattribute__(each)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_children()[key]
        else:
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            return self.get_children()[start:stop:step]


class Data(Genealogy):
    """

    Parameters
    ----------

    origin : object
        The origin (usually the setup used to load the data)
    """

    def __init__(self, origin=None, **kwargs):
        Genealogy.__init__(self, **kwargs)
        self.origin = origin
        self._filename = "None"
        self._data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self.set_data(v)

    @property
    def filename(self):
        return os.path.normpath(self._filename)

    @filename.setter
    def filename(self, v):
        self._filename = v

    def __str__(self):
        s = "origin: " + self.origin + "\n"
        return s

    def set_data(self, v):
        self._data = v


class Curve(object):

    """
    The `Curve`-class takes x and y values of a curve and implements some useful magic-members
    useful when adding, shifting and multiplying curves.

    :param x: array like
    :param y: array like should be same length as x
    """

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = v

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self._x = kwargs.get('x', np.array([]))
        self._y = kwargs.get('y', np.array([]))
        self._yo = kwargs.get('y', np.array([]))
        self._name = kwargs.get('name', 'No-name')
        if len(self._y) != len(self._x):
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

    def save(self, filename=None, mode='txt'):
        if filename is None:
            filename = os.path.join(self.name + '_data.txt')
        if mode == 'txt':
            mfm.io.txt_csv.Csv().save(self, filename)

    def load(self, filename, mode='txt'):
        if mode == 'txt':
            csv = mfm.io.txt_csv.Csv()
            csv.load(filename, unpack=True, skiprows=self.skiprows)
            self.x = csv.data_x
            self.y = csv.data_y
            self._weights = csv.error_y


class DataCurve(Data, Curve):
    """
    DataCurve object may contain experimental data. Here, a set of operations are defined which may be used
    with those objects. The same set of operations as for :py:class:`.Curve` objects apply in addition the
    experimental error is considered.
    """

    def __init__(self, filename="", skiprows=9, **kwargs):
        Curve.__init__(self)
        Data.__init__(self)
        self.filename = filename
        self.skiprows = skiprows
        self.name = kwargs.get('name', None)
        self.x = kwargs.get('x', [])
        self.y = kwargs.get('y', [])
        self._ex = kwargs.get('ex', None)
        self._ey = kwargs.get('ey', None)
        self._weights = kwargs.get('weights', None)
        if os.path.isfile(filename):
            self.load(filename)

    def __str__(self):
        try:
            s = "filename: " + self.filename + "\n"
            s += "length  : %s\n" % len(self)
            s += "x\ty\terror-x\terror-y\n"

            lx = self.x[:3]
            ly = self.y[:3]
            lex = self.ex[:3]
            ley = self.ey[:3]

            ux = self.x[-3:]
            uy = self.y[-3:]
            uex = self.ex[-3:]
            uey = self.ey[-3:]

            for i in range(3):
                x, y, ex, ey = lx[i], ly[i], lex[i], ley[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
            s += "....\n"
            for i in range(3):
                x, y, ex, ey = ux[i], uy[i], uex[i], uey[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
        except:
            s = "Problems with data or global-fit"
        return s

    def set_data(self, filename, x, y, weights):
        self.filename = filename
        self._weights = weights
        self.x = x
        self.y = y
        self._len = len(x)

    def set_weights(self, w):
        self._weights = w

    def __getitem__(self, key):
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
    
    def __len__(self):
        try:
            return len(self.y)
        except AttributeError:
            return len(self.y)

    @property
    def dt(self):
        """
        The derivative of x. If the derivative is constant (often the case in measured data) only the first element
        is returned.
        """
        dt = np.diff(self.x)
        if len(set(dt)) == 1:
            return dt[0]
        else:
            return dt

    @property
    def ex(self):
        """
        Error of the x-values
        """
        if isinstance(self._ex, np.ndarray):
            return self._ex
        else:
            return np.zeros_like(self.x)

    @ex.setter
    def ex(self, v):
        self._ex = v

    @property
    def ey(self):
        """
        Error of the y-values
        """
        if isinstance(self._ey, np.ndarray):
            return self._ey
        else:
            return np.ones_like(self.y)

    @ey.setter
    def ey(self, v):
        self._ey = v

    @property
    def weights(self):
        """
        The weights of the y-values = inverse of the error
        """
        if self._weights is None:
            er = np.copy(self.ey)
            er[er == 0] = 1
            return 1. / er
        else:
            return self._weights

    @weights.setter
    def weights(self, v):
        self._weights = v

    def clean(self):
        new = DataCurve()
        new.ey = self.ey
        new.ex = self.ex
        new.x = self.x
        new.y = self.y
        new.weights = self.weights
        return new


class Surface(Genealogy):

    def __init__(self, fit, **kwargs):
        Genealogy.__init__(self, **kwargs)
        self.fit = fit
        self._activeRuns = []
        self._chi2 = []
        self._parameter = []
        self.parameter_names = []

    def clear(self):
        self._chi2 = []
        self._parameter = []

    def save_txt(self, filename, sep='\t'):
        fp = open(filename, 'w')
        s = ""
        for ph in self.parameter_names:
            s += ph + sep
        s += "\n"
        for l in self.values.T:
            for p in l:
                s += "%.5f%s" % (p, sep)
            s += "\n"
        fp.write(s)
        fp.close()

    @property
    def values(self):
        try:
            re = np.vstack(self._parameter)
            re = np.column_stack((re, self.chi2s))
            return re.T
        except ValueError:
            return np.array([[0], [0]]).T

    @property
    def chi2s(self):
        return np.hstack(self._chi2)