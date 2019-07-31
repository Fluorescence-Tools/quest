"""ChiSURF - Library

The :py:mod:`mfm` module contains most of the functionality of ChiSURF. It contains
various sub-modules:

1. :py:mod:`.mfm.common`
2. :py:mod:`.mfm.curve`
3. :py:mod:`.mfm.fit`
4. :py:mod:`.mfm.fluorescence`
5. :py:mod:`.mfm.genealogy`
6. :py:mod:`.mfm.widgets`
7. :py:mod:`.mfm.io`
8. :py:mod:`.mfm.math`
9. :py:mod:`.mfm.structure`
10. :py:mod:`.mfm.tools`



Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created simply by
resuming unindented text.


"""
import json
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open('./settings/chisurf_settings.json'))
parse_models = json.load(open('./settings/models.json'))
fortune_properties = settings['fortune']
fits = []
n_threads = settings['n_threads']
verbose = settings['verbose']
__version__ = settings['version']

from .curve import *
import widgets
from .structure import *
from . import fluorescence
from . import experiments
from . import math
from . import ui
from . import io
from . import plots
from .fitting import *
from . import tools


rootNode = Genealogy()
rootNode.name = 'rootNode'


def get_data_curves():
    """
    Returns all curves `mfm.DataCurve` except if the curve is names "Global-fit"
    """
    return [d for d in rootNode.get_descendants() if isinstance(d, mfm.DataCurve) and d.name != "Global-fit"]


def find_object_type(l, object_type):
    """Traverse a list recursively a an return all objects of type `object_type` as
    a list

    :param l: list
    :param object_type: an object type
    :return: list of objects with certain object type
    """
    re = []
    for p in l:
        if isinstance(p, object_type):
            re.append(p)
        if isinstance(p, list) or isinstance(p, mfm.fitting.parameter.AggregatedParameters):
            re += find_object_type(p, object_type)
    return re
