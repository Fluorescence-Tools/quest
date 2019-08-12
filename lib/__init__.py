import os.path
package_directory = os.path.dirname(os.path.abspath(__file__))
from .genealogy import *
from .curve import *
from .structure import *
from . import math
from . import ui
from . import io
import json
from . import tools
from typing import List

rootNode = Genealogy()
rootNode.name = 'rootNode'
fits = []


def getDataCurves() -> List[Genealogy]:
    """
    Returns all curves `lib.DataCurve` except if the curve is names "Global-fit"
    """
    return [d for d in rootNode.get_descendants() if isinstance(d, lib.DataCurve) and d.name != "Global-fit"]

