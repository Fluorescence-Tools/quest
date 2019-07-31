"""
This module contains a collection of tools
"""

from .tau2r import *
import mfm.tools.modelling.transfer
from mfm.tools.modelling.transfer import Structure2Transfer
from modelling.screening import FPSScreenTrajectory
from .tau2r import LifetimeCalculator
from potential_enery import PotentialEnergyWidget
from . import fret_lines
from . import kappa2dist
from . import dye_diffusion
from .pdb2labeling import PDB2Label
from trajectory import AlignTrajectoryWidget, JoinTrajectoriesWidget, SaveTopology, RotateTranslateTrajectoryWidget, \
    MDConverter, RemoveClashedFrames
