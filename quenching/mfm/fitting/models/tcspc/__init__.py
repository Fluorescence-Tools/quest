"""
This module contains all time-resolved fluorescence models (TCSPC)

.. automodule:: models.tcspc
   :members:
"""
import mix_model
import fret
import dye_diffusion
import tcspc
import parse
import pddem
import et

models = [
    tcspc.LifetimeModelWidget,
    fret.GaussianModelWidget,
    fret.FRETrateModelWidget,
    fret.WormLikeChainModelWidget,
    fret.GaussianChainModelWidget,
    mix_model.MixModelWidget,
    parse.ParseDecayModelWidget,
    pddem.PDDEMModelWidget,
    dye_diffusion.TransientDecayGenerator
]
