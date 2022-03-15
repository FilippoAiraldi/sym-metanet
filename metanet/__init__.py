__version__ = '1.0.2'


try:
    import casadi
except Exception as ex:
    raise ImportError(f'error while importing casadi: {str(ex)}') from ex

try:
    import numpy
except Exception as ex:
    raise ImportError(f'error while importing numpy: {str(ex)}') from ex


from . import util
from .blocks.nodes import Node
from .blocks.links import Link, LinkWithVms
from .blocks.origins import Origin, OnRamp, MainstreamOrigin
from .blocks.destinations import Destination
from .blocks.networks import Network
from .sim import functional
from .sim.simulations import Simulation
from .sim import io
from . import ctrl as control # hides the additional subfolders in ctrl
