__version__ = '1.0.2'


try:
    import casadi
except Exception as ex:
    raise ImportError('error while importing casadi: ' + str(ex))

try:
    import numpy
except Exception as ex:
    raise ImportError('error while importing numpy: ' + str(ex))


from .nodes import Node
from .links import Link, LinkWithVms
from .origins import OnRamp, MainstreamOrigin
from .destinations import Destination
from .networks import Network
from .simulations import Simulation
from . import control
from . import functional
from . import util
