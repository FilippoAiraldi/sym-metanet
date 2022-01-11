# version
__version__ = '1.0.0'

# check casadi is available
try:
    import casadi
except Exception as ex:
    raise ImportError('error while importing casadi: ' + str(ex))

# add modules
from . import util
from . import metanet
