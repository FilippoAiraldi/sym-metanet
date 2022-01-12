__version__ = '1.0.0'

try:
    import casadi
except Exception as ex:
    raise ImportError('error while importing casadi: ' + str(ex))

from . import util
from . import metanet
