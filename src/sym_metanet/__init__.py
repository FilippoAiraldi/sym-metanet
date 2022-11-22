from sym_metanet.errors import \
    InvalidNetworkError, EngineNotFoundWarning, EngineNotFoundError
import sym_metanet.engines as engines

# try to instantiate default engine here
_notfound = True
for _engine in engines.get_available_engines().keys():
    try:
        engine = engines.use(_engine)
        _notfound = False
        break
    except ImportError:
        continue
if _notfound:
    del engine
    import warnings
    warnings.warn('No available symbolic engine found.', EngineNotFoundWarning)
del _notfound, _engine

from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.origins import \
    Origin, MeteredOnRamp, SimpleMeteredOnRamp
from sym_metanet.blocks.destinations import Destination
from sym_metanet.network import Network
