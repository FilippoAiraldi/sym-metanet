from tkinter import E
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.origins import \
    Origin, MainstreamOrigin, MeteredOnRamp, SimpleMeteredOnRamp
from sym_metanet.blocks.destinations import Destination
from sym_metanet.network import Network
import sym_metanet.engines as engines


# try to instantiate default engine here
_notfound = True
for engine in engines.get_available_engines().keys():
    try:
        engine = engines.use(engine)
        _notfound = False
        break
    except ImportError:
        continue
if _notfound:
    import warnings
    warnings.warn(
        'No available symbolic engine found.', engines.EngineNotFoundWarning)
del _notfound