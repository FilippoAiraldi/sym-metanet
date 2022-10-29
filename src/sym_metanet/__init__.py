from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.origins import \
    Origin, MainstreamOrigin, MeteredOnRamp, SimpleMeteredOnRamp
from sym_metanet.blocks.destinations import Destination
from sym_metanet.network import Network
import sym_metanet.engines as engines


# try to instantiate default engine here
from importlib import import_module
_notfound = True
for engine_class, module in engines.get_available_engines().items():
    try:
        cls = getattr(import_module(module), engine_class)
        _notfound = False
        break
    except ImportError:
        continue

if _notfound:
    import warnings
    warnings.warn('No available symbolic engine found.',
                  engines.EngineNotFoundWarning)
else:
    engine = cls()
    del cls
del _notfound, module, engine_class, import_module
