from tkinter import E
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.origins import \
    Origin, MainstreamOrigin, MeteredOnRamp, SimpleMeteredOnRamp
from sym_metanet.blocks.destinations import Destination
from sym_metanet.network import Network
import sym_metanet.engines as engines


# try to instantiate default engine here
from importlib import import_module
import inspect
_notfound = True
for engine_name, module in engines.get_available_engines().items():
    try:
        module = import_module(module)
        engine_classes = [
            m[1] for m in inspect.getmembers(module, inspect.isclass)
            if issubclass(m[1], engines.core.EngineBase)
            and m[1] != engines.core.EngineBase
        ]
        if not any(engine_classes):
            continue
        engine_class = engine_classes[0]
        _notfound = False
        break
    except ImportError:
        continue

if _notfound:
    import warnings
    warnings.warn('No available symbolic engine found.',
                  engines.EngineNotFoundWarning)
else:
    engine = engine_class()
    del engine_class
del _notfound, engine_name, module, engine_classes, import_module, inspect
