__all__ = [
    "engines",
    "CongestedDestination",
    "Destination",
    "EngineNotFoundWarning",
    "EngineNotFoundError",
    "InvalidNetworkError",
    "Link",
    "LinkWithVsl",
    "MainstreamOrigin",
    "MeteredOnRamp",
    "Network",
    "Node",
    "Origin",
    "SimplifiedMeteredOnRamp",
]

import sym_metanet.engines as engines
from sym_metanet.errors import (
    EngineNotFoundError,
    EngineNotFoundWarning,
    InvalidNetworkError,
)

_notfound = True
for _engine in engines.get_available_engines().keys():
    try:
        engine = engines.use(_engine)
        _notfound = False
        break
    except ImportError:
        continue
if _notfound:
    import warnings

    warnings.warn("No available symbolic engine found.", EngineNotFoundWarning)
del _notfound, _engine

from sym_metanet.blocks.destinations import CongestedDestination, Destination
from sym_metanet.blocks.links import Link, LinkWithVsl
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.origins import (
    MainstreamOrigin,
    MeteredOnRamp,
    Origin,
    SimplifiedMeteredOnRamp,
)
from sym_metanet.network import Network
