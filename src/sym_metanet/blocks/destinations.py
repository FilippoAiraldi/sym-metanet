from sym_metanet.util import NamedObject


class Destination(NamedObject):
    '''
    Highway congestion-free destination, representing a sink where cars can 
    leave the highway with no congestion (i.e., slowing down due to downstream 
    density).
    '''


class CongestedDestination(Destination):
    '''
    Destination with a downstream density scenario to emulate congestions, that
    is, cars cannot exit freely the highway but must slow down and, possibly,
    create a congestion.
    '''
