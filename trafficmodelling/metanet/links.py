import numpy as np

from ..util import NamedClass, SmartList


class Link(NamedClass):
    '''METANET link'''

    @property
    def nb_seg(self) -> int:
        return self.__nb_seg

    @property
    def lanes(self) -> int:
        return self.__lanes

    @property
    def lengths(self):
        return self.__lengths

    @property
    def density(self) -> SmartList:
        return self.__rho

    @property
    def speed(self) -> SmartList:
        return self.__v

    @property
    def flow(self) -> SmartList:
        return self.__q

    @property
    def v_ctrl(self) -> SmartList:
        return self.__v_ctrl

    def __init__(self, nb_seg, lanes, lengths, v_free, rho_crit, a,
                 name=None) -> None:
        '''
        Instanciate a link.

        Parameters
        ----------
            nb_seg : int
                Number of segments in this link.

            lanes : int
                Number of lanes in this link's segments. Must be positive.

            lengths, v_free, rho_crit, a : {float, array}
                Length, free-flow speed, critical density and model parameter 
                of each link's segment. If not scalar, shape is (Ns, 1) with Ns
                the number of segments in this link. Must be positive.

            name : str, optional
                Name of the link.
        '''

        super().__init__(name=name)

        # save unmutable params
        self.__nb_seg = nb_seg
        self.__lanes = lanes
        self.__lengths = lengths

        # save mutable params
        self.v_free = v_free
        self.rho_crit = rho_crit
        self.a = a

        self.reset()

    def reset(self):
        # initialize state (speed and rho) and others (flow, speed limits)
        self.__rho = SmartList()
        self.__rho.append(np.zeros((self.__nb_seg, 1)))
        self.__v = SmartList()
        self.__v.append(np.zeros((self.__nb_seg, 1)))
        self.__q = SmartList()
        self.__v_ctrl = SmartList()
