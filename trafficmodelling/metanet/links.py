import casadi as cs

from ..util import NamedClass, SmartList


class Link(NamedClass):
    '''METANET link'''

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

            lengths, v_free, rho_crit, a : {float}
                Length, free-flow speed, critical density and model parameter 
                of each link's segment. Must be positive.

            name : str, optional
                Name of the link.
        '''

        super().__init__(name=name)

        self.nb_seg = nb_seg
        self.lanes = lanes
        self.lengths = lengths
        self.v_free = v_free
        self.rho_crit = rho_crit
        self.a = a
        self.reset()

    def reset(self):
        # initialize state (speed and rho) and others (flow, speed limits)
        self.density = SmartList()
        self.speed = SmartList()
        self.flow = SmartList()
        self.v_ctrl = SmartList()
        # NOTE: matrix with structural zeroes vs normal zeroes
        self.density.append(cs.DM(self.nb_seg, 1))  # zero matrix
        self.speed.append(cs.DM(self.nb_seg, 1))  # zero matrix
        # self.density.append(cs.vertcat(*[0] * self.nb_seg))
        # self.speed.append(cs.vertcat(*[0] * self.nb_seg))
