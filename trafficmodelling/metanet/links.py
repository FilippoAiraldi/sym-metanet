import casadi as cs
import numpy as np

from typing import Union, List

from ..util import NamedClass, SmartList


class Link(NamedClass):
    '''METANET link'''

    def __init__(
            self, nb_seg: int, lanes: int, lengths: float, v_free: float,
            rho_crit: float, a: float, name: str = None) -> None:
        '''
        Instantiate a link.

        Parameters
        ----------
            nb_seg : int
                Number of segments in this link.

            lanes : int
                Number of lanes in this link's segments. Must be positive.

            lengths, v_free, rho_crit, a : {float}
                Length, free-flow speed, critical density and model parameter 
                of each link's segment. Must be positive.

            vms : set, tuple, list of int, optional
                Which segments of this link are equipped with Variable Messsage
                Signs for speed control. 0-based. Defaults to None. 

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
        self.__reset()

    def __reset(self) -> None:
        # initialize state (speed and rho) and others (flow, speed limits)
        self.density = SmartList.from_list([cs.DM(self.nb_seg, 1)])  # zeroes
        self.speed = SmartList.from_list([cs.DM(self.nb_seg, 1)])  # zeroes
        self.flow = SmartList()


class LinkWithVms(Link):
    def __init__(
            self, nb_seg: int, lanes: int, lengths: float, v_free: float,
            rho_crit: float, a: float, vms: List[int],
            name: str = None) -> None:
        '''
        Instantiate a link with Variable Sign Messages for speed control.

        Parameters
        ----------
            Same as Link.

            vms : set, tuple, list of int, optional
                Which segments of this link are equipped with Variable Messsage
                Signs for speed control. 0-based.

        Raises
        ------
            ValueError : segment outside link
                If the vms variable points to a segment outside the link.
        '''

        super().__init__(nb_seg, lanes, lengths, v_free, rho_crit, a, name)

        if not all(0 <= s < nb_seg for s in vms):
            raise ValueError('Segment with VMS not contained in the link.')

        self.v_ctrl = SmartList()
        self.vms = sorted(vms)
        self.has_vms = np.zeros(nb_seg, dtype=bool)
        for i in self.vms:
            self.has_vms[i] = True
        self.nb_vms = len(self.vms)

    def v_ctrl_at(self, k: int, seg: Union[int, slice] = None):
        '''
        Returns the control speed at time k for a segment (infinity if the 
        segment has no vms). If None, returns for all segments.
        '''

        if seg is None:
            seg = slice(None, None, None)
        elif isinstance(seg, int):
            return (self.v_ctrl[k][self.vms.index(seg)]
                    if self.has_vms[seg] else
                    cs.inf)

        # seg is a slice
        return cs.vertcat(*[self.v_ctrl_at(k, s)
                            for s in range(*seg.indices(self.nb_seg))])

    def __repr__(self) -> str:
        return f'Link (vms: {str(self.vms)[1:-1]}) {self.name}'
