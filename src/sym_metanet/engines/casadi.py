from typing import Dict, Literal, Type, Union
import casadi as cs
from sym_metanet.engines.base import SymEngineBase


CSTYPES: Dict[str, Type] = {
    'SX': cs.SX,
    'MX': cs.MX,
}


class CasadiEngine(SymEngineBase[Union[cs.SX, cs.MX]]):
    '''Symbolic engine implemented with the CasADi framework'''

    def __init__(self, type: Literal['SX', 'MX'] = 'SX') -> None:
        '''Instantiates a CasADi engine.

        Parameters
        ----------
        type : {'SX', 'MX'}, optional
            A string that tells the engine with type of symbolic variables to 
            use. Must be either `'SX'` or `'MX'`, at which point the engine 
            employes `casadi.SX` or `casadi.MX` variables, respectively. By 
            default, `'SX'` is used.

        Raises
        ------
        ValueError
            Raises if the provided string `type` is not valid.
        '''
        super().__init__()
        if type not in CSTYPES:
            raise ValueError(
                f'CasADi symbolic type must be in {{{", ".join(CSTYPES)}}}; '
                f'got {type} instead.')
        self.CSXX = CSTYPES[type]

    def Veq(
        self,
        rho: Union[cs.SX, cs.MX],
        v_free: Union[cs.SX, cs.MX],
        a: Union[cs.SX, cs.MX],
        rho_crit: Union[cs.SX, cs.MX]
    ) -> Union[cs.SX, cs.MX]:
        return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))
