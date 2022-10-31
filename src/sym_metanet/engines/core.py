from abc import ABC, abstractmethod
from typing import Dict, Literal, Tuple, Type, Union
import sym_metanet


class NodesEngineBase(ABC):
    '''
    Abstract class of a symbolic engine for modelling highway nodes via the 
    METANET framework. The methods of this class implement the various 
    equations proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''


class LinksEngineBase(ABC):
    '''
    Abstract class of a symbolic engine for modelling highway link via the 
    METANET framework. The methods of this class implement the various 
    equations proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''

    @staticmethod
    @abstractmethod
    def get_flow(rho, v, lanes):
        '''Computes the flows of the link's segments, according to 
        [1, Equation 3.1].

        Parameters
        ----------
        rho
            Densities of the link's segments.
        v_free
            Speeds of the link's segments.
        lanes
            Number of lanes in the link.

        Returns
        -------
        flow
            The flow in each link's segment.
        '''
        pass

    @staticmethod
    @abstractmethod
    def step_density(rho, q, q_up, lanes, L, T):
        '''Computes the densities of the link's segments at the next time 
        instant, according to [1, Equation 3.2].

        Parameters
        ----------
        rho
            Densities of this link's segments.
        q
            Flows of this link's segments.
        q_up
            Flows of the upstream link's segments.
        lanes
            Number of lanes in the link.
        L
            Link's segment length.
        T
            Sampling time.

        Returns
        -------
        densities_next
            The density in each link's segment at the next time instant.
        '''
        pass

    @staticmethod
    @abstractmethod
    def step_speed(v, v_up, rho, rho_down, Veq, lanes, L, tau, eta, kappa, T,
                   q_ramp=None, delta=None,
                   lanes_drop=None, phi=None, rho_crit=None):
        '''Computes the speeds of the link's segments at the next time instant,
        according to [1, Equation 3.3]. If `q_ramp` and `delta` are provided, 
        then it also accounts for merging phenomum [1, Equation 3.7]. 
        Similarly, if `lane_drop` and `phi` are provided, then it also accounts 
        for lane drops in the incoming link [1, Equation 3.8].

        Parameters
        ----------
        v
            Speeds of this link's segments.
        v_up
            Speeds of the upstream link's segments.
        rho
            Densities of this link's segments.
        rho_down
            Densities of the downstream link's segments.
        Veq
            Equilibrium speed of this link.
        lanes
            Number of lanes in the link.
        L
            Link's segment length.
        tau
            Model parameter for the relaxation term.
        eta
            Model parameter for the anticipation term.
        kappa
            Model parameter for the anticipation term.
        T
            Sampling time.
        q_ramp : optional
            If provided, it represents the flow of an on-ramp attached to the 
            first segment of this link, and is used to compute merging 
            phenomenum of on-ramp traffic with link traffic.
        delta : optional
            Model parameter for merging phenomenum. Necessary only if `q_ramp` 
            is provided as well. 
        lanes_drop : optional
            If provided, it represents the difference of lanes of this link 
            minus the lanes in the next link, and is used to compute the impact
            of the lane drop on speeds.
        phi : optional
            Model parameter for lane drop phenomenum. Necessary only if 
            `lanes_drop` and `rho_crit` are provided as well.
        rho_crit : optional 
            Critical density of the link. Necessary only if `lanes_drop` 
            and `phi` are provided as well.

        Returns
        -------
        speeds_next
            The speed in each link's segment at the next time instant.
        '''
        pass

    @staticmethod
    @abstractmethod
    def Veq(rho, v_free, rho_crit, a):
        '''Computes the equilibrium speed of the link, according to 
        [1, Equation 3.4].

        Parameters
        ----------
        rho
            Densities of the link's segments.
        v_free
            Free-flow speed of the link.
        rho_crit
            Critical density of the link.
        a
            Model parameter in the equilibrium speed exponent. 

        Returns
        -------
        Veq
            The equilibrium speed of the link.
        '''
        pass


class OriginsEngineBase(ABC):
    '''
    Abstract class of a symbolic engine for modelling highway origins via the 
    METANET framework. The methods of this class implement the various 
    equations proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''

    @staticmethod
    @abstractmethod
    def step_queue(w, d, q, T):
        '''Computes the queue of the origin at the next time instant, according
        to [1, Section 3.2.1].

        Parameters
        ----------
        w
            Queue of this origin.
        d
            Demand at this origin.
        q
            Outflow from this origin.
        T
            Sampling time.

        Returns
        -------
        queue_next
            The queue of the origin at the next time instant.
        '''
        pass

    @staticmethod
    @abstractmethod
    def get_flow(d, w, C, r, rho_max, rho_first, rho_crit, T,
                 type: Literal['in', 'out'] = 'out'):
        '''Computes the flows of the origin, according to 
        [1, Equation 3.5], if `type=='in'`, or [1, Equation 3.6], if 
        `type=='out'`.

        Parameters
        ----------
        d
            Demand at the origin.
        w
            Queue of the origin.
        C
            Capacity of the origin.
        r
            Metering rate (control action) of the origin.
        rho_first
            Density of the first segment of the link the origin is attached to.
        rho_max
            Maximum density of the link the origin is attached to.
        rho_crit
            Critical density of the link the origin is attached to.
        T
            Sampling time.
        type : 'in' or 'out', optional
            Whether the metering rate `r` should be inside or outside the min 
            function, by default 'out'. See Equation 3.5 and  3.6 for more 
            details.

        Returns
        -------
        flow
            The flow of the origin.
        '''
        pass


class EngineBase(ABC):
    '''
    Abstract class of a symbolic engine for modelling highways via the 
    METANET framework. The methods of this class implement the various 
    equations proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''

    @property
    @abstractmethod
    def nodes(self) -> Type[NodesEngineBase]:
        pass

    @property
    @abstractmethod
    def links(self) -> Type[LinksEngineBase]:
        pass

    @property
    @abstractmethod
    def origins(self) -> Type[OriginsEngineBase]:
        pass

    @abstractmethod
    def var(self, name: str, shape: Tuple[int, ...], *args, **kwargs):
        '''Creates a variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        shape : Tuple[int, ...]
            Shape of the variable.

        Returns
        -------
        sym variable
            The symbolic variable (T is generic).
        '''
        pass


class EngineNotFoundError(Exception):
    '''
    Exception raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


class EngineNotFoundWarning(Warning):
    '''
    Warning raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


def get_current_engine() -> EngineBase:
    '''Gets the current symbolic engine.

    Returns
    -------
    EngineBase
        The current symbolic engine.
    '''
    return sym_metanet.engine


def get_available_engines() -> Dict[str, Dict[str, str]]:
    '''Returns the available symbolic engines for METANET modelling, for which
    an implementation exists.

    Returns
    -------
    Dict[str, Dict[str, str]]
        The available engines in the form of a `dict` whose keys are the 
        available engine class types, and the values are info on each engine, 
        such as module and class name.
    '''
    return {
        'casadi': {'module': 'sym_metanet.engines.casadi', 'class': 'Engine'}
    }


def use(engine: Union[str, EngineBase], *args, **kwargs) -> EngineBase:
    '''Uses the given symbolic engine for computations.

    Parameters
    ----------
    engine : str or instance of engine
        If a string, then `engine` must represent the class name of one of the 
        available engines, so that it can be instantiated with args and kwargs.
        (see `sym_metanet.engines.get_available_engines`). Otherwise, it must 
        be an instance of an engine itself that inherits from 
        `sym_metanet.engines.EngineBase`.
    args, kwargs
        Passed to the engine constructor in case `engine` is a string.

    Returns
    -------
    EngineBase
        A reference to the new engine, if `engine` is a string, or the 
        reference to the same instance, otherwise. 

    Raises
    ------
    ValueError
        Raises in case `engine` is neither a string nor a `EngineBase` 
        instance.
    EngineNotFoundError
        Raises in case `engine` is a string but matches no available engines.
    '''
    if isinstance(engine, EngineBase):
        sym_metanet.engine = engine
    elif isinstance(engine, str):
        engines = get_available_engines()
        if engine not in engines:
            raise EngineNotFoundError(
                f'Engine class must be in {{{", ".join(engines)}}}; got '
                f'{engine} instead.')
        from importlib import import_module
        engineinfo = engines[engine]
        cls = getattr(import_module(engineinfo['module']), engineinfo['class'])
        sym_metanet.engine = cls(*args, **kwargs)
    else:
        raise ValueError('Expected `engine` to be either a string or an '
                         f'instance of an engine; got {engine} instead.')
    return sym_metanet.engine