from abc import ABC, abstractmethod
from itertools import count
from typing import ClassVar, Generic, Optional

from sym_metanet.util.types import VarType


class ElementBase:
    """Base class for any element for a highway modelled in METANET."""

    __slots__ = "name"
    __ids: dict[type, count] = {}

    def __init__(self, name: Optional[str] = None) -> None:
        """Instantiates the element with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the element. If `None`, one is automatically created from a counter
            of the class' instancies.
        """
        cls = self.__class__
        if cls in self.__ids:
            _id = self.__ids[cls]
        else:
            _id = count(0)
            self.__ids[cls] = _id
        self.name = name or f"{cls.__name__}{next(_id)}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.name}: {self.__class__.__name__}>"


class ElementWithVars(ElementBase, Generic[VarType], ABC):
    """Base class for any element with states, actions or disturbances."""

    __slots__ = ("states", "next_states", "actions", "disturbances")
    _states: ClassVar[set[str]] = set()
    _actions: ClassVar[set[str]] = set()
    _disturbances: ClassVar[set[str]] = set()

    def __init__(self, name: Optional[str] = None) -> None:
        """Instantiates the element with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the element. If `None`, one is automatically created from a counter
            of the class' instancies.
        """
        super().__init__(name)
        self.states: Optional[dict[str, VarType]] = None
        self.next_states: Optional[dict[str, VarType]] = None
        self.actions: Optional[dict[str, VarType]] = None
        self.disturbances: Optional[dict[str, VarType]] = None

    @property
    def has_states(self) -> bool:
        """Gets whether this element has state variables."""
        return self.states is not None

    @property
    def has_next_states(self) -> bool:
        """Gets whether this element has state variables (computed by stepping the
        dynamics)."""
        return self.next_states is not None

    @property
    def has_actions(self) -> bool:
        """Gets whether this element has control action variables."""
        return self.actions is not None

    @property
    def has_disturbances(self) -> bool:
        """Gets whether this element has disturbance variables."""
        return self.disturbances is not None

    @abstractmethod
    def init_vars(self, *args, **kwargs) -> None:
        """Initializes the variable dicts (`states`, `actions`, `disturbances`) of this
        element."""
        raise NotImplementedError(
            f"Variable initialization not supported for {self.__class__.__name__}."
        )

    @abstractmethod
    def step_dynamics(self, *args, **kwargs) -> dict[str, VarType]:
        """Internal method for stepping the element's dynamics by one time step.

        Returns
        -------
        Dict[str, VarType]
            A dict with the states at the next time step.

        Raises
        ------
        RuntimeError
            Raises if the shapes of the old and new states do not match.
        """
        raise NotImplementedError(
            f"Stepping the dynamics not supported for {self.__class__.__name__}."
        )

    def step(self, *args, **kwargs) -> None:
        """Steps the dynamics of this element."""
        assert self.states is not None, "States not initialized."
        next_states = self.step_dynamics(*args, **kwargs)
        if self.next_states is None:
            self.next_states = {}
        for name, state in self.states.items():
            next_state = next_states[name]
            self.next_states[name] = next_state
            if (
                hasattr(next_state, "shape")
                and hasattr(state, "shape")
                and next_state.shape != state.shape
            ):
                raise RuntimeError("Shapes of new and old states do not match.")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.name}: {self.__class__.__name__}>"
