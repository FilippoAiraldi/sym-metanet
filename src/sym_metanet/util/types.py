from typing import Protocol, TypeVar, Union


class SupportsItem(Protocol):
    """An ABC which is indexable."""

    __slots__ = ()

    def __getitem__(self, i):
        pass

    def __setitem__(self, i, item):
        pass


class SupportsAlgebraOps(Protocol):
    """An ABC that supports simple algebraic computations like +, -, etc."""

    __slots__ = ()

    def __add__(
        self, x: Union["SupportsAlgebraOps", int, float]
    ) -> "SupportsAlgebraOps":
        pass

    def __sub__(
        self, x: Union["SupportsAlgebraOps", int, float]
    ) -> "SupportsAlgebraOps":
        pass


class Variable(SupportsAlgebraOps, SupportsItem):
    "Variable that can also be numerical or symbolic, depending on the engine. It must"
    "be indexable as an array, in case of vector quantities need to be represented."


VarType = TypeVar("VarType", bound=Variable)
