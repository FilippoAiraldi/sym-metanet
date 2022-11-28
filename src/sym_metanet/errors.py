class InvalidNetworkError(Exception):
    """Exception raised when the network has an invalid configuration"""


class EngineNotFoundError(Exception):
    """
    Exception raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    """


class EngineNotFoundWarning(Warning):
    """
    Warning raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    """
