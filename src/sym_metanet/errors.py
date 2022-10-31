class InvalidNetworkError(Exception):
    '''Exception raised when the network has an invalid configuration'''


class DuplicateNetworkElementError(InvalidNetworkError):
    '''
    Exception raised when the same element is added more than once to the 
    network
    '''