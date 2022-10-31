class DuplicateNetworkElementError(Exception):
    '''
    Exception raised when the same element is added more than once to the 
    network
    '''


class DuplicateLinkError(DuplicateNetworkElementError):
    '''
    Exception raised when the same link is added more than once to the network
    '''
