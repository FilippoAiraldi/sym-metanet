import os
import numpy as np

from typing import Tuple, Dict, Any

from ..blocks.links import LinkWithVms
from ..blocks.origins import OnRamp
from .simulations import Simulation


def __savepkl(sim: Simulation, filename: str, **other_data) -> None:
    '''
    Save the simulation outcomes to a .pkl file.
    Parameters
    ----------
        flename : str
            The filename where to save the data to.

        other_data : kwargs
            A dictionary of any additional data to be saved to the file.
            The name 'simulation' will be overwritten.
    '''

    import pickle

    data = other_data
    data['simulation'] = sim

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def __loadpkl(filename: str) -> Tuple[Simulation, Dict[Any, Any]]:
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data.pop('simulation'), data


def __savemat(sim: Simulation, filename: str, **other_data) -> None:
    '''
    Save the simulation outcomes to a .mat file. The simulation must have
    been already run.

    Parameters
    ----------
        flename : str
            The filename where to save the data to.

        other_data : kwargs
            A dictionary of any additional data to be saved to the file.
            The name 'simulation' will be overwritten.
    '''

    from scipy.io import savemat

    # attributes (not lists and arrays containing states, demands, etc)
    sim_attr = ('T', 'eta', 'tau', 'kappa', 'delta', 'rho_max', 'alpha')
    net_attr = ('name', )
    link_attr = ('name', 'nb_seg', 'lanes', 'lengths', 'v_free',
                 'rho_crit', 'a')
    link_vms_attr = ('vms', 'nb_vms')
    origin_attr = ('name', )
    onramp_attr = ('capacity', )

    # create link data
    link_data = {}
    for link in sim.net.links:
        d = {
            **{a: getattr(link, a) for a in link_attr},
            'speed': np.hstack(link.speed[:-1]).reshape(link.nb_seg, -1),
            'density': np.hstack(link.density[:-1]
                                 ).reshape(link.nb_seg, -1),
            'flow': np.hstack(link.flow).reshape(link.nb_seg, -1)
        }
        if isinstance(link, LinkWithVms):
            d['v_ctrl'] = np.hstack(link.v_ctrl).reshape(link.nb_vms, -1)
            d.update({a: getattr(link, a) for a in link_vms_attr})
        link_data[link.name] = d

    # create origin data
    origin_data = {}
    for origin in sim.net.origins:
        d = {
            **{a: getattr(origin, a) for a in origin_attr},
            'queue': np.vstack(origin.queue[:-1]).reshape(1, -1),
            'flow': np.vstack(origin.flow).reshape(1, -1),
            'demand': np.vstack(origin.demand).reshape(1, -1)
        }
        if isinstance(origin, OnRamp):
            d['rate'] = np.vstack(origin.rate).reshape(1, -1)
            d.update({a: getattr(origin, a) for a in onramp_attr})
        origin_data[origin.name] = d

    # NOTE: save turnrates as Node-to-Link string to float as a dictionary with
    # tuples cannot be saved to mat
    turnrates = {f'from {node.name} to {link.name}': rate
                 for (node, link), rate in sim.net.turnrates.items()}

    # assemble all the simulation data to be saved
    data = other_data
    data['simulation'] = {
        **{a: getattr(sim, a) for a in sim_attr},
        'network': {
            **{a: getattr(sim.net, a) for a in net_attr},
            'turnrates': turnrates,
            'links': link_data,
            'origins': origin_data
        }
    }
    savemat(filename, data)


# def __loadmat(filename: str) -> Tuple[Simulation, Dict[Any, Any]]:
#     from scipy.io import loadmat
#     data = loadmat(filename, simplify_cells=True)

#     # extract data
#     simdata = data.pop('simulation')
#     netdata = simdata['network']
#     linkdata = netdata['links']
#     origindata = netdata['origins']

#     # start building basic components

#     HOW DO WE KNOW WHAT IS CONNECTED TO WHAT? No destination and no origin
#     information was saved

#     return simdata


def save_sim(sim: Simulation, filename: str,
             **other_data: Dict[Any, Any]) -> None:
    '''
    Save the simulation results to a mat or pkl file. The simulation must have
    been already run.

    Parameters
    ----------
        flename : str
            The filename where to save the data to. The file extension must 
            either be '.pkl' or '.mat'.

        other_data : kwargs
            A dictionary of any additional data to be saved to the file.
            The name 'simulation' will be overwritten.
    '''
    fmt = os.path.splitext(filename)[-1]
    if fmt == '.mat':
        __savemat(sim, filename, **other_data)
    elif fmt == '.pkl':
        __savepkl(sim, filename, **other_data)
    else:
        raise ValueError(
            f'Invalid saving format {fmt}: expected \'pkl\' or \'mat\'.')


def load_sim(filename: str) -> Tuple[Simulation, Dict[Any, Any]]:
    fmt = os.path.splitext(filename)[-1]

    if fmt == '.mat':
        # return __loadmat(filename)
        raise NotImplementedError('Loading .mat not yet implemented.')
    elif fmt == '.pkl':
        return __loadpkl(filename)
    else:
        raise ValueError(
            f'Invalid file extension {fmt}; must be \'pkl\' or \'mat\'.')
