import casadi as cs


def TTS(w, rho, T, lanes, L, reduce='sum'):
    '''
    Total Time Spent cost.

    Parameters
    ----------
        w : {float, numpy or casadi array} 
            Ramp queues at each timestep with shape (Nr, Nt), where Nr is the 
            number of ramps, and Nt the number of timesteps.
        rho : {float, numpy or casadi array} 
            Segment densities at each timestep with shape (Ns, Nt), where Ns is 
            the number of segments, and Nt the number of timesteps.
        T : {float} 
            simulation step size.
        lanes : {float, numpy or casadi array} 
            Number of lanes per segment with same size as rho, if not scalar.
        L : {float, numpy or casadi array} 
            Number of lanes per segment with same size as rho, if not scalar.
        reduce : {'sum', None}, optional
            How to reduce the cost per timestep to a scalar.

    Returns
    -------
        TTS cost : {float, numpy or casadi array} 
            the total-time-spent cost per vehicle in the links and in the 
            queues, reduced along the time axis accordingly.
    '''

    J = T * (cs.sum1(rho * lanes * L) + cs.sum1(w))
    if reduce == 'sum':
        return cs.sum2(J)
    return J
