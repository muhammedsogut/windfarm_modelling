import numpy as np
from copy import deepcopy

from utilities import Data, Logger, importer
from cutoffs import Cosine, dict2cutoff
#NeighborList = importer('NeighborList')

class Gaussian(object):
    
    def __init__(self, cutoff):
        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        p.cutoff = cutoff.todict()
    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator.
        """
        return self.parameters.tostring()



# Calculators #################################################################


# Neighborlist Calculator
class NeighborlistCalculator:
    """For integration with .utilities.Data
        
        For each turbine, a list of neighbors with offset distances
        is returned.
        
        Parameters
        ----------
        cutoff : float
        Radius above which neighbor interactions are ignored.
        ang_cutoff : float
        Angle in radians above which neighbor interactions are ignored.
        """
    def __init__(self, cutoff, ang_cutoff):
        self.globals = {'cutoff': cutoff, 'ang_cutoff':ang_cutoff}
    
    def calculate(self, turbinelist, turbineposition):
        """For integration with .utilities.Data
            
            For each turbine fed to calculate, a list of neighbors with offset
            distances is returned.
            
            Parameters
            ----------
            turbinelist : list of str
            Turbines in the wind farm.
            turbineposition : list of float
            X and Y coordinates of turbines in the wind farm.
            """
        #print(self.globals)
        r_cutoff = self.globals["cutoff"]
        ang_cutoff =  self.globals["ang_cutoff"]
        nturb = len(turbinelist)
        neighborlist = [None] * nturb
        for count_i in range(nturb):
            neighborlist[count_i]=[]
            Ri=turbineposition[count_i]
            for count_j in range(nturb):
                if (count_i != count_j):
                    Rj = turbineposition[count_j]
                    if Rj[0] < Ri[0]: #only upwind turbines
                        Rij = np.linalg.norm(Rj - Ri)
                        if Rij < r_cutoff:  #within cutoff
                            angij = np.mod(np.arctan2(Ri[1]-Rj[1],Ri[0]-Rj[0])+np.pi,2.*np.pi)-np.pi
                            if np.abs(angij)<ang_cutoff: #within angle of indepence
                                neighborlist[count_i].append(count_j)
    
        return neighborlist

class FingerprintCalculator:
    """For integration with .utilities.Data
        
        Parameters
        ----------
        neighborlist : list of str
        List of neighbors.
        cutoff : float
        Radius above which neighbor interactions are ignored.
        """
    def __init__(self,cutoff,Gs):
        self.globals = {'cutoff': cutoff,'Gs': Gs}
                  
    def calculate(self, turbinelist, turbineposition,neighborlist, symbol):
            
        """Makes a list of fingerprints, one per turbine.
            For each turbine fed to calculate, a list of neighbors with offset
            distances is returned.
            
            Parameters
            ----------
            turbinelist : list of str
            Turbines in the wind farm.
            turbineposition : list of float
            neighborlist : list of str
            List of neighbors.
            X and Y coordinates of turbines in the wind farm.
            """
        self.turbine = turbinelist
        self.neighborlist = neighborlist
        fingerprints = []
        nturb = len(turbinelist)
        for index in range(nturb):
            neighborindices = neighborlist[index]
            neighborpositions = turbineposition[neighborindices]
            indexfp = self.get_fingerprint(index, symbol, turbineposition[index], neighborpositions)
            fingerprints.append(indexfp)
            #print(indexfp)
        
        return fingerprints
    def get_fingerprint(self, index, symbol, turbinepos, neighborpositions):
        """Returns the fingerprint of symmetry function values for turbine
        specified by its index.

        neighborpositions is list of neighbors' Cartesian positions.

        Parameters
        ----------
        index : int
            Index of the center turbine.
        turbinepos : numpy.ndarray of float
            Position of current turbine
        neighborpositions : numpy.ndarray of float
            Array of Cartesian turbine positions.

        Returns
        -------
        symbol, fingerprint : list of float
            fingerprints for turbine specified by its index.
        """
        Ri = turbinepos
        num_symmetries = len(self.globals["Gs"][symbol])
        fingerprint = [None] * num_symmetries
        #print(num_symmetries)
        for count in range(num_symmetries):
            G = self.globals["Gs"][symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_G2(self, neighborpositions, G['eta'],G['offset'], self.globals["cutoff"], Ri)
            elif G['type'] == 'G4':
                ridge = calculate_G4(self, neighborpositions,
                                     G['gamma'], G['zeta'], G['eta'],
                                     self.globals["cutoff"], Ri)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
        return fingerprint
# Auxiliary functions #########################################################


def calculate_G2(self, neighborpositions, eta, offset, cutoff, Ri):
    """Calculate G2 symmetry function

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' numbers.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian turbine positions.
    eta : float
        Parameter of Gaussian symmetry functions.
    offset: float
        offset values for gaussians in G2 fingerprints
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center turbine. Should be fed as a list of three floats.

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """
    cutoff=self.globals["cutoff"]
    cutoff=Cosine(cutoff).todict()
    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    ridge = 0.  # One aspect of a fingerprint :)
    num_neighbors = len(neighborpositions)   # number of neighboring turbine
    for count in range(num_neighbors):
        Rj = neighborpositions[count]
        Rij = np.linalg.norm(Rj - Ri)
        args_cutoff_fxn = dict(Rij=Rij)
        if cutoff['name'] == 'Polynomial':
            args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
        ridge += (np.exp(-eta * ((Rij - offset) ** 2.) / (Rc ** 2.)) *
                    cutoff_fxn(**args_cutoff_fxn))
    return ridge
def calculate_G4(self, neighborpositions, gamma, zeta, eta, cutoff, Ri):
    """Calculate G4 symmetry function.

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' numbers.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian turbine positions.
    gamma : float
        Parameter of Gaussian symmetry functions.
    zeta : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center turbine. Should be fed as a list of three floats.

    Returns
    -------
    ridge : float
        G4 fingerprint.
    """
    cutoff=self.globals["cutoff"]
    cutoff=Cosine(cutoff).todict()
    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    ridge = 0.
    counts = range(len(neighborpositions))
    for j in counts:
        Rij_vector = neighborpositions[j] - Ri
        Rij = np.linalg.norm(Rij_vector)
        cos_theta_ij = np.dot(Rij_vector,np.array([1.0,0.])) / Rij
        if cos_theta_ij < -1.:
                # Can occur by rounding error.
            cos_theta_ij = -1.
        term = (1. + gamma * cos_theta_ij) ** zeta
        term *= np.exp(-eta * (Rij ** 2.) /
                        (Rc ** 2.))
        _Rij = dict(Rij=Rij)
        if cutoff['name'] == 'Polynomial':
            _Rij['gamma'] = cutoff['kwargs']['gamma']
        term *= cutoff_fxn(**_Rij)
        ridge += term
#remove with k instedad of 3 turbine we need to use wind direction and 2 turbine
    ridge *= 2. ** (1. - zeta)
    return (0.1-ridge)
def make_symmetry_functions(turbines, type, etas, offsets=None,
                            zetas=None, gammas=None):
    """Helper function to create Gaussian symmetry functions.
    Returns a list of dictionaries with symmetry function parameters
    in the format expected by the Gaussian class.

    Parameters
    ----------
    elements : list of str
        List of element types to be observed in this fingerprint.
    type : str
        Either G2, G4, or G5.
    etas : list of floats
        eta values to use in G2, G4 or G5 fingerprints
    offsets: list of floats
        offset values to use in G2 fingerprints
    zetas : list of floats
        zeta values to use in G4, and G5 fingerprints
    gammas : list of floats
        gamma values to use in G4, and G5 fingerprints

    Returns
    -------
    G : list of dicts
        A list, each item in the list contains a dictionary of fingerprint
        parameters.
    """
    if type == 'G2':
        offsets = [0.] if offsets is None else offsets
        G = [{'type': 'G2', 'eta': eta, 'offset': offset}
             for eta in etas
             for offset in offsets]
        return G
    elif type == 'G4':
        G = []
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    G.append({'type': 'G4',
                            'eta': eta,
                            'gamma': gamma,
                            'zeta': zeta})
        return G
    raise NotImplementedError('Unknown type: {}.'.format(type))


def Kronecker(i, j):
    """Kronecker delta function.

    Parameters
    ----------
    i : int
        First index of Kronecker delta.
    j : int
        Second index of Kronecker delta.

    Returns
    -------
    int
        The value of the Kronecker delta.
    """
    if i == j:
        return 1
    else:
        return 0
