import numpy as np
from copy import deepcopy

from utilities import Data, Logger, importer
from cutoffs import Polynomial, Polynomial_2, dict2cutoff

class Gaussian(object):
    
    def __init__(self, cutoff):
        # If the cutoff is provided as a number, Polynomial function will be used
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
    def __init__(self, cutoff, cone_grad, cone_offset):
        self.globals = {'cutoff': cutoff, 'cone_grad':cone_grad, 'cone_offset':cone_offset}
    
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
        r_cutoff = self.globals["cutoff"]
        cone_grad =  self.globals["cone_grad"]
        cone_offset = self.globals["cone_offset"]
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
                            if np.abs(Ri[1]-Rj[1])<cone_grad*np.abs(Ri[0]-Rj[0])+cone_offset :
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
    def __init__(self,cutoff,Gs,Rct, delta_R, cone_grad, cone_offset):
        self.globals = {'cutoff': cutoff,'Gs': Gs,'Rct': Rct, 'delta_R': delta_R, 'cone_grad':cone_grad, 'cone_offset':cone_offset}
                  
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
        Rct = self.globals["Rct"]
        delta_R = self.globals["delta_R"]
        num_symmetries = len(self.globals["Gs"][symbol])
        fingerprint = [None] * num_symmetries
        #print(num_symmetries)
        for count in range(num_symmetries):
            G = self.globals["Gs"][symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_G2(self, neighborpositions,G['eta'], G['offset'], self.globals["cutoff"], Ri, Rct)
            elif G['type'] == 'G4':
                ridge = calculate_G4(self, neighborpositions,
                                     G['gamma'], G['eta'],
                                     self.globals["cutoff"], Ri, Rct)
            elif G['type'] == 'G6':
                ridge = calculate_G6(self, neighborpositions,
                                     G['gamma'],  G['eta'],
                                     self.globals["cutoff"], Ri, Rct)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
        return fingerprint

# Auxiliary functions #########################################################

def calculate_G2(self, neighborpositions, eta, offset, cutoff, Ri, Rct):
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
        cutoff=Polynomial(6.5).todict()
    Ri : list
        Position of the center turbine. Should be fed as a list of three floats.
    Rct : float
            Distance where cutoff function will start.

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """
    delta_R = self.globals["delta_R"]
    cone_grad =  self.globals["cone_grad"]
    cone_offset = self.globals["cone_offset"]
    cutoff=self.globals["cutoff"]
    cutoff=Polynomial(cutoff,Rct).todict()
    cutoff_2=Polynomial_2(delta_R).todict()
    Rc = cutoff['kwargs']['Rc']
    #Rct = cutoff['kwargs']['Rct']
    cutoff_fxn = dict2cutoff(cutoff)
    cutoff_fxn_2 = dict2cutoff(cutoff_2)
    ridge = 1.  # One aspect of a fingerprint :)
    num_neighbors = len(neighborpositions)   # number of neighboring turbine
    for count in range(num_neighbors):
        Rj = neighborpositions[count]
        Rij = np.linalg.norm(Rj - Ri)
        Rcy = (np.abs(Rj[0]-Ri[0]) * cone_grad) + cone_offset
        Ry = np.abs(Rj[1]-Ri[1])
        args_cutoff_fxn = dict(Rij=Rij)
        args_cutoff_fxn_2 = dict(Rij=Ry, Rc=Rcy)
        ridge *= (1-(np.exp(-eta*(Rij - offset) / (Rc)) *
                    cutoff_fxn(**args_cutoff_fxn) * cutoff_fxn_2(**args_cutoff_fxn_2)))
    return ridge
def calculate_G4(self, neighborpositions, gamma, eta, cutoff, Ri, Rct):
    """Calculate G4 symmetry function.

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' numbers.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian turbine positions.
    gamma : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Polynomial(6.5).todict()
    Ri : list
        Position of the center turbine. Should be fed as a list of three floats.

    Returns
    -------
    ridge : float
        G4 fingerprint.
    """
    delta_R = self.globals["delta_R"]
    cone_grad =  self.globals["cone_grad"]
    cone_offset = self.globals["cone_offset"]
    cutoff=self.globals["cutoff"]
    cutoff=Polynomial(cutoff,Rct).todict()
    cutoff_2=Polynomial_2(delta_R).todict()
    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    cutoff_fxn_2 = dict2cutoff(cutoff_2)
    ridge = 1.
    counts = range(len(neighborpositions))
    for j in counts:
        
        Rj = neighborpositions[j]
        Rij_vector = Rj - Ri
        Rij = np.linalg.norm(Rij_vector)
        Ry = np.abs(Rj[1]-Ri[1])
        Rcy = (np.abs(Rj[0]-Ri[0]) * cone_grad) + cone_offset
        _Rij = dict(Rij=Rij)
        _Rij_2 = dict(Rij=Ry, Rc=Rcy)
        cos_theta_ij = np.dot(Rij_vector,np.array([-1.0,0.])) / Rij
        if cos_theta_ij < -1.:  # Can occur by rounding error.
            cos_theta_ij = -1.
        theta_ij = np.arccos(cos_theta_ij)
        term = np.exp(-gamma * np.abs(theta_ij))
       
        term *= np.exp(-eta * (Rij ** 2.) /
                        (Rc ** 2.))
        term *= cutoff_fxn(**_Rij) * cutoff_fxn_2(**_Rij_2)
        ridge *= (1-term)
    return (ridge)

def calculate_G6(self, neighborpositions, gamma, eta, cutoff, Ri, Rct):
    """Calculate G6 symmetry function.

    Parameters
    ----------
    neighborpositions : numpy.ndarray of float
        Array of Cartesian turbine positions.
    gamma : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Polynomial(6.5).todict()
    Ri : list
        Position of the center turbine. Should be fed as a list of three floats.

    Returns
    -------
    ridge : float
        G6 fingerprint.
    """
    delta_R = self.globals["delta_R"]
    cone_grad =  self.globals["cone_grad"]
    cone_offset = self.globals["cone_offset"]
    cutoff=self.globals["cutoff"]
    cutoff=Polynomial(cutoff,Rct).todict()
    cutoff_2=Polynomial_2(delta_R).todict()
    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    cutoff_fxn_2 = dict2cutoff(cutoff_2)
    ridge = 1.
    counts = range(len(neighborpositions))
    for j in counts:
        
        Rj = neighborpositions[j]
        Rij_vector = Rj - Ri
        Rij = np.linalg.norm(Rij_vector)
        Ry = np.abs(Rj[1]-Ri[1])
        Rcy = (np.abs(Rj[0]-Ri[0]) * cone_grad) + cone_offset
        _Rij = dict(Rij=Rij)
        _Rij_2 = dict(Rij=Ry, Rc=Rcy)
        for k in range(j):
            Rk = neighborpositions[k]
            Rik_vector = Rk - Ri
            Rik = np.linalg.norm(Rik_vector)
            Rky = np.abs(Rk[1] - Ri[1])
            Rcky = (np.abs(Rk[0] - Ri[0]) * cone_grad) + cone_offset
            _Rik = dict(Rij=Rik)
            _Rik_2 = dict(Rij=Rky, Rc=Rcky)
        
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector)/ (Rij*Rik)
            if cos_theta_ijk < -1.: # Can occur by rounding error.
                cos_theta_ijk = -1.
            theta_ijk = np.arccos(cos_theta_ijk)
            term = np.exp(-gamma * np.abs(theta_ijk))
            term *= np.exp(-eta * (max(Rij,Rik) ** 2.) /(Rc ** 2.))
            term *= min(cutoff_fxn(**_Rij),cutoff_fxn(**_Rik)) * min(cutoff_fxn_2(**_Rij_2), cutoff_fxn_2(**_Rik_2))
            ridge *= (1-term)

    return (ridge)
