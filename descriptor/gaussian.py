import numpy as np
from copy import deepcopy

from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
from ..utilities import Data, Logger, importer
from .cutoffs import Cosine, dict2cutoff
NeighborList = importer('NeighborList')
try:
    from .. import fmodules
except ImportError:
    fmodules = None


class Gaussian(object):
    """Class that calculates Gaussian fingerprints (i.e., Behler-style).

    Parameters
    ----------
    cutoff : object or float
        Cutoff function, typically from amp.descriptor.cutoffs.  Can be also
        fed as a float representing the radius above which neighbor
        interactions are ignored; in this case a cosine cutoff function will be
        employed.  Default is a 6.5-Angstrom cosine cutoff.
    Gs : dict or list
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.,
               ...              "offset": 2.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

        You can use amp.model.gaussian.make_symmetry_functions to help create
        these lists of dictionaries.  If you supply a list instead of a
        dictionary, it will assume you want identical symmetry functions for
        each element.
    dblabel : str
        Optional separate prefix/location for database files, including
        fingerprints, fingerprint derivatives, and neighborlists. This file
        location can be shared between calculator instances to avoid
        re-calculating redundant information. If not supplied, just uses the
        value from label.
    elements : list
        List of allowed elements present in the system. If not provided, will
        be found automatically.
    version : str
        Version of fingerprints.
    fortran : bool
        If True, will use fortran modules (if compiled), if False, will not.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.

    Raises
    ------
        RuntimeError
    """

    def __init__(self, cutoff=Cosine(6.5), Gs=None, dblabel=None,
                 elements=None, version=None, fortran=True,
                 mode='atom-centered'):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use Gaussian fingerprints'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               ' newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # Check that the mode is atom-centered.
        if mode != 'atom-centered':
            raise RuntimeError('Gaussian scheme only works '
                               'in atom-centered mode. %s '
                               'specified.' % mode)

        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        if isinstance(cutoff, int) or isinstance(cutoff, float):
            cutoff = Cosine(cutoff)
        # If the cutoff is provided as a dictionary, assume we need to load it
        # with dict2cutoff.
        if type(cutoff) is dict:
            cutoff = dict2cutoff(cutoff)

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.gaussian.Gaussian',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.todict()
        p.Gs = Gs
        p.elements = elements

        self.dblabel = dblabel
        self.fortran = fortran
        if fmodules is None:
            self.fortran = False
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator.
        """
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, parallel=None, log=None,
                               calculate_derivatives=False):
        """Calculates the fingerpints of the images, for the ones not already
        done.

        Parameters
        ----------
        images : dict
            Dictionary of images; the key is a unique ID assigned to each
            image and each value is an ASE atoms object. Typically created
            from amp.utilities.hash_images.
        parallel : dict
            Configuration for parallelization. Should be in same form as in
            amp.Amp.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        calculate_derivatives : bool
            Decides whether or not fingerprintprimes should also be calculated.
        """
        if parallel is None:
            parallel = {'cores': 1}
        log = Logger(file=None) if log is None else log

        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
            self.dblabel = self.parent.dblabel
        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel

        p = self.parameters

        log('Cutoff function: %s' % repr(dict2cutoff(p.cutoff)))

        if p.elements is None:
            log('Finding unique set of elements in training data.')
            p.elements = set([atom.symbol for atoms in images.values()
                              for atom in atoms])
        p.elements = sorted(p.elements)
        log('%i unique elements included: ' % len(p.elements) +
            ', '.join(p.elements))

        if p.Gs is None:
            log('No symmetry functions supplied; creating defaults.')
            p.Gs = make_default_symmetry_functions(p.elements)
        elif not hasattr(p.Gs, 'keys'):
            log('List of symmetry functions supplied; assumed identical for '
                'each element.')
            p.Gs = {element: deepcopy(p.Gs) for element in p.elements}
        log('Number of symmetry functions for each element:')
        for _ in p.Gs.keys():
            log(' %2s: %i' % (_, len(p.Gs[_])))
        for element, fingerprints in p.Gs.items():
            log('{} feature vector functions:'.format(element))
            for index, fp in enumerate(fingerprints):
                if fp['type'] == 'G2':
                    log(' {}: {}, {}, eta = {}, offset = {}'
                        .format(index, fp['type'], fp['element'], fp['eta'],
                                fp['offset']))
                elif fp['type'] == 'G4':
                    log(' {}: {}, ({}, {}), eta={}, gamma={}, zeta={}'
                        .format(index, fp['type'], fp['elements'][0],
                                fp['elements'][1], fp['eta'], fp['gamma'],
                                fp['zeta']))
                elif fp['type'] == 'G5':
                    log(' {}: {}, ({}, {}), eta={}, gamma={}, zeta={}'
                        .format(index, fp['type'], fp['elements'][0],
                                fp['elements'][1], fp['eta'], fp['gamma'],
                                fp['zeta']))
                else:
                    log(str(fp))

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            calc = NeighborlistCalculator(cutoff=p.cutoff['kwargs']['Rc'])
            self.neighborlist = \
                Data(filename='%s-neighborlists' % self.dblabel,
                     calculator=calc)
        self.neighborlist.calculate_items(images, parallel=parallel, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         Gs=p.Gs,
                                         cutoff=p.cutoff,
                                         fortran=self.fortran)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, parallel=parallel, log=log)
        log('...fingerprints calculated.', toc='fp')

        if calculate_derivatives:
            log('Calculating fingerprint derivatives...',
                tic='derfp')
            if not hasattr(self, 'fingerprintprimes'):
                calc = \
                    FingerprintPrimeCalculator(neighborlist=self.neighborlist,
                                               Gs=p.Gs,
                                               cutoff=p.cutoff,
                                               fortran=self.fortran)
                self.fingerprintprimes = \
                    Data(filename='%s-fingerprint-primes'
                         % self.dblabel,
                         calculator=calc)
            self.fingerprintprimes.calculate_items(
                images, parallel=parallel, log=log)
            log('...fingerprint derivatives calculated.', toc='derfp')


# Calculators #################################################################


# Neighborlist Calculator
class NeighborlistCalculator:
    """For integration with .utilities.Data

    For each image fed to calculate, a list of neighbors with offset distances
    is returned.

    Parameters
    ----------
    cutoff : float
        Radius above which neighbor interactions are ignored.
    """
    def __init__(self, cutoff):
        self.globals = Parameters({'cutoff': cutoff})
        self.keyed = Parameters()
        self.parallel_command = 'calculate_neighborlists'

    def calculate(self, image, key):
        """For integration with .utilities.Data

        For each image fed to calculate, a list of neighbors with offset
        distances is returned.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        cutoff = self.globals.cutoff
        n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                         self_interaction=False,
                         bothways=True,
                         skin=0.)
        n.update(image)
        return [n.get_neighbors(index) for index in range(len(image))]


class FingerprintCalculator:
    """For integration with .utilities.Data

    Parameters
    ----------
    neighborlist : list of str
        List of neighbors.
    Gs : dict
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.,
               ...              "offset": 2.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    cutoff : float
        Radius above which neighbor interactions are ignored.
    fortran : bool
        If True, will use fortran modules, if False, will not.
    """
    def __init__(self, neighborlist, Gs, cutoff, fortran):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'
        self.fortran = fortran

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprints = []
        chemical_symbols = np.array(image.get_chemical_symbols())
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighborindices, neighboroffsets = nl[index]
            neighborsymbols = chemical_symbols[neighborindices]
            neighborpositions = (image.positions[neighborindices] +
                                 np.dot(neighboroffsets, image.get_cell()))
            indexfp = self.get_fingerprint(
                index, symbol, neighborsymbols, neighborpositions)
            fingerprints.append(indexfp)

        return fingerprints

    def get_fingerprint(self, index, symbol,
                        neighborsymbols, neighborpositions):
        """Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol.

        neighborsymbols and neighborpositions are lists of neighbors' symbols
        and Cartesian positions, respectively.

        Parameters
        ----------
        index : int
            Index of the center atom.
        symbol : str
            Symbol of the center atom.
        neighborsymbols : numpy.ndarray of str
            Array of neighbors' symbols.
        neighborpositions : numpy.ndarray of float
            Array of Cartesian atomic positions.

        Returns
        -------
        symbol, fingerprint : list of float
            fingerprints for atom specified by its index and symbol.
        """
        Ri = self.atoms[index].position

        num_symmetries = len(self.globals.Gs[symbol])
        fingerprint = [None] * num_symmetries
        neighbornumbers = [atomic_numbers[_] for _ in neighborsymbols]
        for count in range(num_symmetries):
            G = self.globals.Gs[symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_G2(neighbornumbers, neighborsymbols,
                                     neighborpositions, G['element'], G['eta'],
                                     G['offset'], self.globals.cutoff, Ri,
                                     self.fortran)
            elif G['type'] == 'G4':
                ridge = calculate_G4(neighbornumbers, neighborsymbols,
                                     neighborpositions, G['elements'],
                                     G['gamma'], G['zeta'], G['eta'],
                                     self.globals.cutoff, Ri, self.fortran)
            elif G['type'] == 'G5':
                ridge = calculate_G5(neighbornumbers, neighborsymbols,
                                     neighborpositions, G['elements'],
                                     G['gamma'], G['zeta'], G['eta'],
                                     self.globals.cutoff, Ri, self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
        return symbol, fingerprint


class FingerprintPrimeCalculator:
    """For integration with .utilities.Data

    Parameters
    ----------
    neighborlist : list of str
        List of neighbors.
    Gs : dict
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.,
               ...              "offset": 2.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    cutoff : float
        Radius above which neighbor interactions are ignored.
    fortran : bool
        If True, will use fortran modules, if False, will not.
    """

    def __init__(self, neighborlist, Gs, cutoff, fortran):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprint_primes'
        self.fortran = fortran

    def calculate(self, image, key):
        """Makes a list of fingerprint derivatives, one per atom,
        for the fed image.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprintprimes = {}
        chemical_symbols = np.array(image.get_chemical_symbols())
        for atom in image:
            selfsymbol = atom.symbol
            selfindex = atom.index
            selfneighborindices, selfneighboroffsets = nl[selfindex]
            selfneighborsymbols = chemical_symbols[selfneighborindices]
            selfneighborpositions = (image.positions[selfneighborindices] +
                                     np.dot(selfneighboroffsets,
                                            image.get_cell()))

            for direction in range(3):
                # Calculating derivative of fingerprints of self atom w.r.t.
                # coordinates of itself.
                fpprime = self.get_fingerprintprime(
                    selfindex, selfsymbol,
                    selfneighborindices,
                    selfneighborsymbols,
                    selfneighborpositions, selfindex, direction)

                fingerprintprimes[(selfindex, selfsymbol, selfindex,
                                   selfsymbol, direction)] = fpprime
                # Calculating derivative of fingerprints of neighbor atom
                # w.r.t. coordinates of self atom.
                for nindex, nsymbol, noffset in \
                        zip(selfneighborindices,
                            selfneighborsymbols,
                            selfneighboroffsets):
                    # for calculating forces, summation runs over neighbor
                    # atoms of type II (within the main cell only)
                    if (noffset[0] == 0 and noffset[1] == 0 and noffset[2] == 0):
                        nneighborindices, nneighboroffsets = nl[nindex]
                        nneighborsymbols = chemical_symbols[nneighborindices]
                        neighborpositions = (image.positions[nneighborindices] +
                                             np.dot(nneighboroffsets,
                                                    image.get_cell()))

                        # for calculating derivatives of fingerprints,
                        # summation runs over neighboring atoms of type
                        # I (either inside or outside the main cell)
                        fpprime = self.get_fingerprintprime(
                            nindex, nsymbol,
                            nneighborindices,
                            nneighborsymbols,
                            neighborpositions, selfindex, direction)

                        fingerprintprimes[(selfindex, selfsymbol, nindex,
                                           nsymbol, direction)] = fpprime

        return fingerprintprimes

    def get_fingerprintprime(self, index, symbol,
                             neighborindices,
                             neighborsymbols,
                             neighborpositions,
                             m, l):
        """ Returns the value of the derivative of G for atom with index and
        symbol with respect to coordinate x_{l} of atom index m.

        neighborindices, neighborsymbols and neighborpositions are lists of
        neighbors' indices, symbols and Cartesian positions, respectively.

        Parameters
        ----------
        index : int
            Index of the center atom.
        symbol : str
            Symbol of the center atom.
        neighborindices : list of int
            List of neighbors' indices.
        neighborsymbols : numpy.ndarray of str
            Array of neighbors' symbols.
        neighborpositions : numpy.ndarray of float
            Array of Cartesian atomic positions.
        m : int
            Index of the pair atom.
        l : int
            Direction of the derivative; is an integer from 0 to 2.

        Returns
        -------
        fingerprintprime : list of float
            The value of the derivative of the fingerprints for atom with index
            and symbol with respect to coordinate x_{l} of atom index m.
        """

        num_symmetries = len(self.globals.Gs[symbol])
        Rindex = self.atoms.positions[index]
        fingerprintprime = [None] * num_symmetries
        neighbornumbers = [atomic_numbers[_] for _ in neighborsymbols]

        for count in range(num_symmetries):
            G = self.globals.Gs[symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_G2_prime(
                    neighborindices,
                    neighbornumbers,
                    neighborsymbols,
                    neighborpositions,
                    G['element'],
                    G['eta'],
                    G['offset'],
                    self.globals.cutoff,
                    index,
                    Rindex,
                    m,
                    l,
                    self.fortran)
            elif G['type'] == 'G4':
                ridge = calculate_G4_prime(
                    neighborindices,
                    neighbornumbers,
                    neighborsymbols,
                    neighborpositions,
                    G['elements'],
                    G['gamma'],
                    G['zeta'],
                    G['eta'],
                    self.globals.cutoff,
                    index,
                    Rindex,
                    m,
                    l,
                    self.fortran)
            elif G['type'] == 'G5':
                ridge = calculate_G5_prime(
                    neighborindices,
                    neighbornumbers,
                    neighborsymbols,
                    neighborpositions,
                    G['elements'],
                    G['gamma'],
                    G['zeta'],
                    G['eta'],
                    self.globals.cutoff,
                    index,
                    Rindex,
                    m,
                    l,
                    self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])

            fingerprintprime[count] = ridge

        return fingerprintprime

# Auxiliary functions #########################################################


def calculate_G2(neighbornumbers, neighborsymbols, neighborpositions,
                 G_element, eta, offset, cutoff, Ri, fortran):
    """Calculate G2 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled).  See Eq. 13a of the supplementary information of Khorshidi,
    Peterson, CPC(2016).

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_element : str
        Chemical symbol of the center atom.
    eta : float
        Parameter of Gaussian symmetry functions.
    offset: float
        offset values for gaussians in G2 fingerprints
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]

        if len(neighbornumbers) == 0:
            ridge = 0.
        else:
            Rc = cutoff['kwargs']['Rc']
            if cutoff['name'] == 'Cosine':
                cutofffn_code = 1
            elif cutoff['name'] == 'Polynomial':
                cutofffn_code = 2
            else:
                print("Unknown cutoff function specified! \
                Only supports 'Cosine' and 'Polynomial'.")

            args_calculate_g2 = dict(
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_number=G_number,
                    g_eta=eta,
                    offset=offset,
                    rc=Rc,
                    cutofffn_code=cutofffn_code,
                    ri=Ri
                    )
            if cutofffn_code == 2:
                args_calculate_g2['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g2(**args_calculate_g2)

    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        num_neighbors = len(neighborpositions)   # number of neighboring atoms
        for count in range(num_neighbors):
            symbol = neighborsymbols[count]
            Rj = neighborpositions[count]
            if symbol == G_element:
                Rij = np.linalg.norm(Rj - Ri)
                args_cutoff_fxn = dict(Rij=Rij)
                if cutoff['name'] == 'Polynomial':
                    args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
                ridge += (np.exp(-eta * ((Rij - offset) ** 2.) / (Rc ** 2.)) *
                          cutoff_fxn(**args_cutoff_fxn))
    return ridge


def calculate_G4(neighbornumbers, neighborsymbols, neighborpositions,
                 G_elements, gamma, zeta, eta, cutoff, Ri, fortran):
    """Calculate G4 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled).  See Eq. 13c of the supplementary information of Khorshidi,
    Peterson, CPC(2016).

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
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
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G4 fingerprint.
    """
    if len(neighborpositions) == 0:
        return 0.
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])

        Rc = cutoff['kwargs']['Rc']
        if cutoff['name'] == 'Cosine':
            cutofffn_code = 1
        elif cutoff['name'] == 'Polynomial':
            cutofffn_code = 2
        else:
            print("Unknown cutoff function specified! \
            Only supports 'Cosine' and 'Polynomial'.")

        args_calculate_g4 = dict(
                neighbornumbers=neighbornumbers,
                neighborpositions=neighborpositions,
                g_numbers=G_numbers,
                g_gamma=gamma,
                g_zeta=zeta,
                g_eta=eta,
                rc=Rc,
                cutofffn_code=cutofffn_code,
                ri=Ri
                )
        if cutofffn_code == 2:
            args_calculate_g4['p_gamma'] = cutoff['kwargs']['gamma']

        return fmodules.calculate_g4(**args_calculate_g4)
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.
        counts = range(len(neighborpositions))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([neighborsymbols[j], neighborsymbols[k]])
                if els != G_elements:
                    continue
                Rij_vector = neighborpositions[j] - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = neighborpositions[k] - Ri
                Rik = np.linalg.norm(Rik_vector)
                Rjk_vector = neighborpositions[k] - neighborpositions[j]
                Rjk = np.linalg.norm(Rjk_vector)
                cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                if cos_theta_ijk < -1.:
                    # Can occur by rounding error.
                    cos_theta_ijk = -1.
                term = (1. + gamma * cos_theta_ijk) ** zeta
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (Rc ** 2.))
                _Rij = dict(Rij=Rij)
                _Rik = dict(Rij=Rik)
                _Rjk = dict(Rij=Rjk)
                if cutoff['name'] == 'Polynomial':
                    _Rij['gamma'] = cutoff['kwargs']['gamma']
                    _Rik['gamma'] = cutoff['kwargs']['gamma']
                    _Rjk['gamma'] = cutoff['kwargs']['gamma']
                term *= cutoff_fxn(**_Rij)
                term *= cutoff_fxn(**_Rik)
                term *= cutoff_fxn(**_Rjk)
                ridge += term
        ridge *= 2. ** (1. - zeta)
        return ridge


def calculate_G5(neighbornumbers, neighborsymbols, neighborpositions,
                 G_elements, gamma, zeta, eta, cutoff,
                 Ri, fortran):
    """Calculate G5 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled). In G5, the Gaussians and cutoff functions with respect to
    R_ijk are omitted. This symmetry function is more useful for larger atomic
    separations, and useful for angular configurations in which R_jk is larger
    than Rc but still inside the cutoff radius e.g. triplets of 180 degrees.

    For more information see: J. Behler, Int. J. Quantum Chem. 115, 1032
    (2015).

    Parameters
    ----------
    neighbornumbers : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
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
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G5 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        neighbornumbers = \
            [atomic_numbers[symbol] for symbol in neighborsymbols]
        if len(neighborpositions) == 0:
            return 0.
        else:
            Rc = cutoff['kwargs']['Rc']
            if cutoff['name'] == 'Cosine':
                cutofffn_code = 1
            elif cutoff['name'] == 'Polynomial':
                cutofffn_code = 2
            else:
                print("Unknown cutoff function specified! \
                Only supports 'Cosine' and 'Polynomial'.")

            args_calculate_g5 = dict(
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_numbers=G_numbers,
                    g_gamma=gamma,
                    g_zeta=zeta,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn_code=cutofffn_code,
                    ri=Ri
                    )
            if cutofffn_code == 2:
                args_calculate_g5['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g5(**args_calculate_g5)
            return ridge
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.
        counts = range(len(neighborpositions))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([neighborsymbols[j], neighborsymbols[k]])
                if els != G_elements:
                    continue
                Rij_vector = neighborpositions[j] - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = neighborpositions[k] - Ri
                Rik = np.linalg.norm(Rik_vector)
                cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                if cos_theta_ijk < -1:
                    # Can occur by rounding error.
                    cos_theta_ijk = -1.
                term = (1. + gamma * cos_theta_ijk) ** zeta
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2.) /
                               (Rc ** 2.))
                _Rij = dict(Rij=Rij)
                _Rik = dict(Rij=Rik)

                if cutoff['name'] == 'Polynomial':
                    _Rij['gamma'] = cutoff['kwargs']['gamma']
                    _Rik['gamma'] = cutoff['kwargs']['gamma']

                term *= cutoff_fxn(**_Rij)
                term *= cutoff_fxn(**_Rik)
                ridge += term
        ridge *= 2. ** (1. - zeta)
        return ridge


def make_symmetry_functions(elements, type, etas, offsets=None,
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
        G = [{'type': 'G2', 'element': element, 'eta': eta, 'offset': offset}
             for eta in etas
             for element in elements
             for offset in offsets]
        return G
    elif type == 'G4':
        G = []
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            G.append({'type': 'G4',
                                      'elements': els,
                                      'eta': eta,
                                      'gamma': gamma,
                                      'zeta': zeta})
        return G
    elif type == 'G5':
        G = []
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            G.append({'type': 'G5',
                                      'elements': els,
                                      'eta': eta,
                                      'gamma': gamma,
                                      'zeta': zeta})
        return G
    raise NotImplementedError('Unknown type: {}.'.format(type))


def make_default_symmetry_functions(elements):
    """Makes default set of G2 and G4 symmetry functions.


    Parameters
    ----------
    elements : list of str
        List of the elements, as in: ["C", "O", "H", "Cu"].

    Returns
    -------
    G : dict of lists
        The generated symmetry function parameters.
    """
    G = {}
    for element in elements:
        # Radial symmetry functions.
        etas = np.logspace(np.log10(0.05), np.log10(5.), num=4)
        _G = make_symmetry_functions(type='G2', etas=etas, elements=elements)
        # Angular symmetry functions.
        _G += make_symmetry_functions(type='G4', etas=[0.005],
                                      zetas=[1., 4.], gammas=[+1., -1.],
                                      elements=elements)
        G[element] = _G
    return G


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


def dRij_dRml_vector(i, j, m, l):
    """Returns the derivative of the position vector R_{ij} with respect to
    x_{l} of itomic index m.

    See Eq. 14d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    i : int
        Index of the first atom.
    j : int
        Index of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    list of float
        The derivative of the position vector R_{ij} with respect to x_{l} of
        atomic index m.
    """
    if (m != i) and (m != j):
        return [0, 0, 0]
    else:
        dRij_dRml_vector = [None, None, None]
        c1 = Kronecker(m, j) - Kronecker(m, i)
        dRij_dRml_vector[0] = c1 * Kronecker(0, l)
        dRij_dRml_vector[1] = c1 * Kronecker(1, l)
        dRij_dRml_vector[2] = c1 * Kronecker(2, l)
        return dRij_dRml_vector


def dRij_dRml(i, j, Ri, Rj, m, l):
    """Returns the derivative of the norm of position vector R_{ij} with
    respect to coordinate x_{l} of atomic index m.

    See Eq. 14c of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    i : int
        Index of the first atom.
    j : int
        Index of the second atom.
    Ri : float
        Position of the first atom.
    Rj : float
        Position of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    dRij_dRml : list of float
        The derivative of the noRi of position vector R_{ij} with respect to
        x_{l} of atomic index m.
    """
    Rij = np.linalg.norm(Rj - Ri)
    if m == i and i != j:  # i != j is necessary for periodic systems
        dRij_dRml = -(Rj[l] - Ri[l]) / Rij
    elif m == j and i != j:  # i != j is necessary for periodic systems
        dRij_dRml = (Rj[l] - Ri[l]) / Rij
    else:
        dRij_dRml = 0
    return dRij_dRml


def dCos_theta_ijk_dR_ml(i, j, k, Ri, Rj, Rk, m, l):
    """Returns the derivative of Cos(theta_{ijk}) with respect to
    x_{l} of atomic index m.

    See Eq. 14f of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    i : int
        Index of the center atom.
    j : int
        Index of the first atom.
    k : int
        Index of the second atom.
    Ri : float
        Position of the center atom.
    Rj : float
        Position of the first atom.
    Rk : float
        Position of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    dCos_theta_ijk_dR_ml : float
        Derivative of Cos(theta_{ijk}) with respect to x_{l} of atomic index m.
    """
    Rij_vector = Rj - Ri
    Rij = np.linalg.norm(Rij_vector)
    Rik_vector = Rk - Ri
    Rik = np.linalg.norm(Rik_vector)
    dCos_theta_ijk_dR_ml = 0

    dRijdRml = dRij_dRml_vector(i, j, m, l)
    if np.array(dRijdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(dRijdRml, Rik_vector) / (Rij * Rik)

    dRikdRml = dRij_dRml_vector(i, k, m, l)
    if np.array(dRikdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(Rij_vector, dRikdRml) / (Rij * Rik)

    dRijdRml = dRij_dRml(i, j, Ri, Rj, m, l)
    if dRijdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRijdRml / \
            ((Rij ** 2.) * Rik)

    dRikdRml = dRij_dRml(i, k, Ri, Rk, m, l)
    if dRikdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRikdRml / \
            (Rij * (Rik ** 2.))
    return dCos_theta_ijk_dR_ml


def calculate_G2_prime(neighborindices, neighbornumbers, neighborsymbols,
                       neighborpositions, G_element, eta, offset, cutoff,
                       i, Ri, m, l, fortran):
    """Calculates coordinate derivative of G2 symmetry function for atom at
    index i and position Ri with respect to coordinate x_{l} of atom index
    m.

    See Eq. 13b of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ---------
    neighborindices : list of int
        List of int of neighboring atoms.
    neighbornumbers : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_element : dict
        Symmetry functions of the center atom.
    eta : float
        Parameter of Behler symmetry functions.
    offset: float
        Offset for gaussians in G2 fingerprints
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    i : int
        Index of the center atom.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        Coordinate derivative of G2 symmetry function for atom at index a and
        position Ri with respect to coordinate x_{l} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        if len(neighborpositions) == 0:
            ridge = 0.
        else:
            Rc = cutoff['kwargs']['Rc']
            if cutoff['name'] == 'Cosine':
                cutofffn_code = 1
            elif cutoff['name'] == 'Polynomial':
                cutofffn_code = 2
            else:
                print("Unknown cutoff function specified! \
                Only supports 'Cosine' and 'Polynomial'.")

            args_calculate_g2_prime = dict(
                    neighborindices=list(neighborindices),
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_number=G_number,
                    g_eta=eta,
                    rc=Rc,
                    offset=offset,
                    cutofffn_code=cutofffn_code,
                    i=i,
                    ri=Ri,
                    m=m,
                    l=l
                    )
            if cutofffn_code == 2:
                args_calculate_g2_prime['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g2_prime(**args_calculate_g2_prime)
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        num_neighbors = len(neighborpositions)   # number of neighboring atoms
        for count in range(num_neighbors):
            symbol = neighborsymbols[count]
            Rj = neighborpositions[count]
            j = neighborindices[count]
            if symbol == G_element:
                dRijdRml = dRij_dRml(i, j, Ri, Rj, m, l)
                if dRijdRml != 0:
                    Rij = np.linalg.norm(Rj - Ri)
                    args_cutoff_fxn = dict(Rij=Rij)
                    if cutoff['name'] == 'Polynomial':
                        args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
                    term1 = (-2. * eta * (Rij - offset) *
                             cutoff_fxn(**args_cutoff_fxn) / (Rc ** 2.) +
                             cutoff_fxn.prime(**args_cutoff_fxn))
                    ridge += np.exp(- eta * ((Rij - offset) ** 2.) /
                                    (Rc ** 2.)) * term1 * dRijdRml

    return ridge


def calculate_G4_prime(neighborindices, neighbornumbers, neighborsymbols,
                       neighborpositions, G_elements, gamma, zeta, eta,
                       cutoff, i, Ri, m, l, fortran):
    """Calculates coordinate derivative of G4 symmetry function for atom at
    index i and position Ri with respect to coordinate x_{l} of atom index m.

    See Eq. 13d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    neighborindices : list of int
        List of int of neighboring atoms.
    neighbornumbers : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    gamma : float
        Parameter of Behler symmetry functions.
    zeta : float
        Parameter of Behler symmetry functions.
    eta : float
        Parameter of Behler symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    i : int
        Index of the center atom.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        Coordinate derivative of G4 symmetry function for atom at index i and
        position Ri with respect to coordinate x_{l} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        if len(neighborpositions) == 0:
            ridge = 0.
        else:
            Rc = cutoff['kwargs']['Rc']
            if cutoff['name'] == 'Cosine':
                cutofffn_code = 1
            elif cutoff['name'] == 'Polynomial':
                cutofffn_code = 2
            else:
                print("Unknown cutoff function specified! \
                Only supports 'Cosine' and 'Polynomial'.")

            args_calculate_g4_prime = dict(
                    neighborindices=list(neighborindices),
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_numbers=G_numbers,
                    g_gamma=gamma,
                    g_zeta=zeta,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn_code=cutofffn_code,
                    i=i,
                    ri=Ri,
                    m=m,
                    l=l
                    )
            if cutofffn_code == 2:
                args_calculate_g4_prime['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g4_prime(**args_calculate_g4_prime)
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.
        # number of neighboring atoms
        counts = range(len(neighborpositions))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([neighborsymbols[j], neighborsymbols[k]])
                if els != G_elements:
                    continue
                Rj = neighborpositions[j]
                Rk = neighborpositions[k]
                Rij_vector = neighborpositions[j] - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = neighborpositions[k] - Ri
                Rik = np.linalg.norm(Rik_vector)
                Rjk_vector = neighborpositions[k] - neighborpositions[j]
                Rjk = np.linalg.norm(Rjk_vector)
                cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                if cos_theta_ijk < -1.:
                    # Can occur by rounding error.
                    cos_theta_ijk = -1.
                c1 = (1. + gamma * cos_theta_ijk)

                _Rij = dict(Rij=Rij)
                _Rik = dict(Rij=Rik)
                _Rjk = dict(Rij=Rjk)
                if cutoff['name'] == 'Polynomial':
                    _Rij['gamma'] = cutoff['kwargs']['gamma']
                    _Rik['gamma'] = cutoff['kwargs']['gamma']
                    _Rjk['gamma'] = cutoff['kwargs']['gamma']

                fcRij = cutoff_fxn(**_Rij)
                fcRik = cutoff_fxn(**_Rik)
                fcRjk = cutoff_fxn(**_Rjk)
                if zeta == 1:
                    term1 = \
                        np.exp(- eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (Rc ** 2.))
                else:
                    term1 = c1 ** (zeta - 1.) * \
                        np.exp(- eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (Rc ** 2.))
                term2 = 0.
                fcRijfcRikfcRjk = fcRij * fcRik * fcRjk
                dCosthetadRml = dCos_theta_ijk_dR_ml(i,
                                                     neighborindices[j],
                                                     neighborindices[k],
                                                     Ri, Rj,
                                                     Rk, m, l)
                if dCosthetadRml != 0:
                    term2 += gamma * zeta * dCosthetadRml
                dRijdRml = dRij_dRml(i, neighborindices[j], Ri, Rj, m, l)
                if dRijdRml != 0:
                    term2 += -2. * c1 * eta * Rij * dRijdRml / (Rc ** 2.)
                dRikdRml = dRij_dRml(i, neighborindices[k], Ri, Rk, m, l)
                if dRikdRml != 0:
                    term2 += -2. * c1 * eta * Rik * dRikdRml / (Rc ** 2.)
                dRjkdRml = dRij_dRml(neighborindices[j],
                                     neighborindices[k],
                                     Rj, Rk, m, l)
                if dRjkdRml != 0:
                    term2 += -2. * c1 * eta * Rjk * dRjkdRml / (Rc ** 2.)
                term3 = fcRijfcRikfcRjk * term2
                term4 = cutoff_fxn.prime(**_Rij) * dRijdRml * fcRik * fcRjk
                term5 = fcRij * cutoff_fxn.prime(**_Rik) * dRikdRml * fcRjk
                term6 = fcRij * fcRik * cutoff_fxn.prime(**_Rjk) * dRjkdRml

                ridge += term1 * (term3 + c1 * (term4 + term5 + term6))

        ridge *= 2. ** (1. - zeta)

    return ridge


def calculate_G5_prime(neighborindices, neighbornumbers, neighborsymbols,
                       neighborpositions, G_elements, gamma, zeta, eta,
                       cutoff, i, Ri, m, l, fortran):
    """Calculates coordinate derivative of G5 symmetry function for atom at
    index i and position Ri with respect to coordinate x_{l} of atom index m.

    See Eq. 13d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    neighborindices : list of int
        List of int of neighboring atoms.
    neighborsymbols : numpy.ndarray of str
        Array of neighbors' symbols.
    neighborpositions : numpy.ndarray of float
        Array of Cartesian atomic positions.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    gamma : float
        Parameter of Behler symmetry functions.
    zeta : float
        Parameter of Behler symmetry functions.
    eta : float
        Parameter of Behler symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    i : int
        Index of the center atom.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        Coordinate derivative of G5 symmetry function for atom at index i and
        position Ri with respect to coordinate x_{l} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        if len(neighborpositions) == 0:
            ridge = 0.
        else:
            Rc = cutoff['kwargs']['Rc']
            if cutoff['name'] == 'Cosine':
                cutofffn_code = 1
            elif cutoff['name'] == 'Polynomial':
                cutofffn_code = 2
            else:
                print("Unknown cutoff function specified! \
                Only supports 'Cosine' and 'Polynomial'.")

            args_calculate_g5_prime = dict(
                    neighborindices=list(neighborindices),
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_numbers=G_numbers,
                    g_gamma=gamma,
                    g_zeta=zeta,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn_code=cutofffn_code,
                    i=i,
                    ri=Ri,
                    m=m,
                    l=l
                    )
            if cutofffn_code == 2:
                args_calculate_g5_prime['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g5_prime(**args_calculate_g5_prime)
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.
        # number of neighboring atoms
        counts = range(len(neighborpositions))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([neighborsymbols[j], neighborsymbols[k]])
                if els != G_elements:
                    continue
                Rj = neighborpositions[j]
                Rk = neighborpositions[k]
                Rij_vector = neighborpositions[j] - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = neighborpositions[k] - Ri
                Rik = np.linalg.norm(Rik_vector)
                cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                if cos_theta_ijk < -1.:
                    cos_theta_ijk = -1.
                c1 = (1. + gamma * cos_theta_ijk)

                _Rij = dict(Rij=Rij)
                _Rik = dict(Rij=Rik)
                if cutoff['name'] == 'Polynomial':
                    _Rij['gamma'] = cutoff['kwargs']['gamma']
                    _Rik['gamma'] = cutoff['kwargs']['gamma']

                fcRij = cutoff_fxn(**_Rij)
                fcRik = cutoff_fxn(**_Rik)
                if zeta == 1:
                    term1 = \
                        np.exp(- eta * (Rij ** 2. + Rik ** 2.) /
                               (Rc ** 2.))
                else:
                    term1 = c1 ** (zeta - 1.) * \
                        np.exp(- eta * (Rij ** 2. + Rik ** 2.) /
                               (Rc ** 2.))
                term2 = 0.
                fcRijfcRik = fcRij * fcRik
                dCosthetadRml = dCos_theta_ijk_dR_ml(i,
                                                     neighborindices[j],
                                                     neighborindices[k],
                                                     Ri, Rj,
                                                     Rk, m, l)
                if dCosthetadRml != 0:
                    term2 += gamma * zeta * dCosthetadRml
                dRijdRml = dRij_dRml(i, neighborindices[j], Ri, Rj, m, l)
                if dRijdRml != 0:
                    term2 += -2. * c1 * eta * Rij * dRijdRml / (Rc ** 2.)
                dRikdRml = dRij_dRml(i, neighborindices[k], Ri, Rk, m, l)
                if dRikdRml != 0:
                    term2 += -2. * c1 * eta * Rik * dRikdRml / (Rc ** 2.)
                term3 = fcRijfcRik * term2
                term4 = cutoff_fxn.prime(**_Rij) * dRijdRml * fcRik
                term5 = fcRij * cutoff_fxn.prime(**_Rik) * dRikdRml

                ridge += term1 * (term3 + c1 * (term4 + term5))
        ridge *= 2. ** (1. - zeta)

    return ridge


if __name__ == "__main__":
    """Directly calling this module; apparently from another node.

    Calls should come as

        python -m amp.descriptor.gaussian id hostname:port

    This session will then start a zmq session with that socket, labeling
    itself with id. Instructions on what to do will come from the socket.
    """
    import sys
    import tempfile
    import zmq
    from ..utilities import MessageDictionary

    fortran = False if fmodules is None else True
    hostsocket = sys.argv[-1]
    proc_id = sys.argv[-2]
    msg = MessageDictionary(proc_id)

    # Send standard lines to stdout signaling process started and where
    # error is directed. This should be caught by pxssh. (This could
    # alternatively be done by zmq, but this works.)
    print('<amp-connect>')  # Signal that program started.
    sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             suffix='.stderr')
    print('Log and error written to %s<stderr>' % sys.stderr.name)
    sys.stderr.write('initiated\n')
    sys.stderr.flush()

    # Establish client session via zmq; find purpose.
    context = zmq.Context()
    sys.stderr.write('context started\n')
    sys.stderr.flush()
    socket = context.socket(zmq.REQ)
    sys.stderr.write('socket started\n')
    sys.stderr.flush()
    socket.connect('tcp://%s' % hostsocket)
    sys.stderr.write('connection made\n')
    sys.stderr.flush()
    socket.send_pyobj(msg('<purpose>'))
    sys.stderr.write('message sent\n')
    sys.stderr.flush()
    purpose = socket.recv_pyobj()
    sys.stderr.write('purpose received\n')
    sys.stderr.flush()
    sys.stderr.write('purpose: %s \n' % purpose)
    sys.stderr.flush()

    if purpose == 'calculate_neighborlists':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()
        # sys.stderr.write(str(images)) # Just to see if they are there.

        # Perform the calculations.
        calc = NeighborlistCalculator(cutoff=cutoff)
        neighborlist = {}
        # for key in images.iterkeys():
        while len(images) > 0:
            key, image = images.popitem()  # Reduce memory.
            neighborlist[key] = calc.calculate(image, key)

        # Send the results.
        socket.send_pyobj(msg('<result>', neighborlist))
        socket.recv_string()  # Needed to complete REQ/REP.

    elif purpose == 'calculate_fingerprints':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, cutoff,
                                     fortran)
        result = {}
        while len(images) > 0:
            key, image = images.popitem()  # Reduce memory.
            result[key] = calc.calculate(image, key)
            if len(images) % 100 == 0:
                socket.send_pyobj(msg('<info>', len(images)))
                socket.recv_string()  # Needed to complete REQ/REP.

        # Send the results.
        socket.send_pyobj(msg('<result>', result))
        socket.recv_string()  # Needed to complete REQ/REP.

    elif purpose == 'calculate_fingerprint_primes':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintPrimeCalculator(neighborlist, Gs, cutoff,
                                          fortran)
        result = {}
        while len(images) > 0:
            key, image = images.popitem()  # Reduce memory.
            result[key] = calc.calculate(image, key)
            if len(images) % 100 == 0:
                socket.send_pyobj(msg('<info>', len(images)))
                socket.recv_string()  # Needed to complete REQ/REP.

        # Send the results.
        socket.send_pyobj(msg('<result>', result))
        socket.recv_string()  # Needed to complete REQ/REP.

    else:
        socket.close()  # May be needed in python3 / ZMQ.
        raise NotImplementedError('purpose %s unknown.' % purpose)
    socket.close()  # May be needed in python3 / ZMQ.
