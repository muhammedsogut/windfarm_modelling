import numpy as np
from numpy import sqrt, exp
from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
from ..utilities import Data, Logger, importer
from .cutoffs import Cosine, dict2cutoff
NeighborList = importer('NeighborList')


class Bispectrum(object):
    """Class that calculates spherical harmonic bispectrum fingerprints.

    Parameters
    ----------
    cutoff : object or float
        Cutoff function, typically from amp.descriptor.cutoffs.  Can be also
        fed as a float representing the radius above which neighbor
        interactions are ignored; in this case a cosine cutoff function will be
        employed.  Default is a 6.5-Angstrom cosine cutoff.
    Gs : dict
        Dictionary of symbols and dictionaries for making fingerprints.  Either
        auto-genetrated, or given in the following form, for example:

               >>> Gs = {"Au": {"Au": 3., "O": 2.}, "O": {"Au": 5., "O": 10.}}

    jmax : integer or half-integer or dict
        Maximum degree of spherical harmonics that will be included in the
        fingerprint vector. Can be also fed as a dictionary with chemical
        species as keys.
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

    Raises:
    -------
        RuntimeError, TypeError
    """

    def __init__(self, cutoff=Cosine(6.5), Gs=None, jmax=5, dblabel=None,
                 elements=None, version='2016.02', mode='atom-centered'):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2016.02', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use bispectrum fingerprints'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               ' newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # Check that the mode is atom-centered.
        if mode != 'atom-centered':
            raise RuntimeError('Bispectrum scheme only works '
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
            {'importname': '.descriptor.bispectrum.Bispectrum',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.todict()
        p.Gs = Gs
        p.jmax = jmax
        p.elements = elements

        self.dblabel = dblabel
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator."""
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, parallel=None, log=None,
                               calculate_derivatives=False):
        """Calculates the fingerpints of the images, for the ones not already
        done.

        Parameters
        ----------
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the training set of data. This can
            also be the path to an ASE trajectory (.traj) or database (.db)
            file. Energies can be obtained from any reference, e.g. DFT
            calculations.
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
        if calculate_derivatives is True:
            import warnings
            warnings.warn('Zernike descriptor cannot train forces yet. '
                          'Force training automatically turnned off. ')
            calculate_derivatives = False

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

        log('Maximum degree of spherical harmonic bispectrum:')
        if isinstance(p.jmax, dict):
            for _ in p.jmax.keys():
                log(' %2s: %d' % (_, p.jmax[_]))
        else:
            log('jmax: %d' % p.jmax)

        if p.Gs is None:
            log('No coefficient for atomic density function supplied; '
                'creating defaults.')
            p.Gs = generate_coefficients(p.elements)
        log('Coefficients of atomic density function for each element:')
        for _ in p.Gs.keys():
            log(' %2s: %s' % (_, str(p.Gs[_])))

        # Counts the number of descriptors for each element.
        no_of_descriptors = {}
        for element in p.elements:
            count = 0
            if isinstance(p.jmax, dict):
                for _2j1 in range(int(2 * p.jmax[element]) + 1):
                    for j in range(int(min(_2j1, p.jmax[element])) + 1):
                        count += 1
            else:
                for _2j1 in range(int(2 * p.jmax) + 1):
                    for j in range(int(min(_2j1, p.jmax)) + 1):
                        count += 1
            no_of_descriptors[element] = count

        log('Number of descriptors for each element:')
        for element in p.elements:
            log(' %2s: %d' % (element, no_of_descriptors.pop(element)))

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            calc = NeighborlistCalculator(cutoff=p.cutoff['kwargs']['Rc'])
            self.neighborlist = Data(filename='%s-neighborlists'
                                     % self.dblabel,
                                     calculator=calc)
        self.neighborlist.calculate_items(images, parallel=parallel, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         Gs=p.Gs,
                                         jmax=p.jmax,
                                         cutoff=p.cutoff,)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, parallel=parallel, log=log)
        log('...fingerprints calculated.', toc='fp')


# Calculators #################################################################


# Neighborlist Calculator
class NeighborlistCalculator:
    """For integration with .utilities.Data

    For each image fed to calculate, a list of neighbors with offset
    distances is returned.
    """

    def __init__(self, cutoff):
        self.globals = Parameters({'cutoff': cutoff})
        self.keyed = Parameters()
        self.parallel_command = 'calculate_neighborlists'

    def calculate(self, image, key):
        cutoff = self.globals.cutoff
        n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                         self_interaction=False,
                         bothways=True,
                         skin=0.)
        n.update(image)
        return [n.get_neighbors(index) for index in range(len(image))]


class FingerprintCalculator:
    """For integration with .utilities.Data
    """

    def __init__(self, neighborlist, Gs, jmax, cutoff,):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs,
                                   'jmax': jmax})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'

        self.factorial = [1]
        for _ in range(int(3. * jmax) + 2):
            if _ > 0:
                self.factorial += [_ * self.factorial[_ - 1]]

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        nl = self.keyed.neighborlist[key]
        fingerprints = []
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighbors, offsets = nl[index]
            neighborsymbols = [image[_].symbol for _ in neighbors]
            Rs = [image.positions[neighbor] + np.dot(offset, image.cell)
                  for (neighbor, offset) in zip(neighbors, offsets)]
            self.atoms = image
            indexfp = self.get_fingerprint(index, symbol, neighborsymbols, Rs)
            fingerprints.append(indexfp)

        return fingerprints

    def get_fingerprint(self, index, symbol, n_symbols, Rs):
        """Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol.

        n_symbols and Rs are lists of
        neighbors' symbols and Cartesian positions, respectively.

        Parameters
        ----------
        index : int
            Index of the center atom.
        symbol : str
            Symbol of the center atom.
        n_symbols : list of str
            List of neighbors' symbols.
        Rs : list of list of float
            List of Cartesian atomic positions of neighbors.

        Returns
        -------
        symbols, fingerprints : list of float
            fingerprints for atom specified by its index and symbol.
        """

        home = self.atoms[index].position
        cutoff = self.globals.cutoff
        Rc = cutoff['kwargs']['Rc']
        jmax = self.globals.jmax

        if cutoff['name'] == 'Cosine':
            cutoff_fxn = Cosine(Rc)
        elif cutoff['name'] == 'Polynomial':
            # cutoff_fxn = Polynomial(cutoff)
            raise NotImplementedError()

        rs = []
        psis = []
        thetas = []
        phis = []
        for neighbor in Rs:
            x = neighbor[0] - home[0]
            y = neighbor[1] - home[1]
            z = neighbor[2] - home[2]
            r = np.linalg.norm(neighbor - home)
            if r > 10.**(-10.):

                psi = np.arcsin(r / Rc)

                theta = np.arccos(z / r)
                if abs((z / r) - 1.0) < 10.**(-8.):
                    theta = 0.0
                elif abs((z / r) + 1.0) < 10.**(-8.):
                    theta = np.pi

                if x < 0.:
                    phi = np.pi + np.arctan(y / x)
                elif 0. < x and y < 0.:
                    phi = 2 * np.pi + np.arctan(y / x)
                elif 0. < x and 0. <= y:
                    phi = np.arctan(y / x)
                elif x == 0. and 0. < y:
                    phi = 0.5 * np.pi
                elif x == 0. and y < 0.:
                    phi = 1.5 * np.pi
                else:
                    phi = 0.

                rs += [r]
                psis += [psi]
                thetas += [theta]
                phis += [phi]

        fingerprint = []
        for _2j1 in range(int(2 * jmax) + 1):
            j1 = 0.5 * _2j1
            j2 = 0.5 * _2j1
            for j in range(int(min(_2j1, jmax)) + 1):
                value = calculate_B(j1, j2, 1.0 * j, self.globals.Gs[symbol],
                                    Rc, cutoff['name'],
                                    self.factorial, n_symbols,
                                    rs, psis, thetas, phis)
                value = value.real
                fingerprint.append(value)

        return symbol, fingerprint

# Auxiliary functions #########################################################


def calculate_B(j1, j2, j, G_element, cutoff, cutofffn, factorial, n_symbols,
                rs, psis, thetas, phis):
    """Calculates bi-spectrum B_{j1, j2, j} according to Eq. (5) of "Gaussian
    Approximation Potentials: The Accuracy of Quantum Mechanics, without the
    Electrons", Phys. Rev. Lett. 104, 136403.
    """

    mvals = m_values(j)
    B = 0.
    for m in mvals:
        for mp in mvals:
            c = calculate_c(j, mp, m, G_element, cutoff, cutofffn, factorial,
                            n_symbols, rs, psis, thetas, phis)
            m1bound = min(j1, m + j2)
            mp1bound = min(j1, mp + j2)
            m1 = max(-j1, m - j2)
            while m1 < (m1bound + 0.5):
                mp1 = max(-j1, mp - j2)
                while mp1 < (mp1bound + 0.5):
                    c1 = calculate_c(j1, mp1, m1, G_element, cutoff, cutofffn,
                                     factorial, n_symbols, rs, psis, thetas,
                                     phis)
                    c2 = calculate_c(j2, mp - mp1, m - m1, G_element, cutoff,
                                     cutofffn, factorial, n_symbols, rs, psis,
                                     thetas, phis)
                    B += CG(j1, m1, j2, m - m1, j, m, factorial) * \
                        CG(j1, mp1, j2, mp - mp1, j, mp, factorial) * \
                        np.conjugate(c) * c1 * c2
                    mp1 += 1.
                m1 += 1.

    return B

###############################################################################


def calculate_c(j, mp, m, G_element, cutoff, cutofffn, factorial, n_symbols,
                rs, psis, thetas, phis):
    """Calculates c^{j}_{m'm} according to Eq. (4) of "Gaussian Approximation
    Potentials: The Accuracy of Quantum Mechanics, without the Electrons",
    Phys. Rev. Lett. 104, 136403
    """

    if cutofffn is 'Cosine':
        cutoff_fxn = Cosine(cutoff)
    elif cutofffn is 'Polynomial':
        # cutoff_fxn = Polynomial(cutoff)
        raise NotImplementedError

    value = 0.
    for n_symbol, r, psi, theta, phi in zip(n_symbols, rs, psis, thetas, phis):

        value += G_element[n_symbol] * \
            np.conjugate(U(j, m, mp, psi, theta, phi, factorial)) * \
            cutoff_fxn(r)

    return value

###############################################################################


def m_values(j):
    """Returns a list of m values for a given j."""

    assert j >= 0, '2*j should be a non-negative integer.'

    return [j - i for i in range(int(2 * j + 1))]

###############################################################################


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers with n >= k.'
    c = factorial[int(n)] / (factorial[int(k)] * factorial[int(n - k)])
    return c

###############################################################################


def WignerD(j, m, mp, alpha, beta, gamma, factorial):
    """Returns the Wigner-D matrix. alpha, beta, and gamma are the Euler
    angles."""

    result = 0
    if abs(beta - np.pi / 2.) < 10.**(-10.):
        # Varshalovich Eq. (5), Section 4.16, Page 113.
        # j, m, and mp here are J, M, and M', respectively, in Eq. (5).
        for k in range(int(2 * j + 1)):
            if k > j + mp or k > j - m:
                break
            elif k < mp - m:
                continue
            result += (-1)**k * binomial(j + mp, k, factorial) * \
                binomial(j - mp, k + m - mp, factorial)

        result *= (-1)**(m - mp) * \
            sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                 float((factorial[int(j + mp)] * factorial[int(j - mp)]))) / \
            2.**j
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    else:
        # Varshalovich Eq. (10), Section 4.16, Page 113.
        # m, mpp, and mp here are M, m, and M', respectively, in Eq. (10).
        mvals = m_values(j)
        for mpp in mvals:
            # temp1 = WignerD(j, m, mpp, 0, np.pi/2, 0) = d(j, m, mpp, np.pi/2)
            temp1 = 0.
            for k in range(int(2 * j + 1)):
                if k > j + mpp or k > j - m:
                    break
                elif k < mpp - m:
                    continue
                temp1 += (-1)**k * binomial(j + mpp, k, factorial) * \
                    binomial(j - mpp, k + m - mpp, factorial)
            temp1 *= (-1)**(m - mpp) * \
                sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                     float((factorial[int(j + mpp)] *
                            factorial[int(j - mpp)]))) / 2.**j

            # temp2 = WignerD(j, mpp, mp, 0, np.pi/2, 0) = d(j, mpp, mp,
            # np.pi/2)
            temp2 = 0.
            for k in range(int(2 * j + 1)):
                if k > j - mp or k > j - mpp:
                    break
                elif k < - mp - mpp:
                    continue
                temp2 += (-1)**k * binomial(j - mp, k, factorial) * \
                    binomial(j + mp, k + mpp + mp, factorial)
            temp2 *= (-1)**(mpp + mp) * \
                sqrt(float(factorial[int(j + mpp)] * factorial[int(j - mpp)]) /
                     float((factorial[int(j - mp)] *
                            factorial[int(j + mp)]))) / 2.**j

            result += temp1 * exp(-1j * mpp * beta) * temp2

        # Empirical normalization factor so results match Varshalovich
        # Tables 4.3-4.12
        # Note that this exact normalization does not follow from the
        # above equations
        result *= (1j**(2 * j - m - mp)) * ((-1)**(2 * m))
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    return result

###############################################################################


def U(j, m, mp, omega, theta, phi, factorial):
    """Calculates rotation matrix U_{MM'}^{J} in terms of rotation angle omega as
    well as rotation axis angles theta and phi, according to Varshalovich,
    Eq. (3), Section 4.5, Page 81. j, m, mp, and mpp here are J, M, M', and M''
    in Eq. (3).
    """

    result = 0.
    mvals = m_values(j)
    for mpp in mvals:
        result += WignerD(j, m, mpp, phi, theta, -phi, factorial) * \
            exp(- 1j * mpp * omega) * \
            WignerD(j, mpp, mp, phi, -theta, -phi, factorial)
    return result


###############################################################################


def CG(a, alpha, b, beta, c, gamma, factorial):
    """Clebsch-Gordan coefficient C_{a alpha b beta}^{c gamma} is calculated
    acoording to the expression given in Varshalovich Eq. (3), Section 8.2,
    Page 238."""

    if int(2. * a) != 2. * a or int(2. * b) != 2. * b or int(2. * c) != 2. * c:
        raise ValueError("j values must be integer or half integer")
    if int(2. * alpha) != 2. * alpha or int(2. * beta) != 2. * beta or \
            int(2. * gamma) != 2. * gamma:
        raise ValueError("m values must be integer or half integer")

    if alpha + beta - gamma != 0.:
        return 0.
    else:
        minimum = min(a + b - c, a - b + c, -a + b + c, a + b + c + 1.,
                      a - abs(alpha), b - abs(beta), c - abs(gamma))
        if minimum < 0.:
            return 0.
        else:
            sqrtarg = \
                factorial[int(a + alpha)] * \
                factorial[int(a - alpha)] * \
                factorial[int(b + beta)] * \
                factorial[int(b - beta)] * \
                factorial[int(c + gamma)] * \
                factorial[int(c - gamma)] * \
                (2. * c + 1.) * \
                factorial[int(a + b - c)] * \
                factorial[int(a - b + c)] * \
                factorial[int(-a + b + c)] / \
                factorial[int(a + b + c + 1.)]

            sqrtres = sqrt(sqrtarg)

            zmin = max(a + beta - c, b - alpha - c, 0.)
            zmax = min(b + beta, a - alpha, a + b - c)
            sumres = 0.
            for z in range(int(zmin), int(zmax) + 1):
                value = \
                    factorial[int(z)] * \
                    factorial[int(a + b - c - z)] * \
                    factorial[int(a - alpha - z)] * \
                    factorial[int(b + beta - z)] * \
                    factorial[int(c - b + alpha + z)] * \
                    factorial[int(c - a - beta + z)]
                sumres += (-1.)**z / value

            result = sqrtres * sumres

            return result

###############################################################################


def generate_coefficients(elements):
    """Automatically generates coefficients if not given by the user.

    Parameters
    ---------
    elements : list of str
        List of symbols of all atoms.

    Returns
    -------
    G : dict of dicts
    """
    _G = {}
    for element in elements:
        _G[element] = atomic_numbers[element]
    G = {}
    for element in elements:
        G[element] = _G
    return G

###############################################################################


if __name__ == "__main__":
    """Directly calling this module; apparently from another node.
    Calls should come as

    python -m amp.descriptor.example id hostname:port

    This session will then start a zmq session with that socket, labeling
    itself with id. Instructions on what to do will come from the socket.
    """
    import sys
    import tempfile
    import zmq
    from ..utilities import MessageDictionary

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

    # Establish client session via zmq; find purpose.
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://%s' % hostsocket)
    socket.send_pyobj(msg('<purpose>'))
    purpose = socket.recv_pyobj()

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
        socket.send_pyobj(msg('<request>', 'jmax'))
        jmax = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, jmax, cutoff,)
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
