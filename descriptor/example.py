import time
import numpy as np

from ase.calculators.calculator import Parameters
from ..utilities import Data, Logger, importer
from .cutoffs import Cosine
NeighborList = importer('NeighborList')


class AtomCenteredExample(object):

    """Class that calculates fingerprints.

    This is an example class that doesn't do much; it just shows the code
    structure. If making your own module, you can copy and modify this one.

    Parameters
    ----------
    cutoff : object or float
        Cutoff function. Can be also fed as a float representing the radius
        above which neighbor interactions are ignored.  Default is 6.5
        Angstroms.
    anotherparameter : float
        Just an example.
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

    Raises
    ------
        RuntimeError, TypeError
    """

    def __init__(self, cutoff=Cosine(6.5), anotherparameter=12.2, dblabel=None,
                 elements=None, version=None, mode='atom-centered'):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2016.02', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use Example fingerprints'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               ' newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # Check that the mode is atom-centered.
        if mode != 'atom-centered':
            raise RuntimeError('This scheme only works '
                               'in atom-centered mode. %s '
                               'specified.' % mode)

        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        if isinstance(cutoff, int) or isinstance(cutoff, float):
            cutoff = Cosine(cutoff)

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.example.AtomCenteredExample',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.Rc
        p.cutofffn = cutoff.__class__.__name__
        p.anotherparameter = anotherparameter
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
        log = Logger(file=None) if log is None else log

        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
            self.dblabel = self.parent.dblabel
        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel

        p = self.parameters

        log('Cutoff radius: %.2f' % p.cutoff)
        log('Cutoff function: %s' % p.cutofffn)

        if p.elements is None:
            log('Finding unique set of elements in training data.')
            p.elements = set([atom.symbol for atoms in images.values()
                              for atom in atoms])
        p.elements = sorted(p.elements)
        log('%i unique elements included: ' % len(p.elements) +
            ', '.join(p.elements))

        log('anotherparameter: %.3f' % p.anotherparameter)

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            calc = NeighborlistCalculator(cutoff=p.cutoff)
            self.neighborlist = Data(filename='%s-neighborlists'
                                     % self.dblabel,
                                     calculator=calc)
        self.neighborlist.calculate_items(images, parallel=parallel, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         anotherparamter=p.anotherparameter,
                                         cutoff=p.cutoff,
                                         cutofffn=p.cutofffn)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, parallel=parallel, log=log)
        log('...fingerprints calculated.', toc='fp')


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
    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, anotherparamter, cutoff, cutofffn):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'anotherparameter': anotherparamter})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image.
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
        """ Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol.

        n_symbols and Rs are lists of neighbors' symbols and Cartesian
        positions, respectively.

        This function doesn't actually do anything but sleep and return
        a vector of ones.

        Parameters
        ----------
        index : int
            index: Index of the center atom.
        symbol: str
            Symbol of the center atom.
        n_symbols: list of str
            List of neighbors' symbols.
        Rs: list of list of float
            List of Cartesian atomic positions.

        Returns
        -------
        symbols, fingerprints : list of float
            Fingerprints for atom specified by its index and symbol.
        """
        time.sleep(1.0)  # Pretend to do some work.
        fingerprint = [1., 1., 1., 1.]
        return symbol, fingerprint


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
        socket.send_pyobj(msg('<request>', 'cutofffn'))
        cutofffn = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'anotherparameter'))
        anotherparameter = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, anotherparameter, cutoff,
                                     cutofffn)
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
