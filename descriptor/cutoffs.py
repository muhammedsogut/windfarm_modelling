#!/usr/bin/env python
"""
This script contains different cutoff function forms.

Note all cutoff functions need to have a "todict" method
to support saving/loading as an Amp object.

All cutoff functions also need to have an `Rc` attribute which
is the maximum distance at which properties are calculated; this
will be used in calculating neighborlists.

"""

import numpy as np


def dict2cutoff(dct):
    """This function converts a dictionary (which was created with the
    to_dict method of one of the cutoff classes) into an instantiated
    version of the class. Modeled after ASE's dict2constraint function.
    """
    if len(dct) != 2:
        raise RuntimeError('Cutoff dictionary must have only two values,'
                           ' "name" and "kwargs".')
    return globals()[dct['name']](**dct['kwargs'])


class Cosine(object):
    """Cosine functional form suggested by Behler.

    Parameters
    ---------
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return 0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.)

    def prime(self, Rij):
        """Derivative (dfc_dRij) of the Cosine cutoff function with respect to Rij.

        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return -0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc)

    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc}}

    def __repr__(self):
        return ('<Cosine cutoff with Rc=%.3f from amp.descriptor.cutoffs>'
                % self.Rc)


class Polynomial(object):
    """Polynomial functional form suggested by Khorshidi and Peterson.

    Parameters
    ----------
    gamma : float
        The power of polynomial.
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, Rc, Rct, gamma=4):
        self.gamma = gamma
        self.Rc = Rc
        self.Rct = Rct

    def __call__(self, Rij):
        """
        Parameters
        ----------
        Rij : float
            Distance between pair atoms.
        Rct : float
            Distance where cutoff function will start.

        Returns
        -------
        value : float
            The value of the cutoff function.
        """
        #R1=3000
        R1=self.Rct
        if Rij > self.Rc:
            return 0.
        elif Rij < R1:
            return 1.
        else:
            value = 1. + self.gamma * ((Rij-R1) / (self.Rc-R1)) ** (self.gamma + 1) - \
                (self.gamma + 1) * ((Rij-R1) / (self.Rc-R1)) ** self.gamma
            return value

    def todict(self):
        return {'name': 'Polynomial',
                'kwargs': {'Rc': self.Rc,
                           'gamma': self.gamma,
                           'Rct': self.Rct
                           }
                }

    def __repr__(self):
        return ('<Polynomial cutoff with Rc=%.3f and gamma=%i '
                'from amp.descriptor.cutoffs>'
                % (self.Rc, self.gamma))

class Polynomial_2(object):
    """Polynomial functional form suggested by Khorshidi and Peterson.

    Parameters
    ----------
    gamma : float
        The power of polynomial.
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, delta_R, gamma=4):
        self.gamma = gamma
        #self.Rc = Rc
        self.delta_R = delta_R

    def __call__(self, Rij, Rc):
        """
        Parameters
        ----------
        Rij : float
            Distance between pair atoms.
        Rct : float
            Distance where cutoff function will start.

        Returns
        -------
        value : float
            The value of the cutoff function.
        """
        delta_R=self.delta_R
        R1 = Rc-delta_R
        
        if Rij > Rc:
            return 0.
        elif Rij < R1:
            return 1.
        else:
            value = 1. + self.gamma * ((Rij-R1) / delta_R) ** (self.gamma + 1) - \
                (self.gamma + 1) * ((Rij-R1) / delta_R) ** self.gamma
            return value

    def todict(self):
        return {'name': 'Polynomial_2',
                'kwargs': {'gamma': self.gamma,
                           'delta_R': self.delta_R
                           }
                }

    def __repr__(self):
        return ('<Polynomial cutoff with Rc=%.3f and gamma=%i '
                'from amp.descriptor.cutoffs>'
                % (self.Rc, self.gamma))

