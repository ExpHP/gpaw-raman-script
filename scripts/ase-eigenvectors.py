#!/usr/bin/env python3

import argparse
import os
import sys
import pickle
import numpy as np

import ase.io.jsonio

PROG = os.path.basename(sys.argv[0])


SQRT_EIGENVALUE_TO_THZ = 15.6333043  # (1/2pi) * sqrt(ev / amu / Angstrom^2) / THz
THZ_TO_WAVENUMBER = 33.3564095198152
EV_TO_WAVENUMBER = 8065.54429

DEFAULT_SYMMETRIZE = 3
DEFAULT_METHOD = 'frederiksen'

def main():
    parser = argparse.ArgumentParser(
        description='',
    )
    parser.add_argument('GPW', help="gpaw GPW file for atoms")
    parser.add_argument('--name', default='elph', help=
        "name of ASE JSON cache (e.g. 'phonon').  Must already have 'cache.0x+.json' and etc."
        " The default of 'elph' is suitable for gpaw-raman-script's output."
    )
    parser.add_argument('--manual', action='store_true', help='use own code to diagonalize instead of ASE (for debugging)')
    parser.add_argument('--method', default=DEFAULT_METHOD, help='set method for ASE (standard, frederiksen) (no effect for --manual)')
    parser.add_argument('--no-acoustic', action='store_false', dest='acoustic', help='enable acoustic sum rule option for ASE (no effect with --manual)')
    parser.add_argument('--symmetrize', type=int, default=DEFAULT_SYMMETRIZE, help='set force constant symmetrization iterations for ASE (no effect with --manual)')
    parser.add_argument('--eigensolver', choices=list(EIGENSOLVER_CHOICES), default='np-eig', help=f'choices: {", ".join(EIGENSOLVER_CHOICES)}')
    parser.add_argument('--displacement', type=float, required=True, help='displacement distance that was used for forces')
    parser.add_argument('--supercell', type=(lambda s: tuple(int(x) for x in s.split())), default=(1, 1, 1), help='supercell as space-separated string of 3 ints')
    parser.add_argument('-o', '--output', help='output npy file.  Each row will be a column eigenvector.  (this is the transpose of the eigenvector matrix)')
    parser.add_argument('--write-frequencies', help='output npy file for frequencies')
    args = parser.parse_args()

    effectful_args = ['output', 'write_frequencies']
    if not any(getattr(args, a) for a in effectful_args):
        parser.error('Nothing to do! Please supply one of: ' + ', '.join('--' + a.replace('_', '-') for a in effectful_args))

    if args.manual:
        dynmat = manual_dynmat(args)
    else:
        dynmat = ase_dynmat(args)

    eigenvalues, eigenvectors = EIGENSOLVER_CHOICES[args.eigensolver](dynmat)

    perm = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[perm]
    eigenvectors = eigenvectors[:, perm]

    frequencies = np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues) * SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER

    print(frequencies)
    if args.output:
        np.save(args.output, eigenvectors.T)
    if args.write_frequencies:
        np.save(args.write_frequencies, frequencies)

EIGENSOLVER_CHOICES = {
    'np-eig': np.linalg.eig,
    'np-eigh': np.linalg.eigh,
    'np-eigh-u': lambda *args, **kw: np.linalg.eigh(UPLO='U', *args, **kw),
}

def manual_dynmat(args):
    from gpaw import GPAW
    if args.supercell != (1, 1, 1):
        die('supercell not implemented for --manual')
    if args.acoustic is not True:
        warn('--no-acoustic has no effect for --manual')
    if args.symmetrize != DEFAULT_SYMMETRIZE:
        warn('--symmetrize has no effect for --manual')

    masses = GPAW(args.GPW).get_atoms().get_masses()
    natoms = len(masses)

    read_force = lambda s: load_json_forces(f'{args.name}/cache.{s}.json')
    plus_forces = np.array([[read_force(f'{i}{xyz}+') for xyz in 'xyz'] for i in range(len(masses))])
    minus_forces = np.array([[read_force(f'{i}{xyz}-') for xyz in 'xyz'] for i in range(len(masses))])

    fcs = -1.0 * (plus_forces - minus_forces) / (2 * args.displacement)
    dynmat = fcs / np.sqrt(masses)[None, None, :, None] / np.sqrt(masses)[:, None, None, None]
    return dynmat.reshape(natoms * 3, natoms * 3)

def ase_dynmat(args):
    from gpaw import GPAW
    from ase.phonons import Phonons

    calc = GPAW(args.GPW)
    phonon = Phonons(calc.get_atoms(), name=args.name, delta=args.displacement, supercell=args.supercell)
    phonon.read(acoustic=args.acoustic, symmetrize=args.symmetrize, method=args.method)
    return phonon.compute_dynamical_matrix([0, 0, 0], phonon.D_N)

def load_json_forces(path):
    return ase.io.jsonio.read_json(path)['forces']

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
