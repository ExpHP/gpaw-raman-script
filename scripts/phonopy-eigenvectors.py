#!/usr/bin/env python3

import argparse
import os
import sys
import phonopy
import itertools
import pickle
import numpy as np

PROG = os.path.basename(sys.argv[0])

THZ_TO_WAVENUMBER = 33.3564095198152

def main():
    parser = argparse.ArgumentParser(
        description='',
    )
    parser.add_argument('INPUT', nargs='?', default='phonons.sym-{i}.pckl', help='pattern for the force files (.npy or .pckl of arrays in eV/A^2), which must be a python format string that uses "{i}"')
    parser.add_argument('--eq', metavar='EQFILE', help='subtract equilibrium forces from EQFILE (same formats as INPUT)')
    parser.add_argument('-1', dest='start_index', default=0, action='store_const', const=1, help='indicate that file indices start from 1 instead of 0')
    parser.add_argument('--phonopy', required=True, help='phonopy.yaml or phonopy_disp.yaml')
    parser.add_argument('--fc-symmetry', action='store_true', help='make force constants symmetric and apply acoustic sum rule')
    parser.add_argument('-o', '--output', help='output npy file.  Each row will be a column eigenvector.  (this is the transpose of the eigenvector matrix)')
    parser.add_argument('--write-ase-forces', metavar='PREFIX', help='output force files as $PREFIX.0x+.pckl and etc.')
    parser.add_argument('--write-force-constants', help='output npy file for force constants')
    parser.add_argument('--write-frequencies', help='output npy file for frequencies')
    args = parser.parse_args()

    effectful_args = ['output', 'write_force_constants', 'write_frequencies', 'write_ase_forces']
    if not any(getattr(args, a) for a in effectful_args):
        parser.error('Nothing to do! Please supply one of: ' + ', '.join('--' + a.replace('_', '-') for a in effectful_args))

    phonon = phonopy.load(args.phonopy, produce_fc=False)
    ndisp = len(phonon.displacements)
    indices = range(args.start_index, args.start_index + ndisp)
    force_sets = np.array([load_array(args.INPUT.format(i=i)) for i in indices])

    if args.eq:
        force_sets -= load_array(args.eq)

    phonon.set_forces(force_sets)
    phonon.produce_force_constants()
    if args.fc_symmetry:
        phonon.symmetrize_force_constants()
    if args.write_force_constants:
        np.save(args.write_force_constants, phonon.get_force_constants())
    if args.write_ase_forces:
        write_ase_forces(args.write_ase_forces, phonon)

    if not (args.output or args.write_frequencies):
        return

    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    path = [[[0, 0, 0], [0, 0, 0]]]
    labels = ["$\\Gamma$", "$\\Gamma$"]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=2)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=True)

    frequencies = phonon.band_structure.frequencies[0][0] * THZ_TO_WAVENUMBER
    print(frequencies)
    if args.output:
        np.save(args.output, phonon.band_structure.eigenvectors[0][0].T)
    if args.write_frequencies:
        np.save(args.write_frequencies, frequencies)

def load_array(path):
    if path.lower().endswith('.npy'):
        return np.load(path)
    else:
        return pickle.load(open(path, 'rb'))

def save_array(path, array):
    if path.lower().endswith('.npy'):
        np.save(path, array)
    else:
        pickle.dump(array, open(path, 'wb'))

def get_displacement_amplitude(phonon):
    disps = phonon.displacements
    assert len(disps[0]) == 4  # weird api, returns [atom, dx, dy, dz]
    disps = np.array(disps)[:, 1:]  # get rid of the atoms

    disp_norms = np.linalg.norm(disps, axis=1)
    np.testing.assert_allclose(disp_norms.min(), disp_norms.max())  # all should be approximately the same value
    return disp_norms.mean()

def write_ase_forces(prefix, phonon, eq_forces = 0):
    if abs(np.linalg.det(phonon.get_supercell_matrix())) != 1:
        die('Supercells not currently supported for --write-ase-forces (need permutation between ASE and phonopy)')

    fcs = phonon.get_force_constants()
    assert fcs.ndim == 4 and fcs.shape[2:] == (3, 3) and fcs.shape[0] == fcs.shape[1]

    amplitude = get_displacement_amplitude(phonon)
    for atom in range(fcs.shape[0]):
        for axis in range(3):
            xyz = 'xyz'[axis]
            save_array(f'{prefix}.{atom}{xyz}+.pckl', eq_forces - fcs[atom, :, axis, :] * amplitude)
            save_array(f'{prefix}.{atom}{xyz}-.pckl', eq_forces + fcs[atom, :, axis, :] * amplitude)

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
