#!/usr/bin/env python3

import itertools
import argparse
import os
import sys
import pickle
import numpy as np

# FIXME FIXME FIXME
# THIS SCRIPT IS PRODUCING GARBAGE

PROG = os.path.basename(sys.argv[0])
DEFAULT_INPUT_GLOB = 'phonons.*.pckl' # default ASE force filenames

def main():
    parser = argparse.ArgumentParser(
        description='Convert phonons.*.pckl into gamma-dynmat.npz.  Supercells not supported!',
    )
    parser.add_argument('INPUTGLOB', nargs='?', default=DEFAULT_INPUT_GLOB, help='glob for the ASE phonon force files (pickled arrays in eV/A^2), which must have exactly 1 wildcard')
    parser.add_argument('--displacement', type=float, required=True, help='displacement distance that was used, in angstrom')
    parser.add_argument('-o', '--output', required=True, help="output npy file")
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-d', '--phonopy-yaml', help='read masses from this phonopy.yaml file (phonopy_disp.yaml is acceptible)')
    g.add_argument('-s', '--structure', help="use masses for this POSCAR (will use Phonopy's default mass for each element)")
    args = parser.parse_args()

    if args.INPUTGLOB.count('*') != 1:
        parser.error("INPUTGLOB must have exactly one '*'")
    
    if args.structure:
        masses = get_masses_from_poscar(args.structure)
    elif args.phonopy_yaml:
        masses = get_masses_from_phonopy_yaml(args.phonopy_yaml)
    else:
        parser.error('No source for masses; --phonopy-yaml or --structure is required.')
    prefix, suffix = args.INPUTGLOB.split('*')

    force_diffs = read_disp_force_diffs(lambda i, xyz, pm: f'{prefix}{i}{xyz}{pm}{suffix}')
    force_diffs /= (2*args.displacement)
    force_diffs /= np.sqrt(masses)[:, None, None, None]
    force_diffs /= np.sqrt(masses)[None, None, :, None]
    np.save(args.output, force_diffs)

def get_masses_from_phonopy_yaml(path):
    from ruamel.yaml import YAML
    yaml = YAML(typ='rt')

    data = yaml.load(open(path))
    unit = data['physical_unit']['atomic_mass']
    if unit != 'AMU':
        die(f'Unexpected mass unit (got {unit}, expected AMU)')
    return np.array([d['mass'] for d in data['supercell']['points']])

def get_masses_from_poscar(path):
    from phonopy.interface.vasp import read_vasp
    return np.array(read_vasp(path).masses)

# Get the difference between the '+'-displacement forces and the '-'-displacement forces
# for each atom and displacement direction, as a (n, 3, n, 3) shape array.
def read_disp_force_diffs(get_disp_path):
    out = []
    if not os.path.exists(get_disp_path(0, 'x', '+')):
        die(f'No force files found! (File not found: {get_disp_path(0, "x", "+")})')

    for i in itertools.count(0):
        if not os.path.exists(get_disp_path(i, 'x', '+')):
            return np.array(out)

        out.append([
            pickle.load(open(get_disp_path(i, xyz, '+'), 'rb'))
            - pickle.load(open(get_disp_path(i, xyz, '-'), 'rb'))
            for xyz in 'xyz'
        ])

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
