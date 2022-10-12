#!/usr/bin/env python3

import band_dot # https://github.com/ExpHP/band-dot
import numpy as np
import argparse
import os
import sys

PROG = os.path.basename(sys.argv[0])

def main():
    parser = argparse.ArgumentParser(
        description='',
    )
    parser.add_argument('EVECS1')
    parser.add_argument('EVECS2')
    parser.add_argument('--frequencies-2', required=True)
    parser.add_argument('--intensities-1', required=True)

    evecs1 = np.load(args.EVECS1)
    evecs2 = np.load(args.EVECS2)
    system_pair = band_dot.PairOfEigensystems.from_eigenvectors(evecs1, evecs2, threshold=1e-3)
    perm = system_pair.permutation()
    evecs2[perm]


    args = parser.parse_args()

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
