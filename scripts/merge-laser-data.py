#!/usr/bin/env python3

import argparse
import os
import sys
import re
from pathlib import Path
from collections import defaultdict, namedtuple

import numpy as np

PROG = os.path.basename(sys.argv[0])

def main():
    parser = argparse.ArgumentParser(
        description='',
    )
    commands = parser.add_subparsers()
    registered_names = []

    def no_command(*args, **kw):
        nonlocal registered_names
        choice_str = ",".join(registered_names)
        die('no subcommand given! (choices: [","])')

    def register(name):
        registered_names.append(name)
        return commands.add_parser(name)

    parser.set_defaults(func=no_command)

    KIND_CHOICES = ['ModeI', 'RI', 'Contrib']
    KIND_HELP = 'ModeI, RI, or Contrib'

    p = register('merge')
    p.set_defaults(func=main__merge)
    p.add_argument('INPUTPAT', help="input npy path with '{laser}' and '{pol}' where laser frequency and polarization go")
    p.add_argument('--kind', required=True, choices=KIND_CHOICES, help=KIND_HELP)
    p.add_argument('--output', '-o', required=True, help='output npz')
    p.add_argument('--clip-contributions', default=None, type=float, help='remove contributions less than this (relative to global max value)')
    p.add_argument('--delete', action='store_true', help='delete the input files')

    p = register('split')
    p.set_defaults(func=main__split)
    p.add_argument('INPUT', help="input npy")
    p.add_argument('--kind', required=True, choices=KIND_CHOICES, help=KIND_HELP)
    p.add_argument('--output', '-o', required=True, help="output npy pattern with '{laser}' and '{pol}' where laser frequency and polarization go")
    p.add_argument('--delete', action='store_true', help='delete the input file')
    args = parser.parse_args()

    args.func(args)

def main__merge(args):
    pattern = LaserFilePattern(args.INPUTPAT)

    # read all files matching pattern
    data_dict = defaultdict(dict)
    all_input_paths = list(Path('').glob(pattern.glob))
    if not all_input_paths:
        die(f'no input files found for {pattern.glob}')
    for path in all_input_paths:
        path_info = pattern.get_info(str(path))
        read_object = np.load(path)
        if not isinstance(read_object, np.ndarray) and hasattr(read_object, 'f'):
            # convert NpzFile to dict
            read_object = {k: read_object[k] for k in read_object}
        # this is hashing floats, but they come from strings in filenames so they
        # ought to match pretty deterministically... (and they shouldn't parse to NaN)
        data_dict[path_info['pol']][path_info['laser']] = read_object

    # order of keys for output
    canonical_pols = [f'{i}{o}' for i in 'xyz' for o in 'xyz']
    first_subdict = data_dict[sorted(data_dict)[0]]
    canonical_lasers = sorted(first_subdict)

    # check that all polarizations have the same lasers
    for subdict in data_dict.values():
        assert sorted(subdict) == sorted(first_subdict), "not all polarizations have same lasers computed"

    # add missing polarizations
    missing_pols = set(canonical_pols) - set(data_dict)
    if missing_pols:
        zero_subdict = _zero_like_pol_dict(first_subdict, kind=args.kind)
        for pol in sorted(missing_pols):
            data_dict[pol] = zero_subdict
        warn('No data found for polarizations {}; assuming zero'.format(', '.join(sorted(missing_pols))))
    assert sorted(data_dict) == canonical_pols, (sorted(data_dict), canonical_pols)

    def make_big_dense_array():
        # shape [laser][9][...dims]
        big_array = np.array([[data_dict[pol][laser] for pol in canonical_pols] for laser in canonical_lasers])
        # shape [laser][3][3][...dims]
        big_array = big_array.reshape((big_array.shape[0], 3, 3) + big_array.shape[2:])
        return big_array

    if args.kind == 'RI':
        big_array = make_big_dense_array()

        # first row of each original file is raman shift, pull this out
        big_raman_shifts = big_array[:, :, :, 0]  # [laser][3][3][plot_x]
        big_data = big_array[:, :, :, 1]  # [laser][3][3][plot_x]

        assert big_raman_shifts.ndim == 4
        canonical_raman_shifts = big_raman_shifts[0, 0, 0, :]
        assert np.all(big_raman_shifts == canonical_raman_shifts)
        np.savez_compressed(
            Path(args.output),
            laser_freqs=canonical_lasers,
            raman_shifts=canonical_raman_shifts,
            data=big_data,
        )

    elif args.kind == 'ModeI':
        big_array = make_big_dense_array()
        np.savez_compressed(
            Path(args.output),
            laser_freqs=canonical_lasers,
            data=big_array,
        )

    elif args.kind == 'Contrib':
        merged_dict = _merge_contrib(data_dict, canonical_pols, canonical_lasers, args.clip_contributions)
        np.savez_compressed(
            Path(args.output),
            **merged_dict,
        )

    else:
        assert False, args.kind

    if args.delete:
        for path in Path('').glob(pattern.glob):
            path.unlink()


def _zero_like_pol_dict(data_dict_at_single_pol, kind):
    """ Make a dictionary representing all-zero data for a single polarization, given a dict for another polarization. """
    all_lasers = sorted(data_dict_at_single_pol)
    first_laser = min(data_dict_at_single_pol)
    first_obj = data_dict_at_single_pol[first_laser]
    if kind == 'Contrib':
        assert isinstance(first_obj, dict)
        key_classes = _classify_contrib_keys(first_obj)

        zero_subdict = {}
        for key in key_classes.shared:
            zero_subdict[key] = first_obj[key]
        for key in key_classes.by_contrib:
            zero_subdict[key] = np.array([], dtype=first_obj[key].dtype)
        return {laser: zero_subdict for laser in all_lasers}
    else:
        assert isinstance(first_obj, np.ndarray)
        zero_array = np.zeros_like(first_obj)
        return {laser: zero_array for laser in all_lasers}

# This kind is sparse and we have to give it special treatment
def _merge_contrib(data_dict, canonical_pols, canonical_lasers, clip_contributions):
    # clone the dicts so we can add things
    data_dict = {
        pol: {
            laser: dict(data_dict[pol][laser])
            for laser in canonical_lasers
        } for pol in canonical_pols
    }
    all_dicts = [data_dict[pol][laser] for pol in canonical_pols for laser in canonical_lasers]

    
    # Add contrib_pol1 etc. arrays similar to the already existing contrib_band_1 and etc.
    for pol in canonical_pols:
        pol1_index = 'xyz'.index(pol[0])
        pol2_index = 'xyz'.index(pol[1])
        for laser_index, laser in enumerate(canonical_lasers):
            num_contribs_in_this_npz = len(data_dict[pol][laser]['contrib_value'])
            data_dict[pol][laser]['contrib_pol1'] = np.array([pol1_index] * num_contribs_in_this_npz, dtype=int)
            data_dict[pol][laser]['contrib_pol2'] = np.array([pol2_index] * num_contribs_in_this_npz, dtype=int)
            data_dict[pol][laser]['contrib_laser'] = np.array([laser_index] * num_contribs_in_this_npz, dtype=int)
            data_dict[pol][laser]['laser_wavelengths'] = canonical_lasers

    all_keys = set(all_dicts[0])
    key_classes = _classify_contrib_keys(all_keys)

    # Possibly clip contributions to avoid absurd memory usage.
    if clip_contributions is not None:
        global_threshold = clip_contributions * max(max(abs(d['contrib_value']), default=0) for d in all_dicts)
        def do_clipping(pol, laser):
            d = data_dict[pol][laser]
            mask = abs(d['contrib_value']) >= global_threshold
            print(f'{pol} {laser}nm: {sum(mask)} of {len(d["contrib_value"])} elements kept after clipping')
            for key in key_classes.by_contrib:
                d[key] = d[key][mask]

        for laser in canonical_lasers:
            for pol in canonical_pols:
                do_clipping(pol, laser)


    output = {}

    # concatenate contrib values and indices
    for key in key_classes.by_contrib:
        assert all_dicts[0][key].ndim == 1
        output[key] = np.concatenate([d[key] for d in all_dicts])

    # check that coordinate data and dimensions match between all files
    for key in key_classes.shared:
        output[key] = all_dicts[0][key]
        for d in all_dicts:
            np.testing.assert_array_equal(d[key], output[key])

    return output

ContribKeys = namedtuple('ContribKeys', ['shared', 'by_contrib'])

def _classify_contrib_keys(d):
    unused_keys = set(d)

    # data indexed by contribution
    contrib_keys = [key for key in unused_keys if key.startswith('contrib_')]
    unused_keys -= set(contrib_keys)

    # scalars and coordinate labels
    shared_keys = [
        key for key in unused_keys
        if any(key.startswith(prefix) for prefix in ['num_', 'term_', 'k_', 'laser_', 'particle_'])
    ]
    unused_keys -= set(shared_keys)

    if unused_keys:
        die(f'unhandled keys in Contrib npz: {unused_keys}')
    return ContribKeys(shared=shared_keys, by_contrib=contrib_keys)

def main__split(args):
    pattern = LaserFilePattern(args.output)

    # read all files matching pattern
    npz = np.load(args.INPUT)

    canonical_lasers = npz.f.laser_freqs if 'laser_freqs' in npz else npz.f.laser_wavelengths
    canonical_pols = [f'{i}{o}' for i in 'xyz' for o in 'xyz']
    if (canonical_lasers == np.rint(canonical_lasers)).all():
        # use integers for cleaner filepaths
        canonical_lasers = np.array(canonical_lasers, dtype='int')

    if args.kind in ['RI', 'ModeI']:
        big_data = npz.f.data
        for i_pol1, pol1 in enumerate('xyz'):
            for i_pol2, pol2 in enumerate('xyz'):
                for i_laser, laser in enumerate(canonical_lasers):
                    path = pattern.get_path(laser=laser, pol=f'{pol1}{pol2}')
                    data = big_data[i_laser][i_pol1][i_pol2]

                    if args.kind == 'RI':
                        assert data.ndim == 1
                        np.save(path, np.array([npz.f.raman_shifts, data]))
                    elif args.kind == 'ModeI':
                        assert data.ndim == 1
                        np.save(path, data)
                    elif args.kind == 'Contrib':
                        die('splitting Contrib not implemented')
                        # TODO: filter contrib_pol1, contrib_pol2, contrib_laser to generate new files
                    else:
                        assert False, args.kind
    else:
        big_dict = {key: npz[key] for key in npz}
        classes = _classify_contrib_keys(big_dict)
        for i_pol1, pol1 in enumerate('xyz'):
            for i_pol2, pol2 in enumerate('xyz'):
                for i_laser, laser in enumerate(canonical_lasers):
                    path = pattern.get_path(laser=laser, pol=f'{pol1}{pol2}')
                    contrib_mask = np.logical_and(
                        big_dict['contrib_pol1'] == i_pol1,
                        big_dict['contrib_pol2'] == i_pol2,
                        big_dict['contrib_laser'] == i_laser,
                    )
                    out_dict = {}
                    for key in classes.by_contrib:
                        out_dict[key] = big_dict[key][contrib_mask]
                    for key in classes.shared:
                        out_dict[key] = big_dict[key]
                    np.savez_compressed(path, **out_dict)

    if args.delete:
        Path(args.INPUT).unlink()

class LaserFilePattern:
    def __init__(self, string):
        self.pattern = string
        if self.pattern.count('{laser}') != 1 or self.pattern.count('{pol}') != 1:
            die('input pattern must contain "{laser}" and "{pol}"')

        self.glob = self.pattern.replace('{laser}', '*').replace('{pol}', '*')

        # make my life simpler. (can't just str.replace because we want to re.escape the unreplaced parts,
        # so it's just easier to just force them to be in a specific order)
        assert self.pattern.index('{laser}') < self.pattern.index('{pol}')
        part_0, after = self.pattern.split('{laser}', 1)
        part_1, part_2 = after.split('{pol}', 1)
        self.re = re.compile("{}{}{}{}{}".format(
            re.escape(part_0),
            r'(?P<laser>[0-9.e]+)',
            re.escape(part_1),
            r'(?P<pol>[xyz][xyz])',
            re.escape(part_2),
        ))

    def get_path(self, laser, pol):
        return self.pattern.replace('{laser}', f'{laser:03}').replace('{pol}', pol)

    def get_info(self, path):
        match = self.re.match(path)
        if match is None:
            die(f'error parsing freq in {path} from regex {self.re.pattern}')
        return {
            'laser': float(match.group('laser')),
            'pol': match.group('pol'),
        }

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
