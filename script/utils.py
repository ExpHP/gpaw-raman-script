import numpy as np

import sys

def permutation_outer_product(*perms):
    """ Compute the mathematical outer product of a sequence of permutations.

    The result is a permutation that operates on an array whose length is the product of all of the
    input perms.  The last perm will be the fastest index in the output (rearranging items within
    blocks), while the first perm will be the slowest (rearranging the blocks themselves).
    """
    from functools import reduce

    lengths = [len(p) for p in perms]  # na, nb, ..., ny, nz
    strides = np.multiply.accumulate([1] + lengths[1:][::-1])[::-1]   #   ..., nx*ny*nz, ny*nz, nz, 1

    premultiplied_perms = [stride * np.array(perm) for (stride, perm) in zip(strides, perms)]
    permuted_n_dimensional = reduce(np.add.outer, premultiplied_perms)

    # the thing we just computed is basically what you would get if you started with
    #  np.arange(product(lengths)).reshape(lengths) and permuted each axis.
    return permuted_n_dimensional.ravel()


class Tee :
    def __init__(self, *fds):
        self.fds = list(fds)

    def write(self, text):
        for fd in self.fds:
            fd.write(text)

    def flush(self):
        for fd in self.fds:
            fd.flush()

    def closed(self):
        return False

    def __enter__(self, *args, **kw):
        for i, fd in enumerate(self.fds):
            if fd not in [sys.stdout, sys.stderr] and hasattr(fd, '__enter__'):
                self.fds[i] = self.fds[i].__enter__(*args, **kw)
        return self

    def __exit__(self, *args, **kw):
        for fd in self.fds:
            if fd not in [sys.stdout, sys.stderr] and hasattr(fd, '__exit__'):
                fd.__exit__(*args, **kw)
