import numpy as np
import random
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
            if fd is sys.stdout or fd is sys.stderr:
                fd.flush()

    def flush(self):
        for fd in self.fds:
            fd.flush()

    def close(self):
        for fd in self.fds:
            fd.close()

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


def assert_allclose_with_counterexamples(x, y, rtol=1e-7, atol=0, equal_nan=True, max_examples=10, **kw):
    x, y = np.array(x), np.array(y)
    isclose_kw = dict(rtol=rtol, atol=atol, equal_nan=equal_nan)
    allclose_kw = dict(kw, **isclose_kw)
    try:
        np.testing.assert_allclose(x, y, **allclose_kw)
    except:
        close_mask = np.isclose(x, y, **isclose_kw)
        indices = [tuple(idx) for idx in np.indices(x.shape).reshape(len(x.shape), -1).T]
        counterexamples = [
            (index, x[index], y[index])  # yuck, lots of __getitem__
            for index in indices if not close_mask[index]
        ]
        print(x.shape)
        for (ix, x, y) in random.sample(counterexamples, min([max_examples, len(counterexamples)])):
            print(f'{str(ix):10}   {x}   {y}', file=sys.stderr)
        raise
