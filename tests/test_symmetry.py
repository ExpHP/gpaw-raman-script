
from script import symmetry, test_utils
import script
import numpy as np

def test_blah():
    # rotataion matrices for xyz permutation symmetry
    sym = test_utils.SemigroupTree(
        [np.array([[0,1,0],[0,0,1],[1,0,0]])],
        (lambda a,b: a @ b),
        make_hashable=lambda arr: tuple(map(tuple, arr.tolist())),
    )
    print(sym.members)

    derivs = symmetry.expand_derivs_by_symmetry(
            disp_atoms = [0, 0],
            disp_carts = [[0.1, 0, 0], [-0.1, 0, 0]],
            disp_values = np.array([[1., 0, 0], [-1., 0, 0]]),
            callbacks = symmetry.GeneralArrayCallbacks(['cart']),
            oper_cart_rots = sym.members,
            oper_perms = np.array([[0]]*3),
            quotient_perms=None,
    )
    derivs = np.array(derivs.tolist())
    assert np.allclose(derivs, np.array([[
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ]]))


