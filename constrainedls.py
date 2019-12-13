import sys
import numpy as np
from activeset import ConstrainedLS
from numpy.testing import assert_allclose, assert_equal, \
    assert_almost_equal, assert_raises


# input validation and unit testing for ActiveSet base
# class. unit tests verified against Matlab lsqlin().
def main():
    print("Constrained least squares...")
    # initialize CLS object
    cls = ConstrainedLS(atol=1e-7)
    # input validation testing starts here
    # iv test #1 - empty objective matrix
    assert_raises(ValueError, cls, [], [])
    # iv test #2 - objective matrix/vector mismatch
    A = [[0.0372, 0.2869],
         [0.6861, 0.7071],
         [0.6233, 0.6245],
         [0.6344, 0.6170]]
    b = [0.8587, 0.1781, 0.0747]
    assert_raises(ValueError, cls, A, b)
    # iv test #3 - no constraints
    b = [0.8587, 0.1781, 0.0747, 0.8405]
    assert_raises(ValueError, cls, A, b)
    # iv test #4 - bound vector mismatch
    cl = [0, 0, 0]
    assert_raises(ValueError, cls, A, b, cl=cl)
    # iv test #5 - objective/constraint mismatch
    Ce = [[-.01150290, -.09501570, 0],
          [-.21111988, -.29119600, 0]]
    de = [.30503011, .10502311, -.09119810]
    assert_raises(ValueError, cls, A, b, Ce, de)
    # iv test #6 - constraint matrix/vector mismatch
    Ce = [[-.01150290, -.09501570],
          [-.21111988, -.29119600]]
    assert_raises(ValueError, cls, A, b, Ce, de)
    # iv test #7 - constraint matrix/vector mismatch
    Ci = [[-.01150290, -.09501570],
          [-.21111988, -.29119600]]
    di = [.30503011, .10502311, -.09119810]
    assert_raises(ValueError, cls, A, b, Ci=Ci, di=di)
    # iv test #8 - input 1d/2d array
    assert_raises(ValueError, cls, A, b, Ci=Ci, di=Ce)
    # iv test #9 - input 2d array
    assert_raises(ValueError, cls, A, b, Ci=di, di=di)
    # iv test #10 - infeasible x0
    assert_raises(ValueError, cls, A, b, cl=[0, 0], x0=[-1, -1])
    # iv test #11 - infeasible program
    assert_raises(ValueError, cls, A, b, cl=[0, 0], cu=[-1, -1])
    # unit testing starts here
    # unit test #1 - non-negative least sqares
    # objective matrix
    A = [[0.0372, 0.2869],
         [0.6861, 0.7071],
         [0.6233, 0.6245],
         [0.6344, 0.6170]]
    # target vector
    b = [0.8587, 0.1781, 0.0747, 0.8405]
    # lower bound vector
    cl = [0, 0]
    # solve
    x, scr, nit = cls(A, b, cl=cl)
    # check
    print("x(final) = \n", x)
    sln = np.asarray([0, 0.69293]).reshape(2,1)
    assert_allclose(x, sln, rtol=0, atol=1e-5)
    print("score(final) = ", scr)
    assert_almost_equal(scr, 0.83146, decimal=5)
    print("iter = ", nit)
    assert_equal(nit, 2)
    # unit test #2: inequality constraints
    # objective matrix
    A = [[ 1, 2, 0],
         [-8, 3, 2],
         [ 0, 1, 1]]
    # target vector
    b = [3, 2, 3]
    # inequality constraint matrix
    Ci = [[ 1, 2,  1],
          [ 2, 0,  1],
          [-1, 2, -1]]
    # inequality constraint vector
    di = [3, 2, -2]
    # solve
    x, scr, nit = cls(A, b, Ci=Ci, di=di)
    # check
    print("x(final) = \n", x)
    sln = np.asarray([0.12997, -0.06499, 1.74005]).reshape(3,1)
    assert_allclose(x, sln, rtol=0, atol=1e-5)
    print("score(final) = ", scr)
    assert_almost_equal(scr, 10.81565, decimal=5)
    print("iter = ", nit)
    assert_equal(nit, 1)
    # unit test #3: equality + inequality + bound constraints
    # objective matrix
    A = [[0.9501, 0.7620, 0.6153, 0.4057],
         [0.2311, 0.4564, 0.7919, 0.9354],
         [0.6068, 0.0185, 0.9218, 0.9169],
         [0.4859, 0.8214, 0.7382, 0.4102],
         [0.8912, 0.4447, 0.1762, 0.8936]]
    # target vector
    b = [0.0578, 0.3528, 0.8131, 0.0098, 0.1388]
    # inequality constraint matrix
    Ci = [[0.2027, 0.2721, 0.7467, 0.4659],
          [0.1987, 0.1988, 0.4450, 0.4186],
          [0.6037, 0.0152, 0.9318, 0.8462]]
    # inequality constraint vector
    di = [0.5251, 0.2026, 0.6721]
    # equality constraint matrix
    Ce = [[3, 5, 7, 9]]
    # equality constraint vector
    de = [4]
    # upper bound vector
    cu = [2, 2, 2, 2]
    # lower bound vector
    cl = [-0.1, -0.1, -0.1, -0.1]
    # solve
    x, scr, nit = cls(A, b, Ce, de, Ci, di, cu, cl)
    # check
    print("x(final) = \n", x)
    sln = np.asarray([-0.10000, -0.10000, 0.15991, 0.40896]).reshape(4, 1)
    assert_allclose(x, sln, rtol=0, atol=1e-5)
    print("score(final) = ", scr)
    assert_almost_equal(scr, 0.16951, decimal=5)
    print("iter = ", nit)
    assert_equal(nit, 3)
    # unit test #4: equality + bound constraints
    # with supplied initial feasible solution
    # objective matrix
    A = [[-.01150290, -.09501570,  .35119807],
         [-.21111988, -.29119600,  .15501210],
         [-.11111200, -.11950019, -.01111994],
         [ .35119863,  .30119971, -.21150112],
         [ .15119558,  .10501690, -.11111198]]
    # objective vector
    b = [.30503011, .10502311, -.09119810, -.29501510, -.11950052]
    # equality constraint matrix
    Ce = [[1, 1, 1]]
    # equality constraint vector
    de = [1]
    # upper bound vector
    cu = [1, 1, 1]
    # lower bound vector
    cl = [-1, -1, -1]
    # initial feasible solution
    x0 = [0.333333333, 0.333333333, 0.333333333]
    # solve
    x, scr, nit = cls(A, b, Ce, de, cu=cu, cl=cl, x0=x0)
    # check
    print("x(final) = \n", x)
    sln = np.asarray([-0.72381, 0.72381, 1.00000]).reshape(3,1)
    assert_allclose(x, sln, rtol=0, atol=1e-5)
    print("score(final) = ", scr)
    assert_almost_equal(scr, 0.00861, decimal=5)
    print("iter = ", nit)
    assert_equal(nit, 2)
    # unit test #5: equality constraints
    # objective matrix
    A = [[0.9501, 0.7620, 0.6153, 0.4057],
         [0.2311, 0.4564, 0.7919, 0.9354],
         [0.6068, 0.0185, 0.9218, 0.9169],
         [0.4859, 0.8214, 0.7382, 0.4102],
         [0.8912, 0.4447, 0.1762, 0.8936]]
    # target vector
    b = [0.0578, 0.3528, 0.8131, 0.0098, 0.1388]
    # equality constraint matrix
    Ce = [[3, 5, 7, 9]]
    # equality constraint vector
    de = [4]
    # solve
    x, scr, nit = cls(A, b, Ce, de)
    # check
    print("x(final) = \n", x)
    sln = np.asarray([0.01756, -0.59434, 0.51380, 0.36916]).reshape(4, 1)
    assert_allclose(x, sln, rtol=0, atol=1e-5)
    print("score(final) = ", scr)
    assert_almost_equal(scr, 0.02105, decimal=5)
    print("iter = ", nit)
    assert_equal(nit, 1)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
