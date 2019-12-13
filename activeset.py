import sys
import warnings
import numpy as np
from scipy.optimize import linprog

class ActiveSet(object):
    """
    Base class for active set sequential quadratic programming
    """

    # initialize the following static instance variables:
    #   1) tol : floating point comparison tolerance
    #   2) maxiter : maximum number of active set iterations
    def __init__(self, atol=1e-8, maxiter=100):
        self.tol = atol
        self.maxiter = maxiter


    # initialize instance variables for each new run:
    #   1) neq : number of equality constraints
    #   2) nbds : number of bounds constraints
    #   3) A : objective function coefficient matrix
    #   4) b : objective function target vector
    #   5) C : constraint coefficient matrix
    #   6) c : constraint target vector
    #   7) AC_KKT : active constraint matrix in KKT form
    #   8) acidx : active constraint index vector
    #   9) H : objective function Hessian matrix
    #  10) h : objective function Hessian vector
    #  11) Ce : equality constraint matrix
    #  12) ce : equality constraint target vector
    #  13) Ci : inequality constraint matrix
    #  14) ci : inequality constraint target vector
    #  15) cl : lower bound constraints
    #  16) cu : upper bound constraints
    def _init_vars(self):
        self.neq = 0
        self.nbds = 0
        self.A = []
        self.b = []
        self.C = []
        self.d = []
        self.AC_KKT = []
        self.acidx = []
        self.H = []
        self.h = []
        self.Ce = []
        self.ce = []
        self.Ci = []
        self.ci = []
        self.cl = []
        self.cu = []


    # virtual method for calculating objective function
    # Hessian matrix + target vector
    def _calc_Hessians(self):
        raise NotImplementedError('ActiveSet: _calc_Hessians is virtual')


    # virtual method for calculating the current active set
    # Hessian vector
    def _calc_as_Hessian_vector(self, x):
        str = 'ActiveSet: _calc_as_Hessian_vector is virtual'
        raise NotImplementedError(str)


    # virtual method for calculating the objective function
    # for a given solution
    def _calc_objective(self, x):
        raise NotImplementedError('ActiveSet: _calc_objective is virtual')


    # check if solution is feasible
    def _feasible(self, x):
        Cx = np.dot(self.C, x)
        # check equality constraints
        if self.neq:
            if not np.allclose(Cx[:self.neq], self.d[:self.neq], rtol=0,
                               atol=self.tol):
                return 0
        # check inequality constraints
        icm = Cx[self.neq:] <= self.d[self.neq:] + self.tol
        if len(icm[icm == 0]):
            return 0
        return 1


    # add constraint to the active set KKT system
    def _add_active_constraint(self, cidx):
        # save the contraint index
        self.acidx = np.vstack((self.acidx, np.asarray(cidx).reshape(1,1)))
        # add constraint row to KKT
        c = np.zeros(shape=(1, self.AC_KKT.shape[1]))
        c[:, :self.C[cidx].shape[0]] = \
            self.C[cidx].reshape(1, self.C[cidx].shape[0])
        self.AC_KKT = np.vstack((self.AC_KKT, c))
        # add constraint transpose column to KKT
        c = np.zeros(shape=(self.AC_KKT.shape[0], 1))
        c[:self.C[cidx].shape[0]] = \
            self.C[cidx].reshape(self.C[cidx].shape[0], 1)
        self.AC_KKT = np.hstack((self.AC_KKT, c))


    # remove constraint from the active set
    def _remove_active_constraint(self, wcidx):
        # remove the constraint index
        self.acidx = np.delete(self.acidx, (wcidx), axis=0)
        # remove the constraint row/column
        kktidx = wcidx + self.H.shape[0]
        self.AC_KKT = np.delete(self.AC_KKT, (kktidx), axis=0)
        self.AC_KKT = np.delete(self.AC_KKT, (kktidx), axis=1)


    # calculate the active set step length and
    # return the index of the inactive constraint
    # to be added to the active set.
    def _calc_step_length(self, x, p):
        # constraint indeces
        cidx = np.arange(self.C.shape[0])
        # get inactive constraints
        mask = np.isin(cidx, self.acidx, assume_unique=True, invert=True)
        IC = self.C[mask]
        id = self.d[mask]
        # keep track of constraint matrix indeces
        icidx = cidx[mask]
        icidx = icidx.reshape(icidx.shape[0], 1)
        # denominator
        den = np.dot(IC, p)
        # suppress division-by-zero warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            q = (id - np.dot(IC, x)) / den
        # keep quotient when den > 0
        q = q[den > self.tol]
        # check if empty
        if len(q) == 0:
            return 1, -1
        icidx = icidx[den > self.tol]
        minidx = np.argmin(q)
        return q[minidx], icidx[minidx]
   

    # calculate the active set Hessian target vector:
    #   ih = h - H * x
    def _calc_as_Hessian_vector(self, x):
        return self.h - np.dot(self.H, x)


    # uses null-space algorithm (Gill, 1978; Nocedal & Wright, 2006)
    # to solve the current active set KKT system. see details
    # for _solve_as_kkt() below.
    def _null_space_as_kkt(self, x):
        # number of constraints
        m = self.AC_KKT.shape[0] - self.H.shape[0]
        # check if active constraint set is empty
        if m == 0:
            p = np.linalg.solve(self.H, self._calc_as_Hessian_vector(x))
            return p, []
        # number of variables
        n = self.C.shape[1]
        # active constraint vector is set to zero
        d = np.zeros(shape=(m, 1))
        # calculate active set Hessian vector
        h = self._calc_as_Hessian_vector(x)
        # use QR decomposition of active constraint matrix (transpose)
        Q, R = np.linalg.qr(self.AC_KKT[:self.H.shape[0],
                                        self.H.shape[1]:], 'complete')
        # trim full R to get n x n upper diagonal form
        R = R[:m]
        # calculate the step direction vector:
        #   p = Qr * py + Qn * pz
        py = np.linalg.solve(R, d)
        # split Q into null and range spaces
        Qr = Q[:, :m]
        p = np.dot(Qr, py)
        # compute null space portion if the number of variables
        # is greater than the number of active constraints
        if n > m:
            Qn = Q[:, m:]
            Qnt = Qn.T
            # compute Cholesky decomposition for reduced Hessian
            Hz = np.linalg.multi_dot([Qnt, self.H, Qn])
            L = np.linalg.cholesky(Hz)
            # compute null space target vector
            hz = np.dot(Qnt, np.linalg.multi_dot([self.H, Qr, py]) + h)
            pz = np.linalg.solve(L.T, np.linalg.solve(L, hz))
            # update step direction
            p += np.dot(Qn, pz)
        # compute Lagrangians
        l = np.linalg.solve(R, np.dot(Qr.T, np.dot(self.H, -p) + h))
        return p, l


    # returns the new direction vector along with
    # the vector of lagrangian multipliers for the
    # active constraints. the equality subproblem
    # to be solved is the following KKT system:
    #
    #   |  H  AC.T  |   |  wx  |   |  h - H * x_cur  |
    #   |           | * |      | = |                 |
    #   |  AC   0   |   |  wl  |   |        0        |
    #
    # where:
    #   H = objective function Hessian matrix
    #   h = objective function Hessian target vector
    #   AC = current active constraint matrix
    #   x_cur = current feasible solution
    def _solve_as_kkt(self, x):
        # check if active constraint set is empty
        if self.AC_KKT.shape[0] - self.H.shape[0] == 0:
            p = np.linalg.solve(self.H, self._calc_as_Hessian_vector(x))
            return p, []
        # construct the KKT target vector using the current solution
        kkt = np.zeros(shape=(self.AC_KKT.shape[0], 1))
        kkt[:self.A.shape[1]] = self._calc_as_Hessian_vector(x)
        # solve the KKT equations
        wx_wl = np.linalg.solve(self.AC_KKT, kkt)
        # return new direction and lagrangian multipliers
        return wx_wl[:self.A.shape[1]], wx_wl[self.A.shape[1]:]


    # create the initial active constrint set
    def _init_active_set(self, x0):
        # initialize the KKT matrix with Hessian
        self.AC_KKT = self.H
        # equality constraints always in active set.
        # save the indeces into the full matrix.
        # add any inequality constraints to the
        # active working set that are equal to the
        # constraint value, i.e.
        #   x>=0 is active if x=0.
        cx = np.dot(self.C, x0)
        self.acidx = np.ndarray((0, 1))
        for j in range(cx.shape[0]):
            if j < self.neq or np.isclose(cx[j], self.d[j], rtol=0, atol=self.tol):
                self._add_active_constraint(j)


    # find the initial feasible solution
    # (if not user-provided) by solving the
    # equality-constrained linear program
    def _find_feasible_x0(self):
        # trivial solution if only bounds constraints
        if self.nbds == self.C.shape[0]:
            if self.cl.shape[0]:
                x0 = self.cl
            else:
                x0 = self.cu
            if not self._feasible(x0):
                raise ValueError('ActiveSet: cannot find feasible x0')
            return x0
        # remove bounds constraints
        if self.nbds:
            Ce = self.C[:-self.nbds]
            de = self.d[:-self.nbds]
        else:
            Ce = self.C
            de = self.d
        # add slack variables to inequality constraints
        s = np.ones(shape=(Ce.shape[0],))
        Ce = np.hstack((Ce, np.diag(s)[:, self.neq:]))
        # initialize objective constraint
        co = np.zeros(shape=(Ce.shape[1],))
        # add artificial variables to equality constraints
        s = np.ones(shape=(Ce.shape[0],))
        Ce = np.hstack((Ce, np.diag(s)[:, :self.neq]))
        co = np.hstack((co, np.ones(shape=(self.neq,))))
        # add artificial variables to inequality constraints
        # with a negative target
        for i in range(self.neq, Ce.shape[0]):
            if de[i] < 0:
                col = np.zeros(shape=(Ce.shape[0], 1))
                col[i] = 1
                Ce = np.hstack((Ce, col))
                co = np.hstack((co, [1]))
        # convert bounds array to list
        cl = self.cl.reshape(self.cl.shape[0],).tolist()
        cu = self.cu.reshape(self.cu.shape[0],).tolist()
        # create a list of bounds 2-tuples
        bds = []
        for i in range(Ce.shape[1]):
            if i < self.A.shape[1]:
                if len(cl) and len(cu):
                    bd = (cl[i], cu[i])
                elif len(cl):
                    bd = (cl[i], np.Inf)
                elif len(cu):
                    bd = (0, cu[i])
                else:
                    bd = (0, np.Inf)
            else:
                bd = (0, np.Inf)
            bds.append(bd)
        # solve the linear programs
        res = linprog(co, A_eq=Ce, b_eq=de, bounds=bds, method='interior-point')
        if not res.success:
            raise ValueError('ActiveSet: could not find feasible x0')
        x0 = res.x[:self.C.shape[1]].reshape(self.C.shape[1], 1)
        if not self._feasible(x0):
            raise ValueError('ActiveSet: could not find feasible x0')
        return x0


    # input must be a 1d array or a 2d array with a single column
    def _check_1d_array(self, m, mnm):
        if m.ndim > 2 or (m.ndim == 2 and m.shape[1] != 1):
            str = 'ActiveSet: {} must be a 1d array '.format(mnm)
            str += 'or 2d with a single column'
            raise ValueError(str)


    # dimension and emptiness checks for input constraint arrays
    def _check_constraints(self, M, m, mtxnm, mnm, mtxemptyok=True):
        M = np.asarray(M)
        m = np.asarray(m)
        # objective matrix cannot be empty
        if not mtxemptyok and not M.shape[0]:
            raise ValueError('ActiveSet: {} is empty'.format(mtxnm))
        elif M.shape[0]:
            # matrices must have two dimensions
            if M.ndim != 2:
                str = 'ActiveSet: {} must be a 2d array'.format(mtxnm)
                raise ValueError(str)
            # constraint matrices axis=1 must match objective matrix
            if mtxnm != 'A' and M.shape[1] != self.A.shape[1]:
                str = 'ActiveSet: {} (axis=1) must match '.format(mtxnm)
                str += 'A (axis=1)'
                raise ValueError(str)
        # check 1d array dimensions
        self._check_1d_array(m, mnm)
        # matrix/vector pair must have same number of constraints
        # except for vector b which could be empty
        if mnm == 'b' and m.shape[0] and M.shape[0] != m.shape[0]:
            raise ValueError('ActiveSet: b must be empty or match A (axis=0)')
        elif mnm != 'b' and M.shape[0] != m.shape[0]:
            str = 'ActiveSet: {} and {} mismatch (axis=0)'.format(mtxnm, mnm)
            raise ValueError(str)
        if not M.shape[0]:
            M = np.reshape(M, (-1, self.A.shape[1]))
        m = np.reshape(m, (-1, 1))
        if mtxnm != 'A' and M.shape[0]:
            self.C = np.vstack((self.C, M))
            self.d = np.vstack((self.d, m))
        return M, m


    # dimension checks for input bounds arrays
    def _check_bounds(self, b, bnm, upper=True):
        bds = np.asarray(b)
        if bds.shape[0]:
            self._check_1d_array(bds, bnm)
            if bds.shape[0] != self.A.shape[1]:
                str = 'ActiveSet: bounds (axis=0) must match A (axis=1)'
                raise ValueError(str)
        bds = np.reshape(bds, (-1, 1))
        if upper and bds.shape[0]:
            self.C = np.vstack((self.C, np.identity(bds.shape[0])))
            self.d = np.vstack((self.d, bds))
        elif not upper and bds.shape[0]:
            self.C = np.vstack((self.C, -np.identity(bds.shape[0])))
            self.d = np.vstack((self.d, -bds))
        return bds


    # check and prepare inputs for active set algorithm.
    # concatenate constraints in the following order:
    #   equality, inequality, bound.
    # negate lower bound constraints.
    def _prep_inputs(self, A, b, Ce, Ci, de, di, cu, cl):
        # initialize instance variables
        self._init_vars()
        # set objective function instance variables.
        # objective matrix cannot be empty. target
        # vector can be empty otherwise must have the
        # same shape as A (axis=0).
        self.A, self.b = self._check_constraints(A, b, 'A', 'b', mtxemptyok=False)
        # initialize C, d
        self.C = np.reshape(self.C, (-1, self.A.shape[1]))
        self.d = np.reshape(self.d, (-1, 1))
        # set Hessian instance variables
        self._calc_Hessians()
        # add equality constraints
        self.Ce, self.de = self._check_constraints(Ce, de, 'Ce', 'de')
        self.neq = self.Ce.shape[0]
        # add inequality constraints
        self.Ci, self.di = self._check_constraints(Ci, di, 'Ci', 'di')
        # add upper bound constraints
        self.cu = self._check_bounds(cu, 'cu')
        self.nbds = self.cu.shape[0]
        # add lower bound constraints
        self.cl = self._check_bounds(cl, 'cl', upper=False)
        self.nbds += self.cl.shape[0]
        # flag empty input
        if not self.C.shape[0]:
            str = 'ActiveSet: input must have at least one of '
            str += '[eq,ineq,bound] constraints'
            raise ValueError(str)


    def run(self, A, b, Ce, de, Ci, di, cu, cl, x0):
        """
        Active set method for sequential quadratic programming.


        Parameters
        ----------
        A : array_like 2d
            objective function coefficient matrix
        b : array_like 1d or single-column 2d (optional)
            objective function target vector
        Ce : array_like 2d (optional)
            equality constraint coefficient matrix
        de : array_like 1d or single-column 2d (optional)
            equality constraint target vector
        Ci : array_like 2d (optional)
            inequality constraint coefficient matrix
        di : array_like 1d or single-column 2d (optional)
            inequality constraint target vector
        cu : array_like 1d or single-column 2d (optional)
            upper-bound constraint vector
        cl : array_like 1d or single-column 2d (optional)
            lower-bound constraint vector
        x0 : array_like 1d or single-column 2d (optional)
            initial feasible solution

        Returns
        -------
        x : ndarray
            final solution
        scr : float
            objective function value of final solution
        nit : int
            number of active set iterations

        Notes
        -----
        The ActiveSet class implements the primal form of active set method
        to solve the generic quadratic program:

            min f(x)
            s.t. Ax = b
                 Cx <= d

        where f(x) is a generic quadratic objective function to be minimized
        subject to a set of linear equality and inequality constraints. The
        KKT matrix form of the problem to be solved is:

           |  H  C.T  |   |  x  |   |  h  |
           |          | * |     | = |     |
           |  C   0   |   |  l  |   |  d  |

         where:
           H = objective function Hessian matrix
           h = objective function Hessian target vector
           C = constraint coefficient matrix
           d = constraint target vector
           x = solution vector
           l = Lagrangian vector

        An active-set method must start from a feasible initial solution.
        For many problems deriving a feasible initial solution is a trivial
        problem and can be user-provided. Otherwise, an equality-constrained
        linear program is solved. Subsequently, at each iteration of the
        active set method, an equality-constrained KKT subproblem is solved to
        find the next step direction (with an associated calculation to find
        the magnitude of the step). The KKT form of the subproblem is very
        similar to the above except where:
            C = AC = active constraint matrix
            x = p = directional step vector
            h = h - H * x_cur
            d = 0

        Two methods are provided for solving the directional KKT subproblem:
        direct KKT system solution and the more numerically stable null space
        method. Since numerical stability comes at the cost of increased
        computation, the code is currently set up to run the null space method
        only in the case of direct calculation error due to a poorly
        conditioned KKT matrix, etc.

        The ActiveSet class can be thought-of as an abstract base class
        without actually using the abc library. This was done to help
        facilitate inter-operability between python 2.x and 3.x versions.
        The ActiveSet class requires derived classes to define functions
        for objective and Hessian function calculation. Two examples of
        derived classes can be found below: ConstrainedLS() is a
        constrained least squares solver; PortfolioOpt() is an MVO
        portfolio optimizer.

        References
        ----------
        Gill, P. E. & Murray, W. Numerically Stable Methods for Quadratic
        Programming. Mathematical Programming, 14, 1978.

        Nocedal, J. & Wright, S.J. (2006). Numerical Optimization.
        Springer-Verlag. New York, NY.
        
        Wong, E. (2011). Active-Set Methods for Quadratic Programming
        (Doctoral Dissertation). University of Calfiornia, San Diego, CA.
        """
        # construct input
        self._prep_inputs(A, b, Ce, Ci, de, di, cu, cl)
        # get initial feasible solution. check feasibility if
        # user-provided, otherwise solve the equality constrained
        # linear program for a feasible initial solution.
        if not len(x0):
            x0 = self._find_feasible_x0()
        else:
            x0 = np.asarray(x0)
            self._check_1d_array(x0, 'x0')
            x0 = np.reshape(x0, (-1, 1))
            # raise error if user-supplied x0 is infeasible
            if not self._feasible(x0):
                raise ValueError('ActiveSet: supplied x0 is infeasible')
        self._init_active_set(x0)
        # main loop
        cur_x = x0
        for iter in range(self.maxiter):
            # get the next step direction vector p by solving the
            # equality constrained subproblem. try direct KKT
            # computation first. if problems arise try the more
            # numerically stable null space method.
            try:
                p, l = self._solve_as_kkt(cur_x)
            except:
                p, l = self._null_space_as_kkt(cur_x)
            # get the length of p
            len_p = np.linalg.norm(p)
            # if the len(p) = 0 then check the lagrangian multipliers
            # for any candidate constraints that may be removed from
            # the current active set
            if np.isclose(len_p, 0, rtol=0, atol=self.tol):
                # check for inequality constraints. terminate if
                # there are no active inequality constraints.
                if l.shape[0] == self.neq:
                    return cur_x, self._calc_objective(cur_x), iter
                # get the index of the minimum lagrange multiplier
                # amongst inequality constraints. if none of the
                # inequality lagrangian multipliers is negative, the
                # algorithm is terminated with cur_x as the final
                # solution.
                m = np.amin(l[self.neq:])
                if m >= -self.tol:
                    return cur_x, self._calc_objective(cur_x), iter
                nlidx = np.where(l == m)[0][0]
                # remove the inequality constraint with minimum
                # negative lagrangian multiplier. cur_x is not
                # updated.
                self._remove_active_constraint(nlidx);
            # otherwise, check if any constraints
            # can be added to the active set
            else:
                # calculate step length
                alpha, cidx = self._calc_step_length(cur_x, p)
                # if alpha = 0 then return current solution
                if np.isclose(alpha, 0, rtol=0, atol=self.tol):
                    return cur_x, self._calc_objective(A, b, cur_x), iter
                elif alpha < 1:
                    self._add_active_constraint(cidx)
                    cur_x = cur_x + alpha * p
                else:
                    cur_x = cur_x + p
        # issue maxiter warning
        str = "ActiveSet: maximum number of iterations reached ({})".format(self.maxiter)
        warn(str, category=RuntimeWarning)
        return cur_x, self._calc_objective(cur_x), iter


class ConstrainedLS(ActiveSet):
    """
    An active set method for constrained least squares

    Example:
        A = [[0.9501, 0.7620, 0.6153, 0.4057],
             [0.2311, 0.4564, 0.7919, 0.9354],
             [0.6068, 0.0185, 0.9218, 0.9169],
             [0.4859, 0.8214, 0.7382, 0.4102],
             [0.8912, 0.4447, 0.1762, 0.8936]]
        b = [0.0578, 0.3528, 0.8131, 0.0098, 0.1388]
        Ci = [[0.2027, 0.2721, 0.7467, 0.4659],
              [0.1987, 0.1988, 0.4450, 0.4186],
              [0.6037, 0.0152, 0.9318, 0.8462]]
        di = [0.5251, 0.2026, 0.6721]
        Ce = [[3, 5, 7, 9]]
        de = [4]
        cu = [2, 2, 2, 2]
        cl = [-0.1, -0.1, -0.1, -0.1]

        cls = ConstrainedLS(atol=1e-7)
        x, scr, nit = cls(A, b, Ce, de, Ci, di, cu, cl)

        print x
        [[-0.10000], [-0.10000], [0.15991], [0.40896]]

    """

    # objective function Hessians:
    #   H = 2 * A.T * A
    #   h = 2 * A.T * b
    def _calc_Hessians(self):
        At = self.A.T
        self.H = 2 * np.dot(At, self.A)
        self.h = 2 * np.dot(At, self.b)


    # score the solution x:
    #  f(x) = || A * x - b || ** 2
    def _calc_objective(self, x):
        return np.sum((np.dot(self.A, x) - self.b)**2)


    # command requires objective target vector
    def __call__(self, A, b, Ce=[], de=[], Ci=[], di=[], cu=[], cl=[], x0=[]):
        return self.run(A=A, b=b, Ce=Ce, de=de, Ci=Ci, di=di,
                        cu=cu, cl=cl, x0=x0)


class PortfolioOpt(ActiveSet):
    """
    An active set method for portfolio optimization with minimum-risk
    objective function. See source file portfolio.py for example.
    """

    # objective function Hessians:
    #   H = 2 * A
    #   h = 0
    def _calc_Hessians(self):
        self.H = 2 * self.A
        self.h = np.zeros(shape=(self.H.shape[0], 1))


    # score the solution x:
    #   f(x) = x * Q * x
    def _calc_objective(self, x):
        return np.linalg.multi_dot([x.T, self.A, x])[0][0]


    # command does not require objective target vector
    def __call__(self, A, Ce=[], de=[], Ci=[], di=[], cu=[], cl=[], x0=[]):
        return self.run(A=A, b=[], Ce=Ce, de=de, Ci=Ci, di=di,
                        cu=cu, cl=cl, x0=x0)
