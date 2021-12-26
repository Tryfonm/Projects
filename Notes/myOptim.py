import time
import numpy as np
from matplotlib import pyplot as plt


class myUnconstrainedOptimizer():

    def __init__(self, variablesToTrack=['x', 'f', 'J'], grad_tol=1e-4, max_iters=100000):
        """

        Parameters
        ----------
        variablesToTrack
        grad_tol
        max_iters
        """
        self._fun = None
        self.iteratesTillConvergence = 0
        self.variablesToTrack = variablesToTrack
        self.grad_tol = grad_tol  # Convergence tolerance
        self.max_iters = max_iters  # Just in case
        self.xopt = None

    def clearLog(self, ):
        """
        Clears and initializes the log.

        Returns
        -------

        """
        self._log = {}
        for variable in self.variablesToTrack:
            self._log[variable] = []

    def updateLog(self, currentValues, precision=4):
        """
        """
        for varIndex, variable in enumerate(self.variablesToTrack):
            self._log[variable].append(currentValues[varIndex])

    def getLog(self, variables=None):
        """
        Returns a numpy.array if only one variable is requested, else a list of numpy.arrays is returned.

        Parameters
        ----------
        variables

        Returns
        -------

        """
        totalValues = len(self._log['x'])
        if variables != None:
            temp = []
            for variable in variables:
                temp.append(np.array(self._log[variable]).reshape(self.iteratesTillConvergence, -1))
            return temp[0] if len(temp) == 1 else temp
        else:
            return self._log

    @property
    def fun(self):
        return self._fun
    
    @staticmethod
    def checkDimensions(x0, fun):
        """

        Parameters
        ----------

        Returns
        -------

        """

        # Checking the starting point shape and the function's output shape
        try:
            if x0.shape[1]==1 and x0.shape!=(1,) and fun(x0).shape == (1,1):
                return 1
            else:
                print(f'The optimizer expects a starting point x0 with shape [Nx, 1] and the output of f has to be of shape [Nf, 1]')
                return 0
        except:
            # If for someone the output shape cannot be computed
            print("Sth went really bad")
            return 0
            

    def plot2d(self, contourDensity= 50, lims=[-3, 3, -3, 3], figsize=(20, 8)):
        """

        Parameters
        ----------
        lims

        Returns
        -------

        """

        def fun_aug(xy):
            return self._fun(xy).item()

        xlim_l, xlim_r, ylim_l, ylim_r = lims

        x = np.linspace(xlim_l, xlim_r, 100)
        y = np.linspace(ylim_l, ylim_r, 100)
        xy = np.stack((x, y))
        X, Y = np.meshgrid(x, y)
        XY = np.stack((X, Y))
        Z = np.apply_along_axis(fun_aug, axis=0, arr=XY)

        plt.figure(figsize=figsize, dpi=80)
        contours = plt.contour(X, Y, Z, contourDensity, colors='black', linewidths=0.2)
        # plt.clabel(contours, inline=False, fontsize=5)

        plt.imshow(Z, extent=[xlim_l, xlim_r, ylim_l, ylim_r], origin='lower',
                   cmap='RdGy', alpha=0.5)
        plt.colorbar();
        x_iters = self.getLog('x')
        if self.iteratesTillConvergence != 0:
            xs = x_iters[:, 0]
            ys = x_iters[:, 1]
            plt.plot(xs, ys, linestyle='-', marker='.', color='black')
        else:
            print(f'No minimizer has been run yet - Plotting the function only')
        plt.show()

    @staticmethod
    def finite_difference_jacob(fun, x0):
        """
        Returns a tuple of (f, J)

        Parameters
        ----------
        fun
        x0

        Returns
        -------

        """
        if x0.ndim != 2:
            x0 = x0.reshape((-1, 1))
        Nx = x0.shape[0]  # Variable count
        f0 = fun(x0)  # F(x0) at x0
        Nf = f0.shape[0]  # Number of f's
        h = 1e-8
        I = np.eye(Nx)
        J = []

        for k in range(Nx):
            temp = (fun(x0 + (h * I[:, k]).reshape(-1, 1)) - f0) / h
            J.append(temp)

        return f0, np.array(J).transpose().squeeze(0)

    @staticmethod
    def line_search(fun, x0, J, dk, sigma, beta):
        """
        Line search using Armijo conditions and backtracking.

        Parameters
        ----------
        fun:    scalar function
        x0:     initial guess
        J:      jacobian of fun at x0
        dk:     search direction
        sigma:  armijo condition scaling function: (0,1)
        beta:   backtracking parameter: (0,1)

        Returns
        -------

        """
        alpha = 1
        f0 = fun(x0)
        trial_x = x0 + alpha * dk
        while fun(trial_x) > f0 + sigma * alpha * J @ dk:
            alpha = beta * alpha
            trial_x = x0 + alpha * dk
        return trial_x

    @staticmethod
    def params_check(searchMethod, search_parameters):
        if searchMethod == 'fullStep' and search_parameters == None:
            alpha = 1
            return alpha
        elif searchMethod == 'fullStep' and search_parameters != None:
            if type(search_parameters) == list and len(search_parameters) == 1:
                alpha = search_parameters[0]
            elif type(search_parameters) == list and len(search_parameters) != 1:
                raise ValueError(
                    "When choosing 'fullStep' as the search method, the parameter 'search_parameters' expects a single value [alpha]")
            else:
                alpha = search_parameters
            return alpha

        if searchMethod == 'lineSearch' and search_parameters == None:
            sigma = 0.01
            beta = 0.6
            return sigma, beta
        elif searchMethod == 'lineSearch' and search_parameters != None:
            if type(search_parameters) == list and len(search_parameters) == 2:
                sigma, beta = search_parameters[0], search_parameters[1]
                if sigma > 0 and sigma < 1 and beta > 0 and beta < 1:
                    return search_parameters[0], search_parameters[1]
                else:
                    raise ValueError("Both sigma and beta need to be >0 and <1")
            elif type(search_parameters) == list and len(search_parameters) != 2:
                raise ValueError(
                    "When choosing 'lineSearch' as the search method, the parameter 'search_parameters' expects a list with 2 numbers [sigma, beta]")
            return sigma, beta

    def minimize_grad_desc(self, fun, x0, searchMethod='fullStep', search_parameters=None):
        """

        Parameters
        ----------
        fun
        x0
        searchMethod
        search_parameters

        Returns
        -------

        """
        
        self.clearLog()
        params = myUnconstrainedOptimizer.params_check(searchMethod, search_parameters)

        start = time.time()
        x = x0
        if not self.checkDimensions(x0, fun):
            raise ValueError("Something went wrong with the point x0 or the output dimension of f.\
                             \nExpecting a vector x0 of shape [Nx, 1] and output of shape: [Nf, 1]")
        self._fun = fun
        
        for k in range(self.max_iters):
            # Check for divergence
            # ToDo

            # Evaluate the Jacobian
            _, J = myUnconstrainedOptimizer.finite_difference_jacob(self._fun, x)

            # Check for convergence
            if np.linalg.norm(J, np.inf) < self.grad_tol:
                end = time.time()
                self.iteratesTillConvergence = k
                self.xopt = x
                print(f'\nConvergence took {k} iterates | {round(end - start, 2)} sec(s)')
                return self.xopt

            self.updateLog([x, self._fun(x), J])

            # Take a gradient step
            dk = -J.transpose()  # As a reminder: Jacobian is assumed to be equal to the tranposed gradient
            if searchMethod == 'fullStep':
                alpha = params
                x = x + (alpha * dk)
            elif searchMethod == 'lineSearch':
                sigma, beta = params
                x = myUnconstrainedOptimizer.line_search(self._fun, x, J, dk, sigma, beta)

        print('Did not converge')
        return x

    def minimize_bfgs(self, fun, x0, searchMethod, search_parameters=None):
        """

        Parameters
        ----------
        fun
        x0
        searchMethod
        search_parameters

        Returns
        -------

        """

        self.clearLog()
        params = myUnconstrainedOptimizer.params_check(searchMethod, search_parameters)

        start = time.time()
        x = x0
        if not self.checkDimensions(x0, fun):
            raise ValueError("Something went wrong with the point x0 or the output dimension of f.\
                             \nExpecting a vector x0 of shape [Nx, 1] and output of shape: [Nf, 1]")
        self._fun = fun

        # Initialize B
        B = np.eye(x0.shape[0])

        # Evaluate the initial Gradient
        _, J = myUnconstrainedOptimizer.finite_difference_jacob(self._fun, x)

        # Check for correct dimension
        # Todo

        for k in range(self.max_iters):

            # Check for divergence
            # ToDo

            # Check for convergence
            if np.linalg.norm(J, np.inf) < self.grad_tol:
                end = time.time()
                self.iteratesTillConvergence = k
                self.xopt = x
                print(f'\nConvergence took {k} iterates | {round(end - start, 2)} sec(s)')
                return self.xopt

            self.updateLog([x, self._fun(x), J])

            # Take a gradient step
            dk = - np.linalg.pinv(
                B) @ J.transpose()  # As a reminder: Gradient is assumed to be equal to the tranposed Jacobiant

            x_old = x
            if searchMethod == 'fullStep':
                alpha = params
                x = x + (alpha * dk)
            elif searchMethod == 'lineSearch':
                sigma, beta = params
                x = myUnconstrainedOptimizer.line_search(self._fun, x, J, dk, sigma, beta)

            # Evaluate the Gradient at the new point
            J_old = J
            _, J = myUnconstrainedOptimizer.finite_difference_jacob(self._fun, x)

            # Update BFGS hessian approximation
            s = x - x_old
            y = J.transpose() - J_old.transpose()
            Bs = B @ s
            B = B - Bs @ (Bs.T / (s.T @ Bs)) + y @ (y.T / (s.T @ y))

        print('Did not converge')
