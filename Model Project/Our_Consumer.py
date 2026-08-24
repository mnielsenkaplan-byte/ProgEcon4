from types import SimpleNamespace

import numpy as np

from scipy import optimize


class ConsumerClass:
    """ a consumer with nested CES preferences over three goods
    
    Good 1 is food. Goods 2 and 3 are bus trips and train trips, and they sit
    together in a nest.

    The problem is written in *nested* budget shares, in the same two steps as
    the nests themselves:

        s1 = the share of income spent on food
        w  = the share of the remaining (travel) budget spent on the bus

    so that s2 = (1-s1)*w and s3 = (1-s1)*(1-w). Any (s1,w) in the unit square
    is a possible choice, and every possible choice is in the unit square, so
    the constraint set is exactly the box that L-BFGS-B takes as `bounds`.

    """

    def __init__(self,par=None):

        # a. setup
        self.setup()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ set the baseline parameters """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.alpha = 0.60 # weight on food
        par.beta = 0.50 # weight on the bus

        # b. substitution
        par.sigma_A = 0.80 # between food and travel (upper nest)
        par.sigma_B = 0.40 # between bus and train (lower nest)

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a bus trip
        par.p3 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical settings
        par.s_min = 1e-12 # smallest quantity allowed in .ces(), see the note there

    def __str__(self):
        """ print the parameters """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')

        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self,z1,z2,w,sigma):
        """ the CES aggregate of two inputs

        Computes (w*z1**rho + (1-w)*z2**rho)**(1/rho) with rho = 1-1/sigma.

        The inputs are floored at par.s_min, because z**rho is not defined for
        z = 0 when rho < 0, and the corners of the unit square do give zeros.

        Note on the size of par.s_min. It must be *far* below the step that
        L-BFGS-B uses to estimate the gradient, which is `eps` and about 1e-8 by
        default. Otherwise both z and z+eps get floored to the same number near a
        bound, utility comes out exactly equal in the two points, the estimated
        gradient is zero, and the solver stops on the bound and stays there. With
        s_min = 1e-8 that really happens: the answers in section 4 for high tax
        rates come out as zero revenue. 1e-12 is small enough to be invisible and
        large enough to keep z**rho from overflowing.

        Args:

            z1 (float or ndarray): first input
            z2 (float or ndarray): second input
            w (float): weight on the first input
            sigma (float): substitution parameter, must not be 1

        Returns:

            (float or ndarray): the CES aggregate

        """

        par = self.par

        assert not np.isclose(sigma,1.0), 'sigma = 1 gives rho = 0 and a division by zero'

        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)

        rho = 1-1/sigma

        return (w*z1**rho + (1-w)*z2**rho)**(1/rho)

    def utility(self,x1,x2,x3):
        """ nested CES utility of a bundle of quantities

        Two steps: first combine goods 2 and 3 into the travel composite, then
        combine good 1 and the composite into utility. Use .ces() for both.

        Args:

            x1 (float or ndarray): quantity of good 1
            x2 (float or ndarray): quantity of good 2
            x3 (float or ndarray): quantity of good 3

        Returns:

            (float or ndarray): utility

        """

        par = self.par

        #x_B = (par.beta*x2**par.rho_B+(1-par.beta)*x3**par.rho_B)**(1/par.rho_B)
        x_B=self.ces(x2,x3,par.beta,par.sigma_B)

        u = self.ces(x1,x_B,par.alpha,par.sigma_A)
        #u = (par.alpha*x1**par.rho_A + (1-par.alpha)*x_B**par.rho_A)**(1/par.rho_A)

        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,s1,w):
        """ the three budget shares implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three budget shares, which always sum to one

        """

        return s1,(1-s1)*w,(1-s1)*(1-w)

    def quantities(self,s1,w):
        """ the quantities implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three quantities

        """

        par = self.par

        s1,s2,s3 = self.shares(s1,w)

        return s1*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3

    def value_of_choice(self,s1,w):
        """ utility of the bundle implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (float or ndarray): utility

        """

        x1, x2, x3 = self.quantities(s1,w)

        u = self.utility(x1, x2, x3)

        return u

    def objective(self,s):
        """ minus utility, for a minimizer

        Nothing else is needed: the bounds are the whole constraint.

        Args:

            s (ndarray): array of length 2 with (s1,w)

        Returns:

            (float): minus utility

        """

        return -self.value_of_choice(s[0],s[1])

    #################q
    # 3. solving it #
    #################

    def solve_grid(self,N=200,do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares

        Every point of the unit square is a possible choice, so there is nothing
        to mask out here -- a plain np.argmax will do.

        Args:

            N (int): number of grid points for each variable
            do_print (bool): print the solution

        Returns:

            (SimpleNamespace): the grids, the utility values and the best point

        """

        par = self.par
        opt = SimpleNamespace()

        # a. the two grids
        
        s1_vec = np.linspace(0,1,N)
        w_vec = np.linspace(0,1,N)

        s1_grid, w_grid= np.meshgrid(s1_vec,w_vec,indexing="ij")


        # b. utility in every grid point
        
        u_grid= self.value_of_choice(s1_grid, w_grid)


        # c. the best point

        index_max = np.argmax(u_grid)
        i, j=np.unravel_index(index_max, u_grid.shape)

        # d. results
        # opt.s1, opt.w, opt.s2, opt.s3, opt.u
        # opt.s1_grid, opt.w_grid, opt.u_grid (needed for the figures)

        opt.s1 = s1_grid[i,j]
        opt.w = w_grid[i,j]
        opt.s2 = (1-opt.s1)*opt.w
        opt.s3 = (1-opt.s1)*(1-opt.w)
        opt.u = u_grid[i,j]

        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid

        if do_print:
            print(f"s1={opt.s1:.4f}")
            print(f"s2={opt.s2:.4f}")
            print(f"s3={opt.s3:.4f}")
            print(f"w={opt.w:.4f}")
            print(f"u={opt.u:.4f}")

        return opt

    
    # Plot

    def plot_grid(self,opt):
        import matplotlib.pyplot as plt
        from matplotlib import cm


        fig = plt.figure(figsize=(12,6))
        ax=fig.add_subplot(1, 2, 1, projection= "3d")
        h = ax.plot_surface(opt.s1_grid,opt.w_grid,opt.u_grid,cmap=cm.viridis)
        fig.colorbar(h,ax=ax,location="left")
        ax.scatter(opt.s1,opt.w,opt.u)

        ax.set_title("Utility over nested shares")
        ax.set_xlabel("$s_1$")
        ax.set_ylabel("$w$")
        ax.set_zlabel("$u$")

        ax.zaxis.labelpad= 0.1

        ax=fig.add_subplot(1, 2, 2)
        ax.contourf(opt.s1_grid,opt.w_grid,opt.u_grid,cmap=cm.viridis)
        ax.scatter(opt.s1,opt.w)
        ax.set_xlabel("$s_1$")
        ax.set_ylabel("$w$")
        ax.set_title("Utility contour plot")


        fig.tight_layout(pad=3.0)

        return fig

        
    def solve(self,s0=None,do_print=True,**kwargs):
        """ solve with L-BFGS-B

        The bounds are ((0,1),(0,1)) -- the whole constraint set.

        Args:

            s0 (ndarray): starting guess for (s1,w)
            do_print (bool): print the solution
            kwargs: passed on to optimize.minimize, e.g. options={'ftol':1e-10}

        Returns:

            (SimpleNamespace): the solution and the convergence path

        """


        par = self.par
        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = (0.5,0.5)
        s0 = np.asarray(s0,dtype=float)
   
        # b. record the path with a callback
        path = [s0.copy()]
        callback=lambda sk: path.append(sk.copy())

        # c. minimize - pass before
        res= optimize.minimize(self.objective,s0,method='L-BFGS-B', bounds=((0,1),(0,1)),callback=callback, **kwargs)
        
        # d. results
        s1 = res.x[0]
        w = res.x[1]
        s2 = (1-s1)*w
        s3 = (1-s1)*(1-w)
        u = self.utility(s1*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3)
        #path = (s1,w) #I am a bit confused what to write to path
        opt.s1=s1
        opt.w=w
        opt.s2=s2
        opt.s3=s3
        opt.u = u
        opt.path = np.array(path)
        opt.res = res

        return opt
