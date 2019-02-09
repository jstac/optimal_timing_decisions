import numpy as np 
from scipy import interp
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from numpy import maximum as npmax
from numpy import minimum as npmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


class Job_Search_SV_multi(object):
	"""
	A class to store a given parameterization of the generalized
	job search model. (Parametric class problem w.r.t. rho)

	The state process:
	w_t = eta_t + thet_t * xi_t, 
	log(thet_t) = rho* log(thet_{t-1}) + log(u_t),
	where:
		eta_t ~ LN(mu_eta, gam_eta) = v(.)
		xi_t ~ LN(0, gam_xi) = h(.)
		u_t ~ LN(0, gam_u) 

	The value function:
	v^*(w, thet) = max{u(w)/(1 - beta),
	                   c0+ beta*E[v^*(w',thet')|thet]}
    where:
    	E[v^*(w',thet')|thet] 
    	= int 
    	     v^*(w', thet')* f(thet'|thet)* v(eta')* h(xi')
    	  d(thet',eta',xi')

    	w' = eta' + thet' * xi'
    	f(thet'|thet) = LN(rho* thet, gam_u)

    The continuation value operator:
    Q psi(thet) 
      = c0 + 
        beta * E{max{u(w')/(1-beta),psi(thet')}|thet}
    where:
    	E{max{u(w')/(1-beta),psi(thet')}|thet}
    	= int 
    		 max{u(w')/(1-beta),psi(thet')}
    		 * f(thet'|thet)* v(eta')* h(xi')
    	  d(thet',eta',xi')

    Parameters
	----------
	beta : scalar(float), optional(default=0.95)
		   The discount factor
	c0_tilde : scalar(float), optional(default=1.)
		       The unemployment compensation
    sig : scalar(float), optional(default=2.5)
          The coefficient of relative risk aversion
    gam_u : scalar(float), optional(default=1e-4)
    		The variance of the shock process {u_t}
    gam_xi : scalar(float), optional(default=5e-4)
    		 The variance of the transient shock process 
    		 {xi_t}
    mu_eta : scalar(float), optional(default=0.)
             The mean of the process {eta_t}
    gam_eta : scalar(float), optional(default=1e-6.)
              The variance of the process {eta_t}
    rho_min : scalar(float), optional(default=0.)
    		  The minumum of the grid for rho
    rho_max : scalar(float), optional(default=1.)
    		  The maximum of the grid for rho
    rho_size : scalar(int), optional(default=100)
    		  The number of grid points over rho
	thet_min : scalar(float), optional(default=1e-3)
	           The minimum of the grid for thet
	thet_max : scalar(float), optional(default=10.)
	           The maximum of the grid for thet
	thet_size : scalar(int), optional(default=200)
			    The number of grid points over thet
	mc_size : scalar(int), optional(default=1000)
		       The number of Monte Carlo samples
	"""

	def __init__(self, beta=.95, c0_tilde=.6, sig=2.5,
		         gam_u=1e-4, gam_xi=5e-4,
		         mu_eta=0., gam_eta=1e-6,
		         rho_min=0., rho_max=1., rho_size=100, 
		         thet_min=1e-4, thet_max=10., thet_size=200,
		         mc_size=1000):

	    self.beta, self.c0_tilde, self.sig = beta, c0_tilde, sig 
	    self.gam_u, self.gam_xi = gam_u, gam_xi
	    self.mu_eta, self.gam_eta = mu_eta, gam_eta
	    # make grids for rho
	    self.rho_min, self.rho_max = rho_min, rho_max
	    self.rho_size = rho_size
	    self.rho_grids = np.linspace(self.rho_min, self.rho_max,
	    	                         self.rho_size)
        # make grids for theta
	    self.thet_min, self.thet_max = thet_min, thet_max
	    self.thet_size = thet_size
	    self.thet_grids = self.makegrid(self.thet_min, self.thet_max,
	    	                            self.thet_size, 4)
	    # alternatively, just use equal step linspace to 
	    # make theta grids
	    #self.thet_grids = np.linspace(self.thet_min, self.thet_max,
	    #	                           self.thet_size)

	    # make grids
	    self.thet_mesh, self.rho_mesh = np.meshgrid(self.thet_grids,
	    	                                        self.rho_grids)
	    self.grid_points = np.column_stack((self.thet_mesh.ravel(1),
	    	                                self.rho_mesh.ravel(1)))
	    # Monte Carlo draws
	    self.mc_size = mc_size
	    self.draws = np.random.randn(self.mc_size)


	def makegrid(self, amin, amax, asize, ascale):
		"""
		Generates grid a with asize number of points ranging
		from amin to amax. 

		Parameters
		----------
		amin: the minimum grid 
		amax: the maximum grid 
		asize: the number of grid points to be generated

		ascale=1: generates equidistant grids, same as np.linspace
		ascale>1: the grid is scaled to be more dense in the 
				  lower value range

		Returns
		-------
		a : array_like(float, ndim=1, length=asize)
			The generated grid points
		
		"""
		a = np.empty(asize)
		adelta = (amax - amin) / ((asize - 1)**ascale)

		for i in range(asize):
			a[i] = amin + adelta * (i ** ascale)
		return a


	def utility_func(self, x):
		"""
		The utility function (CRRA).
		"""
		sig = self.sig
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = (x**(1 - sig)) / (1 - sig)
		return uw


	def cvals_operator(self, psi):
		"""
		The continuation value operator.
		"""
		c0_tilde, beta = self.c0_tilde, self.beta
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		mc_size, draws = self.mc_size, self.draws
		grid_points = self.grid_points
		utility_func = self.utility_func
		# interpolate to obtain a function
		psi_interp = LinearNDInterpolator(grid_points, psi)
		#psi_interp = NearestNDInterpolator(grid_points, psi) 
        
        # points outside the truncated state space will be
        # evaluated at the nearest points on the state space.
		def psi_f(x,y):
			x = npmax(npmin(x, self.thet_max), self.thet_min)
			return psi_interp(x,y)

		N = len(psi)
		new_psi = np.empty(N)

		for i in range(N):
			thet, rho = grid_points[i, :]
			# Monte Carlo draws used to compute integration
			eta_draws = np.exp(mu_eta + np.sqrt(gam_eta)*draws)
			thet_draws = np.exp(rho * np.log(thet) + np.sqrt(gam_u) * draws)
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			utils = utility_func(eta_draws+ thet_draws* xi_draws)
			integrand_1 = utils / (1. - beta)
			integrand_2 = psi_f(thet_draws, rho* np.ones(mc_size))
			integrand = npmax(integrand_1, integrand_2)

			new_psi[i] = utility_func(c0_tilde) + beta * np.mean(integrand)

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-4,
		                    max_iter=500, verbose=1):
	    """
	    Compute the fixed point.
	    """
	    error = error_tol + 1.
	    iteration = 0

	    while error > error_tol and iteration < max_iter:

	    	Qpsi = Q(psi)
	    	error = max(abs(Qpsi - psi))
	    	psi = Qpsi
	    	iteration += 1

	    	if verbose:
	    		print ("Computing iteration ", iteration, " with error", error)

	    return psi


	def res_rule(self, y):
		"""
		Compute the reservation wage.
		"""
		beta, sig = self.beta, self.sig
		if sig == 1.:
			w_bar = np.exp(y * (1. - beta))
		else:
			w_bar = (y *(1.-beta)* (1.-sig))**(1./ (1.- sig))

		return w_bar



jssv = Job_Search_SV_multi(sig=1.)

N = len(jssv.grid_points)
psi_0 = jssv.utility_func(np.ones(N))

"""
# iterate to obtain psi_star without using compute_fixed_point
print ("")

for i in range(50):
	psi_0 = jssv.cvals_operator(psi_0)
	print ("iteration", i+1, "completed ...")

psi_star = psi_0
"""

# using compute_fixed_point to obtain psi_star
psi_star = jssv.compute_fixed_point(jssv.cvals_operator, psi_0)

res_wage = jssv.res_rule(psi_star)

psi_star_plt = psi_star.reshape((jssv.thet_size, jssv.rho_size))
res_wage_plt = res_wage.reshape((jssv.thet_size, jssv.rho_size))

thet_mesh, rho_mesh = jssv.thet_mesh, jssv.rho_mesh



# =============== Plot figure 4 of the main paper ================ #

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(thet_mesh, rho_mesh, res_wage_plt.T,
	            rstride=2, cstride=3, cmap=cm.jet,
	            alpha=0.5, linewidth=0.25)

ax.tick_params(labelsize=20)

ax.set_xlabel('$\\theta$', fontsize=20, labelpad=10)
ax.set_ylabel('$\\rho$', fontsize=20, labelpad=10)
ax.set_zlabel('wage   ', fontsize=20, labelpad=10)


# the reservation wage function
res_wage_interp = LinearNDInterpolator(jssv.grid_points, res_wage)


"""
# plot the red line at theta = 0.1
xx1 = np.linspace(.1, .1, 100)
yy1 = np.linspace(0., 1., 100)
zz1 = res_wage_interp(xx1, yy1)
ax.plot(xx1, yy1, zz1, '.', color='r', linewidth=1.2, 
         label='$\\theta=0.1$')
"""

# plot the red line at theta = 1
xx2 = np.linspace(1., 1., 100)
yy2 = np.linspace(0., 1., 100)
zz2 = res_wage_interp(xx2, yy2)
ax.plot(xx2, yy2, zz2, '-', color='r', linewidth=1.2, 
	    label='$\\theta=1$')

# plot the red line at theta = 5
res_wage_interp = LinearNDInterpolator(jssv.grid_points, res_wage)
xx3 = np.linspace(5., 5., 100)
yy3 = np.linspace(0., 1., 100)
zz3 = res_wage_interp(xx3, yy3)
ax.plot(xx3, yy3, zz3, '--', color='r', linewidth=1.2,
	    label='$\\theta=5$')

ax.legend(loc='right')

plt.show()











