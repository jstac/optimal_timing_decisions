import numpy as np
from numpy import maximum as npmax
from numpy import minimum as npmin
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm




class Job_Search_adaptive(object):
	"""
	A class to store a given parameterization of the generalized
	job search model.

	The value function:
	v^*(w,mu,gam) = max{ u(w)/(1 - beta), 
	                     c0+ beta*E[v^*(w',mu',gam')|mu,gam]}
	
	The Bayesian updating process:
	Z_t = xi + eps_t, eps ~ N(0, gam_eps), iid
	w_t = e^{Z_t}
	prior: xi ~ N(mu, gam)
	posterior: xi|w' ~ N(mu', gam')
			   gam' = 1/(1/gam + 1/gam_eps)
			   mu' = gam'*(mu/gam + log(w')/gam_eps)

	Agents have constant relative risk aversion:
	u(w) = w^{1 - sig} / (1 - sig), 
	       if sig >=0 and sig is not equal to 1;
	u(w) = log(w), if sig = 1.

	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
		   The discount factor
	c0_tilde : scalar(float), optional(default=0.05)
		       The unemployment compensation
    gam_eps : scalar(float), optional(default=1.)
              The variance of the shock process
    sig : scalar(float), optional(default=4.)
          the coefficient of relative risk aversion
    mu_min : scalar(float), optional(default=-10.)
             The minimum of the grid for mu
    mu_max : scalar(float), optional(default=10.)
             The maximum of the grid for mu
	musize : scalar(int), optional(default=200)
			 The number of grid points over mu
	gam_min : scalar(float), optional(default=1e-3)
	          The minimum of the grid for gam
	gam_max : scalar(float), optional(default=10.)
	          The maximum of the grid for gam
	gamsize : scalar(int), optional(default=100)
			  The number of grid points over gam
	mc_size : scalar(int), optional(default=10000)
		      The number of Monte Carlo samples
	"""
	def __init__(self, beta=.95, c0_tilde=.6, gam_eps=1., sig=4., 
		         mu_min=-10., mu_max=20., mu_size=200,
		         gam_min= 1e-3, gam_max=10., gam_size=100,
		         mc_size=1000):
	    self.c0_tilde, self.beta = c0_tilde, beta
	    self.gam_eps, self.sig = gam_eps, sig
	    self.mu_min, self.mu_max = mu_min, mu_max
	    self.mu_size = mu_size
	    self.gam_min, self.gam_max = gam_min, gam_max
	    self.gam_size = gam_size
	    self.mc_size = mc_size
	    # make grids for mu
	    self.mu_grids = np.linspace(self.mu_min, self.mu_max, 
	    	                        self.mu_size)
	    # make grids for gam
	    self.gam_grids = np.linspace(self.gam_min, self.gam_max, 
	    	                         self.gam_size)
	    self.mu_mesh, self.gam_mesh= np.meshgrid(self.mu_grids,
	    	                                     self.gam_grids)
	    self.grid_points= np.column_stack((self.mu_mesh.ravel(1),
        	                               self.gam_mesh.ravel(1)))
	    # initial Monte Carlo draws
	    self.draws = np.random.randn(mc_size)


	def util_func(self,x):
		"""
		The (CRRA) utility function.
		"""
		sig = self.sig
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		return uw


	def r(self, x):
		"""
		The exit payoff (reward) function.
		r(w) = u(w) / (1 - beta).
		"""
		beta, sig = self.beta, self.sig
		if sig == 1.:
			rw = np.log(x) / (1. - beta)
		else:
			rw = x**(1. - sig) / ((1. - sig)*(1. - beta))
		return rw


	def cval_operator(self, psi):
		"""
		The continuation value operator
		--------------------------------
		Qpsi = c0 + 
			   beta*int max{u(w')/(1-beta),phi(mu',gam')} 
			            * f(w'|mu,gam)
                    dw'
		where:
			   u(w) is the CRRA utility
			   f(w'|mu, gam) = LN(mu, gam + gam_eps)
			   gam' = 1/(1/gam + 1/gam_eps)
			   mu' = gam' * (mu/gam + log(w')/gam_eps)

		The operator Q is a contraction mapping on 
		(b_l Z, rho_l) with unique fixed point psi^*,
		where:
		    l is the weight function defined by model
		    primitives.

		Parameters
		----------
		psi : array_like(float, ndim=1, length=len(grid_points))
			  An approximate fixed point represented as a one-dimensional
			  array.

		Returns
		-------
		new_psi : array_like(float, ndim=1, length=len(grid_points))
				  The updated fixed point.

		"""
		beta, gam_eps, c0_tilde = self.beta, self.gam_eps, self.c0_tilde
		mc_size, draws = self.mc_size, self.draws
		psi_interp = LinearNDInterpolator(self.grid_points, psi)
		
		def psi_f(x, y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmin(npmax(x, self.mu_min), self.mu_max)
			y = npmin(npmax(y, self.gam_min), self.gam_max)
			return psi_interp(x, y)

		N = len(psi)
		new_psi = np.empty(N)

		for i in range(N):
			mu, gam = self.grid_points[i,:]
			# Bayesian updating of gam. "gam_prime" is a scalar.
			gam_prime = 1. / (1. / gam + 1. / gam_eps)
			# MC sampling : log(w')
			# ln_prime : an array of length mc_size
			lnw_prime = mu + np.sqrt(gam + gam_eps) * draws
			# MC sampling : mu'; and
			# Bayesian updating of mu
			# mu_prime : an array of length mc_size
			mu_prime = gam_prime* (mu/gam + lnw_prime/gam_eps)

			# MC sampling : r(w')
			integrand_1 = self.r(np.exp(lnw_prime))
			# MC sampling : psi_f(mu',gam')
			integrand_2 = psi_f(gam_prime * np.ones(mc_size),
				                mu_prime)
			integrand = npmax(integrand_1, integrand_2)	
			# c0 = u(c0_tilde): write the unmemployment
			# compensation in the utility form
			c0 = self.util_func(self.c0_tilde) 
			new_psi[i] = c0 + beta * np.mean(integrand) 

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-4,
		                    max_iter=500, verbose=1):
	    """
	    Compute the fixed point of the operator Q.
	    """
	    error = error_tol + 1.
	    iteration = 0

	    while error > error_tol and iteration < max_iter:
	    	Qpsi = Q(psi)
	    	error = max(abs(Qpsi - psi))
	    	psi = Qpsi
	    	iteration += 1

	    	if verbose:
	    		print ("Compute iteration ", iteration, " with error ", error)

	    return psi


	def res_rule_func(self, y):
		"""
		Recover the reservation rule based on the 
		continuation value function.
		"""
		sig, beta = self.sig, self.beta

		if sig == 1.:
			res_rule = np.exp(y* (1. - beta))
		else:
			base = (y * (1. - beta)* (1. - sig))
			res_rule = base**(1. / (1. - sig))

		return res_rule


	def res_utility(self, y):
		"""
		Recover the reservation utility based on the
		continuation value function.
		"""
		return y * (1 - self.beta)



# =============== Plot figure 2 of the main paper =============== #
# plot against 4 different sig values

sig_selections = [3., 4., 5., 6.]

fig = plt.figure(figsize=(12, 12))  

print ("")
print ("Computation in progress ...")
print ("")

for i, y in enumerate(sig_selections):
	jsa = Job_Search_adaptive(sig=y)
	psi_init = np.ones(len(jsa.grid_points))

	# compute the fixed point (continuation value)
	psi_star = jsa.compute_fixed_point(Q=jsa.cval_operator, psi=psi_init)
	# the reservation utility
	#res_util = jsa.res_utility(psi_star)
	# the reservation wage
	res_rule = jsa.res_rule_func(psi_star) 

	# reshape the solutions for plotting
	psi_star_plt = psi_star.reshape((jsa.mu_size, jsa.gam_size))
	#res_util_plt = res_util.reshape((jsa.mu_size, jsa.gam_size))
	res_rule_plt = res_rule.reshape((jsa.mu_size, jsa.gam_size))

	# plot on the important part of the domain
	mu_mesh, gam_mesh = jsa.mu_mesh, jsa.gam_mesh

	ax = fig.add_subplot(2, 2, i+1, projection='3d')
	ax.plot_surface(mu_mesh, gam_mesh, res_rule_plt.T,
	                rstride=2, cstride=3, cmap=cm.jet,
	                alpha=0.5, linewidth=0.25)

	ax.set_title('$\sigma$ = {}'.format(y),fontsize=15)
	
	ax.set_xlabel('$\mu$', fontsize=14)
	ax.set_ylabel('$\gamma$', fontsize=14)
	ax.set_zlabel('wage', fontsize=12)

	ax.set_xlim((-10, 20))
	ax.set_ylim((0, 10))
	#ax.set_zlim((0.5, 0.7))

	print ("")
	print ("Loop ", i+1, " finished ... ", 3-i, " left ...")
	print ("")

plt.show()



"""
# =================== plot against gam_eps ===================== #

gam_eps_selections = [0.1, 0.5, 1., 1.5]

fig = plt.figure(figsize=(12, 12))  

print ""
print "Computation in progress ..."
print ""

for i, y in enumerate(gam_eps_selections):
	jsa = Job_Search_adaptive(gam_eps=y)
	psi_init = np.ones(len(jsa.grid_points))

	# compute the fixed point (continuation value)
	psi_star = jsa.compute_fixed_point(Q=jsa.cval_operator, psi=psi_init)
	# the reservation utility
	#res_util = jsa.res_utility(psi_star)
	# the reservation wage
	res_rule = jsa.res_rule_func(psi_star) 

	# reshape the solutions for plotting
	psi_star_plt = psi_star.reshape((jsa.mu_size, jsa.gam_size))
	#res_util_plt = res_util.reshape((jsa.mu_size, jsa.gam_size))
	res_rule_plt = res_rule.reshape((jsa.mu_size, jsa.gam_size))

	# plot the figure on the important part of the grid range
	mu_mesh, gam_mesh = jsa.mu_mesh, jsa.gam_mesh

	ax = fig.add_subplot(2, 2, i+1, projection='3d')
	ax.plot_surface(mu_mesh, gam_mesh, res_rule_plt.T,
	                rstride=2, cstride=3, cmap=cm.jet,
	                alpha=0.5, linewidth=0.25)

	ax.set_title('$\gamma_\epsilon$ = {}'.format(y),fontsize=15)
	
	ax.set_xlabel('$\mu$', fontsize=14)
	ax.set_ylabel('$\gamma$', fontsize=14)
	ax.set_zlabel('reservation wage', fontsize=12)

	#ax.set_xlim((-14, 12))
	#ax.set_ylim((0, 25))
	ax.set_zlim((0.5, 0.7))

	print ""
	print "Loop ", i+1, " finished ... ", 3-i, " left ..."
	print ""

plt.show()
"""