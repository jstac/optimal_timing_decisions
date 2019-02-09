import numpy as np
from numpy import maximum as npmax
from numpy import minimum as npmin
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import lognorm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import time



class Firm_Entry(object):
	"""
	The firm entry model of Fajgelbaum etc. (2016)

	x = theta + eps_x, eps_x ~ N(0, gam_x)
	where: 
		  x: output, not observed when the firm makes decisions 
		   	 at the beginning of each period.

	y = theta + eps_y, eps_y ~ N(0, gam_y)
	where: 
		  y: a public signal, observed after the firm makes 
		     decision of whether to invest or not.


	The Bayesian updating process:
	theta ~ N(mu, gam) : prior
	theta|y ~ N(mu', gam') : posterior after observing public 
							 signal y
	where: 
		  gam' = 1 / (1/gam + 1/gam_y)
		  mu' = gam'*(mu/gam + y/gam_y)


	The firm has constant absolute risk aversion:
	u(x) = (1/a) * (1 - exp(-a*x))
	where: a is the coefficient of absolute risk aversion


	f ~ h(f) = LN(mu_f, gam_f) : the entry cost / the investment cost


	The value function:
	V(f,mu,gam) = max{ E[u(x)|mu,gam]-f, beta*E[V(f',mu',gam')|mu,gam] }
	where: 
		   RHS 1st: the value of entering the market and investing
		   RHS 2nd: the expectated value of waiting


	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
	       The discount factor
	a : scalar(float), optional(default=0.2)
	    The coefficient of absolute risk aversion
	mu_min : scalar(float), optional(default=-2.)
	         The minimum grid of mu
	mu_max : scalar(float), optional(default=10.)
	         The maximum grid for mu
	mu_size : scalar(int), optional(default=200)
	          The number of grid points over mu
	gam_min : scalar(float), optional(default=1e-4)
	          The minimum grid for gam
	gam_max : scalar(float), optional(default=10)
	          The maximum grid for gam
	gam_size : scalar(int), optional(default=100)
	           The number of grid points over gam
	mu_f : scalar(float), optional(default=0.)
		   The mean of the cost distribution 
		   {f_t} ~ h(f) = LN(mu_f, gam_f)
	gam_f : scalar(float), optional(default=0.01)
			The variance of the cost distribution 
			{f_t} ~ h(f) = LN(mu_f, gam_f)  
	gam_x : scalar(float), optional(default=0.1)
	   		The variance of eps_x, eps_x ~ N(0, gam_x)
	gam_y : scalar(float), optional(default=0.05)
			The variance of eps_y, eps_y ~ N(0, gam_y)
	mc_size : scalar(int), optional(default=1000)
		      The number of Monte Carlo samples 
	"""

	def __init__(self, beta=.95, a=.2,
		         mu_min=-2., mu_max=10., mu_size=100,
		         gam_min=1e-4, gam_max=10., gam_size=100,
		         mu_f=0., gam_f=.01, gam_x=.1, gam_y=.05,
		         mc_size=1000):

		self.beta, self.a = beta, a
		self.mu_f, self.gam_f = mu_f, gam_f
		self.gam_x, self.gam_y = gam_x, gam_y
		# make grids for mu
		self.mu_min, self.mu_max = mu_min, mu_max
		self.mu_size = mu_size
		self.mu_grids = np.linspace(self.mu_min, self.mu_max,
			                        self.mu_size)
		# make grids for gamma
		self.gam_min, self.gam_max = gam_min, gam_max
		self.gam_size = gam_size
		self.gam_grids = np.linspace(self.gam_min, self.gam_max,
			                         self.gam_size)
		# make grids 
		self.mu_mesh, self.gam_mesh = np.meshgrid(self.mu_grids,
			                                      self.gam_grids)
		self.grid_points = np.column_stack((self.mu_mesh.ravel(1), 
			                                self.gam_mesh.ravel(1)))
		# initial Monte Carlo draws
		self.mc_size = mc_size
		self.draws = np.random.randn(self.mc_size)


	def r(self, x, y, z):
		"""
		The exit payoff function. The expected reward of
		paying the cost f and entering the market.
		r(f, mu, gamma)
		"""
		a, gam_x = self.a, self.gam_x
		part_1 = -a * y + (a**2) * (z + gam_x) / 2.
		return (1. - np.exp(part_1)) / a - x


	def cvals_operator(self, psi):
		"""
		The continuation value operator
		--------------------------------
		Qpsi(mu,gam) 
		= beta * integral( 
			           max{reward,phi(mu',gam')} 
			           * h(f')*l(y|mu,gam) ) 
                 d(f',y)
		where:
			  f ~ h(f) = LN(mu_f, gam_f) : the entry/investment cost
			  gam' = 1/(1/gam + 1/gam_y)
			  mu' = gam' * (mu/gam + y/gam_y)
			  reward = (1/a) - (1/a)*exp[-a*mu' + (a**2)*(gam'+gam_x)/2] - f'
   			  l(y|mu, gam) = N(mu, gam + gam_y)


		The operator Q is a contraction mapping on 
		(b_{ell} Y, rho_{ell}) with unique fixed point psi^*.


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
		beta = self.beta
		gam_y = self.gam_y
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		mu_f, gam_f = self.mu_f, self.gam_f
		mc_size, draws = self.mc_size, self.draws
		grid_points = self.grid_points
		psi_interp = LinearNDInterpolator(grid_points, psi)

		def psi_f(x, y):
			"""
			Notice that the arguments of this function 
			are ordered by : psi_f(mu, gamma).
			"""
			x = npmin(npmax(x, mu_min), mu_max)
			y = npmin(npmax(y, gam_min), gam_max)
			return psi_interp(x, y)

		N = len(psi)
		new_psi = np.empty(N)

		for i in range(N):
			mu, gam = grid_points[i, :]
			# MC draws for y
			y_draws = mu + np.sqrt(gam + gam_y)* draws
			# MC draws for f'
			f_prime = np.exp(mu_f + np.sqrt(gam_f) * draws)

			# MC draws for gamma'
			gam_prime = 1. / (1. / gam + 1. / gam_y)
			# MC draws for mu'
			mu_prime = gam_prime * (mu / gam + y_draws / gam_y)

			# MC draws for r(f',mu',gamma')
			rprime_draws = self.r(f_prime, mu_prime, gam_prime)
			# MC draws for psi(mu',gamma')
			psiprime_draws = psi_f(mu_prime,
				                   gam_prime * np.ones(mc_size))
			# MC draws: max{r(f',mu',gam'), psi(mu',gam')}
			integrand = npmax(rprime_draws, psiprime_draws)
			new_psi[i] = beta * np.mean(integrand)

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-4,
		                    max_iter=500, verbose=1):
		"""
		Compute the fixed point.
		"""
		iteration = 0
		error = error_tol + 1.

		while iteration < max_iter and error > error_tol:
			Qpsi = Q(psi)
			error = max(abs(Qpsi - psi))
			psi = Qpsi
			iteration += 1

			if verbose:
				print ("Computing iteration", iteration," with error", error)

		return psi


	def res_rule(self, psi):
		"""
		The reservation cost function.
		"""
		a, gam_x = self.a, self.gam_x
		grid_points = self.grid_points
		part_1 = -a * grid_points[:,0] + \
		         (a**2)* (grid_points[:,1] + gam_x) / 2.
		part_2 = (1. - np.exp(part_1)) / a

		return part_2 - psi



fe = Firm_Entry()

# compute fixed point via compute_fixed_point
psi_0 = np.ones(len(fe.grid_points))
psi_star = fe.compute_fixed_point(fe.cvals_operator, psi_0)


"""
# compute fixed point without using compute_fixed_point
psi_star = np.ones(len(fe.grid_points))
for i in range(20):
	psi_star = fe.cvals_operator(psi_star)
	print ("Iteration ", i+1, "completed ...")
"""


# compute the reservation cost
res_cost = fe.res_rule(psi_star)
# compute the probability of investment
prob_inv = lognorm.cdf(res_cost, s=np.sqrt(fe.gam_f),
	                   scale=np.exp(fe.mu_f))


psi_star_plt = psi_star.reshape((fe.mu_size, fe.gam_size))
res_cost_plt = res_cost.reshape((fe.mu_size, fe.gam_size))
prob_inv_plt = prob_inv.reshape((fe.mu_size, fe.gam_size))


mu_mesh, gam_mesh = fe.mu_mesh, fe.gam_mesh


# plot figure 3 of the main paper, i.e.,
# plot the perceived probability of investment 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				prob_inv_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.tick_params(labelsize=20)

ax.set_xlabel('$\mu$', fontsize=20, labelpad=10)
ax.set_ylabel('$\gamma$', fontsize=20, labelpad=10)
ax.set_zlabel('probability', fontsize=20, labelpad=10)

# plot the reservation cost
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				res_cost_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=20, labelpad=10)
ax.set_ylabel('$\gamma$', fontsize=20, labelpad=10)
ax.set_zlabel('cost', fontsize=20, labelpad=10)

# plot the continuation value
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				psi_star_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=20, labelpad=10)
ax.set_ylabel('$\gamma$', fontsize=20, labelpad=10)
ax.set_zlabel('continuation value', fontsize=20, labelpad=10)

plt.show()