import numpy as np
#from scipy.interpolate import LinearNDInterpolator
from scipy import interp
from numpy import maximum as npmax 
from numpy import minimum as npmin
import matplotlib.pyplot as plt


class American_Option(object):
	"""
	The American Option.
	"""

	def __init__(self, gam=.04, K=20., rho=.6, b=-.5, sigma=.5,
	             z_min=-10., z_max=10., grid_size=200,
	             mc_size=1000):

	    self.gam, self.K = gam, K
	    self.rho, self.b, self.sigma = rho, b, sigma
	    self.z_min, self.z_max = z_min, z_max
	    self.grid_size = grid_size
	    self.grid_points = np.linspace(self.z_min, self.z_max,
	    	                           self.grid_size)
	    self.mc_size = mc_size
	    self.draws = np.random.randn(self.mc_size)


	def r(self, x):
		"""
		The exit payoff function.
		"""
		gam, K = self.gam, self.K
		return npmax(np.exp(x) - K, 0.)


	def cval_operator(self, psi):
		"""
		The continuation value operator.
		"""
		rho, b, sigma = self.rho, self.b, self.sigma
		z_min, z_max = self.z_min, self.z_max

		# interpolate to get an approximate fixed point function
		psi_interp = lambda x: interp(x, self.grid_points, psi)
		N = len(psi)
		psi_new = np.empty(N)

		for i, z in enumerate(self.grid_points):
			# sample z' from f(z'|z) = N(rho z + b, sigma^2)
			z_prime = rho * z + b + sigma * self.draws
			# samples outside the truncated state space is 
			# replaced by the nearest grids of the state space
			z_prime = npmax(npmin(z_prime, z_max), z_min)

			integrand_1 = self.r(z_prime) # r(z') samples
			integrand_2 = psi_interp(z_prime) # psi(z') samples

			# approximate integral via Monte Carlo integration
			integral = np.mean(npmax(integrand_1, integrand_2))

			psi_new[i] = np.exp(-self.gam) * integral # Q psi(z)

		return psi_new


	def compute_fixed_point(self, Q, psi, max_iter=500,
							error_tol=1e-3, verbose=1):
		"""
		Compute the fixed point of the continuation value
		operator.
		"""
		error = error_tol + 1.0
		iteration = 0

		while error > error_tol and iteration < max_iter:

			psi_new = Q(psi)
			error = np.max(abs(psi_new - psi)) 
			psi = psi_new
			iteration += 1

			if verbose:
				print ("Computing iteration ", iteration, " with error ", error)

		return psi



# ======================= rho > 0 ========================== #

ao = American_Option(gam=.04, K=20., rho=.65, b=-.2, sigma=1.)
psi_init = np.ones(ao.grid_size)

# compute the fixed point (continuation value function)
psi_star = ao.compute_fixed_point(ao.cval_operator, psi_init,
	                              verbose=0)

"""
# an alternative way to compute the fixed point
	# (iterate a fixed number of times)
	# uncomment the next three lines to implement
psi_star = np.ones(ao.grid_size)
for i in range(100):
	psi_star = ao.cval_operator(psi_star)
	print ("Iteration", i+1, "completed ...")
"""

# the exit payoff function
r = ao.r(ao.grid_points)
# the value function 
v_star = npmax(ao.r(ao.grid_points), psi_star)


fig, ax = plt.subplots(figsize=(9,7))
ax.plot(ao.grid_points, psi_star, '--', color='red',
	    linewidth=4, label='CVF')
ax.plot(ao.grid_points, v_star, color='blue',
	    linewidth=3, label='VF')
#ax.plot(ao.grid_points, r, label='exit payoff',
#	    color='green')

ax.tick_params(labelsize=22)

ax.set_xlim(-2,8)
ax.set_ylim(-20, 100)
ax.set_xlabel('$z$', fontsize=28, labelpad=3)

ax.set_title('$\\rho$ = {}'.format(ao.rho), fontsize=24)

ax.legend(loc='upper left',fontsize=24)


# ======================= rho < 0 ========================== #

ao = American_Option(gam=.04, K=20., rho=-.65, b=-.2, sigma=1.)
psi_init = np.ones(ao.grid_size)

# compute the fixed point (continuation value function)
psi_star = ao.compute_fixed_point(ao.cval_operator, psi_init,
	                              verbose=0)

# the exit payoff function
r = ao.r(ao.grid_points)
# the value function 
v_star = npmax(ao.r(ao.grid_points), psi_star)


fig, ax = plt.subplots(figsize=(9,7))
ax.plot(ao.grid_points, psi_star, '--', color='red',
	    linewidth=4, label='CVF')
ax.plot(ao.grid_points, v_star, color='blue',
	    linewidth=3, label='VF')
#ax.plot(ao.grid_points, r, label='exit payoff',
#	    color='green')

ax.tick_params(labelsize=22)

ax.set_xlim(-8,10)
ax.set_ylim(-20, 100)
ax.set_xlabel('$z$', fontsize=28, labelpad=3)

ax.set_title('$\\rho$ = {}'.format(ao.rho), fontsize=24)

#ax.legend(loc='upper left',fontsize=24)

plt.show()