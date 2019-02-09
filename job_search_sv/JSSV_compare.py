import numpy as np 
from scipy import interp
from scipy.interpolate import LinearNDInterpolator
from numpy import maximum as npmax
from numpy import minimum as npmin
import matplotlib.pyplot as plt
import time



class Job_Search_SV(object):
	"""
	A class to store a given parameterization of the generalized
	job search model.

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
	thet_min : scalar(float), optional(default=1e-3)
	           The minimum of the grid for thet
	thet_max : scalar(float), optional(default=10.)
	           The maximum of the grid for thet
	thet_size : scalar(int), optional(default=200)
			    The number of grid points over thet
	mc_size : scalar(int), optional(default=10000)
		      The number of Monte Carlo samples
	"""

	def __init__(self, beta=.95, c0_tilde=.6, sig=2.5,
		         rho=.75, gam_u=1e-4, gam_xi=5e-4,
		         mu_eta=0., gam_eta=1e-6,
		         thet_min=1e-3, thet_max=10., thet_size=100,
		         w_min=1e-3, w_max=10., w_size=100,
		         mc_size=1000):
	    self.beta, self.c0_tilde, self.sig = beta, c0_tilde, sig 
	    self.rho, self.gam_u, self.gam_xi = rho, gam_u, gam_xi
	    self.mu_eta, self.gam_eta = mu_eta, gam_eta
	    self.thet_min, self.thet_max = thet_min, thet_max
	    self.thet_size = thet_size
	    self.w_min, self.w_max = w_min, w_max
	    self.w_size = w_size

	    # ============= make grids for CVI =============== #
	    self.grid_points = np.linspace(self.thet_min, 
	    	                           self.thet_max,
	    	                           self.thet_size)

	    # ============= make grids for VFI =============== #
	    self.thet_grids = np.linspace(self.thet_min, 
	    	                          self.thet_max,
	    	                          self.thet_size)
	    self.w_grids = np.linspace(self.w_min, self.w_max,
	    	                       self.w_size)
	    self.w_mesh, self.thet_mesh = np.meshgrid(self.w_grids,
	    	                                      self.thet_grids)
	    self.grid_points_vfi = \
	        np.column_stack((self.w_mesh.ravel(1), 
	    	                 self.thet_mesh.ravel(1)))
	    # MC draws
	    self.mc_size = mc_size
	    self.draws = np.random.randn(self.mc_size)


	def utility_func(self, x):
		"""
		The utility function (CRRA).
		"""
		sig = self.sig
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		return uw


	def Bellman_operator(self, v):
		"""
		The Bellman operator.
		"""
		beta, c0_tilde, rho = self.beta, self.c0_tilde, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		draws = self.draws
		grid_points_vfi = self.grid_points_vfi
		utility_func = self.utility_func
		# interpolate to obtain a function
		v_interp = LinearNDInterpolator(grid_points_vfi, v)

		def v_f(x,y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmax(npmin(x, self.w_max), self.w_min)
			y = npmax(npmin(y, self.thet_max), self.thet_min)
			return v_interp(x,y)

		N = len(v)
		new_v = np.empty(N)

		for i in range(N):
			w, thet = grid_points_vfi[i, :]
			# MC samples: eta'
			eta_draws = np.exp(mu_eta+ np.sqrt(gam_eta)*draws)
			# MC samples: thet'
			thet_draws = np.exp(rho * np.log(thet) + \
				         np.sqrt(gam_u) * draws)
			# MC samples: xi'
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			# MC samples: w'
			w_draws = eta_draws + thet_draws * xi_draws
			# MC samples: v(w',thet')
			integrand = v_f(w_draws, thet_draws)

			c0 = utility_func(c0_tilde) # c0
			# the continuation value
			cvf = c0 + beta * np.mean(integrand)
			# the exit payoff
			exit_payoff = utility_func(w) / (1 - beta)

			new_v[i] = max(exit_payoff, cvf)

		return new_v


	def compute_cvf(self, v):
		"""
		Compute the continuation value based on the 
		value function.
		"""
		beta, c0_tilde, rho = self.beta, self.c0_tilde, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi
		draws = self.draws
		grid_points_vfi = self.grid_points_vfi
		grid_points = self.grid_points
		utility_func = self.utility_func
		# interpolate to obtain a function
		v_interp = LinearNDInterpolator(grid_points_vfi, v)

		def v_f(x,y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmax(npmin(x, self.w_max), self.w_min)
			y = npmax(npmin(y, self.thet_max), self.thet_min)
			return v_interp(x,y)

		cvf = np.empty(len(grid_points))

		for i, thet in enumerate(grid_points):
			# MC samples: eta'
			eta_draws = np.exp(mu_eta+ np.sqrt(gam_eta)*draws)
			# MC samples: thet'
			thet_draws = np.exp(rho * np.log(thet) + \
				         np.sqrt(gam_u) * draws)
			# MC samples: xi'
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			# MC samples: w'
			w_draws = eta_draws + thet_draws * xi_draws
			# MC samples: v(w',thet')
			integrand = v_f(w_draws, thet_draws)

			c0 = utility_func(c0_tilde) # c0
			# the continuation value
			cvf[i] = c0 + beta * np.mean(integrand)

		return cvf
						

	def cvals_operator(self, psi):
		"""
		The continuation value operator.
		"""
		beta, c0_tilde, rho = self.beta, self.c0_tilde, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		draws = self.draws
		grid_points = self.grid_points
		utility_func = self.utility_func
		# interpolate to obtain a function
		psi_interp = lambda x: interp(x, grid_points, psi)
        
		def psi_f(x):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmax(npmin(x, self.thet_max), self.thet_min)
			return psi_interp(x)

		N = len(psi)
		new_psi = np.empty(N)

		for i, thet in enumerate(grid_points):
			# MC samples: eta'
			eta_draws = np.exp(mu_eta+ np.sqrt(gam_eta)*draws)
			# MC samples: thet'
			thet_draws = np.exp(rho * np.log(thet) + \
				         np.sqrt(gam_u) * draws)
			# MC samples: xi'
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			# MC samples: u(w')
			utils = utility_func(eta_draws+ thet_draws* xi_draws)

			# MC samples: r(w')
			integrand_1 = utils / (1. - beta)
			# MC samples: psi(thet')
			integrand_2 = psi_f(thet_draws)
			# MC samples: max{r(w'), psi(thet')}
			integrand = npmax(integrand_1, integrand_2)

			c0 = utility_func(c0_tilde)
			new_psi[i] = c0 + beta * np.mean(integrand)

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
			w_bar = (y*(1.-sig)*(1.-beta))**(1./ (1. - sig))

		return w_bar



# the setup of the grid sizes in table 1.
thet_list = [200, 200, 200, 300, 300, 300, 400, 400, 400]
w_list = [200, 300, 400, 200, 300, 400, 200, 300, 400]


# ================= Computation Time: CVI =================== #

print ("")
print ("CVI in progress ... ")

time_cvi = np.empty(len(thet_list))

for i, theta_size in enumerate(thet_list):
	start_cvi = time.time() # start the clock

	# compute the fixed point (continuation value)
	jssv = Job_Search_SV(thet_size=theta_size, rho=.75, sig=1.)
	N = len(jssv.grid_points)
	psi_0 = np.ones(N) # initial guess of the solution
	psi_star = jssv.compute_fixed_point(jssv.cvals_operator, 
		                                psi_0, verbose=0)

	# time taken for iteration i
	time_cvi[i] = time.time() - start_cvi  



# ================= Computation Time: VFI =================== # 

print ("")
print ("VFI in progress ... ")

loops = len(w_list)
time_vfi = np.empty(loops)

for i in range(loops):
	start_vfi = time.time() # start the clock

	jssv_vfi = Job_Search_SV(thet_size=thet_list[i], 
		                     w_size=w_list[i], rho=.75, sig=1.)
	# compute the fixed point (value function)
	N = len(jssv_vfi.grid_points_vfi)
	v0 = np.ones(N)
	v_star = jssv_vfi.compute_fixed_point(jssv_vfi.Bellman_operator, 
		                                  v0, verbose=0)
    # calculate the time taken for iteration i
	time_vfi[i] = time.time() - start_vfi



# ============= Computation Time : CVI v.s. VFI =============== #

print ("")
print ("----------------------------------------------")
print ("")
print ("Grid size: ")
print ("")
print ("theta: ", thet_list)
print ("w: ", w_list)
print ("")
print ("Computation time: ")
print ("")
print ("CVI : ", time_cvi)
print ("VFI : ", time_vfi)
print ("")
print ("----------------------------------------------")