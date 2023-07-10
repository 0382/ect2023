import sys
sys.path.append('./../lib')

import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt
import pyDOE as pyDOE
import granada_phases as granada
import auxiliary as aux
import emcee
import corner
import prettyplease
import lec_values as lec_values

# initialize an object for computing T-matrices, phase shifts, 
nn = nn_studio.nn_studio(jmin=0,jmax=0,tzmin=0,tzmax=0)

# initialize an object for the chiral interaction (isospin symmetric LO, NLO in WPC available)
potential = chiral_potential.two_nucleon_potential('NLO',Lambda=500.0)

# give the potential to the nn-analyzer
nn.V = potential

# give initial LECS to the potential (via the nn-analyzer)
lecs = dict(lec_values.nlo_lecs)
nn.lecs = lecs

# define the lab neutron-proton kinetic energies that you want to analyze
nn.Tlabs = granada.Tlabs

## you can inspect results for the channels <ll s j || l s j> in your basis
idxs,selected_channel = nn.lookup_channel_idx(l=0,ll=0,s=0,j=0)

ndim = 2
npts = 3
training_points = pyDOE.lhs(ndim,npts)

lim_lo = [None]*ndim
lim_hi = [None]*ndim

lim_lo[0] =  -0.2 ; lim_hi[0] = +0.00
lim_lo[1] = 1.0 ; lim_hi[1] = +2.0

for j in range(npts):
    for i in range(ndim):
        training_points[j, i] = aux.scale_point(training_points[j, i], lim_lo[i], lim_hi[i])

emulators = []
for Tlab in nn.Tlabs:
    emulators.append(nn.model_order_reduction(channel=selected_channel,Tlab=Tlab,directions=['C_1S0','D_1S0'],training_points=training_points,verbose=True))
    # for LO, with one fewer LEC, use the following call
    #emulators.append(nn.model_order_reduction(channel=selected_channel,Tlab=Tlab,directions=['C_1S0'],training_points=training_points,verbose=True))

ydata = granada.delta_1S0
yerr_granada = granada.delta_1S0_errors
yerr2 = yerr_granada**2

def model(pars):
    # x: (n,) array. Even if n=1
   
    lecs["C_1S0"] = pars[0]
    lecs["D_1S0"] = pars[1] 
        
    delta_model = [] 
    for emulator in emulators:
        T,Tlab,ko,mu = emulator([1.0,pars[0],pars[1]])
        # this is what the 'LO' emulator call looks like
        #T,Tlab,ko,mu = emulator([1.0,pars[0]])
        delta_model.append(nn.compute_phase_shifts(ko,mu,T))

    return np.array(delta_model)

def log_likelihood(par, ydata):
    yth = model(par)
    inv_s2 = 1/yerr2
    return -0.5*(np.sum((ydata-yth)**2*inv_s2 - np.log(inv_s2)))

# a uniform prior with compact support
def log_prior(par):

    #C0  = par
    ## starting with a simple uniform prior for the counterterms
    #if -1.0 < C0 < +1.0:
    #    return 0.0

    C0, C1 = par
    # starting with a simple uniform prior for the counterterms
    if -1.0 < C0 < +1.0 and -1.0 < C1 < +2.0:
        return 0.0
    return -np.inf

def log_posterior(par, ydata):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(par, ydata)
    return ll + lp

# instructions for emcee

#LO is ndim=1
#ndim, nwalkers = 1, 50
ndim, nwalkers = 2, 50

# this is the max number of samples. emcee might stop before pending the
# convergence criteria based on the integrated autocorrelation time.
max_nsamples = 10000

# we always need a starting point. 
guess_mode = np.array([-0.1,1])
#guess_mode = np.array([-0.1])

start_pos = guess_mode + np.array([1e-3]*ndim)*np.random.randn(nwalkers, ndim)

#see https://emcee.readthedocs.io/en/stable/tutorials/monitor/
#for more on monitoring and saving

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[ydata])

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_nsamples)

# This will be useful to testing convergence
old_tau = np.inf

threshold = 50 # emcee recommends 50
# Now we'll sample for up to max_nsamples steps
for sample in sampler.sample(start_pos, iterations=max_nsamples, progress=False):
    # Only check convergence every threshold steps
    if sampler.iteration % threshold:
        continue

    # Compute the autocorrelation time (per direction) so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    print(f'N = {sampler.iteration}; tau = {tau} ; tau*threshold = {tau*threshold} ; tau*threshold/N = {tau*threshold/sampler.iteration} (must be <1) ; dtau = {(np.abs(old_tau - tau) / tau) } (must be <0.01)')
    # Check convergence in all directions and that the tau is stable to within 1%
    converged = np.all(tau * threshold < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

#Now let’s take a look at how the autocorrelation time estimate (averaged across dimensions) changed over the course of this run. In this plot, the 
#tau  estimate is plotted as a function of chain length and, for comparison, the N>threshold\tau threshold is plotted as a dashed line.

plt.figure(1)
n = threshold * np.arange(1, index + 1)
y = autocorr[:index]
plt.plot(n, n / threshold, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");
plt.savefig('tau.pdf')

tau = sampler.get_autocorr_time()
burnin = int(10 * np.max(tau))
print(f'burnin = {burnin}')
samples = sampler.get_chain(discard=burnin,flat=True)
print(f'retained samples = {len(samples)}')
print(f'chain shape = {samples.shape}')

np.savetxt('NLO_chain_1s0.txt',samples)

labels = [r'C1S0',r'D1S0']
#labels = [r'C1S0']
plt.figure(2)
fig = prettyplease.corner(samples, bins=50, labels=labels, plot_estimates=True, colors='green',
                          n_uncertainty_digits=4, title_loc='center', figsize=(7,7))
plt.savefig('uncoupled_posterior_nlo.pdf', bbox_inches='tight')
