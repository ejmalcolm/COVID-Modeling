import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, savgol_filter
import math

import lhsmdu


def ODEmodel(vals, t, alpha, epsilon, tau, mu):

  ###other parameters###
  sig = 1/5 #exposed -> presympt/astmpy
  delt = 1/2 #pre-sympt -> sympt
  gam_I = 1/7 #sympt -> recovered
  gam_A = 1/7 #asympt -> recovered
  nu = gam_I/99 #sympt -> deceased

  k = 0.5 # percentage asymptomatic

  ###unpack###
  # two groups here: U for undistanced and D for distanced
  S_u = vals[0]
  E_u = vals[1]
  A_u = vals[2]
  P_u = vals[3]
  I_u = vals[4]
  R_u = vals[5]
  D_u = vals[6]
  
  S_d = vals[7]
  E_d = vals[8]
  A_d = vals[9]
  P_d = vals[10]
  I_d = vals[11]
  R_d = vals[12]
  D_d = vals[13]

  ###defining lambda###
  b_I = 0.6

  N = S_u+E_u+P_u+A_u+I_u+R_u+D_u + S_d+E_d+P_d+A_d+I_d+R_d+D_d

  # there's a 2x2 square
  # ----U---|---D---|
  # U   1   |   .5  |
  # --------|-------|
  # D   .5  | .125  |
  # --------|-------|

  c_uu = 1
  c_ud = 0.55
  c_dd = 0.0125

  lamb_uu = (c_uu*I_u + c_uu*P_u + c_uu*A_u) * (b_I/N)
  lamb_dd = (c_dd*I_d + c_dd*P_d + c_dd*A_d) * (b_I/N)

  lamb_u = lamb_uu + (c_ud*I_d + c_ud*P_d + c_ud*A_d)*(b_I/N)
  lamb_d = lamb_dd + (c_ud*I_u + c_ud*P_u + c_ud*A_u)*(b_I/N)

  ###stuff to return###

  # transition functions
  def become_distanced(x):
    return max(alpha*( 1 - math.exp( -epsilon*x ) ), 0)

  def become_undistanced(x):
    #return max(tau * math.exp(-mu*x), 0)
    return max(tau*(1- math.exp( -mu*t ) ), 0)

  # undistanced group #
  death_rate = nu*(I_u+I_d)

  dS_udt = (-lamb_u * S_u) - become_distanced(death_rate)*S_u + become_undistanced(death_rate)*S_d
  dE_udt = (lamb_u * S_u) - (sig * E_u)
  dA_udt = (k * sig * E_u) - (gam_A * A_u)
  dP_udt = ( (1-k) * sig * E_u) - (delt * P_u)
  dI_udt = (delt * P_u) - ( (nu + gam_I) * I_u)
  dR_udt = (gam_A * A_u) + (gam_I * I_u)
  dD_udt = nu * I_u

  # distanced group #
  dS_ddt = (-lamb_d * S_d) + become_distanced(death_rate)*S_u - become_undistanced(death_rate)*S_d
  dE_ddt = (lamb_d * S_d) - (sig * E_d)
  dA_ddt = (k * sig * E_d) - (gam_A * A_d)
  dP_ddt = ( (1-k) * sig * E_d) - (delt * P_d)
  dI_ddt = (delt * P_d) - ( (nu + gam_I) * I_d)
  dR_ddt = (gam_A * A_d) + (gam_I * I_d)
  dD_ddt = nu * I_d

  return [dS_udt,dE_udt,dA_udt,dP_udt,dI_udt,dR_udt,dD_udt,
          dS_ddt,dE_ddt,dA_ddt,dP_ddt,dI_ddt,dR_ddt,dD_ddt]

def SDmodel(alpha, epsilon, tau, mu):
    return odeint(ODEmodel, y0, t, (alpha, epsilon, tau, mu))

def gen_time(caseinc, days_after,y0, starting_point=False):
    if starting_point:
      first_index = starting_point
      last_index = first_index+days_after # first day + however many days after we want
      incidence_data = caseinc[first_index:last_index] # 14 days before first case
    else:
      first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
      last_index = first_index+days_after # first day + however many days after we want
      incidence_data = caseinc[first_index-14:last_index] # 14 days before first case
    
    try:
      incidence_data = savgol_filter(incidence_data, 11, 2)
    except:
      print('Error in smoothing data-- proceeding with unsmoothed data.')
    global t
    t = np.linspace(0,len(incidence_data),num=len(incidence_data))
    return incidence_data, t, y0

def SD_curve_fit(initial_guesses):
    resids = lambda params, data: SDmodel(params[0], params[1], params[2], params[3])[:,4] + SDmodel(params[0], params[1], params[2], params[3])[:,11] - data
    op = least_squares(resids, initial_guesses, args=(incidence_data,) )
    return op

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, alpha, epsilon, tau, mu):
    y = SDmodel(alpha, epsilon, tau, mu)
    f5 = plt.figure(5)
    f5.suptitle(f'alpha={round(alpha,3)}, epsilon={round(epsilon,3)}, tau={round(tau,3)}, mu={round(mu,3)}')
    
    plt.plot(t, y[:,4], label='Predicted Symptomatic Undistanced') #plot the model's symptomatic infections
    
    plt.plot(t, incidence_data, label='Actual Symptomatic') #plot the actual symptomatic infections
    
    plt.plot(t, y[:,11], label='Predicted Symptomatic Distanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,4]+y[:,11], label='Total Symptomatic Cases')

def define_dataset(county, days_after, starting_point=False):
    POPULATIONS = {'Bernalillo' : [1, 679121],
        'District of Columbia' : [2, 704749],
        'Denver' : [3, 600158],
        'Fayette' : [4, 323152],
        'Hinds' : [5, 231840],
        'Honolulu' : [6, 974563],
        'Juneau' : [7, 31275],
        'Los Alamos' : [8, 12019],
        'Montgomery' : [9, 98985],
        'Muscogee' : [10, 195769],
        'Orleans' : [11, 343829],
        'Philadelphia' : [12, 1526000],
        'Richmond City' : [13, 230436],
        'San Francisco' : [14, 881549],
        'Wake' : [15, 1111761],
    }
    index = POPULATIONS[county][0]

    # plot already existing case data
    conf_data = np.genfromtxt('city_county_case_time_series_incidence.csv', dtype=str,delimiter=",") #this is the incidence
    print(conf_data[index][2]) #print the name of the county
    pre_incidence = [float(x) for x in conf_data[index][3:]]

    # define y0
    if starting_point:
      starting_infections = pre_incidence[starting_point]
      y0 = [POPULATIONS[county][1],1,0,0,starting_infections/2,0,0, # undistanced population
            0, 0, 0, 0, starting_infections/2, 0, 0] # distanced population
    else:
      y0 = [POPULATIONS[county][1],1,0,0,0,0,0, # undistanced population
            0, 0, 0, 0, 0, 0, 0] # distanced population
    
    return gen_time(pre_incidence, days_after, y0, starting_point) #returns conf incidence, t, y0

# looks at period day averages and finds relative peaks
def detect_peaks(incidence, display=False):
  peaks = find_peaks(incidence, width=50, rel_height=1, threshold=0, prominence=10)[0]
  if display:
    plt.vlines(peaks, 0, max(incidence), color='red', linestyles='dashed')
  return peaks


def LHS_for_guesses(sample_count):

  # we use latin hypercube sampling to obtain initial guesses for curve fitting
  lhsmdu.setRandomSeed(None)
  alpha_tau_sample = lhsmdu.sample(2,sample_count)
  epsilon_mu_sample = lhsmdu.sample(2,sample_count)
  alpha_tau_sample = alpha_tau_sample.tolist()
  epsilon_mu_sample = epsilon_mu_sample.tolist()

  # we then adjust the variables to the correct ranges
  adjusted_sample = []
  # for AT, we adjust to between 1 and 1/30
  for var_dist in alpha_tau_sample:
    var_dist = [(1 + var*(1/30-1)) for var in var_dist]
    adjusted_sample.append(var_dist)
  # for EM, we adjust to between 10 and 0.0001
  # however, we actually use theta where EM = 10^theta, so theta is between 1 and -3
  # prevents overweighting towards the top end of the spectrum
  for var_dist in epsilon_mu_sample:
    var_dist = [(1 + var*(-5-1)) for var in var_dist]
    adjusted_sample.append(var_dist)

  # then, for each pair of 4 variables, we run it through the fit and determine cost
  # for every guess, we check to see if that's the lowest cost generated thus far
  # if it is, we store it, and at the end, that's our result
  lowest_cost = [10000000000000, [0,0,0,0]]

  for i in range(sample_count):
    test_guesses = []
    for var_dist in adjusted_sample:
      test_guesses.append(var_dist[i])
    try:
      # we have to rearrange test_guesses so that it goes alpha, epsilon, tau, mu
      tau = test_guesses[1]
      test_guesses[1] = test_guesses[2]
      test_guesses[2] = tau
      # we then adjust epsilon and mu to be 10^[their value], because it's currently theta
      test_guesses[1] = 10**(test_guesses[1])
      test_guesses[3] = 10**(test_guesses[3])

      cost = SD_curve_fit(test_guesses).cost
      print(f"{test_guesses} cost: {cost}")

      if cost < lowest_cost[0]:
        lowest_cost = [cost, test_guesses]
    except OverflowError as e:
      print("Overflow while running LHS for initial guesses: ", e)
    except ValueError as e:
      print("Residual error while running LHS for initial guesses: ", e)
  
  print(f"LHS suggests that {lowest_cost[1]} is the best set of guesses")

  return lowest_cost[1]

if __name__ == "__main__":
  for label in ['Bernalillo', 'District of Columbia', 'Denver', 'Fayette', 'Hinds', 'Honolulu', 'Juneau', 'Montgomery', 'Muscogee', 'Orleans', 'Philadelphia', 'Richmond City', 'San Francisco', 'Wake']:
    incidence_data, t, y0 = define_dataset(label, starting_point = 0, days_after=500)
    peaks = detect_peaks(incidence_data, display=True)
    # run simulation for each individual peak
    for i, peak in enumerate(peaks):
      incidence_data, t, y0 = define_dataset(label, starting_point=peak-50, days_after=100)
      initial_guesses = LHS_for_guesses(10)
      # graph values
      alpha, epsilon, tau, mu = SD_curve_fit(initial_guesses).x
      plot_for_vals(incidence_data, alpha, epsilon, tau, mu)
      # save graphs
      plt.savefig(f'IndividualSims/{label}/Peak{str(i+1)}')
      plt.clf()
    
    #initial_guesses = LHS_for_guesses(50)
    #initial_guesses = [0.6402235457506377, 0.002642127419530047, 0.12264339694248039, 1.030895939476692] # philadelphia
    #alpha, epsilon, tau, mu = SD_curve_fit(initial_guesses).x
    # plot_for_vals(incidence_data, alpha, epsilon, tau, mu)

    # plt.savefig('Peaks/'+label)
    # plt.clf()
    # plt.legend()
    # plt.show()