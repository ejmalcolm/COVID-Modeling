import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
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
  b_I = 0.386 #alpha

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

  lamb_uu = (c_uu*I_u) * (b_I/N)
  lamb_dd = (c_dd*I_d) * (b_I/N)

  lamb_u = lamb_uu + (c_ud*I_d)*(b_I/N)
  lamb_d = lamb_dd + (c_ud*I_u)*(b_I/N)

  ###stuff to return###

  # transition functions
  def become_distanced(x):
    return max(alpha*( 1 - math.exp( -epsilon*x ) ), 0)

  def become_undistanced(x):
    return max(tau*( 1 - math.exp( -mu*t ) ), 0)

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

def gen_time(caseinc, days_after,y0):
    first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
    last_index = first_index+days_after # first day + however many days after we want
    incidence_data = caseinc[first_index-14:last_index] # 14 days before first case
    global t
    t = np.linspace(0,len(incidence_data),num=len(incidence_data))
    return incidence_data, t, y0

def SD_curve_fit(initial_guesses):
    resids = lambda params, data: (SDmodel(params[0], params[1], params[2], params[3])[:,4] - data)
    op = least_squares(resids, initial_guesses, args=(incidence_data,) )
    return op

#general form to get the optimal B_I from curve fit
def get_optimal_bI():
    op = SD_curve_fit()
    return op.x

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, alpha, epsilon, tau, mu):
    y = SDmodel(alpha, epsilon, tau, mu)
    f5 = plt.figure(5)
    f5.suptitle(f'alpha={round(alpha,3)}, epsilon={round(epsilon,3)}')
    
    plt.plot(t, y[:,4], label='Predicted Symptomatic Undistanced') #plot the model's symptomatic infections
    
    plt.plot(t, incidence_data, label='Actual Symptomatic') #plot the actual symptomatic infections
    
    plt.plot(t, y[:,11], label='Predicted Symptomatic Distanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,4]+y[:,11], label='Total Symptomatic Cases')

def define_dataset(county, days_after):
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
        'San Fransisco' : [14, 881549],
        'Wake' : [15, 1111761],
    }
    index = POPULATIONS[county][0]
    # define y0
    y0 = [POPULATIONS[county][1],1,0,0,0,0,0, # undistanced population
          0, 0, 0, 0, 0, 0, 0] # distanced population
    # plot already existing case data
    conf_data = np.genfromtxt('city_county_case_time_series_incidence.csv', dtype=str,delimiter=",") #this is the incidence
    print(conf_data[index][2]) #print the name of the county
    pre_incidence = [float(x) for x in conf_data[index][3:]]
    return gen_time(pre_incidence,days_after,y0) #returns conf incidence, t, y0


if __name__ == "__main__":
  # you always need to globally define the dataset
  incidence_data, t, y0 = define_dataset('Richmond City', days_after=500)

  # we use latin hypercube sampling to obtain initial guesses for curve fitting
  lhsmdu.setRandomSeed(None)
  sample = lhsmdu.sample(4,2000)
  sample = sample.tolist()
  # by default, the sample is between 0-1, we adjust it to between -100 and 100
  adjusted_sample = []
  for var_dist in sample:
    var_dist = [var*1000-500  for var in var_dist]
    adjusted_sample.append(var_dist)

  # then, for each pair of 4 variables, we run it through the fit and determine cost
  # for every guess, we check to see if that's the lowest cost generated thus far
  # if it is, we store it, and at the end, that's our result
  lowest_cost = [10000000000000, [0,0,0,0]]

  for i in range(2000):
    test_guesses = []
    for var_dist in adjusted_sample:
      test_guesses.append(var_dist[i])
    try:
      cost = SD_curve_fit(test_guesses).cost
      print(f"{test_guesses} cost: {cost}")

      if cost < lowest_cost[0]:
        lowest_cost = [cost, test_guesses]
    except OverflowError as e:
      print("Overflow while running LHS for initial guesses: ", e)
    except ValueError as e:
      print("Residual error while running LHS for initial guesses: ", e)
  
  print(f"LHS suggests that {lowest_cost[1]} is the best set of guesses")

  initial_guesses = lowest_cost[1]
  alpha, epsilon, tau, mu = SD_curve_fit(initial_guesses).x

  plot_for_vals(incidence_data, alpha, epsilon, tau, mu)

  plt.legend()
  plt.show()