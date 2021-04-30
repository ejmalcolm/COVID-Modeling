import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
import math

import lhsmdu
import csv

def ODEmodel(vals, t, max_rate_dist, distance_accel, max_rate_undist, undistance_accel):

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
    return max(max_rate_dist*( 1 - math.exp( -distance_accel*x ) ), 0)

  def become_undistanced(x):
    #return max(max_rate_undist * math.exp(-undistance_accel*x), 0)
    return max(max_rate_undist*(1- math.exp( -undistance_accel*t ) ), 0)

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

def SDmodel(max_rate_dist, distance_accel, max_rate_undist, undistance_accel):
    return odeint(ODEmodel, y0, t, (max_rate_dist, distance_accel, max_rate_undist, undistance_accel))

def gen_time(caseinc, days_after, starting_point=False):
    # find starting and ending points
    if starting_point:
      first_index = starting_point
      last_index = first_index+days_after # first day + however many days after we want
      incidence_data = caseinc[first_index:last_index]
    else:
      first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
      last_index = first_index+days_after # first day + however many days after we want
      incidence_data = caseinc[first_index-14:last_index] # 14 days before first case
    
    # smooth data
    try:
      incidence_data = savgol_filter(incidence_data, 11, 2)
    except:
      print('Error in smoothing data-- proceeding with unsmoothed data.')
    
    # define t
    global t
    t = np.linspace(0,len(incidence_data),num=len(incidence_data))
    
    return incidence_data, t

def SD_curve_fit(initial_guesses):
    resids = lambda params, data: SDmodel(params[0], params[1], params[2], params[3])[:,4] + SDmodel(params[0], params[1], params[2], params[3])[:,11] - data
    op = least_squares( resids, initial_guesses, args=(incidence_data,), bounds=([0, np.inf]))
    return op

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, max_rate_dist, distance_accel, max_rate_undist, undistance_accel):
    y = SDmodel(max_rate_dist, distance_accel, max_rate_undist, undistance_accel)
    f5 = plt.figure(5)
    f5.suptitle(f'max_rate_dist={round(max_rate_dist,3)}, distance_accel={round(distance_accel,3)}, max_rate_undist={round(max_rate_undist,3)}, undistance_accel={round(undistance_accel,3)}')
    
    plt.plot(t, incidence_data, label='Actual Symptomatic') #plot the actual symptomatic infections
    
    
    plt.plot(t, y[:,4], label='Predicted Symptomatic Undistanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,11], label='Predicted Symptomatic Distanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,4]+y[:,11], label='Total Symptomatic Cases')

def define_dataset(county, days_after, starting_point=False, fulldata=None):
    global POPULATIONS
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
    print(f'Day {max(1, starting_point)} to {(0 or starting_point)+days_after}.')

    pre_incidence = [float(x) for x in conf_data[index][3:]]
    
    conf_incidence, t = gen_time(pre_incidence, days_after, starting_point) #returns conf incidence, t
    
    # * define y0
    if starting_point:
      starting_infections = conf_incidence[0]
      y0 = [POPULATIONS[county][1]-starting_infections, 1, 0, 0, starting_infections/2, 0, 0, # undistanced population
            0, 0, 0, 0, starting_infections/2, 0, 0] # distanced population
    else:
      # it's undistance_accelch easier if you're starting from the beginning of the outbreak :)
      y0 = [POPULATIONS[county][1],1,0,0,0,0,0, # undistanced population
            0, 0, 0, 0, 0, 0, 0] # distanced population
    
    return conf_incidence, t, y0

# looks at period day averages and finds relative peaks
def detect_peaks(incidence, display=False):
  peaks = find_peaks(incidence, width=50, rel_height=1, threshold=0, prominence=10)[0]
  if display:
    plt.vlines(peaks, 0, max(incidence), color='red', linestyles='dashed')
  return peaks

def LHS_for_guesses(sample_count):

  # we use latin hypercube sampling to obtain initial guesses for curve fitting
  lhsmdu.setRandomSeed(None)
  max_rate_dist_max_rate_undist_sample = lhsmdu.sample(2,sample_count)
  distance_accel_undistance_accel_sample = lhsmdu.sample(2,sample_count)
  max_rate_dist_max_rate_undist_sample = max_rate_dist_max_rate_undist_sample.tolist()
  distance_accel_undistance_accel_sample = distance_accel_undistance_accel_sample.tolist()

  # we then adjust the variables to the correct ranges
  adjusted_sample = []
  # for AT, we adjust to between 1 and 1/30
  for var_dist in max_rate_dist_max_rate_undist_sample:
    var_dist = [(1 + var*(1/30-1)) for var in var_dist]
    adjusted_sample.append(var_dist)
  # for EM, we adjust to between 10 and 0.0001
  # however, we actually use theta where EM = 10^theta, so theta is between 1 and -3
  # prevents overweighting towards the top end of the spectrum
  for var_dist in distance_accel_undistance_accel_sample:
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
      # we have to rearrange test_guesses so that it goes max_rate_dist, distance_accel, max_rate_undist, undistance_accel
      max_rate_undist = test_guesses[1]
      test_guesses[1] = test_guesses[2]
      test_guesses[2] = max_rate_undist
      # we then adjust distance_accel and undistance_accel to be 10^[their value], because it's currently theta
      test_guesses[1] = 10**(test_guesses[1])
      test_guesses[3] = 10**(test_guesses[3])

      cost = SD_curve_fit(test_guesses).cost
      print(f"LHS attempting guess: {test_guesses} cost: {cost}")

      if cost < lowest_cost[0]:
        lowest_cost = [cost, test_guesses]
    except OverflowError as e:
      print("Overflow while running LHS for initial guesses: ", e)
    except ValueError as e:
      print("Residual error while running LHS for initial guesses: ", e)
  
  print(f"LHS suggests that {lowest_cost[1]} is the best set of guesses")

  return lowest_cost[1]

def full_dynamics():
  global incidence_data
  global y0
  incidence_data, t, y0 = define_dataset('Philadelphia', days_after=10000)
  initial_guesses = LHS_for_guesses(1)
  max_rate_dist, distance_accel, max_rate_undist, undistance_accel = SD_curve_fit(initial_guesses).x

  if True:
    f5 = plt.figure(5)
    f5.suptitle('Full Overview of Philadelphia Simulation Dynamics')
    normalized_data = [x/max(incidence_data) for x in incidence_data]
    plt.plot(t, normalized_data, ':', label='Actual Symptomatic') #plot the actual symptomatic infections


    y = SDmodel(max_rate_dist, distance_accel, max_rate_undist, undistance_accel)
    categories = ['S_u','E_u','A_u','P_u','I_u','R_u','D_u', 'S_d','E_d','A_d','P_d','I_d','R_d','D_d']
    for i, category in enumerate(categories):
      data = [x/max(y[:,i]) for x in y[:,i]]
      style = '-'
      if i > 6:
        style = '--'
      plt.plot(t, data, style, label=category) #plot the model's symptomatic infections
  
  plt.legend()
  plt.show()


if __name__ == "__main__":

  # ! parameter used to control how many guesses are ran for LHS simulations
  # ! computation time increases rapidly as depth increases
  depth = 5

  # * clear results.csv for fresh data
  with open('results.csv', 'w') as f:
    data_row = ['City-County', 'Population', 'Peak #', 'Start Date', 'End Date', 'max_rate_dist', 'distance_accel', 'max_rate_undist', 'undistance_accel']
    writer = csv.writer(f)
    writer.writerow(data_row)

  for label in ['Bernalillo', 'District of Columbia', 'Denver', 'Fayette', 'Hinds', 'Honolulu', 'Juneau', 'Montgomery', 'Muscogee', 'Orleans', 'Philadelphia', 'Richmond City', 'San Francisco', 'Wake']:
    
    # * define the dataset
    # note that the y0 produced here is just 1 infection, 0 everything else (except susceptibles)
    incidence_data, t, y0 = define_dataset(label, days_after=1000)

    # * get the date of the midpoint of each peak
    peaks = detect_peaks(incidence_data, display=False)

    # * run simulation for each individual peak
    for i, peak in enumerate(peaks):
      

      # # to generate the starting values (y0) for each peak, we run the simulation up to the START of the peak
      # incidence_data, t, y0 = define_dataset(label, days_after=(peak-50))
      # initial_guesses = LHS_for_guesses(depth)
      # max_rate_dist, distance_accel, max_rate_undist, undistance_accel = SD_curve_fit(initial_guesses).x
      # y0 = SDmodel(max_rate_dist, distance_accel, max_rate_undist, undistance_accel)[-1]

      # we then define the dataset for the actual peak simulation
      incidence_data, t, y0 = define_dataset(label, starting_point=max(peak-50, 0), days_after=100)

      # obtain parameter values
      initial_guesses = LHS_for_guesses(depth)
      max_rate_dist, distance_accel, max_rate_undist, undistance_accel = SD_curve_fit(initial_guesses).x

      # write parameter values to file
      date_labels = np.genfromtxt('city_county_case_time_series_incidence.csv', dtype=str,delimiter=",")[0][3:]

      start_of_peak = date_labels[peak-50].replace('"', '')
      end_of_peak = date_labels[peak+50].replace('"', '')

      data_row = [label, POPULATIONS[label][1], i+1, start_of_peak, end_of_peak, round(max_rate_dist, 6), round(distance_accel, 6), round(max_rate_undist, 6), round(undistance_accel, 6)]
      with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

      # * graph and save graphs in the IndividualSims folder
      plot_for_vals(incidence_data, max_rate_dist, distance_accel, max_rate_undist, undistance_accel)
      plt.legend()
      plt.savefig(f'IndividualSims/{label}/Peak{str(i+1)}')
      plt.clf()