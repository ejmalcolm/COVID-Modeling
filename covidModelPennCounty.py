import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
import math
import os

import lhsmdu
import csv

def ODEmodel(vals, t, S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel):

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
  def S_become_distanced(x):
    return max(S_max_rate_dist*( 1 - math.exp( -S_distance_accel*x ) ), 0)

  def I_become_distanced(x):
    return max(I_max_rate_dist*( 1 - math.exp( -I_distance_accel*x ) ), 0)

  def become_undistanced(x):
    return max(max_rate_undist*( 1 - math.exp( -undistance_accel*t ) ), 0)

  # undistanced group #
  death_rate = nu*(I_u+I_d)/run_population

  dS_udt = (-lamb_u * S_u) - S_become_distanced(death_rate)*S_u + become_undistanced(death_rate)*S_d
  dE_udt = (lamb_u * S_u) - (sig * E_u)
  dA_udt = (k * sig * E_u) - (gam_A * A_u)
  dP_udt = ( (1-k) * sig * E_u) - (delt * P_u)
  dI_udt = (delt * P_u) - ( (nu + gam_I) * I_u) - I_become_distanced(death_rate)
  dR_udt = (gam_A * A_u) + (gam_I * I_u)
  dD_udt = nu * I_u

  # distanced group #
  dS_ddt = (-lamb_d * S_d) + S_become_distanced(death_rate)*S_u - become_undistanced(death_rate)*S_d
  dE_ddt = (lamb_d * S_d) - (sig * E_d)
  dA_ddt = (k * sig * E_d) - (gam_A * A_d)
  dP_ddt = ( (1-k) * sig * E_d) - (delt * P_d)
  dI_ddt = (delt * P_d) - ( (nu + gam_I) * I_d) + I_become_distanced(death_rate)
  dR_ddt = (gam_A * A_d) + (gam_I * I_d)
  dD_ddt = nu * I_d

  return [dS_udt,dE_udt,dA_udt,dP_udt,dI_udt,dR_udt,dD_udt,
          dS_ddt,dE_ddt,dA_ddt,dP_ddt,dI_ddt,dR_ddt,dD_ddt]

def SDmodel(S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel):
  return odeint(ODEmodel, y0, t, (S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel))

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
    resids = lambda params, data: SDmodel(params[0], params[1], params[2], params[3], params[4], params[5])[:,4] + SDmodel(params[0], params[1], params[2], params[3], params[4], params[5])[:,11] - data
    op = least_squares( resids, initial_guesses, args=(incidence_data,), bounds=([0, np.inf]))
    return op

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel):
    y = SDmodel(S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel)
    f5 = plt.figure(5)
    f5.suptitle(f'S_max_rate_dist={round(S_max_rate_dist,3)}, S_distance_accel={round(S_distance_accel,3)}, max_rate_undist={round(max_rate_undist,3)}, undistance_accel={round(undistance_accel,3)}')
    
    plt.plot(t, incidence_data, label='Actual Symptomatic') #plot the actual symptomatic infections
    
    
    plt.plot(t, y[:,4], label='Predicted Symptomatic Undistanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,11], label='Predicted Symptomatic Distanced') #plot the model's symptomatic infections
    plt.plot(t, y[:,4]+y[:,11], label='Total Symptomatic Cases')

def define_dataset(county, days_after, starting_point=False):
    global POPULATIONS
    POPULATIONS = {
      'Adams' : [1, 102903],
      'Allegheny' : [2, 1213570],
      'Armstrong' : [3, 63501],
      'Beaver' : [4, 162623],
      'Bedford' : [5, 47476],
      'Berks' : [6, 422434],
      'Blair' : [7, 120481],
      'Bradford' : [8, 59381],
      'Bucks' : [9, 629186],
      'Butler' : [10, 188283],
      'Cambria' : [11, 127678],
      'Cameron' : [12, 4339],
      'Carbon' : [13, 64196],
      'Centre' : [14, 161953],
      'Chester' : [15, 530795],
      'Clarion' : [16, 37970],
      'Clearfield' : [17, 78621],
      'Clinton' : [18, 38504],
      'Columbia' : [19, 64462],
      'Crawford' : [20, 83667],
      'Cumberland' : [21, 257848],
      'Dauphin' : [22, 257848],
      'Delaware' : [23, 569779],
      'Elk' : [24, 29510],
      'Erie' : [25, 266096],
      'Fayette' : [26, 127176],
      'Forest' : [27, 7173],
      'Franklin' : [28, 155923],
      'Fulton' : [29, 14540],
      'Greene' : [30, 35377],
      'Huntingdon' : [31, 44642],
      'Indiana' : [32, 83337],
      'Jefferson' : [33, 43047],
      'Juniata' : [34, 24853],
      'Lackawanna' : [35, 208484],
      'Lancaster' : [36, 549234],
      'Lawrence' : [37, 84280],
      'Lebanon' : [38, 142701],
      'Lehigh' : [39, 371236],
      'Luzerne' : [40, 316533],
      'Lycoming' : [41, 112165],
      'McKean' : [42, 39975],
      'Mercer' : [43, 107330],
      'Mifflin' : [44, 45992],
      'Monroe' : [45, 172225],
      'Montgomery' : [46, 838897],
      'Montour' : [47, 18210],
      'Northampton' : [48, 306727],
      'Northumberland' : [49, 90369],
      'Out of PA' : [50, 0],
      'Perry' : [51, 46508],
      'Philadelphia' : [52, 1585010],
      'Pike' : [53, 55867],
      'Potter' : [54, 16332],
      'Schuylkill' : [55, 140447],
      'Snyder' : [56, 40080],
      'Somerset' : [57, 72597],
      'Sullivan' : [58, 6046],
      'Susquehanna' : [59, 39864],
      'Tioga' : [60, 40393],
      'Unassigned' : [61, 0],
      'Union' : [62, 44735],
      'Venango' : [63, 49602],
      'Warren' : [64, 38471],
      'Washington' : [65, 206559],
      'Wayne' : [66, 51293],
      'Westmoreland' : [67, 345779],
      'Wyoming' : [68, 26208],
      'York' : [69, 451480],
    }
    index = POPULATIONS[county][0]

    global POP_DENSITIES 
    POP_DENSITIES = {'Adams': 195.5, 'Allegheny': 1675.6, 'Armstrong': 105.5, 'Beaver': 392.3, 'Bedford': 49.2, 'Berks': 480.4, 'Blair': 241.7, 'Bradford': 54.6, 'Bucks': 1034.7, 'Butler': 233.1,
                    'Cambria': 208.7, 'Cameron': 12.8, 'Carbon': 171.1, 'Centre': 138.7, 'Chester': 664.7, 'Clarion': 66.6, 'Clearfield': 71.3, 'Clinton': 44.2, 'Columbia': 139.3, 'Crawford': 87.7, 'Cumberland': 431.6, 'Dauphin': 510.6,
                    'Delaware': 3040.5, 'Elk': 38.6, 'Erie': 351.1, 'Fayette': 172.8, 'Forest': 18.1, 'Franklin': 193.7, 'Fulton': 33.9, 'Greene': 67.2, 'Huntingdon': 52.5, 'Indiana': 107.5, 'Jefferson': 69.3, 'Juniata': 63.0, 
                    'Lackawanna': 467.1, 'Lancaster': 550.4, 'Lawrence': 254.4, 'Lebanon': 369.1, 'Lehigh': 1012.5, 'Luzerne': 360.4, 'Lycoming': 94.5, 'McKean': 44.4, 'Mercer': 173.4,'Mifflin': 113.6, 'Monroe': 279.2, 
                    'Montgomery': 1655.9, 'Montour': 140.3, 'Northampton': 805.4, 'Northumberland': 206.2, 'Perry': 83.4, 'Philadelphia': 11379.5, 'Pike': 105.3, 'Potter': 16.1, 'Schuylkill': 190.4, 'Snyder': 120.8,
                    'Somerset': 72.4, 'Sullivan': 14.3, 'Susquehanna': 52.7, 'Tioga': 37.0, 'Union': 142.2, 'Venango': 81.5, 'Warren': 47.3, 'Washington': 242.5, 'Wayne': 72.8, 'Westmoreland': 355.4, 
                    'Wyoming': 71.2, 'York': 481.1}

    global PERCENT_GOP
    PERCENT_GOP = {'Adams': 0.663077364, 'Allegheny': 0.391172398, 'Armstrong': 0.755608818, 'Beaver': 0.580874664, 'Bedford': 0.833851506, 'Berks': 0.532894373, 'Blair': 0.710897008, 'Bradford': 0.715797179,
                  'Bucks': 0.47350706, 'Butler': 0.657764126, 'Cambria': 0.680010793, 'Cameron': 0.725819672, 'Carbon': 0.65440063, 'Centre': 0.467470817, 'Chester': 0.408318043,
                  'Clarion': 0.746750546, 'Clearfield': 0.740824047, 'Clinton': 0.673633162, 'Columbia': 0.644037954, 'Crawford': 0.680194411, 'Cumberland': 0.54504656, 'Dauphin': 0.44959651,
                  'Delaware': 0.361944389, 'Elk': 0.715957256, 'Erie': 0.48631475, 'Fayette': 0.666466507, 'Forest': 0.711178939, 'Franklin': 0.708700441, 'Fulton': 0.853080569, 'Greene': 0.710872299,
                  'Huntingdon': 0.746760637, 'Indiana': 0.681572181, 'Jefferson': 0.786350279, 'Juniata': 0.800132692, 'Lackawanna': 0.45186225, 'Lancaster': 0.573756887, 'Lawrence': 0.64292191,
                  'Lebanon': 0.651320176, 'Lehigh': 0.456883796, 'Luzerne': 0.567721642, 'Lycoming': 0.699211602, 'McKean': 0.723730138, 'Mercer': 0.625345622, 'Mifflin': 0.775531266, 
                  'Monroe': 0.463449184, 'Montgomery': 0.364675483, 'Montour': 0.595393396, 'Northampton': 0.48895238, 'Northumberland': 0.683638114, 'Perry': 0.743966378, 
                  'Philadelphia': 0.182728282, 'Pike': 0.589794587, 'Potter': 0.799573082, 'Schuylkill': 0.690605752, 'Snyder': 0.729268936, 'Somerset': 0.775821216, 
                  'Sullivan': 0.727928928, 'Susquehanna': 0.697725674, 'Tioga': 0.745769942, 'Union': 0.61470266, 'Venango': 0.697992213, 'Warren': 0.687221215, 
                  'Washington': 0.606631083, 'Wayne': 0.66150815, 'Westmoreland': 0.635782067, 'Wyoming': 0.667232944, 'York': 0.615555164}

    global PERCENT_OVER_65
    PERCENT_OVER_65 = {'Adams': 15.4, 'Allegheny': 14.9, 'Armstrong': 16.8, 'Beaver': 18.3, 'Bedford': 18.3, 'Berks': 18.1, 'Blair': 14.4, 'Bradford': 18.0, 'Bucks': 17.4, 'Butler': 14.4, 'Cambria': 15.3, 'Cameron': 18.8, 'Carbon': 19.9, 'Centre': 17.3, 'Chester': 11.4, 'Clarion': 12.6, 'Clearfield': 16.9, 'Clinton': 17.8, 'Columbia': 16.8, 'Crawford': 16.6, 'Cumberland': 16.6, 'Dauphin': 15.7, 'Delaware': 13.9, 'Elk': 14.3, 'Erie': 19.0, 'Fayette': 14.6, 'Forest': 17.3, 'Franklin': 17.1, 'Fulton': 16.9, 'Greene': 17.1, 'Huntingdon': 15.1, 'Indiana': 16.0, 'Jefferson': 16.0, 'Juniata': 18.1, 'Lackawanna': 17.2, 'Lancaster': 17.7, 'Lawrence': 15.0, 'Lebanon': 18.8, 'Lehigh': 17.2, 'Luzerne': 15.3, 'Lycoming': 18.1, 'McKean': 16.7, 'Mercer': 16.9, 'Mifflin': 17.9, 'Monroe': 18.4, 'Montgomery': 12.5, 'Montour': 15.0, 'Northampton': 18.4, 'Northumberland': 15.1, 'Perry': 18.8, 'Philadelphia': 13.3, 'Pike': 12.5, 'Potter': 16.2, 'Schuylkill': 19.3, 'Snyder': 18.4, 'Somerset': 15.1, 'Sullivan': 18.8, 'Susquehanna': 24.9, 'Tioga': 17.1, 'Union': 18.3, 'Venango': 14.6, 'Warren': 17.9, 'Washington': 18.6, 'Wayne': 17.5, 'Westmoreland': 19.3, 'Wyoming': 18.9, 'York': 15.9}

    global PERCENT_POVERTY
    PERCENT_POVERTY = {'Adams': 12.5, 'Allegheny': 7.2, 'Armstrong': 13.0, 'Beaver': 12.5, 'Bedford': 11.5, 'Berks': 11.7, 'Blair': 12.1, 'Bradford': 14.6, 'Bucks': 14.1, 'Butler': 4.2, 'Cambria': 8.9, 'Cameron': 15.5, 'Carbon': 13.1, 'Centre': 11.2, 'Chester': 18.0, 'Clarion': 6.3, 'Clearfield': 14.3, 'Clinton': 14.2, 'Columbia': 16.3, 'Crawford': 13.7, 'Cumberland': 16.2, 'Dauphin': 6.7, 'Delaware': 12.6, 'Elk': 9.3, 'Erie': 10.7, 'Fayette': 15.7, 'Forest': 17.3, 'Franklin': 22.4, 'Fulton': 9.3, 'Greene': 11.3, 'Huntingdon': 17.5, 'Indiana': 13.0, 'Jefferson': 17.6, 'Juniata': 14.3, 'Lackawanna': 10.4, 'Lancaster': 14.4, 'Lawrence': 9.4, 'Lebanon': 12.0, 'Lehigh': 9.0, 'Luzerne': 12.4, 'Lycoming': 13.1, 'McKean': 14.7, 'Mercer': 16.1, 'Mifflin': 13.0, 'Monroe': 15.1, 'Montgomery': 10.2, 'Montour': 5.5, 'Northampton': 10.7, 'Northumberland': 8.5, 'Perry': 14.6, 'Philadelphia': 8.8, 'Pike': 24.5, 'Potter': 8.2, 'Schuylkill': 15.7, 'Snyder': 11.9, 'Somerset': 14.0, 'Sullivan': 13.9, 'Susquehanna': 13.2, 'Tioga': 13.8, 'Union': 16.0, 'Venango': 13.9, 'Warren': 17.5, 'Washington': 12.2, 'Wayne': 10.9, 'Westmoreland': 11.2, 'Wyoming': 10.4, 'York': 12.8}

    global TOTAL_INSURED_POP
    TOTAL_INSURED_POP = {'Adams': 81517.0, 'Allegheny': 1158326.0, 'Armstrong': 65340.0, 'Beaver': 162945.0, 'Bedford': 44852.0, 'Berks': 332401.0, 'Blair': 115754.0, 'Bradford': 55848.0, 'Bucks': 559410.0, 'Butler': 161981.0, 'Cambria': 131861.0, 'Cameron': 5399.0, 'Carbon': 53674.0, 'Centre': 111615.0, 'Chester': 400987.0, 'Clarion': 35977.0, 'Clearfield': 72657.0, 'Clinton': 32342.0, 'Columbia': 54714.0, 'Crawford': 77842.0, 'Cumberland': 189865.0, 'Dauphin': 222620.0, 'Delaware': 496855.0, 'Elk': 32615.0, 'Erie': 241051.0, 'Fayette': 128910.0, 'Forest': 4154.0, 'Franklin': 117103.0, 'Fulton': 13077.0, 'Greene': 33101.0, 'Huntingdon': 37091.0, 'Indiana': 75496.0, 'Jefferson': 40989.0, 'Juniata': 20059.0, 'Lackawanna': 189134.0, 'Lancaster': 414964.0, 'Lawrence': 83929.0, 'Lebanon': 105137.0, 'Lehigh': 275045.0, 'Luzerne': 283396.0, 'Lycoming': 103915.0, 'McKean': 39174.0, 'Mercer': 103904.0, 'Mifflin': 41434.0, 'Monroe': 125869.0, 'Montgomery': 696181.0, 'Montour': 16235.0, 'Northampton': 240924.0, 'Northumberland': 82193.0, 'Perry': 40192.0, 'Philadelphia': 1228578.0, 'Pike': 43189.0, 'Potter': 15984.0, 'Schuylkill': 130986.0, 'Snyder': 32475.0, 'Somerset': 68488.0, 'Sullivan': 5599.0, 'Susquehanna': 37607.0, 'Tioga': 35659.0, 'Union': 29908.0, 'Venango': 50730.0, 'Warren': 39257.0, 'Washington': 188114.0, 'Wayne': 41338.0, 'Westmoreland': 337417.0, 'Wyoming': 24949.0, 'York': 350391.0}

    global TOTAL_NURSING_POP
    TOTAL_NURSING_POP = {'Adams': 955.0, 'Allegheny': 7942.0, 'Armstrong': 324.0, 'Beaver': 1297.0, 'Bedford': 335.0, 'Berks': 2452.0, 'Blair': 1955.0, 'Bradford': 383.0, 'Bucks': 3569.0, 'Butler': 1299.0, 'Cambria': 1036.0, 'Cameron': 50.0, 'Carbon': 430.0, 'Centre': 644.0, 'Chester': 2572.0, 'Clarion': 252.0, 'Clearfield': 725.0, 'Clinton': 231.0, 'Columbia': 545.0, 'Crawford': 503.0, 'Cumberland': 1870.0, 'Dauphin': 1377.0, 'Delaware': 4131.0, 'Elk': 247.0, 'Erie': 1962.0, 'Fayette': 823.0, 'Forest': 94.0, 'Franklin': 983.0, 'Fulton': 10.0, 'Greene': 224.0, 'Huntingdon': 134.0, 'Indiana': 439.0, 'Jefferson': 460.0, 'Juniata': 219.0, 'Lackawanna': 2440.0, 'Lancaster': 4311.0, 'Lawrence': 735.0, 'Lebanon': 1303.0, 'Lehigh': 3023.0, 'Luzerne': 3201.0, 'Lycoming': 1009.0, 'McKean': 566.0, 'Mercer': 1080.0, 'Mifflin': 444.0, 'Monroe': 496.0, 'Montgomery': 7095.0, 'Montour': 371.0, 'Northampton': 2099.0, 'Northumberland': 858.0, 'Perry': 278.0, 'Philadelphia': 7877.0, 'Pike': 95.0, 'Potter': 132.0, 'Schuylkill': 1418.0, 'Snyder': 153.0, 'Somerset': 659.0, 'Sullivan': 149.0, 'Susquehanna': 205.0, 'Tioga': 237.0, 'Union': 372.0, 'Venango': 462.0, 'Warren': 376.0, 'Washington': 1149.0, 'Wayne': 323.0, 'Westmoreland': 2131.0, 'Wyoming': 111.0, 'York': 2145.0}

    global run_population

    # plot already existing case data
    conf_data = np.genfromtxt('Data/penn_incidence.csv', dtype=str,delimiter=",") #this is the incidence

    print(conf_data[index][0]) #print the name of the county
    print(f'Day {max(1, starting_point)} to {(0 or starting_point)+days_after}.')

    pre_incidence = [float(x) for x in conf_data[index][1:]]
    
    conf_incidence, t = gen_time(pre_incidence, days_after, starting_point) #returns conf incidence, t
    
    # * define y0
    
    if starting_point:
      starting_infections = conf_incidence[0]
      y0 = [POPULATIONS[county][1]-starting_infections, 1, 0, 0, starting_infections/2, 0, 0, # undistanced population
             0, 0, 0, 0, starting_infections/2, 0, 0] # distanced population
      run_population = POPULATIONS[county][1]-starting_infections
    else:
      # it's much easier if you're starting from the beginning of the outbreak :)
      y0 = [POPULATIONS[county][1],1,0,0,0,0,0, # undistanced population
            0, 0, 0, 0, 0, 0, 0] # distanced population
      run_population = POPULATIONS[county][1]
    
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
  max_rate_dist_max_rate_undist_sample = lhsmdu.sample(3,sample_count)
  distance_accel_undistance_accel_sample = lhsmdu.sample(3,sample_count)
  max_rate_dist_max_rate_undist_sample = max_rate_dist_max_rate_undist_sample.tolist()
  distance_accel_undistance_accel_sample = distance_accel_undistance_accel_sample.tolist()

  # we then adjust the variables to the correct ranges
  adjusted_sample = []
  # for max rate, we adjust to between 1 and 1/30
  for var_dist in max_rate_dist_max_rate_undist_sample:
    var_dist = [(1 + var*(1/30-1)) for var in var_dist]
    adjusted_sample.append(var_dist)
  # for accel, we adjust to between 10 and 0.0001
  # however, we actually use theta where EM = 10^theta, so theta is between 1 and -5
  # prevents overweighting towards the top end of the spectrum
  for var_dist in distance_accel_undistance_accel_sample:
    var_dist = [(1 + var*(-5-1)) for var in var_dist]
    adjusted_sample.append(var_dist)

  # then, for each pair of 4 variables, we run it through the fit and determine cost
  # for every guess, we check to see if that's the lowest cost generated thus far
  # if it is, we store it, and at the end, that's our result
  lowest_cost = [10000000000000, [0,0,0,0,0,0]]
 
  for i in range(sample_count):
    test_guesses = []
    for var_dist in adjusted_sample:
      test_guesses.append(var_dist[i])
    try:
      # we have to rearrange test_guesses so that it goes S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel
      # it is currently [S_distmax, undistmax, I_distmax, S_accel, unaccel, I_accel]
      S_distmax, undistmax, I_distmax, S_accel, unaccel, I_accel = test_guesses
      test_guesses = [S_distmax, S_accel, undistmax, unaccel, I_distmax, I_accel]
      # we then adjust distance accels to be 10^[their value], because it's currently theta
      test_guesses[1] = 10**(test_guesses[1])
      test_guesses[3] = 10**(test_guesses[3])
      test_guesses[5] = 10**(test_guesses[5])

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

def full_dynamics(city_county):
  global incidence_data
  global y0
  incidence_data, t, y0 = define_dataset(city_county, days_after=10000)
  initial_guesses = LHS_for_guesses(10)
  S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel = SD_curve_fit(initial_guesses).x

  if True:
    f5 = plt.figure(5)
    f5.suptitle(f'Full Overview of {city_county} Simulation Dynamics')
    normalized_data = [x/max(incidence_data) for x in incidence_data]
    plt.plot(t, normalized_data, ':', label='Actual Symptomatic') #plot the actual symptomatic infections


    y = SDmodel(S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel)
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
    data_row = ['County', 'Population', 'Percentage over 65', 'Percentage in Poverty', 'Percentage Insured', 'Total Nursing Home Pop', 'Pop/Mile^2', 'Cases per Capita', 'Percentage GOP', 'Peak #', 'Start Date', 'End Date', 'S_max_rate_dist', 'S_distance_accel', 'max_rate_undist', 'undistance_accel', 'I_max_rate_dist', 'I_distance_accel']
    writer = csv.writer(f)
    writer.writerow(data_row)

  for label in  ['Adams', 'Allegheny', 'Armstrong', 'Beaver', 'Bedford', 'Berks', 'Blair', 'Bradford', 'Bucks', 'Butler', 'Cambria', 'Cameron', 'Carbon', 'Centre', 'Chester', 'Clarion',
                'Clearfield', 'Clinton', 'Columbia', 'Crawford', 'Cumberland', 'Dauphin', 'Delaware', 'Elk', 'Erie', 'Fayette', 'Forest', 'Franklin', 'Fulton', 'Greene',
                'Huntingdon', 'Indiana', 'Jefferson', 'Juniata', 'Lackawanna', 'Lancaster', 'Lawrence', 'Lebanon', 'Lehigh', 'Luzerne', 'Lycoming', 'McKean', 'Mercer', 'Mifflin',
                'Monroe', 'Montgomery', 'Montour', 'Northampton', 'Northumberland', 'Perry', 'Philadelphia', 'Pike', 'Potter', 'Schuylkill', 'Snyder', 'Somerset',
                'Sullivan', 'Susquehanna', 'Tioga', 'Union', 'Venango', 'Warren', 'Washington', 'Wayne', 'Westmoreland', 'Wyoming', 'York']:
    
    # * define the dataset
    # note that the y0 produced here is just 1 infection, 0 everything else (except susceptibles)
    incidence_data, t, y0 = define_dataset(label, days_after=1000)

    # * get the date of the midpoint of each peak
    peaks = detect_peaks(incidence_data, display=False)


    # * run simulation for each individual peak
    for i, peak in enumerate(peaks):

      # * we then define the dataset for the actual peak simulation
      incidence_data, t, y0 = define_dataset(label, starting_point=max(peak-50, 0), days_after=100)

      # * obtain parameter values
      initial_guesses = LHS_for_guesses(depth)
      S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel = SD_curve_fit(initial_guesses).x

      # * write parameter values to file
      date_labels = np.genfromtxt('Data/penn_incidence.csv', dtype=str,delimiter=",")[0][3:]

      # * obtain start and end of each peak
      start_of_peak = date_labels[peak-50].replace('"', '')
      try:
        end_of_peak = date_labels[peak+50].replace('"', '')
      except IndexError:
        end_of_peak = date_labels[-1].replace('"', '')

      # * obtain total case count
      y = SDmodel(S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel)
      total_cases = sum( y[:,4]+y[:,11])

      # * write resulting values to file
      # ['County', 'Population',
      # 'Percentage Over 65, Percentage in Poverty, Percentage Insured, Percentage in Nursing Homes,
      # 'Pop/Mile^2', 'Cases per Capita', 'Percentage GOP', 'Peak #', 'Start Date', 'End Date', 
      # 'S_max_rate_dist', 'S_distance_accel', 'max_rate_undist', 'undistance_accel', I_max_rate_dist, 'I_distance_accel']
      print('Writing country: ', label)
      data_row = [label, POPULATIONS[label][1], PERCENT_OVER_65[label], PERCENT_POVERTY[label], TOTAL_INSURED_POP[label]/POPULATIONS[label][1], TOTAL_NURSING_POP[label],
                  POP_DENSITIES[label], total_cases/(POPULATIONS[label][1]), PERCENT_GOP[label], i+1, start_of_peak, end_of_peak, 
                  round(S_max_rate_dist, 6), round(S_distance_accel, 6), round(max_rate_undist, 6), round(undistance_accel, 6), round(I_max_rate_dist, 6), round(I_distance_accel, 6)]

      with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

      # * graph and save graphs in the IndividualSims folder
      plot_for_vals(incidence_data, S_max_rate_dist, S_distance_accel, max_rate_undist, undistance_accel, I_max_rate_dist, I_distance_accel)
      plt.legend()
      try:
        plt.savefig(f'IndividualSimsPenn/{label}/Peak{str(i+1)}')
      except FileNotFoundError:
        os.makedirs(f'IndividualSimsPenn/{label}')
        plt.savefig(f'IndividualSimsPenn/{label}/Peak{str(i+1)}')

      plt.clf()