import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
import math


# def social_bI(alpha, epsilon, N):
#     try:
#         bI = alpha*N + epsilon
#     except ValueError:
#         bI = 0
#     return max(bI, 0)

def ODEmodel(vals, t, alpha, epsilon):

    ###other parameters###
    sig = 1/5 #exposed -> presympt/astmpy
    delt = 1/2 #pre-sympt -> sympt
    gam_I = 1/7 #sympt -> recovered
    gam_A = 1/7 #asympt -> recovered
    nu = gam_I/99 #sympt -> deceased

    k = .5 # percentage asymptomatic

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
    b_Iu = alpha
    b_Au = 0.5*b_Iu
    b_Pu = b_Iu

    b_Id = epsilon
    b_Ad = 0.5*b_Id
    b_Pd = b_Id

    N = S_u+E_u+P_u+A_u+I_u+R_u+D_u + S_d+E_d+P_d+A_d+I_d+R_d+D_d

    lamb = ((b_Au*A_u + b_Pu*P_u + b_Iu*I_u) + (b_Ad*A_d + b_Pd*P_d + b_Id*I_d)) /N

    ###stuff to return###

    # undistanced group #
    dS_udt = (-lamb * S_u)
    dE_udt = (lamb * S_u) - (sig * E_u)
    dA_udt = (k * sig * E_u) - (gam_A * A_u)
    dP_udt = ( (1-k) * sig * E_u) - (delt * P_u)
    dI_udt = (delt * P_u) - ( (nu + gam_I) * I_u)
    dR_udt = (gam_A * A_u) + (gam_I * I_u)
    dD_udt = nu * I_u

    # distanced group #
    dS_ddt = (-lamb * S_d)
    dE_ddt = (lamb * S_d) - (sig * E_d)
    dA_ddt = (k * sig * E_d) - (gam_A * A_d)
    dP_ddt = ( (1-k) * sig * E_d) - (delt * P_d)
    dI_ddt = (delt * P_d) - ( (nu + gam_I) * I_d)
    dR_ddt = (gam_A * A_d) + (gam_I * I_d)
    dD_ddt = nu * I_d


    return [dS_udt,dE_udt,dA_udt,dP_udt,dI_udt,dR_udt,dD_udt,
            dS_ddt,dE_ddt,dA_ddt,dP_ddt,dI_ddt,dR_ddt,dD_ddt]


def SDmodel(alpha, epsilon):
    return odeint(ODEmodel, y0, t, (alpha, epsilon))

def gen_time(caseinc, days_after,y0):
    first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
    last_index = first_index+days_after # first day + however many days after we want
    conf_incidence = caseinc[first_index-14:last_index] # 14 days before first case
    global t
    t = np.linspace(0,len(conf_incidence),num=len(conf_incidence))
    return conf_incidence, t, y0

def SD_curve_fit():
    resids = lambda params, data: (SDmodel(params[0], params[1])[:,4] - data)
    op = least_squares(resids, [0, 0], args=(conf_incidence,) )
    return op

#general form to get the optimal B_I from curve fit
def get_optimal_bI():
    op = SD_curve_fit()
    return op.x

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, alpha, epsilon):
    y = SDmodel(alpha, epsilon)
    f5 = plt.figure(5)
    f5.suptitle(f'b_Iu={alpha}, B_Id={epsilon}')
    
    plt.plot(t, y[:,4], label='Predicted Symptomatic Noncompliant') #plot the model's symptomatic infections
    
    plt.plot(t, conf_incidence, label='Actual Symptomatic') #plot the actual symptomatic infections
    
    plt.plot(t, y[:,11], label='Predicted Symptomatic Compliant') #plot the model's symptomatic infections
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
    #plot already existing case data
    conf_data = np.genfromtxt('city_county_case_time_series_incidence.csv', dtype=str,delimiter=",") #this is the incidence
    print(conf_data[index][2]) #print the name of the county
    pre_incidence = [float(x) for x in conf_data[index][3:]]
    return gen_time(pre_incidence,days_after,y0) #returns conf incidence, t, y0


if __name__ == "__main__":
    # you always need to globally define the dataset
    conf_incidence, t, y0 = define_dataset('San Fransisco', days_after=500)

    alpha, epsilon = SD_curve_fit().x

    cost = SD_curve_fit().cost

    plot_for_vals(conf_incidence, alpha, epsilon)

    plt.legend()
    plt.show()