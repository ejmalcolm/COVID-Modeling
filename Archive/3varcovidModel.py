import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
import math

guesses = [0, 0, 0]
replace_data_with_peaks = False

def social_bI(a, b, c, N, t):
    try:
      bI = a*N+b
    except ValueError as e:
      print(e)
      bI = 0
    return max(bI, 0)

last_peak_time = 0
def store_last_peak(time):
  global last_peak_time
  last_peak_time = time

def ODEmodel(vals, t, alpha, epsilon, omicron, k, c2, mxstep=50000, full_output=1):
# def ODEmodel(vals, t, alpha, epsilon, k, c2, mxstep=50000, full_output=1):
    
    ###other parameters###
    sig = 1/5 #exposed -> presympt/astmpy
    delt = 1/2 #pre-sympt -> sympt
    gam_I = 1/7 #sympt -> recovered
    gam_A = 1/7 #asympt -> recovered
    nu = gam_I/99 #sympt -> deceased
    
    ###unpack###
    S = vals[0]
    E = vals[1]
    A = vals[2]
    P = vals[3]
    I = vals[4]
    R = vals[5]
    D = vals[6]
    N = S+E+P+A+I+R+D

    # use only if we're using dBIdt as one of the outputs
    b_I = max(0, vals[7])
    
    ###defining lambda###
    
    # * alpha = bi_0, epsilon = scaling factor

    b_I = social_bI(alpha, epsilon, omicron, D, t)
    b_I = b_I + omicron*(t-last_peak_time)
    

    b_A = 0.5*b_I
    b_P = c2*b_I
    lamb = (b_A*A + b_P*P + b_I*I)/N
    ###stuff to return###
    dSdt = (-lamb*S)
    dEdt = (lamb*S) - (sig*E)
    dAdt = (k*sig*E) - (gam_A*A)
    dPdt = ((1-k)*sig*E) - (delt*P)
    dIdt = (delt*P) - ((nu + gam_I)*I)
    dRdt = (gam_A*A) + (gam_I*I)
    dDdt = nu*I
    dBIdt = alpha*(dDdt) - epsilon*b_I + omicron*t
    
    # check if this is a peak
    if (-0.01 < dIdt < 0.01):
      # print(t)
      # store_last_peak(t)
      pass

    if 59 < t < 61:
      print(t, dIdt)

    return [dSdt,dEdt,dAdt,dPdt,dIdt,dRdt,dDdt,dBIdt]


def model(beta_I, k, c): #function that allows us to call the odeint solver with more readability
    return odeint(ODEmodel, y0, t, (beta_I, k, c))


def SDmodel(alpha, epsilon, omicron, k, c):
    return odeint(ODEmodel, y0, t, (alpha, epsilon, omicron, k, c))


def gen_time(caseinc, days_after,y0):
    first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
    last_index = first_index+days_after # first day + however many days after we want
    conf_incidence = caseinc[first_index-14:last_index] # 14 days before first case

    # TODO: fake data insertion
    if replace_data_with_peaks:
      conf_incidence = [50*(math.sin(0.025*k)) for k,v in enumerate(conf_incidence)]
      buffer = [0 for x in conf_incidence]
      conf_incidence = (conf_incidence + buffer)*5
      conf_incidence = [max(x, 1) for x in conf_incidence]
    # TODO: fake data insertion

    global t
    t = np.linspace(0,len(conf_incidence),num=len(conf_incidence))
    return conf_incidence, t, y0


#get the return from curve fit
def get_curve_fit(k, c):
    resids = lambda bI, data: (model(bI, k, c)[:,4] - data) #have to use this to fix k and c so that they're not part of the curve fitting function
    op = least_squares(resids, 1.5, args=(conf_incidence,) )
    return op


def SD_curve_fit(k, c):
    resids = lambda params, data: (SDmodel(params[0], params[1], params[2], k, c)[:,4] - data) #have to use this to fix k and c so that they're not part of the curve fitting function
    op = least_squares(resids, guesses, args=(conf_incidence,) )
    return op


#general form to get the optimal B_I from curve fit
def get_optimal_bI(k, c):
    op = SD_curve_fit(k, c)
    return op.x


#plot model output against given dataset for parameter values
def plot_for_vals(dataset, alpha, epsilon, omicron, k, c):
    y = SDmodel(alpha, epsilon, omicron, k, c)
    f5 = plt.figure(5)
    f5.suptitle(f'Alp={round(alpha, 3)}, Eps={round(epsilon, 3)}, Omi={round(omicron, 3)} k={k}, c={c}')
    plt.plot(t, y[:,4], label='Predicted Symptomatic') #plot the model's symptomatic infections
    plt.plot(t, conf_incidence, label='Actual Symptomatic') #plot the actual symptomatic infections
  

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
        'San Francisco' : [14, 881549],
        'Wake' : [15, 1111761],
    }
    index = POPULATIONS[county][0]
    # define y0
    y0 = [POPULATIONS[county][1],1,0,0,0,0,0,1.9] # PHL
    #plot already existing case data
    conf_data = np.genfromtxt('city_county_case_time_series_incidence.csv', dtype=str,delimiter=",") #this is the incidence
    print(conf_data[index][2]) #print the name of the county
    pre_incidence = [float(x) for x in conf_data[index][3:]]
    return gen_time(pre_incidence,days_after,y0) #returns conf incidence, t, y0


def bI_heatmap(): #get a heatmap of bI values for a range of k and c values, ranges: 0.1-1 and 1-5
    #first, we generate an array of bI values
    bI_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            bI = round(get_optimal_bI(k,c)[0],3)
            sublist.append((bI))
        bI_list.append(sublist)
    bI_array = np.array(bI_list)

    #then we plot the heatmap from that array
    plt.figure(1)
    heat_map = sb.heatmap(bI_array, annot=True, fmt='.3g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='Greens', cbar_kws = {'label': 'βᵢ'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def get_R0(k, c):
    bI = get_optimal_bI(k,c)
    bI = 2 ## for fixed
    ODEmodel(y0, t, bI, k, c)
    return R0 #R0 is defined in a global variable, that's the only way to access it for reasons


def R0_heatmap():
    #first, we generate an array of r0 values
    r0_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            # r0 = round(get_R0(k,c)[0],3)
            r0 = round(get_R0(k,c),3) #this is the fixed version
            sublist.append((r0))
        r0_list.append(sublist)
    r0_array = np.array(r0_list)

    #then we plot the heatmap from that array
    plt.figure(2)
    heat_map = sb.heatmap(r0_array, annot=True, fmt='.3g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = [round(x,1) for x in np.arange(1, 5.5, step=.5)],
      cmap='Reds', cbar_kws = {'label': 'R0'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def get_tot(k, c):
    bI = get_optimal_bI(k,c)
    # bI = 2
    total = model(bI, k, c)[-2,-2]
    return total


def total_heatmap():
    #first, we generate an array of r0 values
    total_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            total = round(get_tot(k,c),3)
            sublist.append((total))
        total_list.append(sublist)
    total_array = np.array(total_list)

    #then we plot the heatmap from that array
    plt.figure(3)
    heat_map = sb.heatmap(total_array, annot=True, fmt='.9g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='Blues', cbar_kws = {'label': 'Total Cases'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def plot_avs(k, c): #sympt vs. asympt timeseries
    bI_0, alpha = SD_curve_fit(k, c).x
    y = SDmodel(bI_0, alpha, k,c)
    plt.plot(t, y[:,4], label='(%s, %s) Symptomatic' % (k, c))
    plt.plot(t, y[:,2], label='(%s, %s) Asymptomatic' % (k, c))


def cost_heatmap():
    # first, we curve fit at all k,c values then return the cost function
    total_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            cst = SD_curve_fit(k, c).cost
            sublist.append((cst))
        total_list.append(sublist)
    # get the minimum value of all the cost outputs
    mins = []
    for sublist in total_list:
        for item in sublist:
            mins.append(item)
    min_value = min(mins)
    # divide every value by the minimum
    standardized = []
    for sublist in total_list:
        sublist = [x/min_value for x in sublist]
        standardized.append(sublist)
    total_array = np.array(standardized)


    #then we plot the heatmap from that array
    plt.figure(4)
    heat_map = sb.heatmap(total_array, annot=True, fmt='.5g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = [round(x,1) for x in np.arange(1,5.5,step=.5)],
      cmap='Blues', cbar_kws = {'label': 'Cost Function'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def R0_deriv_plots():
    bI = 2 #fixed value
    sig = 1/15 #exposed -> presympt/astmpy
    delt = 1/2 #pre-sympt -> sympt
    gam_A = 1/20 #asympt -> recovered

    c = np.linspace(0,5)
    k = np.linspace(0,1)
    R0_deriv_k = sig*(((c*bI))/gam_A)
    kslope = np.polyfit(c, R0_deriv_k, 1)[0]
    R0_deriv_c = (bI/delt) + ( (k*sig*bI) /gam_A )
    cslope = np.polyfit(k, R0_deriv_c, 1)[0]
    plt.plot(k, R0_deriv_c, label= f'R0_c, slope {cslope}')
    plt.plot(c, R0_deriv_k, label= f'R0_k, slope {kslope}')


def BI_vs_c_heatmap():
    #first, we generate an array of r0 values
    r0_list = []
    for bI in np.arange(1,5.5,step=.5):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            # k fixed at .2
            ODEmodel(y0, t, bI, .2, c)
            r0 = round(R0, 2)
            sublist.append((r0))
        r0_list.append(sublist)
    r0_array = np.array(r0_list)
    
    #then we plot the heatmap from that array
    heat_map = sb.heatmap(r0_array, annot=True, fmt='.5g',
      yticklabels = np.arange(1, 5.5, step=.5),
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='Reds', cbar_kws = {'label': 'R0'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'βᵢ, infectious transmission rate')
    heat_map.invert_yaxis()


def c1c2_heatmap():
    #first, we curve fit at all k,c values then return the cost function
    total_list = []
    for c1 in np.arange(0, 5.5, step=.5):
        sublist = []
        for c2 in np.arange(1, 11, step=.5):
            total = get_curve_fit(c1, c2).cost
            sublist.append((total))
        total_list.append(sublist)
    total_array = np.array(total_list)

    #then we plot the heatmap from that array
    heat_map = sb.heatmap(total_array, annot=True, fmt='.5g',
      yticklabels = np.arange(0, 5.5, step=.5),
      xticklabels = np.arange(1, 11, step=.5),
      cmap='Greens', cbar_kws = {'label': 'Cost Function'})
    plt.xlabel('C1, where βₐ = c*βᵢ')
    plt.ylabel('C2, where βₚ = c*βᵢ')
    heat_map.invert_yaxis()


def plot_Reffective(k, c2):
    # constants used for R0 calculation
    sig = 1/5 # exposed -> presympt/astmpy
    delt = 1/2 # pre-sympt -> sympt
    gam_I = 1/7 # sympt -> recovered
    gam_A = 1/7 # asympt -> recovered
    nu = gam_I/99 # sympt -> deceased
    # calculate a bI value through fitting to data
    # bI_0, alpha = SD_curve_fit(k, c2).x
    # OR use fixed bI0 and alpha values
    bI_0, alpha = 0.186, -.257
    print(f'Using bI_0 {bI_0} and alpha {alpha}')
    # calculate the death array
    y = SDmodel(bI_0, alpha, k, c2)
    S, E, A, P, I, R, D = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6]
    # calculate a list of R0 values based off the death values
    R0_list = []
    for i in range(len(S)):
        b_I = social_bI(bI_0, alpha, D[i])
        b_I = 2
        b_A = 0.5*b_I #transmission from asympt as a a small fraction of symptomatic infection
        b_P = c2*b_I #transmission from presympt
        R0 = ( (1-k)*sig ) * ( (b_I / (gam_I+nu) ) + ( (b_P) / delt )) + ( k*sig * ( (b_A)/gam_A ) )
        Reffective = R0*( S[i]/(sum([S[i],E[i],A[i],P[i],I[i],R[i],D[i]])) )
        R0_list.append(Reffective)
    # plot the R0 list against time
    f6 = plt.figure(6)
    # plot the R0 values
    plt.plot(t, R0_list, label=f'R0: k,c2,bI_0,alpha,={k},{c2},{round(bI_0, 2)},{round(alpha, 2)}')


if __name__ == "__main__":
    # you always need to globally define the dataset
    conf_incidence, t, y0 = define_dataset('District of Columbia', days_after=200)
    k = .2
    c2 = 1
    
    # curve fitting
    # alpha, epsilon, omicron = SD_curve_fit(k, c2).x
    # cost = SD_curve_fit(k, c2).cost
    # print(f'Cost is {cost}')

    # plotting
    # standard values: alpha = -.112, epsilon = .406, omicron= .001
    alpha = -.112
    epsilon = .406
    omicron = .001
    plot_for_vals(conf_incidence, alpha, epsilon, omicron, k, c2)
    plt.legend()
    plt.show()
