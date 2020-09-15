import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.ticker import PercentFormatter
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
import math


def social_bI(bI_0, alpha, N):
    try:
        bI = bI_0 * math.pow(N, alpha)
        # bI = math.pow(bI_0*N, alpha)
    except ValueError:
        bI = 0
    return max(bI, 0)


# def ODEmodel(vals, t, b_I, k, c, mxstep=50000):
def ODEmodel(vals, t, bI_0, alpha, k, c2, mxstep=50000, full_output=1):
    #unpack
    S = vals[0]
    E = vals[1]
    A = vals[2]
    P = vals[3]
    I = vals[4]
    R = vals[5]
    D = vals[6]
    total_cases = vals[7]
    ###defining lambda###
    dDdT = 1/7/99 * I
    b_I = social_bI(bI_0, alpha, dDdT)
    b_A = 0.5*b_I #transmission from asympt as a a small fraction of symptomatic infection
    b_P = c2*b_I #transmission from presympt
    N = S+E+P+A+I+R+D
    lamb = (b_A*A + b_P*P + b_I*I)/N # susceptible -> infected
    ###other valsameters###
    # k = .2 #percentage of exposed -> asymptomatic
    sig = 1/5 #exposed -> presympt/astmpy
    delt = 1/2 #pre-sympt -> sympt
    gam_I = 1/7 #sympt -> recovered
    gam_A = 1/7 #asympt -> recovered
    nu = gam_I/99 #sympt -> deceased
    ###stuff to return###
    dSdt = (-lamb*S)
    dEdt = (lamb*S) - (sig*E)
    dAdt = (k*sig*E) - (gam_A*A)
    dPdt = ((1-k)*sig*E) - (delt*P)
    dIdt = (delt*P) - ((nu + gam_I)*I)
    dRdt = (gam_A*A) + (gam_I*I)
    dDdt = nu*I
    dTdt = sig*E
    return [dSdt,dEdt,dAdt,dPdt,dIdt,dRdt,dDdt,dTdt]


def model(beta_I, k, c): #function that allows us to call the odeint solver with more readability
    return odeint(ODEmodel, y0, t, (beta_I, k, c))


def SDmodel(bI_0, alpha, k, c):
    return odeint(ODEmodel, y0, t, (bI_0, alpha, k, c))


def gen_time(caseinc, days_after,y0):
    first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
    last_index = first_index+days_after # first day + however many days after we want
    conf_incidence = caseinc[first_index-14:last_index] # 14 days before first case
    global t
    t = np.linspace(0,len(conf_incidence),num=len(conf_incidence))
    return conf_incidence, t, y0


#get the return from curve fit
def get_curve_fit(k, c):
    resids = lambda bI, data: (model(bI, k, c)[:,4] - data) #have to use this to fix k and c so that they're not part of the curve fitting function
    op = least_squares(resids, 1.5, args=(conf_incidence,) )
    return op


def SD_curve_fit(k, c):
    resids = lambda params, data: (SDmodel(params[0], params[1], k, c)[:,4] - data) #have to use this to fix k and c so that they're not part of the curve fitting function
    op = least_squares(resids, [1, 0], args=(conf_incidence,) )
    return op


#general form to get the optimal B_I from curve fit
def get_optimal_bI(k, c):
    # ! change
    op = SD_curve_fit(k, c)
    return op.x


#plot model output against given dataset for parameter values
def plot_for_vals(dataset, bI0, alpha, k, c):
    y = SDmodel(bI0, alpha,k,c)
    f5 = plt.figure(5)
    f5.suptitle(f'bI0={round(bI0, 3)}, alpha={round(alpha, 3)}, k={k}, c={c}')
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
        'San Fransisco' : [14, 881549],
        'Wake' : [15, 1111761],
    }
    index = POPULATIONS[county][0]
    # define y0
    y0 = [POPULATIONS[county][1],1,0,0,0,0,0,0] # PHL
    #plot already existing case data
    conf_data = np.genfromtxt('COVID_city_county_new.csv', dtype=str,delimiter=",") #this is the incidence
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
    bI_0, alpha = SD_curve_fit(k, c2).x
    # OR use fixed bI0 and alpha values
    bI_0, alpha = 0.186, -.257
    print(f'Using bI_0 {bI_0} and alpha {alpha}')
    # calculate the death array
    y = SDmodel(bI0, alpha, k, c2)
    deaths = y[:,6]
    # calculate a list of R0 values based off the death values
    R0_list = []
    output = open('Figures/R-Effective/Reffective_analysis.csv.txt', 'w')
    print('bI_0,alpha,new deaths,bI,bA,bP,R0\n', file=output)
    for death in deaths:
        b_I = social_bI(bI_0, alpha, death)
        b_A = 0.5*b_I #transmission from asympt as a a small fraction of symptomatic infection
        b_P = c2*b_I #transmission from presympt
        R0 = ( (1-k)*sig ) * ( (b_I / (gam_I+nu) ) + ( (b_P) / delt )) + ( k*sig * ( (b_A)/gam_A ) )
        print('%s,%s,%s,%s,%s,%s,%s\n' % (bI_0, alpha, death, b_I, b_A, b_P, R0), file=output)
        R0_list.append(R0)
    # plot the R0 list against time
    f6 = plt.figure(6)
    # plot the R0 values
    plt.plot(t, R0_list, label=f'R0: k,c2,bI_0,alpha,={k},{c2},{round(bI_0, 2)},{round(alpha, 2)}')


if __name__ == "__main__":
    # you always need to globally define the dataset
    conf_incidence, t, y0 = define_dataset('Philadelphia', days_after=150)
    k = .5
    c2 = 5
    bI0, alpha = SD_curve_fit(k, c2).x
    cost = SD_curve_fit(k, c2).cost
    print(f'Cost is {cost}')
    # plot_for_vals(conf_incidence, bI0, alpha, k, c2)
    plot_Reffective(.5,5)
    plt.legend()
    plt.show()
