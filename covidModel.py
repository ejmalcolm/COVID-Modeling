from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sb
from scipy.optimize import curve_fit, least_squares

global asympt
global sympt

def ODEmodel(vals, t, b_I, k, c, mxstep=50000):
    #unpack
    S = vals[0]
    E = vals[1]
    A = vals[2]
    P = vals[3]
    I = vals[4]
    R = vals[5]
    D = vals[6]
    ###defining lambda###
    b_A = c*b_I #transmission from asympt as a multiple "c" of symptomatic infection
    b_P = b_A #transmission from presympt
    # b_I = params[1] #transmission from sympt 
    N = S+E+P+A+I+R+D
    lamb = (b_A*A + b_P*P + b_I*I)/N #susceptible -> infected
    ###other valsameters###
    # k = .2 #percentage of exposed -> asymptomatic
    sig = 1/15 #exposed -> presympt/astmpy
    delt = 1/2 #pre-sympt -> sympt
    nu = 1/15 #sympt -> deceased
    gam_I = 1/20 #sympt -> recovered
    gam_A = 1/20 #asympt -> recovered
    ###stuff to return###
    dSdt = (-lamb*S)
    dEdt = (lamb*S) - (sig*E)
    dAdt = (k*sig*E) - (gam_A*A)
    dPdt = ((1-k)*sig*E) - (delt*P)
    dIdt = (delt*P) - ((nu + gam_I)*I)
    dRdt = (gam_A*A) + (gam_I*I)
    dDdt = nu*I
    total_cases = sig*E
    #calculate R0
    global R0
    R0 = ( (1-k)*sig ) * ( (b_I / (gam_I+nu) ) + ( (c*b_I) / delt )) + ( k*sig * ( (c*b_I)/gam_A ) )
    global R0_deriv_k
    R0_deriv_k = sig*(((c*b_I))/gam_A)
    global R0_deriv_c
    R0_deriv_c = (b_I/delt) + ( (k*sig*b_I) /gam_A )
    return [dSdt,dEdt,dAdt,dPdt,dIdt,dRdt,dDdt, total_cases]

def model(beta_I, k, c): #function that allows us to call the odeint solver with more readability
    return odeint(ODEmodel, y0, t, (beta_I, k, c))

def gen_time(caseinc, days_after):
    first_index = next((i for i, x in enumerate(caseinc) if float(x)), None) # returns the index of the first nonzero
    last_index = first_index+days_after # first day + 3 weeks
    return caseinc[first_index:last_index]

##used for testing all different county data sets systematically
def test_all_sets(k, c):
    conf_data = np.loadtxt('COVID_city_county.csv', dtype=str,delimiter=",") #this is the incidence
    i = 1
    temp_populations = (704749,1526000,343829,881549,12019,974563,31275,600158,195769,323152)
    for _ in range(1,11):
        pop = temp_populations[i-1]
        global y0
        y0 = [pop,1,0,0,0,0,0]
        print(conf_data[i][0], pop) 
        pre_incidence = [float(data) for data in conf_data[i:][0][2:]]
        global conf_incidence 
        conf_incidence = gen_time(pre_incidence,21)
        print(conf_incidence)
        x = get_optimal_bI(k, c)
        print(x)
        i += 1

#testing sweden data
def test_sweden():
    conf_data = np.genfromtxt('global_case_time_series.csv', dtype=str,delimiter=",")
    conf_incidence = []
    total_cases = 0
    for tot_today in conf_data[1][34:]:
        new_cases = int(tot_today)-total_cases
        total_cases += new_cases
        conf_incidence.append(new_cases)
    print(conf_incidence)

#get the return from curve fit
def get_curve_fit(k, c):
    resids = lambda bI, data: (model(bI, k, c)[:,4] - data) #have to use this to fix k and c so that they're not part of the curve fitting function
    op = least_squares(resids, 1.5, args=(conf_incidence,) )
    return op

#general form to get the optimal B_I from curve fit
def get_optimal_bI(k, c):
    op = get_curve_fit(k, c)
    return op.x

#plot model output against given dataset for parameter values
def plot_for_vals(dataset, bI, k, c):
    y = model(bI,k,c)
    plt.plot(t, y[:,4], label='Predicted Symptomatic') #plot the model's symptomatic infections
    plt.plot(t, conf_incidence, label='Actual Symptomatic') #plot the actual symptomatic infections

  
def define_dataset(index, days_after): #change first index to 1=DOC, PHL, NO, SF, Alamos, Honolulu, Juneau, Denver, Muscogee, Fayette
    temp_populations = (704749,1526000,343829,881549,12019,974563,31275,600158,195769,323152)
    global y0
    y0 = [temp_populations[index],1,0,0,0,0,0,0] #define population
    global t
    t = np.linspace(0,days_after,num=days_after)
    #plot already existing case data
    conf_data = np.loadtxt('COVID_city_county.csv', dtype=str,delimiter=",") #this is the incidence
    print(conf_data[index][0]) #print the name of the county
    pre_incidence = [float(x) for x in conf_data[index][2:]]
    return gen_time(pre_incidence,days_after)


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
    f1 = plt.figure(1)
    heat_map = sb.heatmap(bI_array, annot=True, fmt='.3g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='Greens', cbar_kws = {'label': 'βᵢ'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def get_R0(k, c):
    bI = get_optimal_bI(k,c)
    # bI = 2 ## for fixed
    ODEmodel(y0, t, bI, k, c)
    return R0 #R0 is defined in a global variable, that's the only way to access it for reasons


def R0_heatmap():
    #first, we generate an array of r0 values
    r0_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(.1, 1, step=.1):
            r0 = round(get_R0(k,c)[0],3)
            # r0 = round(get_R0(k,c),3) #this is the fixed version
            sublist.append((r0))
        r0_list.append(sublist)
    r0_array = np.array(r0_list)

    #then we plot the heatmap from that array
    f2 = plt.figure(2)
    heat_map = sb.heatmap(r0_array, annot=True, fmt='.3g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      cmap='Reds', cbar_kws = {'label': 'R0'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def get_tot(k, c):
    bI = get_optimal_bI(k,c)
    bI = 2
    total = model(bI, k, c)[-1,-1]
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
    f3 = plt.figure(3)
    heat_map = sb.heatmap(total_array, annot=True, fmt='.9g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='Blues', cbar_kws = {'label': 'Total Cases'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()


def plot_avs(k, c): #sympt vs. asympt timeseries
    bI = get_optimal_bI(k, c)
    y = model(bI,k,c)
    plt.plot(t, y[:,4], label='(%s, %s) Symptomatic' % (k, c))
    plt.plot(t, y[:,2], label='(%s, %s) Asymptomatic' % (k, c))


def R0_derivs(k, c, bI=False):
    if not bI:
        bI = get_optimal_bI(k, c)
    ODEmodel(y0, t, bI, k, c) # calculates the three r0 values
    return f"R0: {R0}\nR0_c: {R0_deriv_c}\nR0_k: {R0_deriv_k}"


def cost_heatmap():
    #first, we curve fit at all k,c values then return the cost function
    total_list = []
    for k in np.arange(.1,1,step=.1):
        sublist = []
        for c in np.arange(1, 5.5, step=.5):
            total = get_curve_fit(k, c).cost
            sublist.append((total))
        total_list.append(sublist)
    total_array = np.array(total_list)

    #then we plot the heatmap from that array
    f4 = plt.figure(4)
    heat_map = sb.heatmap(total_array, annot=True, fmt='.5g',
      yticklabels = [round(x,1) for x in np.arange(.1,1,step=.1)],
      xticklabels = np.arange(1, 5.5, step=.5),
      cmap='BuPu', cbar_kws = {'label': 'Cost Function'})
    plt.xlabel('C, where βₐ = c*βᵢ')
    plt.ylabel(r'K, Percentage of Cases Asymptomatic')
    heat_map.invert_yaxis()

# you always need to globally define the dataset
conf_incidence = define_dataset(2, 21)
op = get_curve_fit(.9, 5)
print(f'BI: {op.x}, Cost: {op.cost}, Status: {op.status}')

# plt.legend()
# plt.show()