import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

new_cases = [3, 8, 28, 75, 221, 291, 255, 235, 190, 125, 70, 28, 12, 5]
t = np.arange(1,15)
plt.plot(t,new_cases, label='Actual',color='y')

def model(vals, t, beta, gamma):
    S = vals[0]
    I = vals[1]
    R = vals[2]
    N = S+I+R
    dS = (-beta*S*I)/N
    dI = ((beta*S*I)/N) - gamma*I
    dR = gamma*I
    return [dS, dI, dR]

#least square regression
from scipy.optimize import curve_fit

def regression_model(time,beta,gamma):
    result = odeint(model,[762,1,0],time,(beta,gamma))[:,1]
    return result

ovals = curve_fit(regression_model, t, np.array(new_cases)) #returns an array of the optimal parameters
print(ovals)

optimum = odeint(model,[762,1,0],t,(ovals[0][0],ovals[0][1]))[:,1]
plt.plot(t,optimum,label='Predicted',linestyle=':',color='r')

plt.legend()
plt.show()