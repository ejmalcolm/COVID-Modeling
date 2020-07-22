# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:53:13 2020

@author: mrobert
"""

from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt


# Basic Polynomial Function
def  model(coef,t):
    return (coef[2]*t**2 + coef[1]*t + coef[0])

# base model with data
def  residuals(coef, t, data):
    return ((coef[2]*t**2 + coef[1]*t + coef[0])-data)

# Fake Data 
data = [1, 10, 20, 19, 12, 18, 22, 30, 2]
t = np.linspace(0,len(data),num=len(data))


#  Practice Optimizing
C0 = [1, 1, 1]
op = least_squares(residuals, C0, args = (t,data))
# Note the function takes in the model, initial guesses, and args takes in the
# remaining arguments of the function model


# Run base Model with fitted parameters
out = model(op.x, t)

plt.plot(t, out)
plt.plot(t,data, 'ro')
plt.show()

# Show Cost Function
print('cost value:', op.cost)
# Show status: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
# for status meanings.
print('status:', op.status)

#######################################################################
# Try again with extra parameters not estimated!

# base model with data
def  model3(coef, t, data, M):
    return (M[0]**M[1]*(coef[2]*t**2 + coef[1]*t + coef[0])-data)

# base model without data for plotting
def  model3b(coef, t, M):
    return (M[0]**M[1]*(coef[2]*t**2 + coef[1]*t + coef[0]))

# Run Optimizer for given M value
M = [2, 3]
op = least_squares(model3, C0, args = (t,data,M))

# Run base Model with fitted parameters
out = model3b(op.x, t, M)

# Plot
fig = plt.figure()
plt.plot(t, out)
plt.plot(t, data, 'ro')
plt.ylabel('Data')
plt.xlabel('Time')
plt.show()

# Show Cost Function
print('cost value:', op.cost)
# Show status: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
# for status meanings.
print('status:', op.status)