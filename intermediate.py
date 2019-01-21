# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:42:21 2016

@author: Pedro Leal
"""

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import numpy as np

from filehandling import output_reader


def tangent_error(x, T_1=None, T_2=None):
    T_2 = x[0]
    T_3 = x[0] + x[1]
    strain_1 = x[2]
    strain_2 = x[3]
    strain_3 = x[4]
    strain_4 = x[5]

    f = []
    for T_i in T:
        f_i = tangent_lines(T_i, T_1, T_2, T_3, T_4, strain_1, strain_2,
                            strain_3, strain_4)
        f.append(f_i)

    f = np.array(f)
    strain_np = np.array(strain)

    rmse = np.sqrt(np.sum((f-strain_np)**2)/len(strain))

    return rmse


def tangent_lines(T, props):
    [T_1, T_2, T_3, T_4, strain_1, strain_2, strain_3, strain_4] = props
    if T < T_2:
        return (strain_2 - strain_1)/(T_2 - T_1)*(T-T_1) + strain_1
    elif T < T_3:
        return (strain_3 - strain_2)/(T_3 - T_2)*(T-T_2) + strain_2
    else:
        return (strain_4 - strain_3)/(T_4 - T_3)*(T - T_4) + strain_4


def fitting(raw_data, transformation='Austenite', error=tangent_error,
            optimizer='differential_evolution'):

    if transformation == 'Austenite':
        T_1, T_4 = raw_data[0][0], raw_data[0][-1]
    elif transformation == 'Martensite':
        T_4, T_1 = raw_data[0][0], raw_data[0][-1]

    bounds = [(30, 120), (10, 60)] + 4*[min(raw_data[1]), max(raw_data[1]), ]
    x0 = [(x[0]+x[1])/2. for x in bounds]

    if optimizer == 'BFGS':
        result = minimize(error, x0, method='BFGS')
    elif optimizer == 'differential_evolution':
        result = differential_evolution(error, bounds, popsize=100,
                                        maxiter=100)

    strains = result.x[2:6]
    temperatures = [T_1, result.x[0], result.x[0] + result.x[1], T_4]
    return(np.vstack(temperatures, strains))


def processing_raw(filename):
    raw_data = output_reader(filename, header=['Temperature', "Strain",
                                               "Stress"],)
    temperature = raw_data['Temperature']
    strain = raw_data['Strain']
    stress = raw_data['Stress']

    i = temperature.index(max(temperature))

    props_austenite = np.vstack(temperature[:i+1], strain[:i+1], stress[:i+1])
    props_martensite = np.vstack(temperature[i:], strain[i:], stress[i:])

    return({'Austenite': props_austenite, 'Martensite': props_martensite})


def tangent_plotting(raw_data, calibrated_props):
    T = calibrated_props[0]
    f = []
    for i in range(len(calibrated_props[0])):
        f.append(tangent_lines(calibrated_props[0][i], calibrated_props))
    f = np.array(f)
    plt.plot(T, f, 'b', label="Tangents")
    T, epsilon = calibrated_props.T
    plt.plot(T, epsilon, 'g', label="Experimental data")


optimizer = 'differential_evolution'
raw_data = processing_raw("filtered_data_50MPa.txt")
for transformation in ['Austenite', 'Martensite']:
    calibrated_props = fitting(raw_data[transformation], transformation)
    tangent_plotting(raw_data['Austenite'], calibrated_props)

plt.grid()
plt.xlim(min(raw_data[0]), max(raw_data[0]))
plt.ylim(min(raw_data[1]), max(raw_data[1]))
plt.legend(loc="lower left")
plt.xlabel("Temperature (C)")
plt.ylabel("Strain (m/m)")
plt.show()
