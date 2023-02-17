import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import csv
import pandas as pd
from scipy.optimize import curve_fit


def linear(X, A, B):
    return A*X + B

def inc_prod(total, x, y, x_inc, y_inc):
    return total*np.sqrt((x_inc/x)**2 + (y_inc/y)**2)

def inc_add(x_inc, y_inc):
    return np.sqrt(x_inc**2 + y_inc**2)



# name of the csv file
data = pd.read_csv("cadmium_red.csv", sep=";")
data_inc = pd.read_csv("cadmium_red_inc.csv", sep=";")

def pre_analysis(index, distance=5.5):
    df = data.iloc[index]
    df = df[1:]
    df_inc = data_inc.iloc[index]

    L_data = []
    L_data_inc = []
    for i in range(len(df)):
        L_data.append(df[i])
        L_data_inc.append(df_inc[i])


    L_data = [x for x in L_data if not pd.isnull(x)]
    L_data_inc = [3*x for x in L_data_inc if not pd.isnull(x)]
    # printing the new list


    L_plot = []
    L_inc = []
    for i in range(int(len(L_data)/3)):
        Rlamb1 = (L_data[3*i])
        Rlamb = (L_data[3*i+1])
        Rlamb2 = (L_data[3*i+2])

        L_plot.append(Rlamb**2/(Rlamb2**2 - Rlamb1**2))
        R_1_inc = inc_prod(Rlamb1*Rlamb1, Rlamb1, Rlamb1, L_data_inc[3*i], L_data_inc[3*i])
        R_2_inc = inc_prod(Rlamb*Rlamb, Rlamb, Rlamb, L_data_inc[3*i+1], L_data_inc[3*i+1])
        R_3_inc = inc_prod(Rlamb2*Rlamb2, Rlamb2, Rlamb2, L_data_inc[3*i+2], L_data_inc[3*i+2])

        R_sum = inc_add(R_3_inc, R_1_inc)
        R_inc = inc_prod(Rlamb**2/(Rlamb2**2 - Rlamb1**2), Rlamb**2, Rlamb2**2 - Rlamb1**2, R_2_inc, R_sum)
        L_inc.append(R_inc)


    popt, pcov = curve_fit(linear, np.linspace(1, len(L_plot), len(L_plot)), L_plot, sigma=L_inc)


    delta = popt[0]
    delta_inc = np.sqrt(np.diag(pcov))
    delta_inc_inv = delta_inc[0]/delta**2


    delta = (643.847*10**(-9))**2/(delta*distance*4*10**(-3))
    delta_inc = 2*delta_inc_inv*(643.847*10**(-9))**2/(distance*4*10**(-3))
    return delta, delta_inc

def main_analysis():
    L_delta = []
    L_delta_inc = []
    for i in range(len(data["lambda0"])):
        delta, delta_inc = pre_analysis(i)
        L_delta_inc.append(delta_inc)
        L_delta.append(delta)

    print(L_delta)
    print(L_delta_inc)
    popt, pcov = curve_fit(linear, data["B (T)"], L_delta, sigma=L_delta_inc)
    B_inc = np.ones(len(data["B (T)"]))*0.001
    plt.errorbar(data["B (T)"], L_delta, xerr=B_inc, yerr=L_delta_inc, fmt=".", color="r", ecolor="black", capsize=2, linewidth=1, markersize=4)
    plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "--")
    plt.plot(data["B (T)"], linear(data["B (T)"], popt[0], popt[1]), linewidth=0.7, color="r")
    plt.title("Cadmium raie 643nm")
    plt.xlabel("B (T)")
    plt.ylabel("R^2/(R2^2-R1^2)")

    print((popt[0]*3*10**8 * 6.62*10**(-34))/((643.847*10**(-9))**2))
    delta_inc = np.sqrt(np.diag(pcov))
    print((delta_inc[0]*3*10**8 * 6.62*10**(-34))/((643.847*10**(-9))**2))


main_analysis()
#print(((delta*4*np.pi*3*10**8)/(0.4184*(643.847*10**(-9))**2))*(10**(-34)))


# plt.errorbar(np.linspace(1, len(L_plot), len(L_plot)), L_plot, yerr=L_inc, fmt=".")
# plt.plot(np.linspace(1, len(L_plot), len(L_plot)), linear(np.linspace(1, len(L_plot), len(L_plot)), popt[0], popt[1]), linewidth=1)
plt.show()