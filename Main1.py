import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import brentq


P_ATM = 101325
gamma = 1.4

with open('data_dict.pkl', 'rb') as f:
    loaded_data_dict = pickle.load(f)


tap_locations = {
    -2: {"x_mm": 0, "y_mm": 3.45, "h_mm": 26.55, "A_Astar": 1.58},
    -1: {"x_mm": 11.6, "y_mm": 8.86, "h_mm": 21.14, "A_Astar": 1.26},
    1: {"x_mm": 24.7, "y_mm": 12.17, "h_mm": 17.83, "A_Astar": 1.06},
    2: {"x_mm": 36.6, "y_mm": 13.21, "h_mm": 16.79, "A_Astar": 1.00},
    3: {"x_mm": 48.3, "y_mm": 12.37, "h_mm": 17.63, "A_Astar": 1.05},
    4: {"x_mm": 61.0, "y_mm": 10.69, "h_mm": 19.31, "A_Astar": 1.15},
    5: {"x_mm": 73.7, "y_mm": 9.40, "h_mm": 20.60, "A_Astar": 1.23},
    6: {"x_mm": 86.4, "y_mm": 8.64, "h_mm": 21.36, "A_Astar": 1.27},
    7: {"x_mm": 99.1, "y_mm": 8.33, "h_mm": 21.67, "A_Astar": 1.28},
    8: {"x_mm": 118.0, "y_mm": 8.18, "h_mm": 21.82, "A_Astar": 1.30},
    9: {"x_mm": 209.5, "y_mm": 8.18, "h_mm": 21.82, "A_Astar": 1.30},
    10: {"x_mm": 300.1, "y_mm": 8.18, "h_mm": 21.82, "A_Astar": 1.30},
    11: {"x_mm": "diffuser", "y_mm": None, "h_mm": None, "A_Astar": None}
}

tap_column_mapping = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 4,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 11,
    11: 10
}


def ComputeFreestreamM(p_total, p_static):
    gamma = 1.4

    term = (p_total / p_static) ** ((gamma - 1) / gamma) - 1

    mach_squared = (2 / (gamma - 1)) * term
 
    mach = math.sqrt(mach_squared)
    
    return mach


def ComputeVBernoulli(p_total, p_static):
    return math.sqrt(2 * (p_total - p_static) / 1.225)


def MapTapNumToColumn(tap_num):
    return tap_column_mapping[tap_num]


def ComputeTheoreticalMachNumber(tap_num):
    A_Astar = tap_locations[tap_num]["A_Astar"]

    def equation(M):
        return (
            (1/M) * ((gamma + 1) / 2) ** (-(gamma + 1) / (2 * (gamma - 1))) *
            ((1 + ((gamma - 1) / 2) * M**2) ) ** ((gamma + 1) / (2 * (gamma - 1))) - A_Astar
        )

    subsonic_solution = fsolve(equation, 0.5)[0]
    supersonic_solution = fsolve(equation, 2.0)[0]

    return subsonic_solution, supersonic_solution



supersonic_M = []
supersonic_theory_M = []
subsonic_M = []
subsonic_theory_M = []
subsonic_M_incomp = []

for key in loaded_data_dict:
    data = loaded_data_dict[key]

    currentLocation = data["currentlocation"]

    currentLocationStaticPavg = np.array(data["dataP"]).mean(axis=0)[MapTapNumToColumn(currentLocation) - 1] + P_ATM

    currentLocationTotalPavg = np.array(data["dataP"]).mean(axis=0)[12 - 1] + P_ATM


    M_infinity = ComputeFreestreamM(currentLocationTotalPavg, currentLocationStaticPavg)

    V_infinity_bernoulli = ComputeVBernoulli(currentLocationTotalPavg, currentLocationStaticPavg)

    a_inf = math.sqrt(1.4 * 287 * 300)

    subsonic_M_theory, supersonic_M_theory = ComputeTheoreticalMachNumber(currentLocation)

    if key.startswith("subsonic"):
        subsonic_M.append(M_infinity)
        subsonic_M_incomp.append(V_infinity_bernoulli / a_inf)
        subsonic_theory_M.append(subsonic_M_theory)

    if key.startswith("supersonic"):
        supersonic_M.append(M_infinity)
        supersonic_theory_M.append(supersonic_M_theory)


x_positions = [tap_locations[i]["x_mm"] for i in range(1, 8)]


df = pd.read_csv("AER303 Lab 2 Uncertainties.csv")

mach_uncerts_subsonic = []
mach_uncerts_supersonic = []

for i in range(1, 8):
    mach_uncerts_subsonic.append(float(df.iloc[i, 4]))
    mach_uncerts_supersonic.append(float(df.iloc[i + 15, 4]))


def PlotWindTunnelCrossSection(ax):
    def evaluate_delavel_lower(x):
        a, b, c, d, f, g, h, i, j, k, l = -22.049311, 2.8329489, -0.32729524, 0.026977414, -0.0014090913, \
                                           0.000046581106, -9.8618389e-7, 1.3246316e-8, -1.0818611e-10, \
                                           4.8369577e-13, -8.8856732e-16
        return (a + b*x + c*x**2 + d*x**3 + f*x**4 + g*x**5 + h*x**6 + i*x**7 +
                j*x**8 + k*x**9 + l*x**10)

    def evaluate_delavel_upper(x):
        a2, b2, c2, d2, f2 = 21.5, -2.86667, 0.420833, -0.0333333, 0.00104167
        return a2 + b2 * x + c2 * x**2 + d2 * x**3 + f2 * x**4

    x_vals_lower = np.linspace(2, 76, 1000)
    y_vals_lower = evaluate_delavel_lower(x_vals_lower)
    x_vals_upper = np.linspace(1.6, 10, 1000)
    y_vals_upper = evaluate_delavel_upper(x_vals_upper)

    generic_lines = [
        {"y": -11.5, "x_min": 76, "x_max": 140},
        {"y": 12, "x_min": 10, "x_max": 140},
        {"y": -17.5, "x_min": 2, "x_max": 140}
    ]
    vertical_line = {"x": 140, "y_min": -17.5, "y_max": -11.5}

    tap_x_values = [24.7, 36.4, 48, 60, 72.6, 85, 96.7, 114.8]
    tap_y_values = [12] * len(tap_x_values)
    tap_labels = [f"tap {i+1}" for i in range(len(tap_x_values))]

    ax.plot(x_vals_lower, y_vals_lower, color="blue", linestyle="solid", linewidth=2)
    ax.plot(x_vals_upper, y_vals_upper, color="blue", linestyle="solid", linewidth=2)

    for line in generic_lines:
        ax.hlines(y=line["y"], xmin=line["x_min"], xmax=line["x_max"],
                  colors="blue", linestyles="solid", linewidth=2)

    ax.vlines(x=vertical_line["x"], ymin=vertical_line["y_min"], ymax=vertical_line["y_max"],
              colors="blue", linestyles="solid", linewidth=2)

    ax.scatter(tap_x_values, tap_y_values, color="red")
    for i, label in enumerate(tap_labels):
        ax.text(tap_x_values[i], tap_y_values[i] + 0.5, label, color="black", ha="center")

    ax.set_title("Wind Tunnel Side Cross Section")
    ax.set_xlabel("x")
    ax.set_ylabel("y")



def PlotCombinedFigure(x_positions, subsonic_M_incomp, mach_uncerts_subsonic, supersonic_M, mach_uncerts_supersonic):
    x_min = min(min(x_positions), 0)
    x_max = 140 


    def save_plot(data_label, y_data, y_uncerts, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        if y_data is not None:
            ax1.errorbar(x_positions, y_data, yerr=y_uncerts, label=data_label, fmt='-o', capsize=8, ecolor='black')
            ax1.set_title(f"Freestream Mach Number with Uncertainties ({data_label})")

            if data_label == "Supersonic":
                ax1.plot(x_positions, supersonic_theory_M, label="Theoretical Mach Number", color="red", linestyle="--")

            if data_label == "Subsonic":
                ax1.plot(x_positions, subsonic_theory_M, label="Theoretical Mach Number", color="red", linestyle="--")

            ax1.legend()

        else:
            ax1.errorbar(x_positions, subsonic_M_incomp, yerr=mach_uncerts_subsonic, label="Subsonic", fmt='-o', capsize=8, ecolor='black')
            ax1.errorbar(x_positions, supersonic_M, yerr=mach_uncerts_supersonic, label="Supersonic", fmt='-o', capsize=8, ecolor='black')
            ax1.set_title("Freestream Mach Number with Uncertainties (Combined)")
            ax1.legend()

        ax1.set_xlabel("X Position (mm)")
        ax1.set_ylabel("Freestream Mach Number")
        ax1.set_xlim(x_min, x_max)

        PlotWindTunnelCrossSection(ax2)
        ax2.set_xlim(x_min, x_max)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    save_plot("Subsonic", subsonic_M_incomp, mach_uncerts_subsonic, "graphs/subsonic_plot.png")
    save_plot("Supersonic", supersonic_M, mach_uncerts_supersonic, "graphs/supersonic_plot.png")
    save_plot("Combined", None, None, "graphs/combined_plot.png")

def PlotTunnelGeometryAlone():
    fig, ax = plt.subplots(figsize=(10, 6))
    PlotWindTunnelCrossSection(ax)
    plt.tight_layout()
    plt.savefig("graphs/tunnel_geometry.png")
    plt.close(fig)


PlotCombinedFigure(x_positions, subsonic_M_incomp, mach_uncerts_subsonic, supersonic_M, mach_uncerts_supersonic)
PlotTunnelGeometryAlone()