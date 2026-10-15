"""Run the Shyue-type or Lee-type JWL Riemann test case.

Usage:
    python run_case.py shyue
    python run_case.py lee
"""

from jwl_riemann_solver import *

def run_case(case_name="shyue", output_dir="."):

    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    cases = {
        "shyue": {
            "label": "Shyue-type", "eos": {"A": 8.545, "B": 0.205, "R1": 4.6, "R2": 1.35, "Gamma": 0.25, "rho0": 1.84},
            "rhoL": 1.7, "uL": 0.0, "pL": 10.0, "rhoR": 1.0, "uR": 0.0, "pR": 0.5, "t": 12.0
        },
        "lee": {
            "label": "Lee-type", "eos": {"A": 632.1, "B": -0.04472, "R1": 11.3, "R2": 1.13, "Gamma": 0.8938, "rho0": 1.905},
            "rhoL": 0.9525, "uL": 0.0, "pL": 1.0, "rhoR": 3.810, "uR": 0.0, "pR": 2.0, "t": 20.0
        }
    }
    case_name = case_name.lower()
    if case_name not in cases:
        raise ValueError("case_name must be 'shyue' or 'lee'")
    cfg = cases[case_name]
    JWL = cfg["eos"]
    rhoL, uL, pL = cfg["rhoL"], cfg["uL"], cfg["pL"]
    rhoR, uR, pR = cfg["rhoR"], cfg["uR"], cfg["pR"]
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    print("\n====================================")
    print(f"{cfg['label'].upper()} TEST")
    print("====================================")

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------

    solution = exact_riemann_solver(

        rhoL,
        uL,
        pL,

        rhoR,
        uR,
        pR,

        JWL
    )

    # ========================================================
    # CONVERGENCE TABLE
    # ========================================================

    print("\n========================================")
    print("CONVERGENCE ANALYSIS")
    print("========================================")

    print(f"{'Iter':<10}{'p_star':<20}{'Pressure Error':<20}{'Velocity Error'}")

    for i in range(len(solution["iteration_history"])):

        print(
            f"{solution['iteration_history'][i]:<10}"
            f"{solution['pressure_history'][i]:<20.12e}"
            f"{solution['pressure_error_history'][i]:<20.12e}"
            f"{solution['velocity_error_history'][i]:.12e}"
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n====================================")
    print("FINAL STAR SOLUTION")
    print("====================================")

    print(f"p_star      = {solution['p_star']}")

    print(f"u_star      = {solution['u_star']}")

    print(f"rhoL_star   = {solution['rhoL_star']}")

    print(f"rhoR_star   = {solution['rhoR_star']}")

    print(f"left_wave   = {solution['left_wave']}")

    print(f"right_wave  = {solution['right_wave']}")

    # ========================================================
    # EXPORT STAR STATE DATA
    # ========================================================

    star_state_data = pd.DataFrame({

        "p_star"      : [solution["p_star"]],

        "u_star"      : [solution["u_star"]],

        "rhoL_star"   : [solution["rhoL_star"]],

        "rhoR_star"   : [solution["rhoR_star"]],

        "left_wave"   : [solution["left_wave"]],

        "right_wave"  : [solution["right_wave"]]

    })

    star_state_data.to_csv(

        "star_state.csv",

        index=False
    )

    print("Star-state data exported.")

    print("\n====================================")
    print("FULL SOLVER COMPLETE")
    print("====================================")
    
    # ============================================================
    # HIGH-RESOLUTION PROFILE GENERATION
    # ============================================================

    import matplotlib.pyplot as plt


    print("\n====================================")
    print("GENERATING HIGH-RESOLUTION PROFILES")
    print("====================================")

    # ------------------------------------------------------------
    # SPATIAL DOMAIN
    # ------------------------------------------------------------

    x_values = np.linspace(0.0,100.0, 2000)

    # ------------------------------------------------------------
    # OUTPUT TIME
    # ------------------------------------------------------------

    t = cfg["t"]

    # ------------------------------------------------------------
    # STORAGE ARRAYS
    # ------------------------------------------------------------

    rho_values = []

    u_values = []

    p_values = []

    e_values = []

    # ------------------------------------------------------------
    # SAMPLE COMPLETE SOLUTION
    # ------------------------------------------------------------

    for x in x_values:

        rho, u, p, e = sample_solution(

            x,
            t,
            50.0,

            solution,

            rhoL,
            uL,
            pL,

            rhoR,
            uR,
            pR,

            JWL
        )

        rho_values.append(rho)

        u_values.append(u)

        p_values.append(p)

        e_values.append(e)

    # ------------------------------------------------------------
    # CONVERT TO NUMPY
    # ------------------------------------------------------------

    rho_values = np.array(rho_values)

    u_values = np.array(u_values)

    p_values = np.array(p_values)

    e_values = np.array(e_values)

    print("Profile generation complete.")

    

    # ========================================================
    # EXPORT SOLUTION PROFILE DATA
    # ========================================================

    

    solution_data = pd.DataFrame({

        "x"        : x_values,

        "density"  : rho_values,

        "velocity" : u_values,

        "pressure" : p_values,

        "energy"   : e_values
    })

    solution_data.to_csv(

        "solution_profiles.csv",

        index=False
    )

    print("Solution profile data exported.")

    # ============================================================
    # DENSITY PLOT
    # ============================================================

    plt.figure(figsize=(8,5))

    plt.plot(x_values, rho_values)

    plt.xlabel("x")

    plt.ylabel(r"Density $\rho$ [g/cm$^3$]")

    plt.title("Density Profile")

    plt.grid(True)

    plt.savefig("density_profile.png", dpi=300, bbox_inches="tight")

    plt.show()


    # ============================================================
    # VELOCITY PLOT
    # ============================================================

    plt.figure(figsize=(8,5))

    plt.plot(x_values, u_values)

    plt.xlabel("x")

    plt.ylabel(r"Velocity $u$ [cm/$\mu$s]")

    plt.title("Velocity Profile")

    plt.grid(True)
    
    plt.savefig("velocity_profile.png", dpi=300, bbox_inches="tight")

    plt.show()

 
    # ============================================================
    # PRESSURE PLOT
    # ============================================================

    plt.figure(figsize=(8,5))

    plt.plot(x_values, p_values)

    plt.xlabel("x")
 
    plt.ylabel(r"Pressure $p$ [Mbar]")

    plt.title("Pressure Profile")

    plt.grid(True)
    
    plt.savefig("pressure_profile.png", dpi=300, bbox_inches="tight")

    plt.show()


    # ============================================================
    # INTERNAL ENERGY PLOT
    # ============================================================

    plt.figure(figsize=(8,5))

    plt.plot(x_values, e_values)

    plt.xlabel("x")

    plt.ylabel("Internal Energy")

    plt.title("Internal Energy Profile")

    plt.grid(True)

    plt.savefig("energy_profile.png", dpi=300, bbox_inches="tight")

    plt.show()

    # ========================================================
    # CONVERGENCE PLOTS
    # ========================================================



    # --------------------------------------------------------
    # PRESSURE ERROR
    # --------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.semilogy(
        solution["iteration_history"],
        solution["pressure_error_history"],
        marker='o',
        linewidth=2
   )

    plt.xlabel("Iteration")

    plt.ylabel("Pressure Error")

    plt.title("Pressure Convergence History")

    plt.grid(True)

    plt.savefig("pressure_convergence.png", dpi=300)

    plt.show()

    # --------------------------------------------------------
    # VELOCITY ERROR
    # --------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.semilogy(
        solution["iteration_history"],
        solution["velocity_error_history"],
        marker='o',
        linewidth=2
    )

    plt.xlabel("Iteration")

    plt.ylabel("Velocity Error")

    plt.title("Velocity Convergence History")

    plt.grid(True)

    plt.savefig("velocity_convergence.png", dpi=300)

    plt.show()

    
    # ========================================================
    # EXPORT CONVERGENCE HISTORY
    # ========================================================

    convergence_data = pd.DataFrame({

        "iteration"       : solution["iteration_history"],

        "p_star"          : solution["pressure_history"],

        "pressure_error"  : solution["pressure_error_history"],

        "velocity_error"  : solution["velocity_error_history"]
    })

    convergence_data.to_csv(

        "convergence_history.csv",

        index=False
    )

    print("Convergence history exported.")


if __name__ == "__main__":
    import sys
    run_case(sys.argv[1] if len(sys.argv) > 1 else "shyue", sys.argv[2] if len(sys.argv) > 2 else ".")
