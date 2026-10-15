"""Exact 1-D Riemann solver for compressible flow with a JWL equation of state.

Extracted from the author's Shyue-type and Lee-type test notebooks. The numerical
solver functions are kept unchanged in this initial refactor.
"""


# FULL EXACT RIEMANN SOLVER
# FOR COMPRESSIBLE FLOW WITH JWL EOS
# Adarsh Kumar Sah
#AE25M803
# Shyu Type test


import numpy as np




# JWL EOS FUNCTIONS


def jwl_f_rho(rho, eos):

    rho = max(rho, 1.0e-12)

    A      = eos["A"]
    B      = eos["B"]
    R1     = eos["R1"]
    R2     = eos["R2"]
    Gamma  = eos["Gamma"]
    rho0   = eos["rho0"]

    term1 = (
        A
        *
        (
            1.0
            -
            (Gamma*rho)/(R1*rho0)
        )
        *
        np.exp(-R1*rho0/rho)
    )

    term2 = (
        B
        *
        (
            1.0
            -
            (Gamma*rho)/(R2*rho0)
        )
        *
        np.exp(-R2*rho0/rho)
    )

    return term1 + term2




def jwl_df_drho(rho, eos):

    h = 1.0e-6 * max(rho, 1.0)

    fp = jwl_f_rho(rho+h, eos)

    fm = jwl_f_rho(rho-h, eos)

    return (fp-fm)/(2.0*h)




def jwl_pressure(rho, e, eos):

    rho = max(rho, 1.0e-12)

    Gamma = eos["Gamma"]

    return (
        jwl_f_rho(rho, eos)
        +
        Gamma*rho*e
    )




def jwl_energy(rho, p, eos):

    rho = max(rho, 1.0e-12)

    Gamma = eos["Gamma"]

    return (
        p
        -
        jwl_f_rho(rho, eos)
    ) / (Gamma*rho)



def jwl_sound_speed(rho, p, eos):

    rho = max(rho, 1.0e-12)

    Gamma = eos["Gamma"]

    f = jwl_f_rho(rho, eos)

    dfdrho = jwl_df_drho(rho, eos)

    a2 = (
        ((Gamma + 1.0) * p) / rho
        -
        (f / rho)
        +
        dfdrho
    )

    if a2 <= 0.0:
        raise ValueError(
            f"Negative sound speed squared: a2={a2}, rho={rho}, p={p}"
        )

    return np.sqrt(a2)



# ACOUSTIC INITIALIZATION

def acoustic_pressure_guess(rhoL,
                            uL,
                            pL,

                            rhoR,
                            uR,
                            pR,

                            eos):

    aL = jwl_sound_speed(rhoL, pL, eos)

    aR = jwl_sound_speed(rhoR, pR, eos)

    ZL = rhoL * aL

    ZR = rhoR * aR

    p_star = (

        ZR*pL

        +

        ZL*pR

        +

        ZL*ZR*(uL-uR)

    ) / (ZL+ZR)

    p_star = max(p_star, 1.0e-8)

    return p_star



# SHOCK DETECTION


def is_shock(p_star, p):

    return p_star > p



# SHOCK RESIDUAL



def shock_jump_residual(rho_star,
                        rho,
                        p,
                        p_star,
                        eos):

    rho_star = max(rho_star, 1.0e-12)

    e_initial = jwl_energy(rho, p, eos)

    e_star = jwl_energy(rho_star, p_star, eos)

    delta_p = p_star - p

    denominator = rho_star - rho

    denominator = np.sign(denominator) * max(
        abs(denominator),
        1.0e-12
    )

    residual = (

        e_initial

        +

        (p/rho)

        +

        0.5
        *
        (rho_star/rho)
        *
        (delta_p/denominator)

        -

        (

            e_star

            +

            (p_star/rho_star)

            +

            0.5
            *
            (rho/rho_star)
            *
            (delta_p/denominator)
        )
    )

    return residual



# POST-SHOCK DENSITY SOLVER

def solve_post_shock_density(rho,
                             p,
                             p_star,
                             eos,
                             tolerance=1.0e-12,
                             max_iterations=100):

    rho1 = 1.1*rho

    rho2 = 1.2*rho

    for iteration in range(max_iterations):

        f1 = shock_jump_residual(

            rho1,
            rho,
            p,
            p_star,
            eos
        )

        f2 = shock_jump_residual(

            rho2,
            rho,
            p,
            p_star,
            eos
        )

        if abs(f2) < tolerance:

            return rho2

        if abs(rho2-rho1) < 1.0e-12:

            return rho2

        denominator = f2-f1

        if abs(denominator) < 1.0e-14:

            return rho2

        rho_new = (

            rho2

            -

            f2
            *
            (
                (rho2-rho1)
                /
                denominator
            )
        )

        rho_new = max(rho_new, 1.0e-8)

        rho1 = rho2

        rho2 = rho_new

    raise RuntimeError(
        "Shock density solve failed."
    )



# SHOCK SPEED

def shock_speed_function(rho,
                      u,
                      p,

                      rho_star,
                      p_star,

                      side):

    term = np.sqrt(

        (rho_star/rho)

        *

        (
            (p_star-p)
            /
            (rho_star-rho)
        )
    )

    if side == "left":

        S = u - term

    else:

        S = u + term

    return S



#STAR VELOCITY

def paper_star_velocity(rho,
                        p,

                        rho_star,
                        p_star,

                        shock_speed,

                        side):

    term = np.sqrt(

        (rho/rho_star)

        *

        (
            (p_star-p)
            /
            (rho_star-rho)
        )
    )

    if side == "left":

        u_star = shock_speed + term

    else:

        u_star = shock_speed - term

    return u_star



# COMPLETE SHOCK SOLVER

def shock_solver(rho,
                 u,
                 p,

                 p_star,

                 eos,
                 side):

    rho_star = solve_post_shock_density(

        rho,
        p,
        p_star,
        eos
    )

    shock_speed = shock_speed_function(

        rho,
        u,
        p,

        rho_star,
        p_star,

        side
    )

    u_star = paper_star_velocity(

        rho,
        p,

        rho_star,
        p_star,

        shock_speed,

        side
    )

    return rho_star, u_star, shock_speed



# RK4 RAREFACTION STEP


def rk4_rarefaction_step(rho1,
                         p1,
                         u1,

                         delta_rho,

                         eos,
                         side):

    sign = -1.0 if side == "left" else 1.0

    

    a1 = jwl_sound_speed(rho1, p1, eos)

    p2 = p1 + 0.5*delta_rho*(a1**2)

    rho2 = max(rho1 + 0.5*delta_rho, 1.0e-12)

    a2 = jwl_sound_speed(rho2, p2, eos)

    p3 = p1 + 0.5*delta_rho*(a2**2)

    rho3 = max(rho1 + 0.5*delta_rho, 1.0e-12)

    a3 = jwl_sound_speed(rho3, p3, eos)

    p4 = p1 + delta_rho*(a3**2)

    rho4 = max(rho1 + delta_rho, 1.0e-12)

    a4 = jwl_sound_speed(rho4, p4, eos)

    
    # PRESSURE UPDATE
    

    p_new = (

        p1

        +

        (delta_rho/6.0)

        *

        (
            (a1**2)

            +

            2.0*(a2**2)

            +

            2.0*(a3**2)

            +

            (a4**2)
        )
    )

    
    # VELOCITY UPDATE
    

    u_new = (

        u1

        +

        sign
        *
        (delta_rho/6.0)

        *

        (

            (a1/rho1)

            +

            2.0*(a2/rho2)

            +

            2.0*(a3/rho3)

            +

            (a4/rho4)
        )
    )

    rho_new = max(rho1 + delta_rho, 1.0e-12)

    return rho_new, p_new, u_new



# INTEGRATE RAREFACTION


def integrate_rarefaction(rho_initial,
                          p_initial,
                          u_initial,

                          delta_rho,

                          eos,
                          side,

                          M):

    rho = rho_initial

    p = p_initial

    u = u_initial

    for m in range(M):

        rho, p, u = rk4_rarefaction_step(

            rho,
            p,
            u,

            delta_rho,

            eos,
            side
        )

    return rho, p, u



# RAREFACTION RESIDUAL


def rarefaction_residual(delta_rho,

                         rho_initial,
                         p_initial,
                         u_initial,

                         p_star_target,

                         eos,
                         side,

                         M):

    rho_final, p_final, u_final = integrate_rarefaction(

        rho_initial,
        p_initial,
        u_initial,

        delta_rho,

        eos,
        side,

        M
    )

    residual = p_star_target - p_final

    return residual



# DELTA_RHO SOLVER


def solve_delta_rho(rho_initial,
                    p_initial,
                    u_initial,

                    p_star_target,

                    eos,
                    side,

                    M,

                    tolerance=1.0e-12,
                    max_iterations=100):

    if p_star_target >= p_initial:
        raise ValueError("Rarefaction requires p_star_target < p_initial")

    
    # R(delta_rho) = p_star_target - I(delta_rho)
    # where I(delta_rho) is the RK4-integrated final pressure.
    

    def residual(delta_rho):

        try:
            rho_final, p_final, u_final = integrate_rarefaction(
                rho_initial,
                p_initial,
                u_initial,
                delta_rho,
                eos,
                side,
                M
            )
        except ValueError:
            # If integration hits nonphysical state, this delta_rho
            # is too negative and produces pressure below admissible range.
            return 1.0e30

        return p_star_target - p_final

    
    # Bracket delta_rho.
    #
    # At delta_rho = 0:
    #     p_final = p_initial > p_star_target
    #     residual = p_star_target - p_initial < 0
    #
    # For sufficiently negative delta_rho:
    #     p_final becomes lower than p_star_target
    #     residual becomes positive
    

    delta_high = 0.0
    f_high = residual(delta_high)

    delta_low = -0.001 * rho_initial / M
    f_low = residual(delta_low)

    expansion_count = 0

    while f_low < 0.0 and expansion_count < 80:

        delta_low *= 2.0
        f_low = residual(delta_low)
        expansion_count += 1

    if f_low < 0.0:
        raise RuntimeError("Could not bracket delta_rho for rarefaction")

    
    # Bisection directly on delta_rho.
    # This solves R(delta_rho) = 0.
    

    for iteration in range(max_iterations):

        delta_mid = 0.5 * (delta_low + delta_high)
        f_mid = residual(delta_mid)

        if abs(f_mid) < tolerance:
            return delta_mid

        if abs(delta_high - delta_low) < tolerance:
            return delta_mid

        if f_mid > 0.0:
            # delta_mid is too negative; pressure fell too far.
            delta_low = delta_mid
        else:
            # pressure still too high; need more negative delta_rho.
            delta_high = delta_mid

    raise RuntimeError("Delta_rho solve failed")

# COMPLETE RAREFACTION SOLVER


def rarefaction_solver(rho_initial,
                       u_initial,
                       p_initial,

                       p_star_target,

                       eos,
                       side,

                       M=300):

    delta_rho = solve_delta_rho(

        rho_initial,
        p_initial,
        u_initial,

        p_star_target,

        eos,
        side,

        M
    )

    rho_star, p_star, u_star = integrate_rarefaction(

        rho_initial,
        p_initial,
        u_initial,

        delta_rho,

        eos,
        side,

        M
    )

    a_initial = jwl_sound_speed(

        rho_initial,
        p_initial,
        eos
    )

    a_star = jwl_sound_speed(

        rho_star,
        p_star,
        eos
    )

    if side == "left":

        head_speed = u_initial - a_initial

        tail_speed = u_star - a_star

    else:

        head_speed = u_initial + a_initial

        tail_speed = u_star + a_star

    return (

        rho_star,
        p_star,
        u_star,

        head_speed,
        tail_speed
    )




# VELOCITY RESIDUAL


def velocity_residual(p_star,

                      rhoL,
                      uL,
                      pL,

                      rhoR,
                      uR,
                      pR,

                      eos):

    
    # LEFT SIDE
    

    if is_shock(p_star, pL):

        rhoL_star, uL_star, SL = shock_solver(

            rhoL,
            uL,
            pL,

            p_star,

            eos,

            side="left"
        )

    else:

        (

            rhoL_star,
            pL_star,
            uL_star,

            SHL,
            STL

        ) = rarefaction_solver(

            rhoL,
            uL,
            pL,

            p_star,

            eos,

            side="left"
        )

    
    # RIGHT SIDE
    

    if is_shock(p_star, pR):

        rhoR_star, uR_star, SR = shock_solver(

            rhoR,
            uR,
            pR,

            p_star,

            eos,

            side="right"
        )

    else:

        (

            rhoR_star,
            pR_star,
            uR_star,

            SHR,
            STR

        ) = rarefaction_solver(

            rhoR,
            uR,
            pR,

            p_star,

            eos,

            side="right"
        )

    
    # RESIDUAL
    
    

    residual = uL_star - uR_star

    return (

        residual,

        uL_star,
        uR_star,

        rhoL_star,
        rhoR_star
    )



# IMPROVED SECOND PRESSURE GUESS


def improved_pressure_guess(p0,

                            rhoL,
                            uL,
                            pL,

                            rhoR,
                            uR,
                            pR,

                            eos):

    
    # COMPUTE STAR VELOCITIES
    

    (

        F0,

        uL_star0,
        uR_star0,

        rhoL_star0,
        rhoR_star0

    ) = velocity_residual(

        p0,

        rhoL,
        uL,
        pL,

        rhoR,
        uR,
        pR,

        eos
    )

   
    # LEFT MEAN LAGRANGIAN WAVESPEED
   

    if abs(uL_star0-uL) > 1.0e-14:

        CbarL = abs(

            (p0-pL)

            /

            (uL_star0-uL)
        )

    else:

        aL = jwl_sound_speed(
            rhoL,
            pL,
            eos
        )

        CbarL = rhoL*aL

   
    # RIGHT MEAN LAGRANGIAN WAVESPEED
   
    if abs(uR_star0-uR) > 1.0e-14:

        CbarR = abs(

            (p0-pR)

            /

            (uR_star0-uR)
        )

    else:

        aR = jwl_sound_speed(
            rhoR,
            pR,
            eos
        )

        CbarR = rhoR*aR

    
    # IMPROVED PRESSURE ESTIMATE
    

    p1 = (

        CbarR*pL

        +

        CbarL*pR

        +

        CbarL*CbarR*(uL-uR)

    ) / (CbarL+CbarR)

    p1 = max(p1, 1.0e-8)

    return p1



# FULL EXACT RIEMANN SOLVER


def exact_riemann_solver(rhoL,
                         uL,
                         pL,

                         rhoR,
                         uR,
                         pR,

                         eos,

                         tolerance=1.0e-12,
                         max_iterations=100):

    
    print("FULL EXACT RIEMANN SOLVER")
    

    print(f"Tolerance = {tolerance:.1e}")

   
    # CONVERGENCE STORAGE
    
    iteration_history = []

    pressure_history = []

    velocity_error_history = []

    pressure_error_history = []

    
    # INITIAL PRESSURE GUESS
   
    p0 = acoustic_pressure_guess(

        rhoL,
        uL,
        pL,

        rhoR,
        uR,
        pR,

        eos
    )

    
    # IMPROVED SECOND GUESS
    

    p1 = improved_pressure_guess(

        p0,

        rhoL,
        uL,
        pL,

        rhoR,
        uR,
        pR,

        eos
    )

    
    # INITIALIZE SECANT ITERATES
    
    p_old = p0

    p_new = p1

   
    # SECANT ITERATION
    

    for iteration in range(max_iterations):

        
        # EVALUATE OLD ITERATE
        

        (

            F_old,

            uL_old,
            uR_old,

            rhoL_old,
            rhoR_old

        ) = velocity_residual(

            p_old,

            rhoL,
            uL,
            pL,

            rhoR,
            uR,
            pR,

            eos
        )

       
        # EVALUATE NEW ITERATE
      
        (

            F_new,

            uL_new,
            uR_new,

            rhoL_new,
            rhoR_new

        ) = velocity_residual(

            p_new,

            rhoL,
            uL,
            pL,

            rhoR,
            uR,
            pR,

            eos
        )

      
        # PRINT ITERATION
        

        print(f"\nIteration = {iteration+1}")

        print(f"p_old = {p_old:.6e}")

        print(f"p_new = {p_new:.6e}")

        print(f"F_old = {F_old:.6e}")

        print(f"F_new = {F_new:.6e}")

       
        # CONVERGENCE
        
        pressure_error = abs(p_new-p_old)

        velocity_error = abs(F_new)

        convergence_measure = (

            pressure_error
            +
            velocity_error
        )

        print(f"Pressure Error = {pressure_error:.6e}")

        print(f"Velocity Error = {velocity_error:.6e}")

        # store convergence data 

        iteration_history.append(iteration + 1)

        pressure_history.append(p_new)

        pressure_error_history.append(pressure_error)

        velocity_error_history.append(velocity_error)

        # ----------------------------------------------------

        if convergence_measure < tolerance:

            print("\nConvergence achieved.")
            print(f"Total iterations = {iteration + 1}")

            p_star = p_new

            break

     
        # STAGNATION DETECTION
       

        if abs(F_new-F_old) < 1.0e-14:

            print("\nResidual stagnation detected.")

            p_star = p_new

            break

        
        # DENOMINATOR SAFETY
        

        denominator = F_new-F_old

        if abs(denominator) < 1.0e-14:

            print("\nSecant denominator collapsed.")

            p_star = p_new

            break

        
        # SECANT UPDATE
       

        p_next = (

            p_new

            -

            F_new

            *

            (
                (p_new-p_old)

                /

                denominator
            )
        )

        p_next = max(p_next, 1.0e-8)

       
        # UPDATE ITERATES
        

        p_old = p_new

        p_new = p_next

    else:

        raise RuntimeError(
            "Outer secant iteration failed."
        )

    
    # FINAL STAR STATE EVALUATION
    
    (

        F_final,

        uL_star,
        uR_star,

        rhoL_star,
        rhoR_star

    ) = velocity_residual(

        p_star,

        rhoL,
        uL,
        pL,

        rhoR,
        uR,
        pR,

        eos
    )

   
    # FINAL STAR VELOCITY
   

    u_star = 0.5*(uL_star+uR_star)

    
    # DETERMINE FINAL WAVE TYPES
    

    if p_star > pL:

        left_wave = "shock"

    else:

        left_wave = "rarefaction"

    

    if p_star > pR:

        right_wave = "shock"

    else:

        right_wave = "rarefaction"


    
    # PRECOMPUTE LEFT RAREFACTION FAN
    

    left_fan_data = None

    if left_wave == "rarefaction":

        delta_rho_left = solve_delta_rho(

            rhoL,
            pL,
            uL,

            p_star,

            eos,

            side="left",

            M=300
        )

        (

            rho_final,
            p_final,
            u_final,

            rho_history,
            p_history,
            u_history,
            xi_history

        ) = integrate_rarefaction_with_history(

            rhoL,
            pL,
            uL,

            delta_rho_left,

            eos,

            side="left",

            M=300
        )

        left_fan_data = {

            "rho_history": rho_history,

            "p_history": p_history,

            "u_history": u_history,

            "xi_history": xi_history
        }


    
    # PRECOMPUTE RIGHT RAREFACTION FAN
    

    right_fan_data = None

    if right_wave == "rarefaction":

        delta_rho_right = solve_delta_rho(

            rhoR,
            pR,
            uR,

            p_star,

            eos,

            side="right",

            M=300
        )

        (

            rho_final,
            p_final,
            u_final,

            rho_history,
            p_history,
            u_history,
            xi_history

        ) = integrate_rarefaction_with_history(

            rhoR,
            pR,
            uR,

            delta_rho_right,

            eos,

            side="right",

            M=300
        )

        right_fan_data = {

            "rho_history": rho_history,

            "p_history": p_history,

            "u_history": u_history,

            "xi_history": xi_history
        }

   
    # STORE SOLUTION
    
    solution = {

        "p_star"      : p_star,

        "u_star"      : u_star,

        "rhoL_star"   : rhoL_star,

        "rhoR_star"   : rhoR_star,

        "left_wave"   : left_wave,

        "right_wave"  : right_wave,

        "left_fan_data"  : left_fan_data,

        "right_fan_data" : right_fan_data,

        # convergence data

        "iteration_history"       : iteration_history,

        "pressure_history"        : pressure_history,

        "pressure_error_history"  : pressure_error_history,

        "velocity_error_history"  : velocity_error_history,

        
    }

    return solution





# INTEGRATE RAREFACTION WITH HISTORY STORAGE


def integrate_rarefaction_with_history(

    rho_initial,
    p_initial,
    u_initial,

    delta_rho,

    eos,
    side,

    M
):

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    rho = rho_initial

    p = p_initial

    u = u_initial

    # --------------------------------------------------------
    # STORAGE ARRAYS
    # --------------------------------------------------------

    rho_history = []

    p_history = []

    u_history = []

    xi_history = []

    # --------------------------------------------------------
    # RK4 LOOP
    # --------------------------------------------------------

    for m in range(M):

        # ----------------------------------------------------
        # SOUND SPEED
        # ----------------------------------------------------

        a = jwl_sound_speed(
            rho,
            p,
            eos
        )

        # ----------------------------------------------------
        # CHARACTERISTIC SPEED
        # Eq. (43)
        # ----------------------------------------------------

        if side == "left":

            xi = u - a

        else:

            xi = u + a

        # ----------------------------------------------------
        # STORE STATE
        # ----------------------------------------------------

        rho_history.append(rho)

        p_history.append(p)

        u_history.append(u)

        xi_history.append(xi)

        # ----------------------------------------------------
        # ADVANCE RK4 STEP
        # ----------------------------------------------------

        rho, p, u = rk4_rarefaction_step(

            rho,
            p,
            u,

            delta_rho,

            eos,
            side
        )

    # --------------------------------------------------------
    # STORE FINAL STATE
    # --------------------------------------------------------

    a = jwl_sound_speed(
        rho,
        p,
        eos
    )

    if side == "left":

        xi = u - a

    else:

        xi = u + a

    rho_history.append(rho)

    p_history.append(p)

    u_history.append(u)

    xi_history.append(xi)

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    rho_history = np.array(rho_history)

    p_history = np.array(p_history)

    u_history = np.array(u_history)

    xi_history = np.array(xi_history)

    return (

        rho,
        p,
        u,

        rho_history,
        p_history,
        u_history,
        xi_history
    )



# RAREFACTION INTERPOLATION

def interpolate_rarefaction(

    xi_target,

    xi_history,

    rho_history,

    p_history,

    u_history
):

    # --------------------------------------------------------
    # SORT XI HISTORY
    # np.interp REQUIRES MONOTONIC x-VALUES
    # --------------------------------------------------------

    sort_index = np.argsort(xi_history)

    xi_history = xi_history[sort_index]

    rho_history = rho_history[sort_index]

    p_history = p_history[sort_index]

    u_history = u_history[sort_index]

    # --------------------------------------------------------
    # LINEAR INTERPOLATION
    # --------------------------------------------------------

    rho = np.interp(

        xi_target,

        xi_history,

        rho_history
    )

    p = np.interp(

        xi_target,

        xi_history,

        p_history
    )

    u = np.interp(

        xi_target,

        xi_history,

        u_history
    )

    return rho, p, u


# ============================================================
# COMPLETE SOLUTION SAMPLER
# ============================================================


def sample_solution(

    x,
    t,
    x0,

    solution,

    rhoL,
    uL,
    pL,

    rhoR,
    uR,
    pR,

    eos
):

    # --------------------------------------------------------
    # AVOID t=0
    # --------------------------------------------------------

    t = max(t, 1.0e-12)

    # --------------------------------------------------------
    # SIMILARITY VARIABLE
    # --------------------------------------------------------

    xi = (x - x0)/t

    # ========================================================
    # STAR STATES
    # ========================================================

    p_star = solution["p_star"]

    u_star = solution["u_star"]

    rhoL_star = solution["rhoL_star"]

    rhoR_star = solution["rhoR_star"]

    left_wave = solution["left_wave"]

    right_wave = solution["right_wave"]

    # ========================================================
    # LEFT SIDE
    # ========================================================

    # --------------------------------------------------------
    # LEFT SHOCK
    # --------------------------------------------------------

    if left_wave == "shock":

        rhoL_star_temp, uL_star_temp, SL = shock_solver(

            rhoL,
            uL,
            pL,

            p_star,

            eos,

            side="left"
        )

        if xi < SL:

            rho = rhoL
            u   = uL
            p   = pL

        elif xi < u_star:

            rho = rhoL_star
            u   = u_star
            p   = p_star

    # --------------------------------------------------------
    # LEFT RAREFACTION
    # --------------------------------------------------------

    else:

        # ----------------------------------------------------
        # USE PRECOMPUTED FAN DATA
        # ----------------------------------------------------

        fan_data = solution["left_fan_data"]

        rho_history = fan_data["rho_history"]

        p_history = fan_data["p_history"]

        u_history = fan_data["u_history"]

        xi_history = fan_data["xi_history"]

        # ----------------------------------------------------
        # FAN SPEEDS
        # ----------------------------------------------------

        SHL = xi_history[0]

        STL = xi_history[-1]

        # ----------------------------------------------------
        # LEFT STATE
        # ----------------------------------------------------

        if xi < SHL:

            rho = rhoL
            u   = uL
            p   = pL

        # ----------------------------------------------------
        # INSIDE FAN
        # ----------------------------------------------------

        elif xi < STL:

            rho, p, u = interpolate_rarefaction(

                xi,

                xi_history,

                rho_history,

                p_history,

                u_history
            )

        # ----------------------------------------------------
        # LEFT STAR REGION
        # ----------------------------------------------------

        elif xi < u_star:

            rho = rhoL_star
            u   = u_star
            p   = p_star

    # ========================================================
    # RIGHT SIDE
    # ========================================================

    if xi >= u_star:

        # ----------------------------------------------------
        # RIGHT SHOCK
        # ----------------------------------------------------

        if right_wave == "shock":

            rhoR_star_temp, uR_star_temp, SR = shock_solver(

                rhoR,
                uR,
                pR,

                p_star,

                eos,

                side="right"
            )

            if xi < SR:

                rho = rhoR_star
                u   = u_star
                p   = p_star

            else:

                rho = rhoR
                u   = uR
                p   = pR

        # ----------------------------------------------------
        # RIGHT RAREFACTION
        # ----------------------------------------------------

        else:

            # ------------------------------------------------
            # USE PRECOMPUTED FAN DATA
            # ------------------------------------------------

            fan_data = solution["right_fan_data"]

            rho_history = fan_data["rho_history"]

            p_history = fan_data["p_history"]

            u_history = fan_data["u_history"]

            xi_history = fan_data["xi_history"]

            # ------------------------------------------------
            # FAN SPEED RANGE
            

            xi_min = min(xi_history)
            xi_max = max(xi_history)

            # ------------------------------------------------
            # RIGHT STAR REGION
            # ------------------------------------------------

            if xi < xi_min:

                rho = rhoR_star
                u   = u_star
                p   = p_star
 
            # ------------------------------------------------
            # INSIDE RIGHT RAREFACTION FAN
            # ------------------------------------------------

            elif xi <= xi_max:

                rho, p, u = interpolate_rarefaction(

                    xi,

                    xi_history,

                    rho_history,

                    p_history,

                    u_history
            )

            # ------------------------------------------------
            # RIGHT INITIAL STATE
            # ------------------------------------------------

            else:

                rho = rhoR
                u   = uR
                p   = pR

    # ========================================================
    # INTERNAL ENERGY
    # ========================================================

    e = jwl_energy(

        rho,
        p,
        eos
    )

    return rho, u, p, e
# ============================================================
