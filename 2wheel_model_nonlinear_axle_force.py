import numpy as np
import matplotlib.pyplot as plt


def pacejka_lateral(alpha, B, C, D, E):
    """
    Simplified Pacejka Magic Formula for lateral force for non-linear axle characteristics.

    alpha = slip angle (rad)
    B = stiffness factor (???)
    C = shape factor (???)
    D = peak force (N)
    E = curvature factor (???)
    """
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))


def nonlinear_twowheel(state, control, params):
    """
    Non-linear kinematics for the dynamic 2-wheel model. Longitudinal speed still held constant.
    ??? Do we choose to use m/s or km/h? Same question with rad vs. deg. ???

    Inputs:
    ------------------------------
    state = [X, Y, psi, vy, r]

    X, Y: global position
    psi: yaw angle (rad)
    vy: lateral velocity (m/s)
    r: yaw rate (rad/s)
    -------------------------------
    control = [delta]

    delta: steering angle (rad ???)
    -------------------------------
    params = {???}

    m: mass (kg)
    Iz: yaw inertia (kg/m^2)
    lf, lr: front and rear distances from cog - centre of gravity (m)
    vx: longitudinal velocity (m/s)
    g: acceletion at Earth surface - can we not just hardcode this? xd
    mu: friction coefficient (w/o unit)
    delta_max:  steering limit (rad)
    Bf, Br: front and rear tyre stiffness factor (???)
    Cf, Cr: front and rear tyre shape factor (???)
    Ef, Er: front and rear tyre curvature factor (???)
    ------------------------------
    """

    X, Y, psi, vy, r = state  #VS says X and Y not accessed ??

    delta = control[0]

    m = params["m"]
    Iz = params["Iz"]
    lf = params["lf"]
    lr = params["lr"]
    vx = params["vx"]
    vx_safe = max(vx, 0.1)
    g = params["g"]
    mu = params["mu"]
    delta_max = params["delta_max"]
    Bf = params["Bf"]
    Br = params["Br"]
    Cf = params["Cf"]    
    Cr = params["Cr"]
    Ef = params["Ef"]
    Er = params["Er"]


    # Steering limit
    delta = np.clip(delta, -delta_max, delta_max)

    # Static axle normal loads
    wheelbase = lf + lr
    Fzf = m * g * lr / wheelbase
    Fzr = m * g * lf / wheelbase

    # Peak lateral forces
    Df = mu * Fzf
    Dr = mu * Fzr

    # Slip angles
    alpha_f = delta - np.arctan((vy + lf * r) / vx_safe)
    alpha_r = -np.arctan((vy - lr * r) / vx_safe)

    # Lateral axel forces
    Fyf = pacejka_lateral(alpha_f, Bf, Cf, Df, Ef)
    Fyr = pacejka_lateral(alpha_r, Br, Cr, Dr, Er)

    # ODEs
    dvy = (Fyf * np.cos(delta) + Fyr) / m - vx * r
    dr = (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz

    # Position update
    dpsi = r

    dX = vx * np.cos(psi) - vy * np.sin(psi)
    dY = vx * np.sin(psi) + vy * np.cos(psi)

    # Outputs
    beta = np.arctan2(vy, vx_safe)
    ay = dvy + vx * r

    state_dot = np.array([dX, dY, dpsi, dvy, dr])

    outputs = {
        "delta": delta,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "Fyf": Fyf,
        "Fyr": Fyr,
        "Fzf": Fzf,
        "Fzr": Fzr,
        "beta": beta,
        "ay": ay
    }

    return state_dot, outputs

#-----------------------------------
# Example run (from ChatGPT)
params = {
    "m": 250.0,                     # vehicle mass [kg]
    "Iz": 150.0,                    # yaw moment of inertia [kg m^2]
    "lf": 0.8,                      # COG to front axle [m]
    "lr": 0.7,                      # COG to rear axle [m]
    "vx": 15.0,                     # constant forward speed [m/s]
    "g": 9.81,                      # gravity [m/s^2]
    "mu": 1.2,                      # friction coefficient
    "delta_max": np.radians(45),    # max steering angle [rad]

    # Front axle Pacejka parameters
    "Bf": 8.0,
    "Cf_shape": 1.3,
    "Ef": -1.6,

    # Rear axle Pacejka parameters
    "Br": 8.0,
    "Cr_shape": 1.3,
    "Er": -1.6
}


# -------------------------
# Initial state
# -------------------------
# [X, Y, psi, vy, r]
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


# -------------------------
# Simulation settings
# -------------------------
dt = 0.01
t_end = 5.0
time = np.arange(0, t_end, dt)


# -------------------------
# Storage
# -------------------------
history = {
    "time": [],
    "X": [],
    "Y": [],
    "psi": [],
    "vy": [],
    "r": [],
    "delta": [],
    "alpha_f": [],
    "alpha_r": [],
    "Fyf": [],
    "Fyr": [],
    "beta": [],
    "ay": []
}


# -------------------------
# Steering input function
# -------------------------
def steering_input(t):
    """
    Example steering input:
    0 deg at start, then step to 5 deg after 0.5 s.
    """
    if t < 0.5:
        return np.radians(0)
    else:
        return np.radians(5)


# -------------------------
# Simulation loop
# -------------------------
for t in time:
    delta = steering_input(t)
    control = np.array([delta])

    state_dot, outputs = nonlinear_twowheel(state, control, params)

    # Euler integration
    state = state + state_dot * dt

    # Save states
    history["time"].append(t)
    history["X"].append(state[0])
    history["Y"].append(state[1])
    history["psi"].append(state[2])
    history["vy"].append(state[3])
    history["r"].append(state[4])

    # Save outputs
    for key in outputs:
        if key in history:
            history[key].append(outputs[key])


# Convert to arrays
for key in history:
    history[key] = np.array(history[key])


# -------------------------
# Plots
# -------------------------
plt.figure()
plt.plot(history["X"], history["Y"])
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Vehicle path")
plt.axis("equal")
plt.grid(True)

plt.figure()
plt.plot(history["time"], np.degrees(history["delta"]))
plt.xlabel("Time (s)")
plt.ylabel("Steering angle δ (deg)")
plt.title("Steering input")
plt.grid(True)

plt.figure()
plt.plot(history["time"], history["r"])
plt.xlabel("Time (s)")
plt.ylabel("Yaw rate r (rad/s)")
plt.title("Yaw rate")
plt.grid(True)

plt.figure()
plt.plot(history["time"], np.degrees(history["alpha_f"]), label="front slip angle α_f")
plt.plot(history["time"], np.degrees(history["alpha_r"]), label="rear slip angle α_r")
plt.xlabel("Time (s)")
plt.ylabel("Slip angle (deg)")
plt.title("Slip angles")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(history["time"], history["Fyf"], label="front lateral force Fy_f")
plt.plot(history["time"], history["Fyr"], label="rear lateral force Fy_r")
plt.xlabel("Time (s)")
plt.ylabel("Lateral force (N)")
plt.title("Axle lateral forces")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(history["time"], history["ay"])
plt.xlabel("Time (s)")
plt.ylabel("Lateral acceleration a_y (m/s²)")
plt.title("Lateral acceleration")
plt.grid(True)

plt.show()
