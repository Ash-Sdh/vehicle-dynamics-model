import numpy as np

#Firstly, I write the model as it is, then rewrite it in a suitable way for CasADI

def two_wheel_physics(state, control, params):
    """
    state = [vy, r] - list: lateral velocity (m/s) and yaw rate (rad/s) 
    control = [delta] - list: front steering angle (rad)
    params = {m, Iz, lf, lr, Cf, Cr, vx} - dictionary: mass (kg), yaw inertia (kg/m^2), front- and rear distances form cog (m), front- and rear cornering stiffnes (N/rad), longitudinal velocity (m/s)
    """
    vy, r = state
    delta = control[0]
    m = params['m']
    Iz = params['Iz']
    lf = params['lf']
    lr = params['lr']
    Cf = params['Cf']
    Cr = params['Cr']
    vx = params['vx']

    # Slip angles
    alpha_f = delta - (vy + lf * r) / vx
    alpha_r = (vy - lr * r) / vx                    #Gemini says there should be a minus in front, idk

    # Linear tyre model
    Fyf = Cf * alpha_f
    Fyr = Cr * alpha_r

    # Equations of motion
    dvy = (Fyf * np.cos(delta) + Fyr) / m - vx * r
    dr  = (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz

    return np.array([dvy, dr])


# Example Vehicle Setup
car_params = {
    'm': 250.0,    'Iz': 150.0, 
    'lf': 0.8,     'lr': 0.7, 
    'Cf': 40000.0, 'Cr': 40000.0, 
    'vx': 15.0  # 54 km/h
}

# Initial conditions [vy, r]
x = np.array([0.0, 0.0])
u = np.array([np.radians(2)]) # 2 degrees steering
dt = 0.01

# Simple Euler Integration (for clarity)
for step in range(100):
    x_dot = bicycle_model_dynamics(x, u, car_params)
    x = x + x_dot * dt
    
    if step % 20 == 0:
        print(f"Time {step*dt:.2f}s | Yaw Rate: {x[1]:.3f} rad/s")
