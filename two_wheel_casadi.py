import casadi as ca

def get_bicycle_model():
    vy = ca.SX.sym('vy')  
    r  = ca.SX.sym('r') 
    states = ca.vertcat(vy, r)
    n_states = states.size1()

    delta = ca.SX.sym('delta')
    controls = ca.vertcat(delta)
    n_controls = controls.size1()

    m   = ca.SX.sym('m')   
    Iz  = ca.SX.sym('Iz')  
    lf  = ca.SX.sym('lf')  
    lr  = ca.SX.sym('lr')  
    Cf  = ca.SX.sym('Cf')  
    Cr  = ca.SX.sym('Cr')  
    vx  = ca.SX.sym('vx')  #constant for this 
    params = ca.vertcat(m, Iz, lf, lr, Cf, Cr, vx)

    # Slip angles 
    alpha_f = delta - (vy + lf * r) / vx
    alpha_r = (vy - lr * r) / vx  #same dilemma with the minus

    # Linear tyre model for lateral forces
    Fyf = Cf * alpha_f
    Fyr = Cr * alpha_r

    # ODE Equations: x_dot = f(x, u, p)
    dvy = (Fyf + Fyr) / m - vx * r  #idk how to get the cos(delta) in casadi
    dr  = (lf * Fyf - lr * Fyr) / Iz  #same here
    
    rhs = ca.vertcat(dvy, dr)

    # Create the CasADi function (fully from Gemini)
    f = ca.Function('f', [states, controls, params], [rhs],              ['x', 'u', 'p'], ['x_dot'])
    
    return f

