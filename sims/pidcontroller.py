


class PIDController: 
    """
    PID control system for MountainCarContinuous-v0 opengym ai env. 

    Value to control is bounded between [-1,1] and it is the force you can apply to the car 
    signal received is x-position of the car. Desired signal is 0.45, while we are not at 0.45 adjust accordingly 

    """

    def __init__(self,ki = .01, kp = .1, kd = 1,TARGET = 0.45,env_signal_bounds = [-1.2,0.6],control_signal_bounds = [-1,1] ):
        self.ki = ki
        self.kp = kp
        self.kd = kd 
        self.target = TARGET
        self.current_error = 0 
        self.cumulative_error = 0 
        self.env_signal_bounds = env_signal_bounds
        self.control_signal_bounds = control_signal_bounds
        self.v = 0 
    
    def error_term(self,signal):
        return self.target - signal 

    def control_signal(self,signal):
        e = self.error_term(signal)
        self.cumulative_error += e 
        de_dt = e - self.current_error
        u_t =  (self.kp*e) + (self.ki*self.cumulative_error) + (self.kd*de_dt)
        self.current_error = e 
        return u_t
    
    def action_signal(self,env_signal):
        x,v = env_signal
        u = self.control_signal(x)
        action = max(-1,u) if u < 0 else min(1,u)
        dv_dt = v - self.v 
        slipping = (v <= 0 ) and (dv_dt > 0)
        self.v = v 
        return -action if slipping else action #if the car is slipping, let it, let it regularize 
    
    
        


