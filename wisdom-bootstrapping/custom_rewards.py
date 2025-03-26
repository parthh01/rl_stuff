import numpy as np

# Define your reward shaping function
def lunar_lander_reward_shaping(prev_obs, action, reward, obs, done, truncated, info):
    """
    Example reward shaping for LunarLander-v2
    
    LunarLander obs: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]
    """
    # Encourage staying level (small angle and angular velocity)
    angle = abs(obs[4])
    angular_vel = abs(obs[5])
    level_bonus = 0.1 * (1.0 - angle) - 0.05 * angular_vel
    
    # Encourage gentle descent
    y_vel = obs[3]
    descent_bonus = 0.1 * (1.0 - abs(y_vel)) if y_vel < 0 else 0
    
    # Encourage staying centered
    x_pos = abs(obs[0])
    centered_bonus = 0.1 * (1.0 - x_pos)
    
    # Return shaping component (not the full reward)
    return level_bonus + descent_bonus + centered_bonus




def cartpole_reward_shaping(prev_obs, action, reward, obs, done, truncated, info):
    """
    Reward shaping function for CartPole-v1 environment.
    
    The standard CartPole reward is +1 for each timestep the pole remains upright.
    This shaping function adds additional rewards based on:
    1. Pole angle - reward staying close to vertical
    2. Pole angular velocity - reward low angular velocity
    3. Cart position - reward staying near the center
    4. Cart velocity - reward low cart velocity
    
    Args:
        prev_obs: Previous observation [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        action: Action taken (0 = left, 1 = right)
        reward: Original reward from the environment
        obs: Current observation [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        done: Whether the episode is done
        truncated: Whether the episode was truncated
        info: Additional information
        
    Returns:
        Shaping reward component
    """
    # If the episode is done and not due to reaching max steps, it means the pole fell
    if done and not truncated:
        return -5.0  # Penalty for failing
    
    # Extract observations
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    # Calculate shaping components
    
    # 1. Pole angle - reward being vertical (angle close to 0)
    # Convert to degrees for easier interpretation
    angle_deg = abs(pole_angle * 180 / np.pi)
    angle_reward = 0.1 * (1.0 - angle_deg / 12.0)  # Max angle is ~12 degrees
    
    # 2. Pole angular velocity - reward low velocity
    vel_reward = 0.05 * (1.0 - min(1.0, abs(pole_vel) / 1.0))
    
    # 3. Cart position - reward staying near center
    # Cart position ranges from -2.4 to 2.4
    pos_reward = 0.05 * (1.0 - min(1.0, abs(cart_pos) / 2.4))
    
    # 4. Cart velocity - reward low velocity
    cart_vel_reward = 0.05 * (1.0 - min(1.0, abs(cart_vel) / 2.0))
    
    # Combine all shaping components
    shaping_reward = angle_reward + vel_reward + pos_reward + cart_vel_reward
    
    return shaping_reward

def bipedal_walker_reward_shaping(prev_obs, action, reward, obs, done, truncated, info):
    """
    Minimal reward shaping for BipedalWalker-v3 environment.
    
    This function provides subtle hints to guide learning without
    overwhelming the intrinsic reward structure of the environment.
    
    The default BipedalWalker reward is well-designed, so we'll only
    add small nudges in the right direction.
    """
    # Extract relevant observations
    hull_angle = obs[0]  # Hull angle
    horizontal_speed = obs[2]  # Horizontal velocity
    
    # 1. Small bonus for staying upright (hull angle close to 0)
    # This helps early learning to not fall over immediately
    upright_bonus = 0.05 * (1.0 - min(1.0, abs(hull_angle) * 2))
    
    # 2. Small bonus for moving forward at a reasonable speed
    # Encourages initial steps in the right direction
    speed_bonus = 0.05 * min(1.0, max(0, horizontal_speed))
    
    # 3. Small penalty for falling (in addition to the environment's -100)
    # This helps emphasize the importance of not falling
    fall_penalty = -1.0 if done and not truncated and reward <= -100 else 0.0
    
    # Keep the shaping reward small relative to the environment reward
    shaping = upright_bonus + speed_bonus + fall_penalty
    
    return shaping

def car_racing_reward_shaping(prev_obs, action, reward, obs, done, truncated, info):
    """
    Simple reward shaping for CarRacing-v3 environment.
    
    This function focuses on a single key driving behavior:
    - Encouraging the car to slow down when turning (cornering)
    
    Args:
        prev_obs: Previous observation (96x96 RGB image)
        action: Action taken [steering, gas, brake] or discrete action
        reward: Original reward from the environment
        obs: Current observation (96x96 RGB image)
        done: Whether the episode is done
        truncated: Whether the episode was truncated
        info: Additional information
        
    Returns:
        Shaping reward component
    """
    # Check if we're using continuous or discrete actions
    continuous_actions = hasattr(action, "__len__")
    
    # Extract action information
    if continuous_actions:
        steering = action[0]  # -1 (left) to 1 (right)
        gas = action[1]       # 0 to 1
        brake = action[2]     # 0 to 1
    else:
        # Discrete actions
        steering = -1 if action == 1 else (1 if action == 2 else 0)
        gas = 1 if action == 3 else 0
        brake = 1 if action == 4 else 0
    
    # Calculate the absolute steering angle
    abs_steering = abs(steering)
    
    # Reward for appropriate cornering behavior:
    # 1. Reward reducing speed (using brake or reducing gas) during sharp turns
    # 2. Penalize high speed during sharp turns
    
    if abs_steering > 0.3:  # If turning moderately to sharply
        # Reward braking during turns
        brake_reward = 0.1 * brake * abs_steering
        
        # Penalize high gas during sharp turns
        # The penalty increases with steering angle and gas amount
        gas_penalty = -0.2 * gas * abs_steering
        
        # Combine the cornering rewards
        cornering_reward = brake_reward + gas_penalty
    else:
        # No cornering reward when going straight
        cornering_reward = 0.0
    
    return cornering_reward
