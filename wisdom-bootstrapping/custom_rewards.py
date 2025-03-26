

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
