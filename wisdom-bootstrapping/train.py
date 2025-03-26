import gym
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import copy 
import os 
from custom_rewards import *

SEED = 42 

# Custom reward shaping wrapper
class RewardShapingWrapper(Wrapper):
    def __init__(self, env, reward_shaping_fn):
        """
        Wrapper that applies reward shaping to any gym environment
        
        Args:
            env: The environment to wrap
            reward_shaping_fn: Function that takes (obs, action, reward, next_obs, done, info)
                              and returns the shaped reward
        """
        super().__init__(env)
        self.reward_shaping_fn = reward_shaping_fn
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        shaping_component = self.reward_shaping_fn(self.last_obs, action, reward, obs, done, truncated, info)
        shaped_reward = reward + shaping_component
        
        # Store the original reward in info for evaluation purposes
        info['original_reward'] = reward
        info['shaping_component'] = shaping_component
        
        self.last_obs = obs
        return obs, shaped_reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        return self.last_obs

# Custom callback to track training metrics
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0, track_original_rewards=False):
        super().__init__(verbose)
        self.rewards = []
        self.original_rewards = []  # Track original rewards separately
        self.correlation_logs = []
        self.scaling_factors = []
        self.track_original_rewards = track_original_rewards
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Extract actor and critic losses from the model's logger every step
        if len(self.model.logger.name_to_value) > 0:
            critic_loss = self.model.logger.name_to_value.get('train/value_loss', None)
            
        
        # Process episode info when an episode ends
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_count += 1
                
                # For shaped environments, track the original reward if available
                if self.track_original_rewards and 'original_reward' in info:
                    self.original_rewards.append(info["original_reward"])
                    # We still track the shaped reward for comparison
                    self.rewards.append(info["episode"]["r"])
                else:
                    # For unshaped environments, just track the normal reward
                    self.rewards.append(info["episode"]["r"])
                
                # Capture correlation log if available
                if 'correlation_log' in info and info['correlation_log']:
                    self.correlation_logs.append(info['correlation_log'])
                    # Log to the training output
                    if self.verbose > 0:
                        print(info['correlation_log'])
                        
                # Capture scaling factor if available
                if 'scaling_factor' in info:
                    self.scaling_factors.append(info['scaling_factor'])
                    
        return True

# Enhanced reward shaping wrapper with learnability
class AdaptiveRewardShapingWrapper(Wrapper):
    def __init__(self, env, reward_shaping_fn, adaptation_rate=0.01, window_size=100):
        """
        Wrapper that applies adaptive reward shaping to any gym environment
        
        Args:
            env: The environment to wrap
            reward_shaping_fn: Function that takes (obs, action, reward, next_obs, done, truncated, info)
                              and returns the shaped reward component
            adaptation_rate: How quickly to adjust the scaling factor (0-1)
            window_size: Number of episodes to consider for adaptation
        """
        super().__init__(env)
        self.reward_shaping_fn = reward_shaping_fn
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # Tracking variables
        self.scaling_factor = 1.0  # Initial scaling for the advice
        self.episode_rewards = []  # Original rewards per episode
        self.episode_advice_values = []  # Sum of advice values per episode
        self.episode_future_returns = []  # Future returns after each episode
        
        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_advice_value = 0.0
        self.episode_count = 0
        
        # History of advice and rewards for correlation analysis
        self.advice_history = []  # Stores advice values at each step
        self.reward_history = []  # Stores rewards at each step
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Calculate shaping component (advice)
        advice = self.reward_shaping_fn(self.last_obs, action, reward, obs, done, truncated, info)
        
        # Apply scaling factor to advice
        scaled_advice = self.scaling_factor * advice
        shaped_reward = reward + scaled_advice
        
        # Track original reward and advice for this episode
        self.current_episode_reward += reward
        self.current_episode_advice_value += advice
        
        # Store advice and reward for correlation analysis
        self.advice_history.append(advice)
        self.reward_history.append(reward)
        
        # Store the original reward in info for evaluation purposes
        info['original_reward'] = reward
        info['shaping_component'] = advice
        
        # If episode is done, update tracking and possibly adapt scaling
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_advice_values.append(self.current_episode_advice_value)
            
            # Calculate future returns for previous episodes if possible
            if len(self.episode_rewards) > 1:
                # The future return for the previous episode is the current episode's reward
                self.episode_future_returns.append(self.current_episode_reward)
            
            self.episode_count += 1
            
            # Adapt scaling factor if we have enough episodes
            if len(self.episode_future_returns) >= self.window_size:
                self._adapt_scaling_factor()
            
            # Reset episode tracking
            self.current_episode_reward = 0.0
            self.current_episode_advice_value = 0.0
            
            # Add adaptation info to the info dict
            info['scaling_factor'] = self.scaling_factor
        
        self.last_obs = obs
        return obs, shaped_reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        return self.last_obs
    
    def _adapt_scaling_factor(self):
        """Adapt scaling factor based on actual performance improvement."""
        if len(self.episode_rewards) < self.window_size:
            return

        recent_future_returns = self.episode_future_returns[-self.window_size:]  # Returns that followed
        delta_return = np.mean(recent_future_returns) - np.mean(self.episode_rewards[-2*self.window_size:-self.window_size])
        # Update rule based on actual performance change
        eta = self.adaptation_rate  # Learning rate for adaptation
        delta = 0.1  # Trust region clipping
        lambda_max = 2.0  # Max allowable shaping factor

        update = eta * np.clip(delta_return, -delta, delta)
        self.scaling_factor = max(0.0, min(lambda_max, self.scaling_factor + update))

# Create environment
ENV_NAME = "LunarLander-v2"
#ENV_NAME = "CartPole-v1"
#base_env = gym.make(ENV_NAME)
base_env = gym.make(ENV_NAME,continuous=True, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5)
#env = RewardShapingWrapper(base_env, cartpole_reward_shaping)
env = AdaptiveRewardShapingWrapper(base_env, lunar_lander_reward_shaping, adaptation_rate=0.05, window_size=20)
env = Monitor(env)



def evaluate_reward_shaping(base_env, reward_shaping_fn, timesteps=1e5, runs=3, env_name="Unknown",policy="MlpPolicy",device="cpu"):
    """
    Evaluates the efficacy of a reward shaping function by comparing training with and without it.
    
    Args:
        base_env: The base gym environment or a function that creates a new instance of the environment
        reward_shaping_fn: The reward shaping function to evaluate
        timesteps: Number of timesteps to train for each run
        runs: Number of runs to average over for more reliable results
        env_name: Name of the environment (for plotting and saving)
        
    Returns:
        Dictionary containing training metrics and plots comparing performance
    """
    results = {
        "no_shaping": {"rewards": [], "final_eval_rewards": []},
        "with_shaping": {"rewards": [], "original_rewards": [], "final_eval_rewards": []},
        "adaptive_shaping": {"rewards": [], "original_rewards": [], "scaling_factors": [], "final_eval_rewards": []}
    }
    
    # Check if base_env is a callable (function) or an actual environment
    if callable(base_env):
        create_env = base_env
    else:
        # If it's an actual environment, create a function that returns a new instance
        # with the same parameters
        env_spec = base_env.unwrapped.spec
        create_env = lambda: gym.make(env_spec.id, **env_spec.kwargs)
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(f"models/{env_name}", exist_ok=True)
    
    # Create directory for saving videos
    os.makedirs(f"videos/{env_name}", exist_ok=True)
    
    # Run multiple training sessions and average results
    avg_no_shaping_rewards = []
    avg_with_shaping_rewards = []
    avg_adaptive_shaping_rewards = []
    for run in range(runs):
        print(f"Starting run {run+1}/{runs}")
        
        # Train without reward shaping
        print("Training without reward shaping...")
        env_no_shaping = Monitor(create_env())
        model_no_shaping = PPO(policy, env_no_shaping, verbose=0,seed=SEED,device=device)
        callback_no_shaping = MetricsCallback()
        model_no_shaping.learn(total_timesteps=int(timesteps), callback=callback_no_shaping)
        results["no_shaping"]["rewards"].append(callback_no_shaping.rewards)
        
        # Save the model
        model_path = f"models/{env_name}/no_shaping_run_{run+1}.zip"
        model_no_shaping.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Train with fixed reward shaping
        print("Training with fixed reward shaping...")
        env_shaping = Monitor(RewardShapingWrapper(create_env(), reward_shaping_fn))
        model_shaping = PPO(policy, env_shaping, verbose=0,seed=SEED,device=device)
        callback_shaping = MetricsCallback(track_original_rewards=True)
        model_shaping.learn(total_timesteps=int(timesteps), callback=callback_shaping)
        # Store both shaped and original rewards
        results["with_shaping"]["rewards"].append(callback_shaping.rewards)
        results["with_shaping"]["original_rewards"].append(callback_shaping.original_rewards)
        
        # Save the model
        model_path = f"models/{env_name}/with_shaping_run_{run+1}.zip"
        model_shaping.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Train with adaptive reward shaping
        print("Training with adaptive reward shaping...")
        env_adaptive = Monitor(AdaptiveRewardShapingWrapper(
            create_env(), reward_shaping_fn, adaptation_rate=0.05, window_size=20))
        model_adaptive = PPO(policy, env_adaptive, verbose=0,seed=SEED,device=device)
        callback_adaptive = MetricsCallback(track_original_rewards=True)
        model_adaptive.learn(total_timesteps=int(timesteps), callback=callback_adaptive)
        # Store both shaped and original rewards
        results["adaptive_shaping"]["rewards"].append(callback_adaptive.rewards)
        results["adaptive_shaping"]["original_rewards"].append(callback_adaptive.original_rewards)
        results["adaptive_shaping"]["scaling_factors"].append(callback_adaptive.scaling_factors)
        
        # Save the model
        model_path = f"models/{env_name}/adaptive_shaping_run_{run+1}.zip"
        model_adaptive.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Final evaluation of all three models (10 episodes each)
        print(f"Performing final evaluation for run {run+1}...")
        
        # Evaluate no shaping model
        eval_env = Monitor(create_env())
        model = PPO.load(f"models/{env_name}/no_shaping_run_{run+1}.zip")
        eval_rewards = evaluate_model(model, eval_env, num_episodes=10)
        results["no_shaping"]["final_eval_rewards"].append(eval_rewards)
        print(f"No shaping model average reward: {np.mean(eval_rewards):.2f}")
        
        # Evaluate fixed shaping model (on unshaped environment for fair comparison)
        eval_env = Monitor(create_env())
        model = PPO.load(f"models/{env_name}/with_shaping_run_{run+1}.zip")
        eval_rewards = evaluate_model(model, eval_env, num_episodes=10)
        results["with_shaping"]["final_eval_rewards"].append(eval_rewards)
        print(f"Fixed shaping model average reward: {np.mean(eval_rewards):.2f}")
        
        # Evaluate adaptive shaping model (on unshaped environment for fair comparison)
        eval_env = Monitor(create_env())
        model = PPO.load(f"models/{env_name}/adaptive_shaping_run_{run+1}.zip")
        
        # Record video for the adaptive shaping model in the last run
        if run == runs - 1:
            video_path = f"videos/{env_name}/adaptive_shaping_run_{run+1}"
            print(f"Recording video of adaptive shaping agent to {video_path}")
            eval_rewards = evaluate_model(
                model, 
                eval_env, 
                num_episodes=10, 
                record_video=True, 
                video_path=video_path
            )
        else:
            eval_rewards = evaluate_model(model, eval_env, num_episodes=10)
            
        results["adaptive_shaping"]["final_eval_rewards"].append(eval_rewards)
        print(f"Adaptive shaping model average reward: {np.mean(eval_rewards):.2f}")
    
    # Average results across runs
    for approach in results:
        metrics_to_process = list(results[approach].keys())  # Create a fixed list of keys
        for metric in metrics_to_process:
            if metric == "scaling_factors":
                continue  # Skip averaging scaling factors
            
            # Make sure we have data for this metric
            if not all(len(run_data) > 0 for run_data in results[approach][metric]):
                print(f"Warning: Missing data for {approach}.{metric}")
                continue
                
            # Find the minimum length across all runs for this metric
            min_length = min(len(run_data) for run_data in results[approach][metric])
            
            # Truncate all runs to the minimum length and compute average
            truncated_runs = [run_data[:min_length] for run_data in results[approach][metric]]
            results[approach][metric + "_avg"] = np.mean(truncated_runs, axis=0)
            results[approach][metric + "_std"] = np.std(truncated_runs, axis=0)
    
    # Plot comparison
    plot_reward_shaping_comparison(results, env_name)
    
    # Save results
    save_results(results, env_name, timesteps)
    
    # Print final evaluation summary
    print("\nFinal Evaluation Summary:")
    for approach in ["no_shaping", "with_shaping", "adaptive_shaping"]:
        avg_reward = np.mean([np.mean(rewards) for rewards in results[approach]["final_eval_rewards"]])
        std_reward = np.std([np.mean(rewards) for rewards in results[approach]["final_eval_rewards"]])
        print(f"{approach}: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return results

def evaluate_model(model, env, num_episodes=10, record_video=False, video_path=None):
    """
    Evaluate a trained model for a specified number of episodes
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate in
        num_episodes: Number of episodes to evaluate
        record_video: Whether to record a video of the evaluation
        video_path: Path to save the video (if recording)
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    # Set up video recording if requested
    if record_video and video_path:
        from gym.wrappers import RecordVideo
        env = RecordVideo(env, video_path)
    
    for _ in range(num_episodes):
        # Handle both old and new gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            # New Gym API returns (obs, info)
            obs, _ = reset_result
        else:
            # Old Gym API returns just obs
            obs = reset_result
            
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # Handle both old and new gym API
            if len(step_result) == 5:  # New Gym API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old Gym API: obs, reward, done, info
                obs, reward, done, info = step_result
                
            episode_reward += reward
                
        episode_rewards.append(episode_reward)
        
    return episode_rewards

def plot_reward_shaping_comparison(results, env_name):
    """Plot comparison of training with and without reward shaping"""
    window_size = 20  # For moving average
    
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards - using original rewards for fair comparison
    plt.subplot(2, 2, 1)
    plot_metric_with_std(
        results["no_shaping"]["rewards_avg"], 
        results["no_shaping"]["rewards_std"],
        'b', "No Shaping", window_size
    )
    
    # Use original rewards for shaped methods if available
    if "original_rewards_avg" in results["with_shaping"]:
        plot_metric_with_std(
            results["with_shaping"]["rewards_avg"], 
            results["with_shaping"]["rewards_std"],
            'g', "With Shaping (Original)", window_size
        )
    else:
        plot_metric_with_std(
            results["with_shaping"]["rewards_avg"], 
            results["with_shaping"]["rewards_std"],
            'g', "With Shaping", window_size
        )
        
    if "original_rewards_avg" in results["adaptive_shaping"]:
        plot_metric_with_std(
            results["adaptive_shaping"]["rewards_avg"], 
            results["adaptive_shaping"]["rewards_std"],
            'r', "Adaptive Shaping (Original)", window_size
        )
    else:
        plot_metric_with_std(
            results["adaptive_shaping"]["rewards_avg"], 
            results["adaptive_shaping"]["rewards_std"],
            'r', "Adaptive Shaping", window_size
        )
    
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Reward Comparison - {env_name}")
    plt.legend()
   
    
    # Plot learning curves (cumulative rewards) - using original rewards for fair comparison
    plt.subplot(2, 2, 2)
    
    # Determine which rewards to use for shaped methods
    with_shaping_rewards = results["with_shaping"]["rewards_avg"] 
    adaptive_shaping_rewards = results["adaptive_shaping"]["rewards_avg"] 
    
    # Find the minimum length across all reward arrays to ensure they match
    min_length = min(
        len(results["no_shaping"]["rewards_avg"]),
        len(with_shaping_rewards),
        len(adaptive_shaping_rewards)
    )
    
    # Create x array with the correct length
    x = np.arange(min_length)
    
    # Truncate all reward arrays to the same length
    no_shaping_rewards = results["no_shaping"]["rewards_avg"][:min_length]
    with_shaping_rewards = with_shaping_rewards[:min_length]
    adaptive_shaping_rewards = adaptive_shaping_rewards[:min_length]
    
    # Plot cumulative rewards
    plt.plot(x, np.cumsum(no_shaping_rewards), 'b', label="No Shaping")
    plt.plot(x, np.cumsum(with_shaping_rewards), 'g', label="With Shaping (Original)")
    plt.plot(x, np.cumsum(adaptive_shaping_rewards), 'r', label="Adaptive Shaping (Original)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Curve Comparison (Original Rewards)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"reward_shaping_comparison_{env_name}.png")
    plt.show()

def plot_metric_with_std(mean_values, std_values, color, label, window_size):
    """Helper function to plot a metric with its standard deviation and moving average"""
    x = np.arange(len(mean_values))
    
    # Plot raw mean with std deviation
    plt.fill_between(x, mean_values - std_values, mean_values + std_values, 
                     color=color, alpha=0.2)
    
    # Plot moving average
    if len(mean_values) >= window_size:
        ma = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')
        ma_x = np.arange(window_size-1, len(mean_values))
        plt.plot(ma_x, ma, color=color, linewidth=2, label=label)
    else:
        plt.plot(x, mean_values, color=color, linewidth=2, label=label)

def save_results(results, env_name, timesteps):
    """Save results to file"""
    filename = f"reward_shaping_results_{env_name}_{int(timesteps)}.txt"
    with open(filename, "w") as f:
        for approach in results:
            f.write(f"=== {approach} ===\n")
            for metric in results[approach]:
                if metric.endswith("_avg") or metric.endswith("_std"):
                    continue  # Skip averaged metrics in raw data output
                
                # For each run, write summary statistics
                f.write(f"{metric}:\n")
                for i, run_data in enumerate(results[approach][metric]):
                    if len(run_data) > 0:
                        f.write(f"  Run {i+1}: mean={np.mean(run_data):.4f}, "
                                f"std={np.std(run_data):.4f}, "
                                f"min={np.min(run_data):.4f}, "
                                f"max={np.max(run_data):.4f}\n")
            f.write("\n")

# Example usage in main code
if __name__ == "__main__":
    # Create environment
    # ENV_NAME = "LunarLander-v2"
    # env = gym.make(ENV_NAME,continuous=True, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    # results = evaluate_reward_shaping(
    #     base_env=env,
    #     reward_shaping_fn=lunar_lander_reward_shaping,
    #     timesteps=5e4,  # Reduced for faster evaluation
    #     runs=1,
    #     env_name=ENV_NAME
    # )
    #ENV_NAME="CartPole-v1"
    # ENV_NAME="BipedalWalker-v3"
    # env = gym.make(ENV_NAME,render_mode="rgb_array")
    # results = evaluate_reward_shaping(
    #     base_env=env,
    #     reward_shaping_fn=bipedal_walker_reward_shaping,
    #     timesteps=1e5,  # Reduced for faster evaluation
    #     runs=10,
    #     env_name=ENV_NAME,
    # )
    ENV_NAME="CarRacing-v2"
    env = gym.make(ENV_NAME,render_mode="rgb_array")
    results = evaluate_reward_shaping(
        base_env=env,
        reward_shaping_fn=car_racing_reward_shaping,
        timesteps=5e4,  # Reduced for faster evaluation
        runs=10,
        env_name=ENV_NAME,
        policy="CnnPolicy",
        device="mps"
    )

    
 