#ifndef AGENT_H
#define AGENT_H

#include "game.h"
#include <vector>
#include <random> 
#include <iostream>
#include <tuple>
#include <torch/torch.h>
#include "nn.h"

using namespace std; 

struct PPOAgent {
    SimpleNN actor; 
    SimpleNN actor_old; 
    torch::optim::Adam actor_optimizer; 
    SimpleNN critic; 
    torch::optim::Adam critic_optimizer; 
    torch::Device device; 
    float gamma = 0.95;
    float clip_ratio = 0.2; 
    int num_policy_updates = 4; //per iteration
    int env_size; 

    PPOAgent(int hidden_layers=2, int hidden_nodes=4,int env_size=3, torch::Device device=torch::kCPU, float gamma=0.95, float lr = 0.01, float clip_ratio=0.2, string filename="") : 
        env_size(env_size),
        actor(SimpleNN((env_size*env_size) + 1, hidden_layers, hidden_nodes, env_size*env_size, device, filename != "" ? filename + "_actor.pt" : "")), 
        actor_old(SimpleNN((env_size*env_size) + 1, hidden_layers, hidden_nodes, env_size*env_size, device,"")),
        critic(SimpleNN((env_size*env_size) + 1, hidden_layers, hidden_nodes, 1, device, filename != "" ? filename + "_critic.pt" : "")), 
        actor_optimizer(torch::optim::Adam(actor.parameters(), lr)), 
        critic_optimizer(torch::optim::Adam(critic.parameters(), lr)), 
        device(device), 
        gamma(gamma),   
        clip_ratio(clip_ratio){}

    tuple<vector<int>, torch::Tensor, torch::Tensor> get_action(Game& env){
        vector<int> valid_moves = env.get_valid_moves();
        
        // Convert to tensor and move to device
        torch::Tensor action_mask = torch::from_blob(valid_moves.data(), {env_size*env_size}, torch::kFloat32).clone().to(device);
        
        // Get state tensor and ensure it's on the correct device
        torch::Tensor state = env.get_state_tensor().to(device);
        
        // Get action log probabilities
        torch::Tensor action_log_probs = actor.forward(state, action_mask);
        
        // Convert to probabilities
        torch::Tensor probs = torch::exp(action_log_probs);
        
        // Create vector of valid move indices
        vector<int> valid_indices;
        for(int i = 0; i < valid_moves.size(); i++) {
            if(valid_moves[i] == 1) {
                valid_indices.push_back(i);
            }
        }
        
        if(valid_indices.empty()) {
            cout << "Error: No valid moves available!" << endl;
            throw runtime_error("No valid moves available");
        }
        
        int chosen_index;
        if (torch::rand({1}).item<float>() < 0.1) { // 10% exploration
            // Random valid move
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dis(0, valid_indices.size() - 1);
            chosen_index = valid_indices[dis(gen)];
        } else {
            // Get probabilities only for valid moves
            vector<float> valid_probs;
            for(int idx : valid_indices) {
                valid_probs.push_back(probs[idx].item<float>());
            }
            
            // Find index of maximum probability among valid moves
            auto max_it = max_element(valid_probs.begin(), valid_probs.end());
            int max_valid_idx = distance(valid_probs.begin(), max_it);
            chosen_index = valid_indices[max_valid_idx];
        }
        
        // Convert to row and column
        int row = chosen_index / env_size;
        int col = chosen_index % env_size;
        
        
        return make_tuple(
            vector<int>{row, col},
            action_log_probs,
            action_mask
        );
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor, float> state_action_reward(Game& env){
        torch::Tensor state = env.get_state_tensor(); 
        vector<int> action; 
        torch::Tensor action_log_probs; 
        torch::Tensor action_mask; 
        tie(action, action_log_probs, action_mask) = get_action(env); 
        float reward = env.play_move(action[0], action[1]); 
        return make_tuple(state, action_log_probs, action_mask, reward); 
    }

    tuple<vector<torch::Tensor>, vector<torch::Tensor> ,vector<torch::Tensor>, vector<float> > self_play(Game& env){
        vector<torch::Tensor> states; 
        vector<torch::Tensor> actions; 
        vector<torch::Tensor> action_masks; 
        torch::Tensor action_mask; 
        vector<float> rewards; 
        torch::Tensor state; 
        float reward; 
        torch::Tensor action; 

        while (!env.over){
            tie(state, action, action_mask, reward) = state_action_reward(env); 
            states.push_back(state); 
            actions.push_back(action);
            action_masks.push_back(action_mask);
            rewards.push_back(reward);
            if (env.over) break; 
            tie(state, action, action_mask, reward) = state_action_reward(env); 
            states.push_back(state); 
            actions.push_back(action);
            action_masks.push_back(action_mask);
            rewards.push_back(reward);
        }
        // construct discounted rewards
        vector<float> discounted_rewards(states.size(), 0);
        discounted_rewards[states.size()-1] = rewards[states.size()-1];
        for (int i = states.size()-2;i >= 0;i--){
            discounted_rewards[i] = -(rewards[i] + gamma * discounted_rewards[i+1]);
        }
        return make_tuple(states,actions,action_masks,discounted_rewards); 
    }

    void save_model(string filename){
        actor.save_model(filename + "_actor");
        critic.save_model(filename + "_critic");
    }; 

    void learn(int total_steps, int batch_size) {
        int current_steps = 0;
        int iterations = 0;
        int games = 0;
        
        while (current_steps < total_steps) {
            // Collect batch of experience
            vector<torch::Tensor> batch_states;
            vector<torch::Tensor> batch_actions;
            vector<torch::Tensor> batch_action_masks;
            vector<float> batch_rewards;
            
            while (batch_states.size() < batch_size) {
                Game env = generate_new_game(env_size);
                vector<torch::Tensor> episode_states;
                vector<torch::Tensor> episode_actions;
                vector<torch::Tensor> episode_action_masks;
                vector<float> episode_rewards;
                
                tie(episode_states, episode_actions, episode_action_masks, episode_rewards) = self_play(env);
                
                batch_states.insert(batch_states.end(), episode_states.begin(), episode_states.end());
                batch_actions.insert(batch_actions.end(), episode_actions.begin(), episode_actions.end());
                batch_action_masks.insert(batch_action_masks.end(), episode_action_masks.begin(), episode_action_masks.end());
                batch_rewards.insert(batch_rewards.end(), episode_rewards.begin(), episode_rewards.end());
                games++;
                current_steps += episode_states.size();
            }

            // Limit batch size to what we've collected
            int actual_batch_size = min(batch_size, (int)batch_states.size());

            // Convert to tensors and move to device
            auto states_tensor = torch::stack(vector<torch::Tensor>(batch_states.begin(), 
                batch_states.begin() + actual_batch_size)).to(device);
            auto actions_tensor = torch::stack(vector<torch::Tensor>(batch_actions.begin(), 
                batch_actions.begin() + actual_batch_size)).to(device);
            auto masks_tensor = torch::stack(vector<torch::Tensor>(batch_action_masks.begin(), 
                batch_action_masks.begin() + actual_batch_size)).to(device);
            
            // Convert rewards to tensor
            vector<float> batch_rewards_subset(batch_rewards.begin(), batch_rewards.begin() + actual_batch_size);
            auto rewards_tensor = torch::from_blob(
                batch_rewards_subset.data(), 
                {actual_batch_size, 1}, 
                torch::kFloat32
            ).clone().to(device);

            // Normalize rewards (create a new tensor instead of modifying in-place)
            auto rewards_mean = rewards_tensor.mean();
            auto rewards_std = rewards_tensor.std() + 1e-8;
            auto normalized_rewards = (rewards_tensor - rewards_mean) / rewards_std;

            // Update old policy at the beginning
            actor_old.clone_model(actor);
            
            // Get old action probabilities (before any updates)
            auto old_log_probs = actor_old.forward(states_tensor, masks_tensor).detach();
            
            // Declare loss variables
            torch::Tensor actor_loss;
            torch::Tensor critic_loss;
            
            for (int update = 0; update < num_policy_updates; update++) {
                auto values = critic.forward(states_tensor);
                auto advantages = normalized_rewards - values.detach();
                auto adv_mean = advantages.mean();
                auto adv_std = advantages.std() + 1e-8;
                auto normalized_advantages = (advantages - adv_mean) / adv_std;
                auto curr_log_probs = actor.forward(states_tensor, masks_tensor);
                auto ratio = torch::exp(curr_log_probs - old_log_probs);
                auto surr1 = ratio * normalized_advantages;
                auto surr2 = torch::clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * normalized_advantages;
                actor_loss = -torch::min(surr1, surr2).mean();


                actor_optimizer.zero_grad();
                actor_loss.backward(torch::Tensor(), true, false);
                actor_optimizer.step();
                
                critic_loss = torch::mse_loss(values, normalized_rewards);
                
                critic_optimizer.zero_grad();
                critic_loss.backward(torch::Tensor(), true, false);
                critic_optimizer.step();
            }
            
            if (iterations % 100 == 0) {
                cout << "Iteration: " << iterations 
                     << " Actor Loss: " << actor_loss.item<float>() 
                     << " Critic Loss: " << critic_loss.item<float>() 
                     << " Avg Reward: " << rewards_mean.item<float>()
                     << " Games played: " << games << endl;
                save_model("current");
                cout << "Model saved as current" << endl;
            }
            iterations++;
        }
        
        save_model("final");
        cout << "Training complete after " << iterations << " iterations, " << current_steps << " steps" << endl;
    }

}; 

vector<int> get_random_move(Game& game){
    vector<int> valid_moves = game.get_valid_moves(); 
    vector<vector<int> > moves;
    int n = game.state.size(); 
    if (valid_moves.empty()){
        cout << "No Valid Moves Left" << endl;
        return vector<int>(2, -1); 
    }
    for (int i=0;i < valid_moves.size();i++){
        if (valid_moves[i] == 1){
            moves.push_back(vector<int>{i/n,i%n}); 
        }
    }; 

    random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, moves.size()-1); 
    return moves[dis(gen)]; 
}
#endif