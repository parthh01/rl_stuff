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
    torch::optim::Adam actor_optimizer; 
    SimpleNN critic; 
    torch::optim::Adam critic_optimizer; 
    torch::Device device; 

    PPOAgent(int hidden_layers, int hidden_nodes, Game& env, torch::Device device) : 
        actor(SimpleNN((env.state.size()*env.state.size()) + 1, hidden_layers, hidden_nodes, env.state.size()*env.state.size(), device)), 
        critic(SimpleNN((env.state.size()*env.state.size()) + 1, hidden_layers, hidden_nodes, 1, device)), 
        actor_optimizer(torch::optim::Adam(actor.parameters(), 0.001)), 
        critic_optimizer(torch::optim::Adam(critic.parameters(), 0.001)), 
        device(device)
    {
    }

    vector<int> get_action(Game& env){
        vector<int> valid_moves = env.get_valid_moves(); 
        int n = env.state.size(); 
        torch::Tensor action_mask = torch::from_blob(valid_moves.data(), {n*n}, torch::kInt32).to(device); 
        torch::Tensor action_log_probs = actor.forward(env.get_state_tensor(), action_mask); 
        torch::Tensor action = torch::argmax(action_log_probs); 
        return vector<int>{action.item().toInt()/n, action.item().toInt()%n}; 
    }

    tuple<vector<torch::Tensor>, vector<vector<int> >, vector<int>> self_play(Game& env,bool first=true){
        vector<torch::Tensor> states; 
        vector<vector<int> > actions; 
        vector<int> rewards;
        int n = env.state.size(); 
        vector<int> action; 
        if (first){
            action = get_action(env); 
            env.play_move(action[0], action[1]); 
        }
        while (env.winner == ' '){
            vector<int> action = get_action(env); 
            states.push_back(env.get_state_tensor()); 
            actions.push_back(action);
            int reward = env.play_move(action[0], action[1]); 
            rewards.push_back(first ? reward : -reward); 
            if (env.winner != ' ') break; 
            action = get_action(env); 
            env.play_move(action[0], action[1]); 
        }

        return make_tuple(states,actions,rewards); 
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
    if (moves.empty()){
        cout << "No Valid Moves Left" << endl;
        return vector<int>(2, -1); 
    }

    random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, moves.size()-1); 
    return moves[dis(gen)]; 
}
#endif