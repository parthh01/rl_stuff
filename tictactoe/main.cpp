#include "game.h"
#include "agent.h"
#include "nn.h"
#include <torch/torch.h>

using namespace std; 

// HYPERPARAMETERS
float GAMMA = 0.99; 
float LR = 1e-4; 
float BATCH_SIZE = 128; 
float EPSILON = 0.2; 
int TOTAL_STEPS = 1e6; 

int main(int argc, char* argv[]){
    torch::Device device(torch::kCPU);

    if (torch::mps::is_available()) {
        torch::Device device(torch::kMPS);
        std::cout << "MPS is available!" << std::endl;
    } else {
        std::cout << "MPS is not available." << std::endl;
    }
	if (argc != 2){
		cout << "Usage: " << argv[0] << " <board_size>" << endl; 
		return 1; 
	}
	int n = stoi(argv[1]); 
    vector<int> player_move; 
    vector<int> agent_move; 
    vector<int> valid_moves; 
    vector<torch::Tensor> states; 
    vector<torch::Tensor> actions; 
    torch::Tensor action_mask; 
    vector<torch::Tensor> action_masks; 
    torch::Tensor action_log_probs; 
    vector<float> discounted_rewards; 
	Game game = generate_new_game(n); 
    //PPOAgent agent(2,4,n,device,GAMMA);
    PPOAgent agent(2, 256, n, device, GAMMA, LR, EPSILON,"current");  // Increase hidden layer size
    // print the number of parameters of the agent
    cout << "Agent Parameters: " << endl; 
    cout << "Actor: " << count_parameters(agent.actor,false) << endl; 
    cout << "Critic: " << count_parameters(agent.critic,false) << endl; 
    //agent.save_model("current");
    tie(states, actions, action_masks, discounted_rewards) = agent.self_play(game);
    game.print_board(); 
    cout << "game was won by: " << game.winner << endl; 
    cout << "rewards: " << discounted_rewards << endl; 
    agent.learn(TOTAL_STEPS, BATCH_SIZE);
    game.print_state_tensor(states.back(), n); 

    cout << "States: " << states.size() << endl; 
    cout << "Actions: " << actions.size() << endl; 
    cout << "Rewards: " << discounted_rewards << endl; 
    game = generate_new_game(n);
	while (!game.over){
		game.print_board(); 
		player_move = game.get_player_move(); 
		game.play_move(player_move[0], player_move[1]); 
        if (game.over) break; 
		tie(agent_move, action_log_probs, action_mask) = agent.get_action(game); 
		game.play_move(agent_move[0], agent_move[1]); 
	}
    game.print_board(); 
    cout << "Game Over" << endl; 
    cout << "Winner: " << game.winner << endl; 
    return 0; 
}
