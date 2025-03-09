#include "game.h"
#include "agent.h"
#include "nn.h"
#include <torch/torch.h>

using namespace std; 


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
    SimpleNN sample_actor((n*n) + 1,2,4,n*n,device);

	Game game = generate_new_game(n); 
	while (game.winner == ' '){
		game.print_board(); 
		player_move = game.get_player_move(); 
		game.play_move(player_move[0], player_move[1]); 
        if (game.winner != ' ') break; 
        valid_moves = game.get_valid_moves(); 
        torch::Tensor action_mask = torch::from_blob(valid_moves.data(), {n*n}, torch::kInt32).to(device); 
        torch::Tensor action_log_probs = sample_actor.forward(game.get_state_tensor(), action_mask); 
        //print the action log probs 
        cout << "Action Log Probs: " << action_log_probs << endl;
        cout << "Action Mask Tensor: " << action_mask << endl;
        // get the action with the highest probability 
        torch::Tensor action = torch::argmax(action_log_probs); 
        agent_move = vector<int>{action.item().toInt()/n, action.item().toInt()%n}; 
		//agent_move = get_random_move(game); 
		game.play_move(agent_move[0], agent_move[1]); 
	}
    game.print_board(); 
    cout << "Game Over" << endl; 

    return 0; 
}
