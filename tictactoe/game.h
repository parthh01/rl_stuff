#ifndef GAME_H
#define GAME_H

#include <vector> 
#include <cassert> 
#include <iostream>
#include <torch/torch.h>
#include <string>

using namespace std; 

struct Game {
	vector<vector<char> > state; 
	bool turn = true; 
	char winner = ' ';
	bool over = false;
	int WIN_REWARD = 10;

	torch::Tensor get_state_tensor() const {
		int n = state.size(); 
		torch::Tensor state_tensor = torch::zeros((n*n) + 1, torch::kFloat32); // the board state + 1 feature denoting which char the player is
		for (int i=0;i < n;i++){
			for (int j=0;j < n;j++){
				state_tensor[i*n + j] = state[i][j] == 'X' ? 1 : -1; 
			}
		}
		state_tensor[n*n] = turn ? 1 : 0; // essentially a one hot encoding of the character who's turn it is
		return state_tensor; 
	}

	vector<int> get_valid_moves() const {
		int n = state.size(); 
		vector<int> moves(n*n, 0); 
		for (int i=0;i < n;i++){
			for (int j=0;j < n;j++){
				if (state[i][j] == ' '){
					moves[i*n + j] = 1; 
				}
			}
		}
		return moves; 
	}


	void visualize_board(vector<vector<char> > board) const {
		int n = board.size(); 
		for (int i=0;i < n;i++){
			for(int j=0;j < n;j++){
				cout << " " << state[i][j] << " ";
				if (j < n-1){
					cout << "|"; 
				}
			}
			cout << endl; 
			if (i < n-1){
				for (int j=0;j < n;j++){
					cout << "---"; 
					if (j < n-1){
						cout << "+"; 
					}
				}
				cout << endl; 
			}
		}
	}

	void print_board() const {
		visualize_board(state); 
	}

	void check_winner() {
		int n = state.size(); 
		for (int i=0;i < n;i++){
			if (state[i][0] != ' '){
				bool row_win = true; 
				for (int j=1;j < n;j++){
					if (state[i][j] != state[i][0]){
						row_win = false; 
						break; 
					}
				}
				if (row_win) {
					winner = state[i][0]; 
					over = true; 
					return; 
				}
			}
		}

		//Check Columns 
		for (int j=0;j < n;j++){
			if (state[0][j] != ' '){
				bool col_win = true; 
				for (int i=1;i < n;i++){
					if (state[i][j] != state[0][j]){
						col_win = false; 
						break; 
					}
				}
				if (col_win) {
					winner = state[0][j]; 
					over = true; 
					return; 
				}
			}
		}

		//Check Diagonals 
		if (state[0][0] != ' '){
			bool diag_win = true; 
			for (int i=1;i < n;i++){
				if (state[i][i] != state[0][0]){
					diag_win = false; 
					break; 
				}
			}
			if (diag_win) {
				winner = state[0][0]; 
				over = true; 
				return; 
			}
		}

		if (state[0][n-1] != ' '){
			bool anti_diag_win = true; 
			for (int i=1;i < n;i++){
				if (state[i][n-i-1] != state[0][n-1]){
					anti_diag_win = false; 
					break; 
				}
			}
			if (anti_diag_win) {
				winner = state[0][n-1]; 
				over = true; 
				return; 
			}
		}

		// check if there are no empty spaces left
		for (int i=0;i < n;i++){
			for (int j=0;j < n;j++){
				if (state[i][j] == ' '){
					return; 
				}
			}
		}
		over = true; 
		return; 
	}

	vector<int> get_player_move() const {
		int row, col; 
		char player = turn ? 'X' : 'O'; 
		while (true){
			cout << "Player " << player << " enter your move (row col): "; 
			cin >> row >> col; 
			// check if the move is valid 
			if (row < 0 || row >= state.size() || col < 0 || col >= state[0].size() || state[row][col] != ' '){
				cout << "Invalid move. Try again." << endl; 
				continue; 
			}
			break; 
		}
		vector<int> result;
		result.push_back(row);
		result.push_back(col);
		return result;
	}

	float play_move(int row, int col){
		char current_player = turn ? 'X' : 'O';	
		if (!(row >= 0 && row < state.size() && col >= 0 && col < state[0].size() && state[row][col] == ' ')) {
			cout << "Invalid Move: row=" << row << ", col=" << col << endl;
			cout << "Current board state:" << endl;
			cout << "Valid moves: " << get_valid_moves() << endl;
			print_board();
			assert(false && "Invalid move detected");
		}
		state[row][col] = current_player; 
		turn = !turn; 
		check_winner(); 
		return winner == ' ' ? 0 : (winner == current_player ? WIN_REWARD : -WIN_REWARD); 
	}

	void print_state_tensor(torch::Tensor& state_tensor, int n) {
		vector<vector<char>> board(n, vector<char>(n, ' ')); 
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int val = state_tensor[i*n + j].item<int>();
				board[i][j] = val == 1 ? 'X' : val == -1 ? 'O' : ' ';
			}
		}
		visualize_board(board);
	}

}; 

Game generate_new_game(int n){
	Game new_game; 
	vector<vector<char> > board(n,vector<char>(n,' ')); 
	new_game.state = board; 
	new_game.turn = true; 
	new_game.winner = ' '; 
	new_game.over = false;
	return new_game; 
}; 













#endif