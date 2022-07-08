
from tensorflow.keras import models 
from game import Game 
import numpy as np 
import os
import tensorflow as tf 

class Player:

    def __init__(self,model_dirpath,white=False):
        self.value_network = models.load_model(model_dirpath)
        self.color = white #white is true black is false 


    def policy_evaluation(self,env):
        """
        returns the best move according to the policy for whoever's turn it currently is in the game. 
        """
        action_space = env.action_space()
        heuristic = []
        for a in action_space:
            env.board.push(a)
            state = env.state_serialization()
            heuristic.append(self.value_network.predict(tf.cast(state,tf.float16))[0][0])
            env.board.pop()
            a_idx = np.argmax(heuristic) if env.board.turn else np.argmin(heuristic)

        return action_space[a_idx]



if __name__ == "__main__":
    game = Game() 
    ai = Player('models/v1') 
    game.play(ai)