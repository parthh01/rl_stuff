
import numpy as np 
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import progressbar 
import random 
import os 
import shutil 




class Agent: 
    
    def __init__(self,
    env,
    params = {
        'eps': 0.1, 
        'gamma': 0.9, 
        'lambda': 0.5,
        'alpha': 0.01
    },
    MODEL_DIR = 'models',
    model_path = None):
        self.env = env 
        self.input_shape = env.state_serialization().shape[1:] # first dimension is input_size
        self.params = params 
        self.value_network = models.load_model(model_path) if model_path else self.build_model()
        self.target_network = models.load_model(model_path) if model_path else self.build_model()
        self.target_network.set_weights(self.value_network.get_weights()) 
        self.MODEL_DIR = MODEL_DIR


    def build_model(self):
        model = Sequential() 
        model.add(Conv2D(16,(3,3),activation = 'relu',input_shape=self.input_shape,data_format='channels_first'),)
        model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=self.input_shape,data_format='channels_first'))
        model.add(Conv2D(64,(3,3),activation = 'relu',input_shape=self.input_shape,data_format='channels_first',strides=2))
        model.add(Flatten())
        model.add(Dense(32,activation='relu'))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.params['alpha']))
        #print(self.model.summary())
        return model 


    def e_greedy_policy(self,learning=True):
        action_space = self.env.action_space()
        if (np.random.rand() <= self.params['eps']) and learning: 
            return random.choice(action_space) 

        Q_a = []
        for move in action_space: 
            self.env.board.push(move)
            state = self.env.state_serialization()
            state = tf.cast(state,tf.float16) #for some reason uint8 not allowed 
            Q_a.append(self.value_network.predict(state)[0][0])
            self.env.board.pop() #undo the move 
            a_idx = np.argmax(Q_a) if self.env.board.turn else np.argmin(Q_a)
        return action_space[a_idx]

    def save_model(self,model_name = 'v0'):
        dirpath = os.path.join(self.MODEL_DIR,model_name)
        if os.path.exists(dirpath) and os.path.isdir(dirpath): shutil.rmtree(dirpath)
        self.value_network.save(dirpath)



if __name__ == "__main__":
    from game import Game 
    game = Game()
    agent = Agent(game)
    q_a = agent.e_greedy_policy()
    print(q_a)
    agent.save_model()