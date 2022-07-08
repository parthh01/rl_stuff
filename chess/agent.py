
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
    model_name = None):
        self.env = env 
        self.input_shape = env.state_serialization().shape[1:] # first dimension is input_size
        self.params = params 
        self.value_network = models.load_model(os.path.join(MODEL_DIR,model_name)) if model_name else self.build_model()
        self.target_network = models.load_model(os.path.join(MODEL_DIR,model_name)) if model_name else self.build_model()
        self.sync_networks()
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
    
    def sync_networks(self):
        self.target_network.set_weights(self.value_network.get_weights())


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
    

    def learn(self,num_episodes):
        training_history = []
        for e in range(num_episodes):
            self.env.board.reset()
            #color = random.choice([1,0]) #1 is white but self play so doesnt matter 
            ply = 0 
            #episode_replay = []
            terminal = False 
            mse = []
            while not terminal: 
                s = self.env.state_serialization()
                a = self.e_greedy_policy() # uses current state by default 
                self.env.board.push(a)
                r = self.env.reward_function(self.env.board)
                s_prime = self.env.state_serialization() 
                terminal = self.env.board.outcome() is not None
                target_val = self.target_network.predict(tf.cast(s_prime,tf.float16))[0][0]
                t = r if terminal else r + (self.params['gamma']*target_val)
                hist = self.value_network.fit(s,np.array([[t]]),epochs=1,verbose=0)
                #episode_replay.append((s,a,r,s_prime,terminal))
                ply += 1
                mse.append(hist.history['loss'])
            
            training_history.append(np.mean(mse))
            self.sync_networks()

            if (e+1) %(max(1,num_episodes//10)) == 0: print(f"FINISHED TRAINING EPISODE {e+1} ({ply} ply's) (avg mse: {training_history[-1]})")
        
        print('saving model...')
        self.save_model('v1')
        print('model saved')


                
            




                



if __name__ == "__main__":
    from game import Game 
    game = Game()
    agent = Agent(game,model_name='v1') 
    agent.learn(3)
    #agent.save_model('v1')