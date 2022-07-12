from typing import Tuple, List
from tensorflow.keras import models,Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import numpy as np 
import tqdm
import collections 
import statistics
import os
import shutil


class BaseNetworkClass(Model):

    def __init__(self,model_name,*args,**kwargs):
        super().__init__()
        tf.config.run_functions_eagerly(True)
        self.MODEL_DIR = 'models'
        self.model_name = model_name
        self.dirpath = os.path.join(self.MODEL_DIR,self.model_name)
        tf.keras.utils.set_random_seed(9)
        self.kernel_initializer = tf.keras.initializers.GlorotNormal
    
    def instantiate_network(self,num_actions,hidden_layers,target=False):
        if self.model_exists():
            self.model = models.load_model(self.dirpath)
            return self.model.layers
        return self.build_network(num_actions,hidden_layers,target=target)


    def model_exists(self):
        return os.path.exists(self.dirpath) and os.path.isdir(self.dirpath)

    def sync_target_networks(self):
        for i in range(len(self.target_network)):
            self.target_network[i].set_weights(self.network[i].get_weights())
    
    def save_model(self):
        if self.model_exists(): shutil.rmtree(self.dirpath)
        self.save(self.dirpath)
    
    def build_network(self):
        pass 
        

class ActorNetwork(BaseNetworkClass):

    def __init__(self,model_name,num_actions, hidden_layers: List = [512,512,512],params = {'lr': 0.01, 'tau': 0.001}):
        super().__init__(model_name)
        self.params = params 
        #self.state_input_layer = InputLayer(input_shape=sample_state.shape)
        self.network = self.instantiate_network(num_actions,hidden_layers)
        self.target_network = self.instantiate_network(num_actions,hidden_layers,target=True)
        self.optimizer = Adam(learning_rate=self.params['lr'])
        self.sync_target_networks()

    
    def build_network(self,num_actions,hidden_layers: List = [128],target: bool = False):
        layers = [] #tf.keras.Input(shape=state_input_shape,name='state_input')
        i = 0 
        for l in hidden_layers:
            layers.append(Dense(l,activation = 'tanh',trainable = not target,kernel_initializer = self.kernel_initializer,name = f"l{i}_{'target' if target else ''}"))
            i += 1
        layers.append(Dense(num_actions,activation='tanh',trainable = not target,kernel_initializer = self.kernel_initializer,name = f"l{i}_{'target' if target else ''}"))
        return layers



    def call(self,state: tf.Tensor,target: bool = False) -> tf.Tensor: 
        layers = self.target_network if target else self.network
        A = layers[0](state)
        for l in layers[1:]:
            A = l(A)
        return A 
    

class CriticNetwork(BaseNetworkClass):

    def __init__(
        self,
        model_name,
        state_input_shape, 
        num_actions,
        hidden_layers: List =[512,512,512],
        params = {
            'gamma': 0.99,
            'lr': 0.01,
            'tau': 0.001
        }
        ):
        """
        actor/policy network will take the current state as an input, and output a tensor representing the action. 
        critic network takes the current state and action from the policy network to produce the Q value of the (s,a) pair
        critic network loss (r + gamma*critic(s_prime)) computed against the TARGET CRITIC (cached previous version of critic network to stabilize training)
        policy network loss is the (-ve) of the critic value from the critic network (AFTER critic has just been backpropped) 
        policy network is backpropped according to this loss 

        considerations: 
        polyak averaging for target network update? 

        """

        super().__init__(model_name)
        

        self.params = params 
        self.state_input = tf.keras.Input(shape=state_input_shape,name='state_input')
        self.action_input = tf.keras.Input(shape=(num_actions,),name='action_input')
        self.network = self.instantiate_network(0,hidden_layers)
        self.target_network = self.instantiate_network(0,hidden_layers,target=True)
        self.sync_target_networks()

        self.optimizer = Adam(learning_rate=self.params['lr'])
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    
    def build_network(self,num_actions,hidden_layers,target: bool = False):
        layers = [Concatenate(axis=-1,trainable = not target)] 
        i = 0
        for l in hidden_layers:
            layers.append(Dense(l,activation = 'tanh',trainable = not target,kernel_initializer = self.kernel_initializer,name = f"l{i}_{'target' if target else ''}"))
            i += 1
        layers.append(Dense(1,trainable = not target,kernel_initializer = self.kernel_initializer,name = f"l{i}_{'target' if target else ''}"))
        return layers

    def call(self,state: tf.Tensor,action: tf.Tensor,target:bool = False) -> tf.Tensor:
        layers = self.target_network if target else self.network
        A = layers[0]([state,action]) 
        for l in layers[1:]:
            A = l(A)
        return A 
    
    
        



            




            


