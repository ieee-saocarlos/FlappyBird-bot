import random
import datetime
import itertools
import numpy as np
import tensorflow as tf

class Passaro:
    def __init__(self, name):
        
        self.score = 0.0
        self.name = name
        self.vivo = True
        self.passaro_mov = 0
        self.action_space = 2
        observation = [0.5, 0.82, 0.4]
        input_shape = len(observation)
        units1 = 2
        units2 = 5
        units3 = 1

        self.brain = tf.keras.models.Sequential([
#            tf.keras.layers.Dense(units = units1, input_shape = [input_shape],
#                                  activation = "relu",
#                                  kernel_initializer = tf.keras.initializers.RandomUniform(
#                                      minval = -6, maxval = 6, seed = None),
#                                  bias_initializer = tf.keras.initializers.RandomUniform(
#                                      minval = -6, maxval = 6, seed = None)
#                                 ),
            
#            tf.keras.layers.Dense(units = units2, activation = "relu",
#                                 kernel_initializer = tf.keras.initializers.RandomUniform(
#                                     minval=-6, maxval=6, seed=None),
#                                 bias_initializer = tf.keras.initializers.RandomUniform(
#                                     minval=-6, maxval=6, seed=None)),
            
            tf.keras.layers.Dense(units = units3, input_shape = [input_shape],
                                  activation = "tanh",
                                  kernel_initializer = tf.keras.initializers.RandomUniform(
                                      minval=-10, maxval=10, seed=None),
#								  use_bias = False,
                                  bias_initializer = tf.keras.initializers.RandomUniform(
                                      minval=-10, maxval=10, seed=None)
                                 )
        ])

        self.weights_dict = {i: weight.shape for i, weight in enumerate(self.brain.get_weights())}
        self.shapes = [value[0] * value[1] if len(value) == 2 else value[0] for value in self.weights_dict.values()]
        index = np.cumsum(self.shapes)
        self.start_end = [(0, index[i]) if i == 0 else (index[i-1], index[i]) for i in range(len(index))]

    def reborn(self, solution):

        tensor_list = [
            np.reshape(
                solution[index[0]:index[1]], self.weights_dict[i]
                ) 
            for i, index in enumerate(self.start_end)
        ]

        self.brain.set_weights(tensor_list)

        return self
    
    def weights_vector(self):
        
        weights_array = [list(weight.flatten()) for weight in self.brain.get_weights()]
        return list(itertools.chain(*weights_array))
        
    def update(self):
        self.rect.centery += self.passaro_mov
        return self.rect.centery
        
    def survived(self):
        self.score += 0.005
    
    def passed(self):
        self.score += 0.25 
        
    def think(self, observations):
        self.observations = observations
        with tf.device("/gpu:1"):
            thinking = self.brain(tf.Variable(self.observations))
        return thinking