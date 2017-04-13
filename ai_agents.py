# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:06:56 2017

@author: matthew
"""
from naive_players import Player
import numpy as np
from numpy import random
import tensorflow as tf
from dots_and_boxes import DotsAndBoxes

class TDLearner(Player):
    """DQN"""    
    def __init__(self, name, alpha=1e-4, epsilon=0.1, gamma=0.95,lmbda=1, d_size=20000, update_size=100):
        """Initializes the SARSA agent.
        Attributes:
            weights: weights used by the linear function approximator. One weight matrix per action.
            epsilon: epsilon value for an epsilon greedy improvement policy (0-1)
            alpha: learning rate [0-1]
            lmbda: lambda for SARSA(lambda) improvement policy - determines strength of eligibility trace [0-1]
            eligibility: eligibility table. One per action.
            gamma: Decay rate of reward
            action: Next action to be taken in the environment
        """
        super().__init__()
        self.name = name
        self.learning = True
        
        # Hyper-parameter initialization
        self.epsilon = epsilon
        self.alpha = alpha
        self.lmbda = lmbda
        self.gamma = gamma

        # Deep replay table
        self.transition_count = 0
        self.replay_table = None
        self.replay_size = d_size
        self.update_size = update_size

        # Variables needed for Q function interaction
        self.sess = None
        self.Q_values = None
        self.input_matrix = None
        self.loss = None
        self.saver = None
        self.target_Q = None
        self.optimizer = None
        self.update_model = None
        self.output_gradient = None
        self.convolutional_gradient = None
        self.keep_prob = None
        
        # For TD-Update
        self.last_state = None
        self.last_action = None

        #Logging
        self.log_file = None

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, environment):
        """Sets the environment
        If there is no Q-model, builds one based on the
        environments action space"""
        self._environment = environment
        if self.sess is None:
            self._build_model()

    def act(self):
        """Chooses an action according to the DQN and performs a TD update if possible
        """
        state = self._environment.state
        # Generate a feature vector
        feature_vector = self.generate_input_vector(state, True)
        
         # Choose an action
        action = self.choose_action(feature_vector)
        
        reward, next_state = self._environment.step(action)
        
        if self.learning:
            my_turn_next = self._environment.current_player == self
            
            if next_state is not None:
                next_feature_vector = self.generate_input_vector(state, my_turn_next)
            else:
                next_feature_vector = None
    
            # TD-Update with 0
            self.td_update(feature_vector, action, next_feature_vector, reward)
        
        return action
        
    def observe(self, previous_state, action, reward):
        """Observes the action that the opponent took"""
        
        if self.learning:
            feature_vector = self.generate_input_vector(previous_state,False)
            
            my_turn_next = self._environment.current_player == self
            
            if self._environment.state is not None:
                next_feature_vector = self.generate_input_vector(self._environment.state,my_turn_next)
            else:
                next_feature_vector = None
            
            self.td_update(feature_vector, action, next_feature_vector, reward)


    def generate_input_vector(self, state, my_turn):
        """Generates an input vector based on an environment state."""
        is_my_turn = int(my_turn)
        r, c, d = state.shape
        input_vector = np.resize(state, [1, r, c, d + 1])
        input_vector[:,:,:,-1] = is_my_turn
        
        return input_vector

            
    def choose_action(self, feature_vector):
        """Chooses an action based on an epsilon-greedy SARSA policy"""
        if random.random() < self.epsilon and self.learning:
            chosen_action = np.random.choice(self._environment.valid_actions)
        else:
            with open('Q_log.txt','w') as file:
                q_values = self.get_Q_values(feature_vector)[0]
                print ("Q: {}".format(q_values))
                max_valid_q = q_values[self._environment.valid_actions].max()
                best_actions = np.where(q_values == max_valid_q)[0]
                chosen_action = random.choice([action for action in best_actions if action in self._environment.valid_actions])
        return chosen_action

    def get_Q_values(self, feature_vector):
        """Returns the Q_values of a state"""
        q_values = self.sess.run(self.Q_values, feed_dict={self.input_matrix: feature_vector})
        return q_values
       
    def td_update(self, current_state, last_action, next_state, reward):
        """Updates the Q_function according to the SARSA update algorithm"""
        # Update the replay table
        self.replay_table[self.transition_count % self.replay_size] = (current_state, last_action, next_state, reward)
        self.transition_count = (self.transition_count + 1)

        # Don't start learning until transition table has some data
        if self.transition_count >= self.update_size * 20:
            if self.transition_count == self.update_size * 20:
                print (self.name)
                print("Replay Table is Ready\n")
            
            # Get a random subsection of the replay table for mini-batch update
            random_tbl = random.choice(self.replay_table[:min(self.transition_count,self.replay_size)],size=self.update_size)
            feature_vectors = np.vstack(random_tbl['state'])
            actions = random_tbl['action']
            next_feature_vectors = np.vstack(random_tbl['next_state'])
            rewards = random_tbl['reward']
            
            # Get the indices of the non-terminal states
            non_terminal_ix = np.where([~np.any(np.isnan(next_feature_vectors),axis=(1,2,3))])[1]
                                       
            q_current = self.get_Q_values(feature_vectors)
            # Default q_next will be all zeros (this encompasses terminal states)
            q_next = np.zeros([self.update_size,len(self._environment.action_list)])
            q_next[non_terminal_ix] = self.get_Q_values(next_feature_vectors[non_terminal_ix])
            
            # The target should be equal to q_current in every place
            target = q_current.copy()
            
            # Only actions that have been taken should be updated with the reward
            # This means that the target - q_current will be [0 0 0 0 0 0 x 0 0....] 
            # so the gradient update will only be applied to the action taken
            # for a given feature vector.
            target[np.arange(len(target)), actions] += (rewards + self.gamma*q_next.max(axis=1))
            
            # Logging
            if self.log_file is not None:
                print ("Current Q Value: {}".format(q_current),file=self.log_file)
                print ("Next Q Value: {}".format(q_next),file=self.log_file)
                print ("Current Rewards: {}".format(rewards),file=self.log_file)
                print ("Actions: {}".format(actions),file=self.log_file)
                print ("Targets: {}".format(target),file=self.log_file)
                
                # Log some of the gradients to check for gradient explosion
                loss, output_grad, conv_grad = self.sess.run([self.loss,self.output_gradient,self.convolutional_gradient],
                                                             feed_dict={self.target_Q: target, self.input_matrix: feature_vectors})
                print ("Loss: {}".format(loss),file=self.log_file)
                print ("Output Weight Gradient: {}".format(output_grad),file=self.log_file)
                print ("Convolutional Gradient: {}".format(conv_grad),file=self.log_file)
            
            # Update the model
            self.sess.run(self.update_model, feed_dict={self.target_Q: target, self.input_matrix: feature_vectors})

    def save_model(self,checkpoint_name=None, global_step=None):
        """Saves a model and returns the name of the checkpoint"""
        self.saver.save(self.sess, checkpoint_name, global_step=global_step)

    def load_model(self,model_dir):
        """Restores a model from checkpoint"""
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def get_variable(self,var_name):
        """Returns one of the neural net variables"""
        variable = self.sess.run(var_name)
        return variable

    def list_variables(self):
        """Lists present variables in the neural net"""
        return [v.name for v in tf.trainable_variables()]
        
    def _build_model(self):
        """Builds the neural net used as the Q Function"""
        # Sets up the replay table for the DQN
        r,c,d = self._environment.state.shape
        self.replay_table = np.rec.array(np.zeros(self.replay_size, 
                                                  dtype=[('state','(1,{},{},{})float32'.format(r,c,d+1)),
                                                              ('action', 'int8'),
                                                              ('next_state', '(1,{},{},{})float32'.format(r,c,d+1)),
                                                              ('reward','float32')]))
    
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        # Set relevent parameters
        output_size = len(self._environment.action_list)
        conv1_shape = [3, 3, 5, 16]
        conv2_shape = [1, 1, conv1_shape[-1], 32]
        fc_size = 256

        # Input placeholder
        row, column, depth = self._environment.state.shape
        self.input_matrix = tf.placeholder(tf.float32, [None, r, c, d + 1],name='X')

        # Set up weights and biases for convolutional layers
        W1 = tf.Variable(tf.truncated_normal(conv1_shape, stddev=0.1),name='W1')
        B1 = tf.Variable(tf.zeros(conv1_shape[-1]),name='B1')
        W2 = tf.Variable(tf.truncated_normal(conv2_shape, stddev=0.1),name='W2')
        B2 = tf.Variable(tf.zeros([conv2_shape[-1]]),name='B2')

        # Helper function for convolutional layers
        def conv2d(x,W,strides=[1, 1, 1, 1]):
            return tf.nn.conv2d(x, W, strides=strides, padding='SAME')
        
        # Create convolutional layers
        h1 = tf.nn.relu(tf.add(conv2d(self.input_matrix, W1, strides=[1, 1, 1, 1]), B1),name='Conv1')
        h2 = tf.nn.relu(tf.add(conv2d(h1, W2), B2),name='Conv2')

        # Create flattened layer
        h2_shape = h2.get_shape().as_list()[1:]
        flattened = tf.reshape(h2, [-1, np.prod(h2_shape)], name='Flattened')

        # Create FC and output variables
        W3 = tf.Variable(tf.truncated_normal([flattened.get_shape().as_list()[1], fc_size], stddev=0.1),name='W3')
        B3 = tf.Variable(tf.zeros([fc_size]),name='B3')
        W4 = tf.Variable(tf.truncated_normal([fc_size, fc_size], stddev=0.1), name='W4')
        B4 = tf.Variable(tf.zeros([fc_size]), name='B4')
        outputW = tf.Variable(tf.truncated_normal([fc_size, output_size], stddev=0.1),name='outputW')
        outputB = tf.Variable(tf.zeros([output_size]),name='outputB')

        # Create FC and output layer
        h3 = tf.nn.relu(tf.add(tf.matmul(flattened, W3), B3),name='FC')
        h4 = tf.nn.relu(tf.add(tf.matmul(h3,W4), B4), name='FC2')
        output = tf.add(tf.matmul(h4, outputW), outputB, name='output')
        
        # Q values are represented by the output tensor
        self.Q_values = output

        # Create update structure
        self.target_Q = tf.placeholder(tf.float32,[None,output_size],'Target')
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
        self.optimizer = tf.train.RMSPropOptimizer(self.alpha)
        
         # Clip gradients to prevent gradient explosion
        gradients = self.optimizer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(grad,-1.,1.), var) for grad, var in gradients]
        self.update_model = self.optimizer.apply_gradients(clipped_gradients)
        
        # Saves and loads models
        self.saver = tf.train.Saver()
        
        # Specific gradients for logging purposes only
        self.output_gradient = self.optimizer.compute_gradients(self.loss, [outputW])
        self.convolutional_gradient = self.optimizer.compute_gradients(self.loss, [W1])
        
        # Initialize all variables
        init = tf.global_variables_initializer()

        # Run variable initialization
        self.sess.run(init)
        

if __name__ == '__main__':
    s = TDLearner('train')
    s2 = TDLearner('test')
    s2.learning = False
    
    env = DotsAndBoxes()
    env.player1 = s
    env.player2 = s2
    env.play()
        
        
        
        
        