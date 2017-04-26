import numpy as np
import tensorflow as tf

class DQN_CMM:
    """
    A DQN which uses a conditional minimax while training.
    """

    def __init__(self,input_shape,n_outputs,replay_tbl_size=20000,update_size=100,alpha=1e-4,gamma=0.6):
        """
        Initializes DQN parameters
        :param input_shape: A tuple in form (height,width,depth) as the shape of the input
        :param replay_tbl_size: The total number of records to maintain in the replay table (int)
        :param replay_tbl_step: The number of steps at which to
        :param alpha: The learning rate (float)
        """
        self.input_shape = input_shape
        self.n_outputs = n_outputs

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
        self.alpha = alpha
        self.gamma = gamma

        # Variables needed for replay table
        self.transition_count = 0
        self.replay_table = None
        self.replay_size = replay_tbl_size
        self.update_size = update_size


        # Build the network architecture
        self._build_network()

    def record_state(self,state):
        """
        Records the given state in the replay table
        :param state: The input state as defined by the replay table
        """
        self.replay_table[self.transition_count % self.replay_size] = state
        self.transition_count += 1

    def predict(self,feature_vector):
        q_values = self.sess.run(self.Q_values, feed_dict={self.input_matrix:feature_vector})
        return q_values

    def train(self):
        """
        Train the network based on replay table information
        """
        if self.transition_count >= self.replay_size/2:
            if self.transition_count == self.replay_size/2:
                print("Replay Table Ready")

            random_tbl = np.random.choice(self.replay_table[:min(self.transition_count, self.replay_size)],
                                       size=self.update_size)

            # Get the information from the replay table
            feature_vectors = np.vstack(random_tbl['state'])
            actions = random_tbl['action']
            next_feature_vectors = np.vstack(random_tbl['next_state'])
            rewards = random_tbl['reward']
            next_turn_vector = random_tbl['had_next_turn']

            # Get the indices of the non-terminal states
            non_terminal_ix = np.where([~np.any(np.isnan(next_feature_vectors), axis=(1, 2, 3))])[1]
            next_turn_vector[next_turn_vector == 0] = -1

            q_current = self.predict(feature_vectors)
            # Default q_next will be all zeros (this encompasses terminal states)
            q_next = np.zeros([self.update_size, self.n_outputs])
            q_next[non_terminal_ix] = self.predict(next_feature_vectors[non_terminal_ix])

            # The target should be equal to q_current in every place
            target = q_current.copy()

            # Apply hyperbolix tangent non-linearity to reward
            rewards = np.tanh(rewards)

            # Only actions that have been taken should be updated with the reward
            # This means that the target - q_current will be [0 0 0 0 0 0 x 0 0....]
            # so the gradient update will only be applied to the action taken
            # for a given feature vector.
            # The next turn vector controls for a conditional minimax. If the opponents turn is next,
            # The value of the next state is actually the negative maximum across all actions. If our turn is next,
            # The value is the maximum.
            target[np.arange(len(target)), actions] += (rewards + self.gamma * next_turn_vector * q_next.max(axis=1))

            #Update the model
            self.sess.run(self.update_model,feed_dict={self.input_matrix:feature_vectors,self.target_Q:target})

    def save_model(self,checkpoint_name=None):
        self.saver.save(self.sess, checkpoint_name)

    def load_model(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def _build_network(self):
        """Builds the DQN Architecture t used as the Q Function"""
        # Sets up the replay table for the DQN
        r, c, d = self.input_shape
        self.replay_table = np.rec.array(np.zeros(self.replay_size,
                                                  dtype=[('state', '(1,{},{},{})float32'.format(r, c, d)),
                                                         ('action', 'int8'),
                                                         ('next_state', '(1,{},{},{})float32'.format(r, c, d)),
                                                         ('reward', 'float32'),
                                                         ('had_next_turn', 'int8')]))
        # Set up the graph
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Set relevent parameters
        conv1_shape = [3, 3, 4, 16]
        conv2_shape = [1, 1, conv1_shape[-1], 32]
        fc_size = 256

        # Input placeholder
        self.input_matrix = tf.placeholder(tf.float32, [None, r, c, d], name='X')

        # Set up weights and biases for convolutional layers
        W1 = tf.Variable(tf.truncated_normal(conv1_shape, stddev=0.1), name='W1')
        B1 = tf.Variable(tf.zeros(conv1_shape[-1]), name='B1')
        W2 = tf.Variable(tf.truncated_normal(conv2_shape, stddev=0.1), name='W2')
        B2 = tf.Variable(tf.zeros([conv2_shape[-1]]), name='B2')

        # Helper function for convolutional layers
        def conv2d(x, W, strides=[1, 1, 1, 1]):
            return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

        # Create convolutional layers
        h1 = tf.nn.elu(tf.add(conv2d(self.input_matrix, W1, strides=[1, 1, 1, 1]), B1), name='Conv1')
        h2 = tf.nn.elu(tf.add(conv2d(h1, W2), B2), name='Conv2')

        # Create flattened layer
        h2_shape = h2.get_shape().as_list()[1:]
        flattened = tf.reshape(h2, [-1, np.prod(h2_shape)], name='Flattened')

        # Create FC and output variables
        W3 = tf.Variable(tf.truncated_normal([flattened.get_shape().as_list()[1], fc_size], stddev=0.1), name='W3')
        B3 = tf.Variable(tf.zeros([fc_size]), name='B3')
        W4 = tf.Variable(tf.truncated_normal([fc_size, fc_size], stddev=0.1), name='W4')
        B4 = tf.Variable(tf.zeros([fc_size]), name='B4')
        outputW = tf.Variable(tf.truncated_normal([fc_size, self.n_outputs], stddev=0.1), name='outputW')
        outputB = tf.Variable(tf.zeros([self.n_outputs]), name='outputB')

        # Create FC and output layer
        h3 = tf.nn.elu(tf.add(tf.matmul(flattened, W3), B3), name='FC')
        h4 = tf.nn.elu(tf.add(tf.matmul(h3, W4), B4), name='FC2')
        output = tf.tanh(tf.add(tf.matmul(h4, outputW), outputB), name='output')

        # Q values are represented by the output tensor
        self.Q_values = output

        # Create update structure
        self.target_Q = tf.placeholder(tf.float32, [None, self.n_outputs], 'Target')
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
        self.optimizer = tf.train.RMSPropOptimizer(self.alpha)

        # Clip gradients to prevent gradient explosion
        gradients = self.optimizer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.update_model = self.optimizer.apply_gradients(clipped_gradients)

        # Saves and loads models
        self.saver = tf.train.Saver()

        # Specific gradients for logging purposes only
        self.output_gradient = self.optimizer.compute_gradients(self.loss, [outputW])
        self.convolutional_gradient = self.optimizer.compute_gradients(self.loss, [W1])

        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)