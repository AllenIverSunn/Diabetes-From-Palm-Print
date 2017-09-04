import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class NNClassifier:
    def __init__(self):
        pass

    def create_placeholders(self, n_x, n_y):
        X = tf.placeholder(tf.float32, [n_x, None], name='X')
        Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
        return X, Y

    def initialize_params(self, layers):
        '''
        @params:
        -layers: An array which contains neurons of each layer, layers[0] represents the input
                layer, and the last one represents the output layer.
        @returns:
        -params: A dictionary which contains weight and bias of each layer.
        '''
        tf.set_random_seed(1)
        params = {}
        for i, layer in enumerate(layers):
            if i == len(layers)-1:
                break
            params['W'+str(i+1)] = tf.get_variable('W'+str(i+1), [layers[i+1], layers[i]],\
                                                  initializer=tf.contrib.layers.xavier_initializer(seed=1))
            params['b'+str(i+1)] = tf.get_variable('b'+str(i+1), [layers[i+1], 1],\
                                                  initializer=tf.zeros_initializer)
        return params

    def forward_propogation(self, X, params, activation_funcs):
        '''
        @params:
        -X: Input matrix X
        -params: Obtained using initialize_params
        -activation_funcs: An function object array containing the names of activation function
                            correspondig to each layer. For the last layer, we want z_L instead
                            of a_L, so we just pass a whatever function to the last layer.
        @returns:
        -temp_z: The output of NN, which is y_hat in the course.
        '''
        depth = len(list(params.keys())) / 2
        temp_a = X
        for i in np.arange(0, depth):
            W = params['W'+str(int(i+1))]
            b = params['b'+str(int(i+1))]
            print('temp', temp_a.shape)
            print('W', W.shape)
            temp_z = tf.add(tf.matmul(W, temp_a), b)
            temp_a = activation_funcs[int(i)](temp_z)
        return temp_z

    def compute_cost(self, z_L, Y):
        '''
        @params:
        -z_L: The output value of the NN.
        -Y: Data labels.
        '''
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(z_L),\
                                                                        labels=tf.transpose(Y)))
        return cost

    def shuffle_minibatch(self, X_train, Y_train, minibatch_size):
        '''
        @params:
        -X_train, Y_train: Training data.
        -minibatch_size: The size of each minibatch.
        @returns:
        -X_minibatch, Y_minibatch: Shuffled minibatches.
        '''
        X_train_new = X_train.copy()
        Y_train_new = Y_train.copy()
        index = np.arange(X_train.shape[1])
        np.random.shuffle(index)
        X_train_new = X_train_new[:, index]
        Y_train_new = Y_train_new[:, index]
        X_minibatches = []
        Y_minibatches = []
        num_minibatches = X_train.shape[1] // minibatch_size
        for i in np.arange(0, num_minibatches-1, 1):
            X_minibatches.append(X_train_new[:, i*minibatch_size: (i+1)*minibatch_size])
            Y_minibatches.append(Y_train_new[:, i*minibatch_size: (i+1)*minibatch_size])
        X_minibatches.append(X_train_new[:, (num_minibatches-1)*minibatch_size:])
        Y_minibatches.append(Y_train_new[:, (num_minibatches-1)*minibatch_size:])
        return X_minibatches, Y_minibatches

    def model(self, X_train, Y_train, X_test, Y_test, layers, activation_funcs, learning_rate=0.01,
              mini_batch_size=64, num_epochs=1500, print_cost=True):
        '''
        @params:
        -X_train, Y_train, X_test, Y_test: These params are training data.
        -layers: Param which is about to be passed to initialize_params.
        -learning_rate: Default as 0.01.
        -mini_batch_size: Default as 64.
        -num_epochs: Default as 1500.
        -print_cost: Whether to print the process of trainging. Default as True.
        '''
        tf.set_random_seed(1)
        n_x, m = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        X, Y = self.create_placeholders(n_x, n_y)
        params = self.initialize_params(layers)
        z_L = self.forward_propogation(X, params, activation_funcs)
        cost = self.compute_cost(z_L, Y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            num_minibatches = int(m / mini_batch_size)
            for epoch in np.arange(0, num_epochs, 1):
                epoch_cost = 0
                # seed = 2
                X_minibatches, Y_minibatches = self.shuffle_minibatch(X_train, Y_train, mini_batch_size)
                count = 0
                for x_minibatch, y_minibatch in zip(X_minibatches, Y_minibatches):
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: x_minibatch, Y: y_minibatch})
                    # print(epoch_cost)
                    epoch_cost += minibatch_cost / num_minibatches
                    # print(str(epoch)+': '+str(count))
                    count += 1
                if print_cost and (epoch % 10 == 0):
                    print("Cost after epoch "+str(epoch)+': '+str(epoch_cost))
                if print_cost and (epoch % 5 == 0):
                    costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate = %f' % str(learning_rate))
        plt.show()
        params = sess.run(params)
        print('Params have been trained.')
        correct_prediction = tf.equal(tf.argmax(z_L), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Train accuracy: ', accuracy.eval({X: X_train, Y: Y_train}))
        print('Test accuracy: ', accuracy.eval({X: X_test, Y: Y_test}))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train = mnist.train.images.T
    Y_train = mnist.train.labels.T
    X_test = mnist.test.images.T
    Y_test = mnist.test.labels.T
    classifier = NNClassifier()
    classifier.model(X_train, Y_train, X_test, Y_test, [784, 20, 5, 10], [tf.nn.relu, tf.nn.relu, tf.nn.relu])
