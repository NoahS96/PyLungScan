import tensorflow as tf
import numpy as np
import itertools
import math, os

class CNeuralNetwork:
    PX_SIZE = 0
    SLICE_COUNT = 0
    N_CLASSES = 2
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-3
    keep_rate = 0.8
    saver_path = None

    x = tf.placeholder('float')
    y = tf.placeholder('float')
    
    # __init__
    #   Parameters:
    #       image_size  -   The number of pixels an image has along the x and y axis
    #       slice_count -   The number of images along the z axis
    #       saver_path  -   Location to save the training model
    def __init__(self, image_size, slice_count, saver_path=None):
        self.PX_SIZE = image_size
        self.SLICE_COUNT = slice_count
        self.saver_path = saver_path


    # __conv3d__
    #   Parameters:
    #       x   -   The current data frame being processed  
    #       W   -   The weight of the synapse
    def __conv3d__(self, x, W):
        return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

    
    # __maxpool3d__
    #   Parameters:
    #       x   -   The current data frame being processed
    def __maxpool3d__(self, x):
        #                               Size of window
        return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


    # __convolutional_neural_network__
    #   Parameters:
    #       x   -   The data to be processed
    def __convolutional_neural_network__(self, x):
        
        # Get neuron count based on the image size, slice count, and number of pooling layers.
        neuron_count = ( math.ceil((self.PX_SIZE/(2*self.N_CLASSES))) * math.ceil((self.PX_SIZE/(2*self.N_CLASSES))) 
            * math.ceil((self.SLICE_COUNT/(2*self.N_CLASSES))) )

        weights = { 'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])), 
                    'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
                    #'W_fc':tf.Variable(tf.random_normal([neuron_count, 1024])),
                    'W_fc':tf.Variable(tf.random_normal([54080, 1024])),
                    'out':tf.Variable(tf.random_normal([1024, self.N_CLASSES]))}

        biases = {  'b_conv1':tf.Variable(tf.random_normal([32])),
                    'b_conv2':tf.Variable(tf.random_normal([64])),
                    'b_fc':tf.Variable(tf.random_normal([1024])),
                    'out':tf.Variable(tf.random_normal([self.N_CLASSES]))}
        
        # First convolutional layer
        x = tf.reshape(x, shape=[-1, self.PX_SIZE, self.PX_SIZE, self.SLICE_COUNT, 1])
        conv1 = tf.nn.relu(self.__conv3d__(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self.__maxpool3d__(conv1)

        # Second convolutional layer
        conv2 = tf.nn.relu(self.__conv3d__(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self.__maxpool3d__(conv2)

        # Dense layer
        #fc = tf.reshape(conv2, [-1, neuron_count])
        fc = tf.reshape(conv2, [-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    
    # train_neural_network
    #   Parameters:
    #       patient_data    -   Numpy array with image at 0 and label at 1
    #       num_epochs      -   Number of training sessions
    def train_neural_network(self, patient_generator, num_epochs):
        prediction = self.__convolutional_neural_network__(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(cost)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # If a path is specified, try to open it and restore the training model. If it fails
            # the file is empty so continue and save to it afterwards.
            if self.saver_path is not None and os.path.isfile(self.saver_path):
                try:
                    saver.restore(sess, self.saver_path)
                    print('Successfully restored training model')
                except Exception:
                    print('Couldn\'t restore from save path')

            total_loss = 0
            num_runs = 0
            patient_generator, epoch_generator = itertools.tee(patient_generator)


            for epoch in range(num_epochs):
                epoch_loss = 0

                epoch_generator, patient_generator  = itertools.tee(epoch_generator)
                for patient in patient_generator:
                    X = patient[0]
                    Y = patient[1]
                    _, loss = sess.run([optimizer, cost], feed_dict={self.x:X, self.y:Y})
                
                    epoch_loss += loss 
                    num_runs += 1
            
                print('Epoch [', epoch+1, '/', num_epochs, '] : loss ', epoch_loss)
                total_loss += epoch_loss

                if self.saver_path is not None:
                    print("Saving the training model")
                    saver.save(sess, self.saver_path)

            return total_loss



