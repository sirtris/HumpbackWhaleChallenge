import tensorflow as tf
import pandas as pd
import numpy
import random

#Basic implementation of a neural network in tensorflow
class testClassifier:
    def __init__(self,train_df,train_dir,test_dir):
        self.train_df = train_df
        self.train_dir = train_dir
        self.test_dir = test_dir

    def classify(self): #TODO: get source
        #Define data

        X = [[[0,0],[0,1]],[[1,0],[0,1]]]
        Y = [[1],[0]]#]

        #Define network input
        x_ = tf.placeholder(tf.float32,shape=[2,2,2]) #TODO: Change to image shape
        y_ = tf.placeholder(tf.float32,shape=[2,1]) #TODO: Change 4 to nr of images

        #Define the neural network classifier.
        hu = 3 #nr of hidden units
        w1 = tf.Variable(tf.random_uniform([2,hu],-1.0,1.0)) #1st layer weights
        b1 = tf.Variable(tf.zeros([hu])) #1st layer bias
        mm1 = tf.scan(lambda a, x: tf.matmul(x,w1),x_) #matrix multiplication for 1st layer
        o = tf.nn.sigmoid(mm1 + b1) #1st layer output
        w2 = tf.Variable(tf.random_uniform([hu,1],-1.0,1.0))
        b2 = tf.Variable(tf.zeros([1])) #1st layer bias
        mm2 = tf.scan(lambda a, x: tf.matmul(x,w2),o) #matrix multiplication for 2nd layer
        y = tf.nn.sigmoid(mm2 + b2) #output
        epochs = 5000 #nr of epochs

        #Define the cost and step functions
        cost = tf.reduce_sum(tf.square(y_ - y),reduction_indices=[0])
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

        #Start the sessions
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #Train the network
        for i in range(epochs):
            sess.run(train_step,feed_dict={x_: X, y_: Y})
            #print('Epoch = ',i,', Cost',sess.run(cost, feed_dict={x_: X, y_: Y}))

        #View the results
        correct_prediction = tf.cast(abs(y_-y),'float')
        accuracy = tf.reduce_mean(correct_prediction)
        yy,aa = sess.run([y,accuracy],feed_dict={x_:X,y_:Y})
        print('Output = ' + str(yy) + '\nAccuracy = ' + str(aa))

def main():
    c = testClassifier(pd.read_csv('./data/train.csv'),'data/train/','data/test/')
    c.classify()

if(__name__ == '__main__'):
    main()