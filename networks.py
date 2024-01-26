import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation
import tensorflow as tf

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=128):
        super(ActorNetwork, self).__init__()
        
        self.conv1 = Conv1D(32, kernel_size=5, strides=2, activation='relu')
        self.conv2 = Conv1D(32, kernel_size=3, strides=2, activation='relu')
        self.Flatten = Flatten()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(2, activation=None)
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        self.log_std = tf.Variable(-0.5 * tf.ones((1, 2), dtype=tf.float32))


    def call(self, state):
        range = state[:,:3,:]
        range_input = tf.transpose(range, perm=[0, 2, 1])
        x = self.conv1(range_input)
        x = self.conv2(x)
        x = self.Flatten(x)
        dist = state[:,3:,0]
        x = self.fc1(x)
        x = tf.concat([x,dist],axis=-1)
        x = self.fc2(x)
        x = self.fc3(x)
        x = tf.concat([self.sigmoid(x[:, 0:1]), self.tanh(x[:, 1:])], axis=1)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=128):
        super(CriticNetwork, self).__init__()
        self.conv1 = Conv1D(32, kernel_size=5, strides=2, activation='relu')
        self.conv2 = Conv1D(32, kernel_size=3, strides=2, activation='relu')
        self.Flatten = Flatten()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):

        range = state[:,:3,:]
        range_input = tf.transpose(range, perm=[0, 2, 1])
        x = self.conv1(range_input)
        x = self.conv2(x)
        x = self.Flatten(x)
        x = self.fc1(x)
        dist = state[:,3:,0]
        x = tf.concat([x,dist],axis=-1)
        x = self.fc2(x)
        q = self.q(x)

        return q
