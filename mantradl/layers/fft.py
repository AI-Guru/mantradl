from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Spectrum(Layer):

    def __init__(self, **kwargs):
        super(Spectrum, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[1]
        self.output_dim = input_shape[1]
        self.trainable_weights = []


    def call(self, x, mask=None):
        return K.tf.fft(K.tf.complex(x, K.tf.zeros_like(x)))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



class MagnitudeSpectrum(Layer):

    def __init__(self, **kwargs):
        super(MagnitudeSpectrum, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[1]
        self.output_dim = input_shape[1]
        self.trainable_weights = []


    def call(self, x, mask=None):
        return K.tf.abs(K.tf.fft(
            K.tf.complex(x, K.tf.zeros_like(x))))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
