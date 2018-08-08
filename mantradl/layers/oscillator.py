from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math

class Oscillator(Layer):

    def __init__(
        self,
        output_size,
        sample_rate,
        **kwargs):

        self.output_size = output_size
        self.sample_rate = sample_rate

        super(Oscillator, self).__init__(**kwargs)


    def build(self, input_shape):

        self.oscillators = input_shape[1]

        super(Oscillator, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):

        frames_range = np.arange(self.output_size)

        def create_waves(triples):
            waves = K.map_fn(create_wave, triples)
            wave = K.sum(waves, axis=0) / self.oscillators
            return wave

        def create_wave(triple):

            frequency = triple[0]
            amplitude = triple[1]
            phase = triple[2]
            samples = phase + frames_range * 2 * math.pi * frequency / self.sample_rate
            wave = amplitude * K.sin(samples)
            return wave

        waves = K.map_fn(create_waves, x)
        return waves


    def compute_output_shape(self, input_shape):
        return (None, self.output_size)
