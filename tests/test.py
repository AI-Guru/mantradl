import unittest
from mantradl import layers as mdllayers
from keras import models, layers
import logging
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


class TestMisc(unittest.TestCase):

    #def test_sincnet(self):

        # Create the model.
    #    model = models.Sequential()
    #    model.add(mdllayers.SincConv1D(2, 25, 16000, input_shape=(1024, 1)))
    #    model.add(layers.Flatten())
    #    model.add(layers.Dense(10))
    #    model.summary()

    def test_oscillator(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.Oscillator(4096, 44100, input_shape=(2, 3)))
        model.summary()

        data = [[440.0, 1.0, 0], [880.0, 1.0, 0]]
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        prediction = model.predict(data)[0]
        print(prediction.shape)
        plt.plot(prediction)
        plt.show()



if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
