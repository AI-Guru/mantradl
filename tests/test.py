import unittest
from mantradl import layers as mdllayers
from keras import models, layers
import logging
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

class TestMisc(unittest.TestCase):

    def test_sincnet(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.SincConv1D(2, 25, 16000, input_shape=(1024, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10))
        model.summary()


    def test_oscillator(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.Oscillator(44100, 44100, input_shape=(3, 3)))
        model.summary()

        data = [[440.0, 1.0, 0], [440.0 * 2, 1.0, 0], [440.0 * 3, 1.0, 0]]
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        prediction = model.predict(data)[0]
        print(prediction.shape)
        #librosa.output.write_wav("test.wav", prediction, sr=44100)
        #plt.plot(prediction)
        #plt.show()
        plot_mel(prediction)


    def test_spectrum(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.Spectrum(input_shape=(1024, 1)))
        model.summary()
        prediction = model.predict(np.random.random((1, 1024, 1)))
        print(prediction.shape)


    def test_magnitude_spectrum(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.MagnitudeSpectrum(input_shape=(1024, 1)))
        model.summary()
        prediction = model.predict(np.random.random((1, 1024, 1)))
        print(prediction.shape)


    def test_magnitude_crazy(self):

        # Create the model.
        model = models.Sequential()
        model.add(mdllayers.Oscillator(1024, 44100, input_shape=(3, 3)))
        model.add(mdllayers.Spectrum())
        model.summary()

        data = [[440.0, 1.0, 0], [880.0, 1.0, 0], [880.0 + 440.0, 0.75, 0]]
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        prediction = model.predict(data)[0]
        print(prediction.shape)
        #plt.plot(prediction)
        #plt.show()


def plot_mel(samples):

    spectrogram = librosa.feature.melspectrogram(y=samples, sr=44100)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
