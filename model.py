import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Activation,Conv2D, BatchNormalization, MaxPool2D,Softmax,Dropout, Input, ReLU, \
    Concatenate, Dense, Flatten
from tensorflow.keras.models import load_model





class LPRNet:
    def __init__(self, num_classes, pattern_size=128, dropout=0.5, input_shape=(24,94, 3),
                 include_STN=False):
        self.num_classes = num_classes
        self.pattern_size = pattern_size
        self.dropout = dropout
        self.input_shape = input_shape
        self.input_block = self.mixed_input_block
        self.basic_block = self.basic_blocks
        self.model = self._build()

    def _build(self):
        inputs = Input(self.input_shape)
        x = self.input_block()(inputs)
        x = self.basic_block(x.get_shape().as_list()[3], 256)(x)
        x = self.convolution_block(x.get_shape().as_list()[3], 256, 2)(x)

        x = Dropout(self.dropout)(x)
        x = conv2D_batchnorm(256, [4, 1])(x)
        x = Dropout(self.dropout)(x)

        classes = conv2D_batchnorm(self.num_classes, [1, 13], padding="same")(x)
        pattern = Conv2D(128, [1,1])(classes)
        x = Concatenate()([classes, pattern])
        outs = conv2D_batchnorm(self.num_classes, [1, 1], padding="same")(x)

        return Model(inputs = inputs, outputs = outs)

    @staticmethod
    def basic_blocks(channel_in, channel_out):
        return Sequential([conv2D_batchnorm(channel_out // 4, [1, 1], padding="same"),
                           conv2D_batchnorm(channel_out // 4, [3, 1], padding="same"),
                           conv2D_batchnorm(channel_out // 4, [1, 3], padding="same"),
                           conv2D_batchnorm(channel_out // 4, [1, 1], padding="same")
                           ])







    def mixed_input_block(self):
        return Sequential([conv2D_batchnorm(64, [3, 3], padding="same"),
                           MaxPool2D([3, 3], strides=[1, 1]),
                           self.basic_block(64, 128),
                           MaxPool2D([3, 3], strides=[2, 1])
                           ])

    # Convolution block for CNN
    def convolution_block(self, channel_in, channel_out, stride):
        return Sequential([self.basic_block(channel_in, channel_out),
                           MaxPool2D([3, 3], strides=(stride, 1))
                           ])

    def train(self):
        raise NotImplemented
    def predict(self, x,classnames):
        pred = self.model.predict(x)
        return pred

    def decode_pred(self,pred,classnames ):
        samples, times = pred.shape[:2]
        input_length = tf.convert_to_tensor([times] * samples)
        decodeds, logprobs = tf.keras.backend.ctc_decode(pred, input_length, greedy=True, beam_width=100, top_paths=1)
        decodeds = np.array(decodeds[0])

        results = []
        for d in decodeds:
            text = []
            for idx in d:
                if idx == -1:
                    break
                text.append(classnames[idx])
            results.append(''.join(text).encode('utf-8'))
        return results

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save(self, filepath):
        self.model.save(filepath)

    def summary(self):
        self.model.summary()


def conv2D_batchnorm(*args, **kwargs):
    return Sequential([Conv2D(*args, **kwargs),
                   BatchNormalization(),
                   ReLU()])

if __name__ == '__main__':
    model = LPRNet(37)
    # model.load_weights("saved_models/model_best.pb")
    model = load_model("saved_models/model_best.pb")
    model.save("./saved_models/test-save.hdf5")
    # model.summary()
