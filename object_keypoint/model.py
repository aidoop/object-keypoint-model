import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    AveragePooling2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.python.keras.layers.core import Dropout


class ObjectKeyDetector(Model):
    def __init__(self):
        super(ObjectKeyDetector, self).__init__()
        self.conv1_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=1)
        self.conv1_2 = Conv2D(filters=64, kernel_size=(5, 3), strides=1)
        self.conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=1)
        self.maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=1, padding="valid")
        self.bn_1 = BatchNormalization()
        self.dropout_1 = Dropout(rate=0.2)

        self.conv2_1 = Conv2D(filters=128, kernel_size=(5, 5), strides=2)
        self.conv2_2 = Conv2D(filters=128, kernel_size=(5, 5), strides=2)
        self.conv2_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=2)
        self.maxpool_2 = MaxPool2D(pool_size=(2, 2), strides=1, padding="valid")
        self.bn_2 = BatchNormalization()
        self.dropout_2 = Dropout(rate=0.2)

        self.avg_pool = AveragePooling2D(pool_size=(7, 7), strides=1)

        self.flatten = Flatten()

        self.dense1 = Dense(128, activation="relu")
        self.dropout_2 = Dropout(rate=0.5)

        self.dense2 = Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool_1(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.avg_pool(x)

        x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dropout_2(x)
        x = self.dense2(x)
        return x

    def summary(self, input_shape=(128, 128, 3)):
        x = Input(shape=input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


if __name__ == "__main__":
    # to figure out gpu memory issue
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    key_model = ObjectKeyDetector()

    # build
    key_model.build(input_shape=(None, 128, 128, 3))
    key_model.summary()
