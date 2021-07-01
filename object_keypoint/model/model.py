import tensorflow as tf


class ObjectKeyDetector(tf.keras.Model):
    def __init__(self):
        super(ObjectKeyDetector, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.conv1_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.conv1_3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.maxpool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=1, padding="valid"
        )
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.2)

        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.conv2_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.conv2_3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.maxpool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=1, padding="same"
        )
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.2)

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.5)

        self.dense2 = tf.keras.layers.Dense(2, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool_1(x)
        x = self.dropout_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.maxpool_2(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)
        # x = self.avg_pool(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout_2(x)
        x = self.dense2(x)
        return x


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
    key_model.build(input_shape=(None, 64, 64, 3))
    key_model.summary()
