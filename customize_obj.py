from tensorflow.keras.layers import Dropout
from deoxys.customize import custom_layer


@custom_layer
class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
