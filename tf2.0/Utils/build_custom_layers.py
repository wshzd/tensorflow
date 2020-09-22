import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

# MNIST dataset parameters.
num_classes = 10  # 0 to 9 digits
num_features = 784  # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 500
batch_size = 256
display_step = 50

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


#############  Create a custom layer   ###########

# Create a custom layer, extending TF 'Layer' class.
# Layer compute: y = relu(W * x + b)
class CustomLayer1(layers.Layer):

    # Layer arguments.
    def __init__(self, num_units, **kwargs):
        # Store the number of units (neurons).
        self.num_units = num_units
        super(CustomLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Note: a custom layer can also include any other TF 'layers'.
        shape = tf.TensorShape((input_shape[1], self.num_units))
        # Create weight variables for this layer.
        self.weight = self.add_weight(name='W',
                                      shape=shape,
                                      initializer=tf.initializers.RandomNormal,
                                      trainable=True)
        self.bias = self.add_weight(name='b',
                                    shape=[self.num_units])
        # Make sure to call the `build` method at the end
        super(CustomLayer1, self).build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.weight)
        x = x + self.bias
        return tf.nn.relu(x)

    def get_config(self):
        base_config = super(CustomLayer1, self).get_config()
        base_config['num_units'] = self.num_units
        return base_config


##############    Create another custom layer     #########
# Create a custom layer, extending TF 'Layer' class.
# Custom layer: 2 Fully-Connected layers with residual connection.
class CustomLayer2(layers.Layer):

    # Layer arguments.
    def __init__(self, num_units, **kwargs):
        self.num_units = num_units
        super(CustomLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.num_units))

        self.inner_layer1 = layers.Dense(1)
        self.inner_layer2 = layers.Dense(self.num_units)

        # Make sure to call the `build` method at the end
        super(CustomLayer2, self).build(input_shape)

    def call(self, inputs):
        x = self.inner_layer1(inputs)
        x = tf.nn.relu(x)
        x = self.inner_layer2(x)
        return x + inputs

    def get_config(self):
        base_config = super(CustomLayer2, self).get_config()
        base_config['num_units'] = self.num_units
        return base_config


################    Build Model   ###########
# Create TF Model.
class CustomNet(Model):

    def __init__(self):
        super(CustomNet, self).__init__()
        # Use custom layers created above.
        self.layer1 = CustomLayer1(64)
        self.layer2 = CustomLayer2(64)
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    # Set forward pass.
    def __call__(self, x, is_training=False):
        x = self.layer1(x)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Build neural network model.
custom_net = CustomNet()


# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int64)
    crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(crossentropy)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = custom_net(x, is_training=True)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, custom_net.trainable_variables)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, custom_net.trainable_variables))


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = custom_net(batch_x, is_training=False)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
