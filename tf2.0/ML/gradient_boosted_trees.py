# Ignore all GPUs (current TF GBDT does not support GPU).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
import numpy as np
import copy

# Dataset parameters.
num_classes = 2 # Total classes: greater or equal to $23,000, or not (See notes below).
num_features = 13 # data features size.

# Training parameters.
max_steps = 2000
batch_size = 256
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

# GBDT parameters.
num_batches_per_layer = 1000
num_trees = 10
max_depth = 4

# Prepare Boston Housing Dataset.
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# For classification purpose, we build 2 classes: price greater or lower than $23,000
def to_binary_class(y):
    for i, label in enumerate(y):
        if label >= 23.0:
            y[i] = 1
        else:
            y[i] = 0
    return y

y_train_binary = to_binary_class(copy.deepcopy(y_train))
y_test_binary = to_binary_class(copy.deepcopy(y_test))

#################  GBDT Classifier    #################
# Build the input function.
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary,
    batch_size=batch_size, num_epochs=None, shuffle=True)
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test_binary,
    batch_size=batch_size, num_epochs=1, shuffle=False)
test_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary,
    batch_size=batch_size, num_epochs=1, shuffle=False)
# GBDT Models from TF Estimator requires 'feature_column' data format.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]

gbdt_classifier = tf.estimator.BoostedTreesClassifier(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns,
    n_classes=num_classes,
    learning_rate=learning_rate,
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul,
    l2_regularization=l2_regul
)

# 训练模型
gbdt_classifier.train(train_input_fn, max_steps=max_steps)
# 评估模型
gbdt_classifier.evaluate(test_train_input_fn)
gbdt_classifier.evaluate(test_input_fn)

################   GBDT Regressor     #################
# Build the input function.
# train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x={'x': x_train}, y=y_train,
#     batch_size=batch_size, num_epochs=None, shuffle=True)
# test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x={'x': x_test}, y=y_test,
#     batch_size=batch_size, num_epochs=1, shuffle=False)
# # GBDT Models from TF Estimator requires 'feature_column' data format.
# feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]
#
# gbdt_regressor = tf.estimator.BoostedTreesRegressor(
#     n_batches_per_layer=num_batches_per_layer,
#     feature_columns=feature_columns,
#     learning_rate=learning_rate,
#     n_trees=num_trees,
#     max_depth=max_depth,
#     l1_regularization=l1_regul,
#     l2_regularization=l2_regul
# )
#
# # 训练模型
# gbdt_regressor.train(train_input_fn, max_steps=max_steps)
# # 评估模型
# gbdt_regressor.evaluate(test_train_input_fn)
# gbdt_regressor.evaluate(test_input_fn)

