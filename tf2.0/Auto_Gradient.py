# 在即时执行模式下，TensorFlow 引入了 tf.GradientTape() 这个 “求导记录器” 来实现自动求导。以下代码展示了如何使用 tf.GradientTape() 计算函数 y(x) = x^2 在 x = 3 时的导数：
import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)

# tf.Variable() 中指定 initial_value 参数来指定初始值。这里将变量 x 初始化为 3.。变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数
# tf.GradientTape() 是一个自动求导的记录器。只要进入了 with tf.GradientTape() as tape 的上下文环境，则在该环境中计算步骤都会被自动记录。比如在上面的示例中，计算步骤 y = tf.square(x) 即被自动记录。
# 离开上下文环境后，记录将停止，但记录器 tape 依然可用，因此可以通过 y_grad = tape.gradient(y, x) 求张量 y 对变量 x 的导数

# 输出：
# tf.Tensor(9.0, shape=(), dtype=float32)
# tf.Tensor(6.0, shape=(), dtype=float32)


# 在机器学习中，更加常见的是对多元函数求偏导数，以及对向量或矩阵的求导。这些对于 TensorFlow 也不在话下。
# 以下代码展示了如何使用 tf.GradientTape() 计算函数 L(w, b) = ||Xw + b - y||^2 在 w = (1, 2)^T, b = 1 时分别对 w, b 的偏导数。

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)

# 输出：
# tf.Tensor(125.0, shape=(), dtype=float32)
# tf.Tensor(
# [[ 70.]
# [100.]], shape=(2, 1), dtype=float32)
# tf.Tensor(30.0, shape=(), dtype=float32)

# 这里， tf.square() 操作代表对输入张量的每一个元素求平方，不改变张量形状。 tf.reduce_sum() 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过 axis 参数来指定求和的维度，不指定则默认对所有元素求和）。
# TensorFlow 中有大量的张量操作 API，包括数学运算、张量形状操作（如 tf.reshape()）、切片和连接（如 tf.concat()）等多种类型

