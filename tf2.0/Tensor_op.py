import tensorflow as tf

# 在 TensorFlow 1.X 版本中， 必须 在导入 TensorFlow 库后调用 tf.enable_eager_execution() 函数以启用即时执行模式。
# 在 TensorFlow 2 中，即时执行模式将成为默认模式，无需额外调用 tf.enable_eager_execution() 函数（不过若要关闭即时执行模式，则需调用 tf.compat.v1.disable_eager_execution() 函数）。

# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# 查看矩阵A的形状、类型和值
print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy())    # 输出[[1. 2.]
                    #      [3. 4.]]

# TensorFlow 的大多数 API 函数会根据输入的值自动推断张量中元素的类型（一般默认为 tf.float32 ）。
# 不过你也可以通过加入 dtype 参数来自行指定类型，例如 zero_vector = tf.zeros(shape=(2), dtype=tf.int32) 将使得张量中的元素类型均为整数。
# 张量的 numpy() 方法是将张量的值转换为一个 NumPy 数组.

C = tf.add(A, B)    # 计算矩阵A和B的和
D = tf.matmul(A, B) # 计算矩阵A和B的乘积


