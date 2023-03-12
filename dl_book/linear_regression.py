
from statistics import linear_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

import tensorboard

x = [22, 41, 52, 23, 41, 54, 24, 46, 56, 27, 47, 57, 28, 48, 58,  9,
     49, 59, 30, 49, 63, 32, 50, 67, 33, 51, 71, 35, 51, 77, 40, 51, 81]
y = [131, 139, 128, 128, 171, 105, 116, 137, 145, 106, 111, 141, 114,
     115, 153, 123, 133, 157, 117, 128, 155, 122, 183,
     176,  99, 130, 172, 121, 133, 178, 147, 144, 217]

x = np.asarray(x, np.float32)
y = np.asarray(y, np.float32)

x_hat = x - np.mean(x)
y_hat = y - np.mean(y)

alpha = np.sum(x_hat * y_hat) / np.sum(x_hat * x_hat)
betta = np.mean(y) - alpha * np.mean(x)
alpha, betta
mse = np.sum((y - (alpha * x + betta)) ** 2) / len(x)

reg = linear_regression(x, y)

model = LinearRegression()
model.fit(x.reshape((len(x), 1)), y)
model.coef_, model.intercept_
prediction = model.predict(x.reshape((len(x), 1)))


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# tf.__version__
# a_ = tf.Variable(0.0, name='a')
# b_ = tf.Variable(139.0, name='b')
# x_ = tf.constant(x, name='x')
# y_ = tf.constant(y, name='y')

# y_hat = a_*x_ + b_
# loss_ = tf.reduce_mean(tf.square(y_hat - y_))

# writer = tf.summary.FileWriter("logs/linreg/", tf.get_default_graph())
# writer.close()

# sess = tf.Session()
# res_val = sess.run([loss_,], {a_:0, b_:139})
# sess.close()

# train_op = tf.train.GradientDescentOptimizer(0.0004).minimize(loss_)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(100000):
#         _, a_val, b_val, mse = sess.run([train_op, a_, b_, loss_])
#         if i % 1000 == 0:
#             print(f'{i}: a={a_val:.2f}, b={b_val:.2f}, mse={mse:.2f}')





@tf.function
def fn(a_, b_):
    x_ = tf.constant(x, name='x')
    y_ = tf.constant(y, name='y')

    y_hat = a_*x_ + b_
    loss_ = tf.reduce_mean(tf.square(y_hat - y_))
    return loss_

a_ = tf.Variable(0.0, name='a')
b_ = tf.Variable(139.0, name='b')

writter = tf.summary.create_file_writer("logs/linreg2/")
tf.summary.trace_on(graph=True, profiler=True)
z = fn(a_, b_)
with writter.as_default():
    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir="logs/linreg2/")

res_val = fn(0, 139)
res_val

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0004)
for i in range(100000):
    with tf.GradientTape() as tape:
        loss = fn(a_, b_)
    grads = tape.gradient(loss, [a_, b_])
    optimizer.apply_gradients(zip(grads, [a_, b_]))
    if i % 1000 == 0:
        print(f'{i}: a={a_.numpy():.2f}, b={b_.numpy():.2f}, mse={loss.numpy():.2f}')


plt.scatter(x, y)
plt.plot(x, alpha * x + betta, 'r-')
plt.plot(x, a_.numpy() * x + b_.numpy(), 'y-')
plt.scatter(x, prediction, c='g')
plt.show()

