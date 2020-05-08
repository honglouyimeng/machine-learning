import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
# 加载数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# 打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
# 将数据分为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
# 转换x的数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
# from_tensor_slices函数使输入特征和标签一一对应，将数据分批次，每个批次batch组数据
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
# 生成神经网络参数，4个输入特征，3个输出特征
# tf.Variable()标记参数可训练
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
# 超参数设置
lr = 0.1
train_loss_results = []
train_acc = []
test_acc = []
epoch = 500
loss_all = 0
# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])
        # 实现梯度更新
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    # 每个epoch打印loss信息
    print('epoch {},loss: {}'.format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0
    # 测试部分
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新过的参数预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print('test_acc:', acc)
    print('--------------------------')
plt.title('loss function curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_loss_results, label='$loss$')
plt.legend()
plt.show()

plt.title('acc curve')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(test_acc, label='$accuracy$')
plt.legend()
plt.show()
