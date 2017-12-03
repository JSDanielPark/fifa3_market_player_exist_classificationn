import tensorflow as tf
import numpy as np
from os import listdir
from scipy import misc
from common import input_size, \
    input_y, \
    input_x, \
    output_size
from net import Net

tf.set_random_seed(777)

# 경로 설정
train_0_path = './dataset/train/0/'
train_1_path = './dataset/train/1/'
test_0_path = './dataset/test/0/'
test_1_path = './dataset/test/1/'


# 학습용 버퍼 생성
x_stack = np.empty(0).reshape(0, input_size)
y_stack = np.empty(0).reshape(0, output_size)

# 매물이 존재하지 않는 이미지데이터 로딩 및 가공
li = listdir(train_0_path)
for file in li:
    f = misc.imread(train_0_path + file)
    f = misc.imresize(f, (input_x, input_y))
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [0]])

# 매물이 존재하는 이미지데이터 로딩 및 가공
li = listdir(train_1_path)
for file in li:
    f = misc.imread(train_1_path + file)
    f = misc.imresize(f, (input_x, input_y))
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [1]])


# 하이퍼 파라미터 (데이터가 적어서 배치사이즈를 데이터크기에 맞춤)
learning_rate = 0.001
training_epochs = 30
batch_size = 44


# initialize
with tf.Session() as sess:
    net = Net(sess, learning_rate)
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning stared. It takes sometime.')
    for epoch in range(training_epochs):
        c, _, h = net.update(x_stack, y_stack, 0.7)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
        print('Acc: {}'.format(h))
        # 정확도가 어느정도 이상 나오면 종료
        if h > 0.98:
            break

    print('Learning Finished!')

    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, 1)

    li = listdir(test_0_path)
    for file in li:
        f = misc.imread(test_0_path + file)
        f = misc.imresize(f, (input_x, input_y))
        f = np.reshape(f, [input_size])
        x_stack = np.vstack([x_stack, f])
        y_stack = np.vstack([y_stack, [0]])

    li = listdir(test_1_path)
    for file in li:
        f = misc.imread(test_1_path + file)
        f = misc.imresize(f, (input_x, input_y))
        f = np.reshape(f, [input_size])
        x_stack = np.vstack([x_stack, f])
        y_stack = np.vstack([y_stack, [1]])

    acc = net.calc_accuracy(x_stack, y_stack)
    print('Final Acc: {}'.format(acc))
    net.save()
