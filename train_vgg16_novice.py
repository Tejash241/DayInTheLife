import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import cv2
import random
from sklearn.utils import shuffle
tf.logging.set_verbosity(tf.logging.ERROR)

# ======================= The Data ======================================
def one_hot(label_strings, n_classes, class_map):
	y = np.zeros((len(label_strings), n_classes))
	for i, label in enumerate(label_strings):
		y[i, class_map[label]] = 1

	return y

def get_data(data_folder, train_valid_split=.95):
	all_img_paths = glob(os.path.join(data_folder, '*', '*.jpg'))
	random.shuffle(all_img_paths)
	train_img_paths = all_img_paths[:int(train_valid_split*len(all_img_paths))]
	valid_img_paths = all_img_paths[int(train_valid_split*len(all_img_paths)):]
	X_train, y_train, X_valid, y_valid = [], [], [], []
	for img_file in train_img_paths:
		if '.' not in img_file.split('/')[-2].split('_')[-1]:
			X_train.append(cv2.imread(img_file)) # bgr order
			y_train.append(img_file.split('/')[-2].split('_')[-1])

	for img_file in valid_img_paths:
		if '.' not in img_file.split('/')[-2].split('_')[-1]:
			X_valid.append(cv2.imread(img_file)) # bgr order
			y_valid.append(img_file.split('/')[-2].split('_')[-1])

	y_set = set(y_train)
	n_classes = len(y_set)
	print n_classes, 'n_classes'
	class_map = dict(zip(y_set, range(n_classes)))
	return np.array(X_train, dtype=np.float32), one_hot(y_train, n_classes, class_map), np.array(X_valid, dtype=np.float32), one_hot(y_valid, n_classes, class_map)

def get_augmented_data(X, y, crop_probab=0.5):
	pass

# ======================== The Model =====================================
def get_conv_filter_pretrained(data_dict, name):
    return tf.constant(data_dict[name][0], name="filter")

def get_bias_pretrained(data_dict, name):
    return tf.constant(data_dict[name][1], name="biases")

def xavier_init(shape, name):
	if len(shape)==2:
		N = (shape[0]+shape[1])/2.0
	else:
		N = shape[0]

	with tf.variable_scope(name):
		W = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=np.sqrt(1.0/N)), name=name)

	return W

def get_bias(shape, name):
	with tf.variable_scope(name):
		b = tf.Variable(tf.zeros(shape), name=name)

	return b

def conv2d(img, name, strides=1):
	with tf.variable_scope(name):
		w = get_conv_filter_pretrained(data_dict, name)
		b = get_bias_pretrained(data_dict, name)
		conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,1,1,1], padding='SAME'), b), name=name)

	return conv

def maxpool2d(img, name, k=1):
	with tf.variable_scope(name):
		pool = tf.nn.max_pool(img, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name)

	return pool

def global_average_pooling(img):
	return tf.reduce_mean(img, [1,2])

def build_model(X, data_dict, keep_prob): # X should be bgr order
	conv1_1 = conv2d(X, name='conv1_1')
	conv1_2 = conv2d(conv1_1, name='conv1_2')
	pool1 = maxpool2d(conv1_2, name='pool1', k=2)

	conv2_1 = conv2d(pool1, name='conv2_1')
	conv2_2 = conv2d(conv2_1, name='conv2_2')
	pool2 = maxpool2d(conv2_2, name='pool2', k=2)

	conv3_1 = conv2d(pool2, name='conv3_1')
	conv3_2 = conv2d(conv3_1, name='conv3_2')
	conv3_3 = conv2d(conv3_2, name='conv3_3')
	pool3 = maxpool2d(conv3_3, name='pool3', k=2)

	conv4_1 = conv2d(pool3, name='conv4_1')
	conv4_2 = conv2d(conv4_1, name='conv4_2')
	conv4_3 = conv2d(conv4_2, name='conv4_3')
	pool4 = maxpool2d(conv4_3, name='pool4', k=2)

	conv5_1 = conv2d(pool4, name='conv5_1')
	conv5_2 = conv2d(conv5_1, name='conv5_2')
	conv5_3 = conv2d(conv5_2, name='conv5_3')
	pool5 = maxpool2d(conv5_3, name='pool5', k=2)

	gap = global_average_pooling(pool5)
	Wout = xavier_init((512, n_classes), name='wout')
	bout = tf.Variable(tf.zeros(n_classes), name='bout')
	fc1 = tf.nn.bias_add(tf.matmul(gap, Wout), bout, name='fc1')
	return fc1


# ======================= HYPERPARAMETERS =======================
data_dict = np.load('vgg16.npy', encoding='latin1').item()
# print data_dict.keys()
# print data_dict['conv5_3'][0].shape, data_dict['conv5_3'][1].shape, 'conv5_3'

X_train, y_train, X_valid, y_valid = get_data(os.path.join('.', 'images'))
VGG_MEAN = [103.939, 116.779, 123.68]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DEVICE = '/cpu:0'
N_EPOCHS = 50
n_classes = y_train.shape[1]
n_batches_train = y_train.shape[0]/BATCH_SIZE
n_batches_valid = y_valid.shape[0]/BATCH_SIZE

for channel in range(3):
	X_train[:,:,channel] -= VGG_MEAN[channel]
	X_valid[:,:,channel] -= VGG_MEAN[channel]

# ======================== THE TRAINING ==========================
with tf.Graph().as_default():
	X = tf.placeholder(tf.float32, [None, 224, 224, 3])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32)

	outputs = build_model(X, data_dict, keep_prob)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)	

	corrects = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

	saver = tf.train.Saver()
	init = tf.global_variables_initializer()
	train_loss, valid_loss = [], []
	train_acc, train_acc = [], []
	with tf.Session() as sess, tf.device(DEVICE):
		sess.run(init)
		for epoch in range(N_EPOCHS):
			for batch_id in range(n_batches_train):
				batch_X = X_train[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
				batch_y = y_train[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
				preds, loss, acc, _ = sess.run([outputs, cost, accuracy, optimizer], feed_dict={X:batch_X, y:batch_y, keep_prob:100.})
				train_loss.append(loss)
				train_acc.append(acc)
				print 'Epoch {} Iteration {} Loss {} Accuracy {}'.format(epoch, batch_id, loss, acc)

			print 'Validating now...'
			for batch_id in range(n_batches_valid):
				batch_X = X_valid[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
				batch_y = y_valid[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
				preds, loss, acc = sess.run([outputs, cost, accuracy], feed_dict={X:batch_X, y:batch_y, keep_prob:100.})
				valid_loss.append(loss)
				valid_acc.append(acc)
				print 'Epoch {} Iteration {} Validation Loss {} Validation Accuracy {}'.format(epoch, batch_id, loss, acc)

			saver.save(sess, 'birds_epoch{}_validloss{}_validacc{}'.format(epoch, np.mean(valid_loss), np.mean(valid_acc)))
