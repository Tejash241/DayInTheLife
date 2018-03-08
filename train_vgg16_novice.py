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
	all_img_folders = glob(os.path.join(data_folder, '*'))
	train_img_paths = []
	valid_img_paths = []
	for folder in all_img_folders:
		this_bird_images = glob(os.path.join(folder, '*.jpg'))
		split_index = int(train_valid_split*len(this_bird_images))
		# print split_index, len(this_bird_images), folder
		train_img_paths.extend(this_bird_images[:split_index])
		valid_img_paths.extend(this_bird_images[split_index:])

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
	reverse_class_map = dict(zip(range(n_classes), y_set))
	return train_img_paths, valid_img_paths, np.array(X_train, dtype=np.float32), one_hot(y_train, n_classes, class_map), np.array(X_valid, dtype=np.float32), one_hot(y_valid, n_classes, class_map), reverse_class_map

def get_augmented_data(X, y, augment_prob=0.4):
	n_samples = int(augment_prob*X.shape[0]) # sample augment_prob UNIQUE images
	augmented_X, augmented_y = np.zeros((X.shape)), np.zeros((y.shape))
	augmented_X, augmented_y = augmented_X[:n_samples,:,:,:], augmented_y[:n_samples,:]
	indices = random.sample(range(X.shape[0]), n_samples)
	with tf.Session() as sess:
		for idx, i in enumerate(indices):
			augmented_X[idx] = tf.image.flip_left_right(X[i]).eval(session=sess)
			augmented_y[idx] = y[i]

	return np.array(augmented_X), np.array(augmented_y)

# ======================== The Model =====================================
def get_conv_filter_pretrained(data_dict, name):
    return tf.Variable(data_dict[name][0], name=name+"/filter", trainable=False)

def get_bias_pretrained(data_dict, name):
    return tf.Variable(data_dict[name][1], name=name+"/bias", trainable=False)

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

def conv2d(img, name, stride=1):
	with tf.variable_scope(name):
		w = get_conv_filter_pretrained(data_dict, name)
		b = get_bias_pretrained(data_dict, name)
		conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,stride,stride,1], padding='SAME'), b), name=name)

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

	gap = global_average_pooling(conv5_3)
	Wout = xavier_init((512, n_classes), name='wout')
	bout = tf.Variable(tf.zeros(n_classes), name='bout')
	fc1 = tf.nn.bias_add(tf.matmul(gap, Wout), bout, name='fc1')
	fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
	return fc1

def draw_CAM(idx, this_sample, sess, n_classes=1, folder='./results/'):
	last_conv = tf.get_default_graph().get_tensor_by_name('conv5_3/conv5_3:0')
	last_weights = tf.get_default_graph().get_tensor_by_name('wout/wout:0')
	last_biases = tf.get_default_graph().get_tensor_by_name('bout:0')
	last_conv_reshaped = tf.reshape(last_conv, [-1, 512])

	preds, feat_map, Wk_c, b_c = sess.run([outputs, last_conv_reshaped, last_weights, last_biases], feed_dict={X:this_sample, keep_prob:1.})
	top_class = np.argmax(preds, axis=1)

	class_map = tf.nn.bias_add(tf.matmul(feat_map, Wk_c[:,top_class]), b_c[top_class])
	attention = sess.run([class_map], feed_dict={X:this_sample, keep_prob:1.})[0]
	# print [n.name for n in tf.get_default_graph().as_graph_def().node]
	# print attention.shape

	cam = attention.reshape((14, 14))
	cam = cam - np.min(cam)
	cam_img = cam / np.max(cam)
	cam_img = np.uint8(255 * cam_img)
	original_img = cv2.imread(valid_img_paths[idx])
	print valid_img_paths[idx], reverse_class_map[int(top_class)]
	cv2.imwrite(folder+'original_{}_{}.jpg'.format(idx, valid_img_paths[idx].split('/')[-2]), original_img)
	cam_img = cv2.resize(cam_img, (original_img.shape[0], original_img.shape[1]))
	# cam_img = cam_img.T
	heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
	result = heatmap * 0.3 + original_img * 0.5
	cv2.imwrite(folder+'CAM_{}_{}.jpg'.format(idx, reverse_class_map[int(top_class)]), result)

# ======================= HYPERPARAMETERS =======================
print 'Loading VGG weights ...'
data_dict = np.load('vgg16.npy', encoding='latin1').item()

print 'Getting data ...'
train_img_paths, valid_img_paths, X_train, y_train, X_valid, y_valid, reverse_class_map = get_data(os.path.join('.', 'images'))
#X_train, y_train = shuffle(X_train, y_train)
#X_valid, y_valid = shuffle(X_valid, y_valid)

# print 'Augmenting data ...'
# augmented_Xtrain, augmented_ytrain = get_augmented_data(X_train, y_train)
# X_train = np.concatenate((X_train, augmented_Xtrain), axis=0)
# y_train = np.concatenate((y_train, augmented_ytrain), axis=0)

# augmented_Xvalid, augmented_yvalid = get_augmented_data(X_valid, y_valid)
# X_valid = np.concatenate((X_valid, augmented_Xvalid), axis=0)
# y_valid = np.concatenate((y_valid, augmented_yvalid), axis=0)

# print X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, 'augmented'

VGG_MEAN = [103.939, 116.779, 123.68]
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 50
KEEP_PROB = 1.0
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
	with tf.Session() as sess:
		sess.run(init)
		# for epoch in range(N_EPOCHS):
		# 	train_loss, valid_loss = [], []
		# 	train_acc, valid_acc = [], []
		# 	for batch_id in range(n_batches_train):
		# 		batch_X = X_train[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
		# 		batch_y = y_train[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
		# 		preds, loss, acc, _ = sess.run([outputs, cost, accuracy, optimizer], feed_dict={X:batch_X, y:batch_y, keep_prob:KEEP_PROB})
		# 		train_loss.append(loss)
		# 		train_acc.append(acc)
		# 		print 'Epoch {} Iteration {} Loss {} Accuracy {}'.format(epoch, batch_id, loss, acc)

		# 	print 'Validating now...'
		# 	for batch_id in range(n_batches_valid):
		# 		batch_X = X_valid[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
		# 		batch_y = y_valid[BATCH_SIZE*batch_id:BATCH_SIZE*(batch_id+1)]
		# 		preds, loss, acc = sess.run([outputs, cost, accuracy], feed_dict={X:batch_X, y:batch_y, keep_prob:1.})
		# 		valid_loss.append(loss)
		# 		valid_acc.append(acc)
		# 		print 'Epoch {} Iteration {} Validation Loss {} Validation Accuracy {}'.format(epoch, batch_id, loss, acc)

		# 	print 'Avg Training Loss {} Accuracy {} Test Loss {} Accuracy {}'.format(np.mean(train_loss), np.mean(train_acc), np.mean(valid_loss), np.mean(valid_acc))
		# 	saver.save(sess, './models/birds_epoch{}_validloss{}_validacc{}'.format(epoch, np.mean(valid_loss), np.mean(valid_acc)))

# =================== Visualizing and CAM =====================================
		print 'Visualizing now...'
		saver.restore(sess, './models/birds_epoch19_validloss1.04835760593_validacc0.744140625')
		# samples = random.sample(range(X_train.shape[0]), 10) # take 10 random samples from the training_set
		samples = range(100)
		for idx in samples:
			this_sample = X_valid[idx].reshape((1,) + X_valid[idx].shape)
			draw_CAM(idx, this_sample, sess)

			
