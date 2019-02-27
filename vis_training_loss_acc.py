
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_loss_acc(log_file):
	losses = []
	accs = []
	with open(log_file, 'r') as f:
		f = f.readlines()
		head = True
		for line in f:
			if head:
				head = False
				continue
			loss = line.split('tensor(')[1]
			loss = loss.split(',')[0]

			acc = line.split('tensor(')[2]
			acc = acc.split(',')[0]
			losses.append(float(loss))
			accs.append(float(acc))
	return losses, accs

def vis(train, test, text, title=None):
	fig = plt.figure()
	ax = plt.axes()

	plt.plot(range(len(train)), train, 'b--', label='train_'+text)
	plt.plot(range(len(test)), test, 'g', label='test_'+text)

	# legend = ax.legend(loc='upper right', shadow=True)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	if text == 'loss':
		ax.set_ylim([0.0, 1.75])
	if text == 'acc':
		ax.set_ylim([0.75, 1.0])

	plt.title(title)
	plt.show()



base_folder = '/media/qcxu/qcxuDisk/ActionDetectionInVideos/'
model_folder = 'C3D_ResNet_skeleton/data/results-scratch-18-static_BG-30-skeleton-concatenate-iter6/'
train_losses, train_accs = get_loss_acc(os.path.join(base_folder, model_folder, 'train.log'))
test_losses, test_accs = get_loss_acc(os.path.join(base_folder, model_folder, 'val.log'))

vis(train_losses, test_losses, 'loss', model_folder)
vis(train_accs, test_accs, 'acc', model_folder)