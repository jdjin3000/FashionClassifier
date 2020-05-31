import glob
import os
import shutil
import numpy as np
import random

def split_data(data, prob):
	results = ([], [])
	for row in data:
		results[0 if random.random() < prob else 1].append(row)

	return np.array(results[0]), np.array(results[1])

def remove_data(lists):
	counts =[]

	for lists_ in lists:
		counts.append(len(lists_))

	print(counts)

	cutline = min(counts)
	print(cutline)
	for lists_ in lists:
		num = len(lists_) - cutline
		print(num)
		if num < 1:
			continue

		for i in range(num):
			rand = random.choice(lists_)
			os.remove(rand)
			del lists_[lists_.index(rand)]





categories = list(set(glob.iglob('**')) -{'val', 'train_test_spliter.py'})
lists = []

for category in categories:
	lists.append(list(glob.iglob(category+'/*')))

remove_data(lists)

lists = []

for category in categories:
	lists.append(list(glob.iglob(category+'/*')))

for _ in lists:
	train, val = split_data(_, 0.7)

	for filename in val:
		if not os.path.exists(os.path.join('val',filename.split('\\')[0])):
			os.mkdir(os.path.join('val',filename.split('\\')[0]))

		shutil.copy(filename, os.path.join('val', filename))
		os.remove(filename)