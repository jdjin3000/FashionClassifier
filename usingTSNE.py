from sklearn.datasets import load_files
#from sklearn.manifold import TSNE
from tsnecuda import TSNE
from skimage.io import imread_collection
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np

anno_path = './Anno'
eval_path = './Eval'
classify_path ='./classify'
bboxForTsne_path = './bboxForTsne/train'

n_classes = 46
color = [np.random.rand(3,) for i in range(n_classes)]

def load_images_from_subdirectories():
    data = []
    target = []
    subdirectories = {dir_:idx for (idx, dir_) in enumerate(os.listdir(bboxForTsne_path)) if os.path.isdir(os.path.join(bboxForTsne_path,dir_))}
    for sub_ in subdirectories.keys():
        for img_ in imread_collection(os.path.join(bboxForTsne_path, sub_) +'/*.jpg'):
        	target.append(sub_)
        	data.append(resize(img_,(300, 300)).ravel())

    return np.array(data), target, subdirectories

data, target, target_num = load_images_from_subdirectories()

tsne = TSNE()
data_tsne = tsne.fit_transform(data)

#np.save('tsne.npy', tsne)

for x, y, tg in zip(data_tsne[:, 0], data_tsne[:, 1], target):
	plt.scatter(x, y, c=color[target_num[tg]])

plt.xlim(data_tsne[:, 0].min(), data_tsne[:, 0].max()) # 최소, 최대
plt.ylim(data_tsne[:, 1].min(), data_tsne[:, 1].max()) # 최소, 최대
plt.xlabel('t-SNE 특성0') # x축 이름
plt.ylabel('t-SNE 특성1') # y축 이름
plt.savefig('./result.png')