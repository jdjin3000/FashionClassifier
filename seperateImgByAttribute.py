import shutil
import os
import numpy as np
import cv2
from six.moves import range

anno_path = './Anno'
eval_path = './Eval'
classify_path ='./classify'
bbox_path = './bboxForTsne'

def seperateImages():
    #Training Set, Validation Set, Testing Set 분리
    with open(os.path.join(eval_path, 'list_eval_partition.txt'), 'r') as eval_file, \
        open(os.path.join(anno_path, 'list_bbox.txt'), 'r') as bbox_file:
        eval_data = [line.rstrip('\n') for line in eval_file][2:]
        eval_list = [line.split() for line in eval_data]

        bbox_data = [line.rstrip('\n') for line in bbox_file][2:]
        bbox_list = [line.split() for line in bbox_data]

        #(이미지 경로, 폴더 이름)
        lists = [(os.path.join('.',eval_[0]), eval_[0].split('/')[1]) \
        for (eval_, bbox_) in zip(eval_list, bbox_list)]
    i=0
    name ='Shirt'
    for element in lists:
        i += 1
        if name in element[1]:
            shutil.copy(os.path.join('.', element[0]), 'BUNRYU/'+ name + '/' + name + str(i)+'.jpg')
    print("1. Sperating Image to"+ classify_path +" has done Successfully!")

        

seperateImages()
