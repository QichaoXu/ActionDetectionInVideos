
import numpy as np
import os
import cv2
import time
import json


def vis_skeleton(json_file, inputpath=None, inputlist=None):

    with open(json_file, 'r') as data_file:
        skeleton = json.load(data_file)
    print(skeleton)

    if inputlist:
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
        # for root, dirs, files in os.walk(inputpath):
        #     im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')


if __name__ == "__main__":
    json_file = '/media/qcxu/qcxuDisk/ActionDetectionInVideos/AlphaPose/output/alphapose-results.json'
    inputpath = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/others/clips/Video_12_3_1'
    vis_skeleton(json_file, inputpath)