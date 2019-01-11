from Openpose_skeleton import Openpose_skeleton
import os
import cv2
import time

import threading
from multiprocessing import Queue


class SkeletonToolThreads(threading.Thread):
    def __init__(self, queue_clip_all):

        threading.Thread.__init__(self)
        self.skeleton_det = Openpose_skeleton()
        self.queue_clip_all = queue_clip_all

    def run(self):
        while True:
            print(self.queue_clip_all.qsize())
            clip_all = self.queue_clip_all.get()
            if isinstance(clip_all, str) and clip_all == 'quit':
                break

            self.skeleton_det.run(clip_all, sample_rate=1)
            print(1)


if __name__ == "__main__":

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']

    # base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
    # __action__ = ['normal', 'clean', 'pick', 'scratch']

    # get skeleton
    queue_clip_all = Queue()
    skeleton_det = SkeletonToolThreads(queue_clip_all)
    skeleton_det.start()
    for act in __action__:

        base_in_clip_folder = base_folder + act + '/clips/'
        base_skeleton_folder = base_folder + act + '/skeletons/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):

            if sub != 'Video_11_1_1':
                continue

            print(act, sub)
            in_clip_folder = base_in_clip_folder + sub
            skeleton_folder = base_skeleton_folder + sub

            imglist = []
            for img_name in os.listdir(in_clip_folder):
                if img_name.endswith('jpg'):
                    imglist.append(cv2.imread(os.path.join(in_clip_folder, img_name)))

            queue_clip_all.put(imglist)
            # skeleton_list = skeleton_det.run(imglist, sample_rate=1)
            # skeleton_det.save_skeleton(skeleton_list, skeleton_folder)
