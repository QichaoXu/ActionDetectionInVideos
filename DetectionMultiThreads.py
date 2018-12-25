
# from Alphapose_skeleton import Alphapose_skeleton
from skeleton_tools import skeleton_tools
from Action_Recognition import Action_Recognition

import time
import os
import cv2
from PIL import Image

import threading
from multiprocessing import Queue


class SkeletonToolThreads(threading.Thread):
    def __init__(self, queue_skeleton, st, reg, 
        is_vis, waitTime, is_static_BG, thres):

        threading.Thread.__init__(self)
        self.queue_skeleton = queue_skeleton

        self.st = st
        self.is_static_BG = is_static_BG
        self.waitTime = waitTime

        self.queue_clip_all = Queue()
        self.action_recognition_threads = ActionRecognitionThreads(self.queue_clip_all, st, reg,
            is_vis, waitTime, thres)
        self.action_recognition_threads.start()

    def run(self):
        while True:
            imglist, skeleton = self.queue_skeleton.get()
            print('SkeletonToolThreads get', len(skeleton))
            if isinstance(skeleton, str) and skeleton == 'quit':
                break

            im_name_all, kp_preds_all, kp_scores_all = self.st.get_valid_skeletons(
                'None', in_skeleton_list=skeleton, is_savejson=False)
            clip_all = self.st.get_hand_clip('None', 'None', 'None', 'None.json',
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                is_save=False, is_vis=False, is_static_BG=self.is_static_BG, is_labeled=False, 
                is_heatmap=False, waitTime=self.waitTime)

            print('SkeletonToolThreads put', len(clip_all))
            self.queue_clip_all.put([clip_all, im_name_all, kp_preds_all, kp_scores_all, imglist])

        self.queue_clip_all.put(['quit', 'quit', 'quit', 'quit', 'quit'])
        self.action_recognition_threads.join()
        print('=================== finish SkeletonToolThreads ===================')


class ActionRecognitionThreads(threading.Thread):
    def __init__(self, queue_clip_all, st, reg,
        is_vis, waitTime, thres):

        threading.Thread.__init__(self)
        self.queue_clip_all = queue_clip_all

        self.reg = reg

        self.queue_result_labels = Queue()
        self.skeleton_vis_threads = SkeletonVisThreads(self.queue_result_labels, st,
            is_vis, waitTime, thres)
        self.skeleton_vis_threads.start()

    def run(self):
        while True:
            clip_all, im_name_all, kp_preds_all, kp_scores_all, imglist = self.queue_clip_all.get()
            print('ActionRecognitionThreads get', len(clip_all))
            if isinstance(clip_all, str) and clip_all == 'quit':
                break

            result_labels = []
            for clip in clip_all:
                clip_PIL = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in clip]
                if clip_PIL is None or len(clip_PIL) == 0:
                    result_labels.append([0, [0.0, 0.0, 0.0]])
                else:
                    label, probs = self.reg.run(clip_PIL)
                    result_labels.append([label, probs])

            print('ActionRecognitionThreads put', len(result_labels))
            self.queue_result_labels.put([result_labels, im_name_all, kp_preds_all, kp_scores_all, imglist])

        self.queue_result_labels.put(['quit', 'quit', 'quit', 'quit', 'quit'])
        self.skeleton_vis_threads.join()
        print('=================== finish ActionRecognitionThreads ===================')


class SkeletonVisThreads(threading.Thread):
    def __init__(self, queue_result_labels, st, is_vis, waitTime, thres):
        threading.Thread.__init__(self)
        self.queue_result_labels = queue_result_labels

        self.st = st
        self.is_vis = is_vis
        self.thres = thres
        self.waitTime = waitTime

        # self.queue_img_out_all = Queue()

    def run(self):
        while True:
            result_labels, im_name_all, kp_preds_all, kp_scores_all, imglist = self.queue_result_labels.get()
            print('SkeletonVisThreads get', len(result_labels))
            if isinstance(result_labels, str) and result_labels == 'quit':
                break

            img_out_all = self.st.vis_skeleton('None', 'None', 'None.json',
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                result_labels=result_labels, is_save=False, is_vis=self.is_vis, thres=self.thres,
                waitTime=self.waitTime)

            print('SkeletonVisThreads put', len(img_out_all))
            # self.queue_img_out_all.put(img_out_all)

        print('=================== finish SkeletonVisThreads ===================')


class DetectionMultiThreads(threading.Thread):
    def __init__(self, queue_imglist, reg_model_file, skeleton_opt, cuda_id_list, 
        sample_rate=1, is_vis=False, waitTime=5, is_static_BG=False, thres=0.5):

        threading.Thread.__init__(self)

        skeleton_cuda_id, reg_cuda_id = cuda_id_list

        if skeleton_opt == 'MSRA':
            from MSRA_skeleton import MSRA_skeleton
            skeleton_det = MSRA_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Alphapose':
            from Alphapose_skeleton import Alphapose_skeleton
            skeleton_det = Alphapose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Openpose':
            from Openpose_skeleton import Openpose_skeleton
            skeleton_det = Openpose_skeleton(cuda_id=skeleton_cuda_id)
        else:
            raise Exception('Error: ' + skeleton_opt + ' could not be found')

        st = skeleton_tools()
        reg = Action_Recognition(reg_model_file, reg_cuda_id)
        print('=================== Initialized ===================\n\n')

        self.queue_imglist = queue_imglist

        self.skeleton_det = skeleton_det
        self.sample_rate = sample_rate

        self.queue_skeleton = Queue()
        self.skeleton_tool_threads = SkeletonToolThreads(self.queue_skeleton, st, reg,
            is_vis, waitTime, is_static_BG, thres)
        self.skeleton_tool_threads.start()

    def run(self):
        while True:
            imglist = self.queue_imglist.get()
            print('SkeletonDetThreads get', len(imglist))
            if isinstance(imglist, str) and imglist == 'quit':
                break

            skeleton = self.skeleton_det.run(imglist, self.sample_rate)

            print('SkeletonDetThreads put', len(skeleton))
            self.queue_skeleton.put([imglist, skeleton])

        self.queue_skeleton.put(['quit', 'quit'])
        self.skeleton_tool_threads.join()
        print('=================== finish DetectionMultiThreads ===================')


if __name__ == "__main__":

    reg_model_file = 'results-scratch-18/save_200.pth'
    detection = detection(reg_model_file, skeleton_opt='Openpose', is_vis=False, is_static_BG=False, thres=0.9)

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/others/clips/Video_11_1_1'
    imglist = []
    for img_name in os.listdir(base_folder):
        if img_name.endswith('jpg'):
            imglist.append(cv2.imread(os.path.join(base_folder, img_name)))

    detection.run(imglist)

