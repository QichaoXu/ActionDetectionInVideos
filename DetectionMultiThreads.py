
from skeleton_tools import skeleton_tools

import time
import os
import cv2
from PIL import Image

import threading
from multiprocessing import Queue


class SkeletonToolThreads(threading.Thread):
    def __init__(self, queue_skeleton, queue_clip_all, action_recognition_threads, 
        st, is_static_BG, is_heatmap):

        threading.Thread.__init__(self)
        self.queue_skeleton = queue_skeleton

        self.st = st
        self.is_static_BG = is_static_BG
        self.is_heatmap = is_heatmap

        self.queue_clip_all = queue_clip_all
        self.action_recognition_threads = action_recognition_threads

    def run(self):
        while True:
            imglist, skeleton, clip_id = self.queue_skeleton.get()
            if isinstance(skeleton, str) and skeleton == 'quit':
                break

            im_name_all, kp_preds_all, kp_scores_all = self.st.get_valid_skeletons(
                'None', in_skeleton_list=skeleton, is_savejson=False)
            clip_all, heatmap_all = self.st.get_hand_clip('None', 'None', 'None', 'None.json',
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                is_save=False, is_vis=False, is_static_BG=self.is_static_BG, is_labeled=False, 
                is_heatmap=self.is_heatmap)

            self.queue_clip_all.put([clip_all, heatmap_all, im_name_all, kp_preds_all, kp_scores_all, imglist, clip_id])
            print('queue_clip_all put', self.queue_clip_all.qsize(), clip_id)

        self.queue_clip_all.put(['quit', 'quit', 'quit', 'quit', 'quit', 'quit', 'quit'])
        self.action_recognition_threads.join()
        print('=================== finish SkeletonToolThreads ===================')


class ActionRecognitionThreads(threading.Thread):
    def __init__(self, queue_clip_all, queue_result_labels, skeleton_vis_threads, reg, is_heatmap):

        threading.Thread.__init__(self)
        self.queue_clip_all = queue_clip_all

        self.reg = reg
        self.is_heatmap = is_heatmap

        self.queue_result_labels = queue_result_labels
        self.skeleton_vis_threads = skeleton_vis_threads

    def run(self):
        while True:
            clip_all, heatmap_all, im_name_all, kp_preds_all, kp_scores_all, imglist, clip_id = self.queue_clip_all.get()
            if isinstance(clip_all, str) and clip_all == 'quit':
                break

            result_labels = []
            for i, clip in enumerate(clip_all):
                clip_PIL = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in clip]
                if clip_PIL is None or len(clip_PIL) == 0:
                    result_labels.append([0, [0.0, 0.0, 0.0]])
                else:
                    if self.is_heatmap:
                        heatmap_PIL = [Image.fromarray(img[:, :, 0]) for img in heatmap_all[i]]
                        label, probs = self.reg.run(clip_PIL, heatmap_PIL)
                    else:
                        label, probs = self.reg.run(clip_PIL)
                    result_labels.append([label, probs])

            self.queue_result_labels.put([result_labels, im_name_all, kp_preds_all, kp_scores_all, imglist, clip_id])
            print('queue_result_labels put', self.queue_result_labels.qsize(), clip_id)

        self.queue_result_labels.put(['quit', 'quit', 'quit', 'quit', 'quit', 'quit'])
        self.skeleton_vis_threads.join()
        print('=================== finish ActionRecognitionThreads ===================')


class SkeletonVisThreads(threading.Thread):
    def __init__(self, queue_result_labels, queue_img_out_all, st, thres, out):
        threading.Thread.__init__(self)
        self.queue_result_labels = queue_result_labels

        self.st = st
        self.thres = thres

        self.out = out
        self.queue_img_out_all = queue_img_out_all
        self.out_show_id = 0

    def run(self):
        while True:
            result_labels, im_name_all, kp_preds_all, kp_scores_all, imglist, clip_id = self.queue_result_labels.get()
            if isinstance(result_labels, str) and result_labels == 'quit':
                break

            img_out_all = self.st.vis_skeleton('None', 'None', 'None.json',
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                result_labels=result_labels, is_save=False, is_vis=False, thres=self.thres)

            self.queue_img_out_all.put([img_out_all, clip_id])
            self.out_show_id += 1

            if self.out is not None:
                for img_out in img_out_all:
                    self.out.write(img_out)
                print('SkeletonVisThreads write', len(img_out_all))

        if self.out is not None:
            self.out.release()
        self.queue_img_out_all.put(['quit', 'quit'])
        print('=================== finish SkeletonVisThreads ===================')


class DetectionMultiThreads(threading.Thread):
    def __init__(self, queue_imglist, queue_img_out_all, reg_model_file, skeleton_opt, cuda_id_list, 
        sample_duration, sample_rate=1, is_static_BG=False, is_heatmap=False, thres=0.5, out=None):

        threading.Thread.__init__(self)

        skeleton_cuda_id, reg_cuda_id = cuda_id_list

        if skeleton_opt == 'MSRA':
            from MSRApose_skeleton import MSRApose_skeleton
            skeleton_det = MSRApose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Alphapose':
            from Alphapose_skeleton import Alphapose_skeleton
            skeleton_det = Alphapose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Openpose':
            from Openpose_skeleton import Openpose_skeleton
            skeleton_det = Openpose_skeleton(cuda_id=skeleton_cuda_id)
        else:
            raise Exception('Error: ' + skeleton_opt + ' could not be found')

        st = skeleton_tools()

        if is_heatmap:
            from Action_Recognition_Skeleton import Action_Recognition_Skeleton as Action_Recognition
        else:
            from Action_Recognition import Action_Recognition
        reg = Action_Recognition(reg_model_file, sample_duration, reg_cuda_id)
        print('=================== Network Initialized ===================\n\n')

        self.queue_imglist = queue_imglist

        self.skeleton_det = skeleton_det
        self.sample_rate = sample_rate

        queue_result_labels = Queue()
        skeleton_vis_threads = SkeletonVisThreads(queue_result_labels, queue_img_out_all, 
            st, thres, out)
        skeleton_vis_threads.start()

        queue_clip_all = Queue()
        action_recognition_threads = ActionRecognitionThreads(queue_clip_all, queue_result_labels, 
            skeleton_vis_threads, reg, is_heatmap)
        action_recognition_threads.start()

        self.queue_skeleton = Queue()
        self.skeleton_tool_threads = SkeletonToolThreads(self.queue_skeleton, queue_clip_all, action_recognition_threads, 
            st, is_static_BG, is_heatmap)
        self.skeleton_tool_threads.start()

        print('=================== Threads Initialized ===================')

    def run(self):
        while True:
            imglist, clip_id = self.queue_imglist.get()
            if isinstance(imglist, str) and imglist == 'quit':
                break

            skeleton = self.skeleton_det.run(imglist, self.sample_rate)

            self.queue_skeleton.put([imglist, skeleton, clip_id])
            print('queue_skeleton put', self.queue_skeleton.qsize(), clip_id)

        self.queue_skeleton.put(['quit', 'quit', 'quit'])
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

