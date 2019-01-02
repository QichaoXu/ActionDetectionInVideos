
from skeleton_tools import skeleton_tools
from Action_Recognition import Action_Recognition

import time
import os
import cv2
from PIL import Image


class Detection:
    def __init__(self, reg_model_file, skeleton_opt, cuda_id_list, 
        sample_duration, sample_rate=1, is_vis=False, waitTime=5, is_static_BG=False, thres=0.5):

        skeleton_cuda_id, reg_cuda_id = cuda_id_list

        if skeleton_opt == 'MSRA':
            from MSRApose_skeleton import MSRApose_skeleton
            self.skeleton_det = MSRApose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Alphapose':
            from Alphapose_skeleton import Alphapose_skeleton
            self.skeleton_det = Alphapose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'Openpose':
            from Openpose_skeleton import Openpose_skeleton
            self.skeleton_det = Openpose_skeleton(cuda_id=skeleton_cuda_id)
        else:
            raise Exception('Error: ' + skeleton_opt + ' could not be found')

        self.st = skeleton_tools()
        self.reg = Action_Recognition(reg_model_file, sample_duration, cuda_id=reg_cuda_id)
        print('=================== Initialized ===================\n\n')

        self.sample_rate = sample_rate
        self.is_static_BG = is_static_BG
        self.waitTime = waitTime
        self.is_vis = is_vis
        self.thres = thres

        self.time_st = 0.0
        self.time_reg = 0.0
        self.time_vis = 0.0

    def run(self, imglist):
        ### imglist: list of images read by opencv2

        # detect skeleton
        skeleton = self.skeleton_det.run(imglist, sample_rate=self.sample_rate)

        # prepare hand clip
        time1 = time.time()
        im_name_all, kp_preds_all, kp_scores_all = self.st.get_valid_skeletons(
            'None', in_skeleton_list=skeleton, is_savejson=False)
        # self.st.vis_skeleton('None', 'None', 'None.json',
        #     im_name_all, kp_preds_all, kp_scores_all, imglist,
        #     result_labels=None, is_save=False, is_vis=True, thres=0.3)
        clip_all, _ = self.st.get_hand_clip('None', 'None', 'None', 'None.json',
            im_name_all, kp_preds_all, kp_scores_all, imglist,
            is_save=False, is_vis=False, is_static_BG=self.is_static_BG, is_labeled=False, 
            is_heatmap=False, waitTime=self.waitTime)
        self.time_st += (time.time() - time1)

        # run action recornition
        time1 = time.time()
        result_labels = []
        for clip in clip_all:
            clip_PIL = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in clip]
            if clip_PIL is None or len(clip_PIL) == 0:
                result_labels.append([0, [0.0, 0.0, 0.0]])
            else:
                label, probs = self.reg.run(clip_PIL)
                result_labels.append([label, probs])
        self.time_reg += (time.time() - time1)

        # visualize result
        time1 = time.time()
        img_out_all = self.st.vis_skeleton('None', 'None', 'None.json',
            im_name_all, kp_preds_all, kp_scores_all, imglist,
            result_labels=result_labels, is_save=False, is_vis=self.is_vis, thres=self.thres,
            waitTime=self.waitTime)
        self.time_vis += (time.time() - time1)

        return img_out_all

    def print_runtime(self):
        print('\n\n=================== Time Analysis ===================')

        time_pure_det, time_sk_det = self.skeleton_det.runtime()

        time_total = time_sk_det + self.time_st + self.time_reg + self.time_vis
        print('time pure skeleton:', '{:.4f}'.format(time_pure_det))
        print('time_total:', '{:.4f}'.format(time_total), '{:.4f}'.format(time_total / time_total))
        print('time_skeleton:', '{:.4f}'.format(time_sk_det), '{:.4f}'.format(time_sk_det / time_total))
        print('time_tool:', '{:.4f}'.format(self.time_st), '{:.4f}'.format(self.time_st / time_total))
        print('time_action:', '{:.4f}'.format(self.time_reg), '{:.4f}'.format(self.time_reg / time_total))
        print('time_visualise:', '{:.4f}'.format(self.time_vis), '{:.4f}'.format(self.time_vis / time_total))


if __name__ == "__main__":

    reg_model_file = 'results-scratch-18/save_200.pth'
    detection = Detection(reg_model_file, skeleton_opt='Openpose', 
        is_vis=False, waitTime=5, is_static_BG=False, thres=0.9)

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/others/clips/Video_11_1_1'
    imglist = []
    for img_name in os.listdir(base_folder):
        if img_name.endswith('jpg'):
            imglist.append(cv2.imread(os.path.join(base_folder, img_name)))

    detection.run(imglist)

