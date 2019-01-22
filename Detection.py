
from skeleton_tools import skeleton_tools
from Action_Recognition import Action_Recognition

import time
import os
import cv2
from PIL import Image


class Detection:
    def __init__(self, reg_model_file, model_type, skeleton_opt, cuda_id_list, 
        sample_duration, sample_rate=1, is_static_BG=False, is_heatmap=False, thres=0.5):

        skeleton_cuda_id, reg_cuda_id = cuda_id_list

        if skeleton_opt == 'MSRA':
            from MSRApose_skeleton import MSRApose_skeleton
            self.skeleton_det = MSRApose_skeleton(cuda_id=skeleton_cuda_id, fast_yolo=False)
        elif skeleton_opt == 'Alphapose':
            from Alphapose_skeleton import Alphapose_skeleton
            self.skeleton_det = Alphapose_skeleton(cuda_id=skeleton_cuda_id, fast_yolo=False)
        elif skeleton_opt == 'Openpose':
            from Openpose_skeleton import Openpose_skeleton
            self.skeleton_det = Openpose_skeleton(cuda_id=skeleton_cuda_id)
        elif skeleton_opt == 'TFOpenpose':
            from TFOpenpose_skeleton import TFOpenpose_skeleton
            self.skeleton_det = TFOpenpose_skeleton(cuda_id=skeleton_cuda_id)
        else:
            raise Exception('Error: ' + skeleton_opt + ' could not be found')

        self.st = skeleton_tools()
        self.reg = Action_Recognition(reg_model_file, sample_duration, model_type, cuda_id=reg_cuda_id)
        print('=================== Initialized ===================\n\n')

        self.sample_rate = sample_rate
        self.is_static_BG = is_static_BG
        self.is_heatmap = is_heatmap
        self.thres = thres

        self.time_st = 0.0
        self.time_reg = 0.0
        self.time_vis = 0.0

    def run(self, imglist, out_clip_folder=None):
        ### imglist: list of images read by opencv2

        # detect skeleton
        skeleton = self.skeleton_det.run(imglist, sample_rate=self.sample_rate)

        # prepare hand clip
        time1 = time.time()
        im_name_all, kp_preds_all, kp_scores_all = self.st.get_valid_skeletons(
            'None', in_skeleton_list=skeleton, is_savejson=False)
        clip_all, heatmap_all = self.st.get_hand_clip('None', out_clip_folder, 'None', 'None.json',
            im_name_all, kp_preds_all, kp_scores_all, imglist,
            is_save=False, is_vis=False, is_static_BG=self.is_static_BG, is_labeled=False, 
            is_heatmap=self.is_heatmap)
        self.time_st += (time.time() - time1)

        # prepare hand clip for action recognition
        time1 = time.time()
        empty_list = []
        clip_PIL_batch = []
        heatmap_PIL_batch = []
        for i, clip in enumerate(clip_all):
            if len(clip) == 0:
                empty_list.append(i)
                continue
            clip_PIL = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in clip]
            clip_PIL_batch.append(clip_PIL)
            if self.is_heatmap:
                heatmap_PIL = [Image.fromarray(img[:, :, 0]) for img in heatmap_all[i]]
                heatmap_PIL_batch.append(heatmap_PIL)

        # run action recornition
        if len(clip_PIL_batch) == 0:
            result_labels = [0, [0.0, 0.0, 0.0]]
        else:
            if self.is_heatmap:
                result_labels = self.reg.run(clip_PIL_batch, heatmap_PIL_batch)
            else:
                result_labels = self.reg.run(clip_PIL_batch)
        for i in empty_list:
            result_labels.insert(i, [0, [0.0, 0.0, 0.0]])

        self.time_reg += (time.time() - time1)

        # visualize result
        time1 = time.time()
        img_out_all, _, _, _ = self.st.vis_skeleton('None', 'None', 'None.json',
            im_name_all, kp_preds_all, kp_scores_all, imglist,
            result_labels=result_labels, is_save=False, is_vis=True, thres=self.thres)
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


# def extract_hand_clip_from_clip_folders():
    
#     T = 48

#     skeleton_opt = 'Alphapose' # 'MSRA'  # 'Alphapose'  # 'Openpose' # 'TFOpenpose'

#     reg_model_file = 'results-scratch-18/save_200.pth'
#     detection = Detection(reg_model_file, 'skeleton', skeleton_opt=skeleton_opt, cuda_id_list=[0,1],
#         sample_duration=T, sample_rate=1, is_static_BG=True, is_heatmap=True, thres=0.5)

#     act = 'scratch'
#     base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_scratch/image_scratch/'
#     dst_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_scratch/clip_scratch/'
#     for sub in os.listdir(base_folder):
#         sub_folder = os.path.join(base_folder, sub)
#         image_name_list = [img_name for img_name in os.listdir(sub_folder) if img_name.endswith('.jpg')]
#         N = len(image_name_list)

#         for i in range( N // T)
#         imglist = []
#         for j in range(N):
#             img_name = os.path.join(sub_folder, 'image_{:05d}.jpg'.format(j+1))
#             imglist.append(cv2.imread(img_name))

#         out_clip_folder = os.path.join(dst_folder, act+'_'+sub)
#         print(sub, N, out_clip_folder)
#         detection.run(imglist, out_clip_folder)

#     detection.print_runtime()


def extract_hand_clip_from_image_folders():
    T = 48
    skeleton_opt = 'Alphapose' # 'MSRA'  # 'Alphapose'  # 'Openpose' # 'TFOpenpose'

    reg_model_file = 'results-scratch-18/save_200.pth'
    detection = Detection(reg_model_file, 'skeleton', skeleton_opt=skeleton_opt, cuda_id_list=[1,1],
        sample_duration=T, sample_rate=1, is_static_BG=True, is_heatmap=True, thres=0.5)

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_normal/image_normal/'
    dst_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_normal/clip_normal/'
    for sub in os.listdir(base_folder):
        sub_folder = os.path.join(base_folder, sub)
        image_name_list = [img_name for img_name in os.listdir(sub_folder) if img_name.endswith('.jpg')]
        N = len(image_name_list) // T - 1
        print(sub, N)

        for i in range(N):
            imglist = []
            for j in range(i*T, (i+1)*T, 1):
                img_name = os.path.join(sub_folder, 'image_{:05d}.jpg'.format(j+1))
                imglist.append(cv2.imread(img_name))

            out_clip_folder = os.path.join(dst_folder, sub+'_'+str(i+1))
            print(N, out_clip_folder)
            detection.run(imglist, out_clip_folder)

    detection.print_runtime()


if __name__ == "__main__":

    # None
    # test()
    # extract_hand_clip_from_clip_folders()
    extract_hand_clip_from_image_folders()