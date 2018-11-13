import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt
import matplotlib.pyplot as plt
from PIL import Image
import math
import copy

from dataloader import Image_loader, VideoDetectionLoader, DataWriter, crop_from_dets, Mscoco, DetectionLoader
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction_batch
from SPPE.src.utils.img import load_image
import os
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json


args = opt
args.dataset = 'coco'


class skeleton_tools:
    def __init__(self):
        None

    def get_skeleton(self, inputpath, outputpath):

        # update inputpath in opt
        print(inputpath)
        args.inputpath = inputpath

        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

        # Load input images
        im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
        dataset = Image_loader(im_names, format='yolo')

        # Load detection loader
        print('Loading YOLO model..')
        test_loader = DetectionLoader(dataset).start()

        # Load pose model
        pose_dataset = Mscoco()
        if args.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        pose_model.cuda()
        pose_model.eval()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # Init data writer
        writer = DataWriter(args.save_video).start()
        time.sleep(10)
        data_len = dataset.__len__()
        im_names_desc = tqdm(range(data_len))
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inp, orig_img, im_name, boxes, scores) = test_loader.read()
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue
                print("test loader:", test_loader.len())
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps, pt1, pt2 = crop_from_dets(inp, boxes)
                inps = Variable(inps.cuda())

                hm = pose_model(inps)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)

                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                print("writer:" , writer.len())
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

            # TQDM
            im_names_desc.set_description(
                'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

        print('===========================> Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
        while(writer.running()):
            pass
        writer.stop()
        final_result = writer.results()
        # write_json(final_result, outputpath, 'alphapose-results.json')

        out_file = open(os.path.join(outputpath, 'skeleton.txt'), 'w')
        for im_res in final_result:
            im_name = im_res['imgname']

            out_file.write(im_name)
            if im_res['result'] is not None:
                for human in im_res['result']:
                    kp_preds = human['keypoints']
                    kp_scores = human['kp_score']

                    for n in range(kp_scores.shape[0]):
                        out_file.write(' ' + str(int(kp_preds[n, 0])))
                        out_file.write(' ' + str(int(kp_preds[n, 1])))
                        out_file.write(' ' + str(round(float(kp_scores[n]), 2)))
            out_file.write('\n')
        out_file.close()


    def __plot_skeleton(self, img, kp_preds, kp_scores, skeleton_size, target_kps):

        l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),       # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # Hand
            (17, 11), (17, 12),                         # Body
            (11, 13), (12, 14), (13, 15), (14, 16)]     # Leg

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
        
        # Nose, LEye, REye, LEar, REar
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77),
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)]
        

        for h in range(len(kp_scores) // skeleton_size): # number of human
            kp_preds_h = kp_preds[h*2*skeleton_size : (h+1)*2*skeleton_size]
            kp_scores_h = kp_scores[h*skeleton_size : (h+1)*skeleton_size]

            kp_preds_h += [(int(kp_preds_h[10])+int(kp_preds_h[12]))/2, (int(kp_preds_h[11])+int(kp_preds_h[13]))/2]
            kp_scores_h += [(float(kp_scores_h[5]) + float(kp_scores_h[6]))/2]
            
            cor_x, cor_y = int(kp_preds_h[-2]), int(kp_preds_h[-1])
            cv2.putText(img, str(h), (int(cor_x), int(cor_y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            part_line = {}
            for n in range(len(kp_scores_h)):
                # if float(kp_scores_h[n]) <= 0.05:
                #     continue

                cor_x, cor_y = int(kp_preds_h[2*n]), int(kp_preds_h[2*n+1])
                part_line[n] = (cor_x, cor_y)
                cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)

            # Draw limbs
            for i, (start_p, end_p) in enumerate(l_pair):
                if i not in target_kps:
                    continue
                    
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    cv2.line(img, start_xy, end_xy, line_color[i], int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))
        return img

    def __get_pred_score(self, kp):
        kp_preds = []
        kp_scores = []
        for i in range(len(kp)):
            if (i+1) % 3 == 0:
                kp_scores += [float(kp[i])]
            else:
                kp_preds += [int(kp[i])]
        return kp_preds, kp_scores

    def __calc_pose_distance(self, h1, h2):
        assert(len(h1) == len(h2))
        dis = 0
        for k in range(10, 20, 1):
            dis += abs(int(h1[k]) - int(h2[k]))
        return dis

    # h1: new, h2: previous
    def __pose_match(self, h1, h2, h2_map, skeleton_size):
        n1 = len(h1) // skeleton_size
        n2 = len(h2) // skeleton_size

        h1_map = []
        n_new = max(h2_map) + 1

        h2_visited = [False for i in range(n2)]
        for i in range(n1):
            min_dis = 10000000
            opt_id = 0
            for j in range(n2):
                if not h2_visited[j]:
                    h1_tmp = h1[i*skeleton_size: (i+1)*skeleton_size]
                    h2_tmp = h2[j*skeleton_size: (j+1)*skeleton_size]
                    dis = self.__calc_pose_distance(h1_tmp, h2_tmp)
                    if (min_dis > dis):
                        min_dis = dis
                        opt_id = j
            if min_dis > 5000:
                h1_map += [n_new]
                n_new += 1
            else:
                h2_visited[opt_id] = True
                h1_map += [h2_map[opt_id]]

        return h1_map

    def __validate_skeletons(self, kp_preds_all, kp_scores_all, kp_maps_all, skeleton_size, is_labeled):
        
        # get number of valid humans from list of frames
        num_valid_human = 1000
        for kp_preds in kp_preds_all:
            if kp_preds is None:
                return None, None
            num_valid_human = min(num_valid_human, len(kp_preds) // (2*skeleton_size))

        if is_labeled:
            num_valid_human = 1

        kp_preds_out_all = []
        kp_scores_out_all = []
        for i, kp_preds in enumerate(kp_preds_all):
            kp_scores = kp_scores_all[i]
            kp_maps = kp_maps_all[i]

            kp_preds_out = []
            kp_scores_out = []
            for h in range(num_valid_human):
                for match_id in range(len(kp_maps)):
                    if kp_maps[match_id] == h:
                        break
                kp_preds_out += kp_preds[match_id*2*skeleton_size : (match_id+1)*2*skeleton_size]
                kp_scores_out += kp_scores[match_id*skeleton_size : (match_id+1)*skeleton_size]
            kp_preds_out_all.append(kp_preds_out)
            kp_scores_out_all.append(kp_scores_out)

        return kp_preds_out_all, kp_scores_out_all


    def __get_valid_skeletons(self, skeleton_folder, skeleton_size, is_labeled):

        skeleton_size = 17

        in_skeleton_file = os.path.join(skeleton_folder, 'skeleton.txt')
        print(in_skeleton_file)
        in_skeleton_list = [line.split() for line in open(in_skeleton_file, 'r')]

        im_name_all = []
        kp_preds_all = []
        kp_scores_all = []
        kp_maps_all = []

        pre_map = None
        pre_preds = None
        pre_scores = None
        for line_id in range(len(in_skeleton_list)):

            line = in_skeleton_list[line_id]
            im_name_all.append(line[0])

            kp_preds, kp_scores = self.__get_pred_score(line[1:])
            if len(kp_preds) == 0:
                kp_maps_all.append(pre_map)
                kp_preds_all.append(pre_preds)
                kp_scores_all.append(pre_scores)
                continue

            if pre_preds is None:
                new_map = [i for i in range(len(kp_scores)//skeleton_size)]
            else:
                new_map = self.__pose_match(kp_preds, pre_preds, pre_map, 2*skeleton_size)
            pre_map = new_map
            pre_preds = kp_preds
            pre_scores = kp_scores

            kp_maps_all.append(new_map)
            kp_preds_all.append(kp_preds)
            kp_scores_all.append(kp_scores)

        kp_preds_all, kp_scores_all = \
            self.__validate_skeletons(kp_preds_all, kp_scores_all, kp_maps_all, skeleton_size, is_labeled)
        return im_name_all, kp_preds_all, kp_scores_all

    def __get_hand_cors(self, kp_preds_all, kp_scores_all, skeleton_size, target_kps):
        num_valid_human = len(kp_preds_all[0]) // (2*skeleton_size)
        assert(num_valid_human == len(kp_scores_all[0]) // skeleton_size)

        hand_cors = []
        for h in range(num_valid_human):
            x1, y1, x2, y2 = 10000, 10000, 0, 0
            for i, kp_preds in enumerate(kp_preds_all):
                kp_scores = kp_scores_all[i]

                kp_preds_h = kp_preds[h*2*skeleton_size : (h+1)*2*skeleton_size]
                kp_scores_h = kp_scores[h*skeleton_size : (h+1)*skeleton_size]
                for n in target_kps:
                    if float(kp_scores_h[n]) <= 0.05:
                        continue
                    x1 = min(x1, kp_preds_h[2*n]-20)
                    y1 = min(y1, kp_preds_h[2*n+1]-20)
                    x2 = max(x2, kp_preds_h[2*n]+20)
                    y2 = max(y2, kp_preds_h[2*n+1]+20)
            hand_cors.append([x1, y1, x2, y2])

        return hand_cors


    def vis_skeleton(self, in_clip_folder, skeleton_folder, is_labeled, is_save=False):

        skeleton_size = 17
        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = \
            self.__get_valid_skeletons(skeleton_folder, skeleton_size, is_labeled)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        vis_out_folder = os.path.join(skeleton_folder, 'vis')
        if not os.path.exists(vis_out_folder):
            os.makedirs(vis_out_folder)

        for i, im_name in enumerate(im_name_all):
            img = cv2.imread(os.path.join(in_clip_folder, im_name))
            img_out = self.__plot_skeleton(img, kp_preds_all[i], kp_scores_all[i], skeleton_size, target_kps)
            cv2.imshow('skeletons', img_out)
            cv2.waitKey(5)

            if is_save:
                if im_name == 'image_00000.jpg':
                    im_name = 'image_00045.jpg'
                cv2.imwrite(os.path.join(vis_out_folder, im_name), img_out)

    def get_hand_clip(self, in_clip_folder, skeleton_folder, out_clip_folder, is_labeled):

        skeleton_size = 17
        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = \
            self.__get_valid_skeletons(skeleton_folder, skeleton_size, is_labeled)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        hand_cors = self.__get_hand_cors(kp_preds_all, kp_scores_all, skeleton_size, target_kps)

        for human_id, hand_cor in enumerate(hand_cors):
            x1, y1, x2, y2 = hand_cor

            out_folder = out_clip_folder + '_' + str(human_id+1)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            for i, im_name in enumerate(im_name_all):
                img = cv2.imread(os.path.join(in_clip_folder, im_name))
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, img.shape[1])
                y2 = min(y2, img.shape[0])
                img_out = img[y1:y2, x1:x2, :]
                cv2.imshow('skeletons', img_out)
                cv2.waitKey(5)

                if im_name == 'image_00000.jpg':
                    im_name = 'image_00045.jpg'
                cv2.imwrite(os.path.join(out_folder, im_name), img_out)
            
            im_names = [img for img in sorted(os.listdir(out_folder)) if img.endswith('jpg')]
            print(out_folder, len(im_names))

def demo_video():
    
    base_folder = '/media/qcxu/qcxuDisk/scratch_dataset/'
    __action__ = ['scratch']#, 'others', 'pick', 'scratch']

    st = skeleton_tools()

    for act in __action__:

        if act == 'others':
            is_labeled = False
        else:
            is_labeled = True

        base_in_clip_folder = base_folder + act + '/clips/'
        base_skeleton_folder = base_folder + act + '/skeletons/'
        base_out_clip_folder = base_folder + 'hand/' + act + '/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):
            # if sub != 'Video_32_11_1':
            #     continue

            if act == 'others' and sub_id % 4 != 0:
                continue

            in_clip_folder = base_in_clip_folder + sub
            skeleton_folder = base_skeleton_folder + sub
            out_clip_folder = base_out_clip_folder + sub

            # st.get_skeleton(in_clip_folder, skeleton_folder)
            # st.vis_skeleton(in_clip_folder, skeleton_folder, is_labeled, is_save=False)
            st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder, is_labeled)


def count_clips()
    base_folder = '/media/qcxu/qcxuDisk/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']

    for act in __action__:

        base_out_clip_folder = base_folder + 'hand/' + act + '/'

        for sub_id, sub in enumerate(os.listdir(base_out_clip_folder)):
            out_clip_folder = base_out_clip_folder + sub
            im_names = [img for img in sorted(os.listdir(out_clip_folder)) if img.endswith('jpg')]
            print(out_clip_folder, len(im_names))
            assert(len(im_names) != 45)

if __name__ == "__main__":

    demo_video()