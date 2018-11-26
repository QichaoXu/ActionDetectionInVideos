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

import json
from yolo.darknet import Darknet

args = opt
args.dataset = 'coco'


class skeleton_tools:
    def __init__(self):
        self.skeleton_size = 17

        # Load yolo detection model
        print('Loading YOLO model..')
        self.det_model = Darknet("yolo/cfg/yolov3.cfg")
        self.det_model.load_weights('models/yolo/yolov3.weights')
        self.det_model.cuda()
        self.det_model.eval()

        # Load pose model
        print('Loading pose model..')
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()



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
        test_loader = DetectionLoader(dataset, self.det_model).start()

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

                hm = self.pose_model(inps)
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

    def __plot_skeleton(self, img, kp_preds, kp_scores, target_kps, result_labels):

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
        
        if result_labels is None:

            for h in range(len(kp_scores) // self.skeleton_size): # number of human
                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]

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

        else:

            for h in range(len(kp_scores) // self.skeleton_size): # number of human
                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]

                kp_preds_h += [(int(kp_preds_h[10])+int(kp_preds_h[12]))/2, (int(kp_preds_h[11])+int(kp_preds_h[13]))/2]
                kp_scores_h += [(float(kp_scores_h[5]) + float(kp_scores_h[6]))/2]
                
                cor_x, cor_y = int(kp_preds_h[-2]), int(kp_preds_h[-1])


                if result_labels[h][0] != 'others' and result_labels[h][1] > 0.75: 
                    cv2.putText(img, result_labels[h][0]+':{:.3f}'.format(result_labels[h][1]), (int(cor_x), int(cor_y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

                    part_line = {}
                    for n in range(len(kp_scores_h)):
                        # if float(kp_scores_h[n]) <= 0.05:
                        #     continue

                        cor_x, cor_y = int(kp_preds_h[2*n]), int(kp_preds_h[2*n+1])
                        part_line[n] = (cor_x, cor_y)

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
    def __pose_match(self, h1, h2, h2_map):
        n1 = len(h1) // (2*self.skeleton_size)
        n2 = len(h2) // (2*self.skeleton_size)

        h1_map = []
        n_new = max(h2_map) + 1

        h2_visited = [False for i in range(n2)]
        for i in range(n1):
            min_dis = 10000000
            opt_id = 0
            for j in range(n2):
                if not h2_visited[j]:
                    h1_tmp = h1[i*2*self.skeleton_size: (i+1)*2*self.skeleton_size]
                    h2_tmp = h2[j*2*self.skeleton_size: (j+1)*2*self.skeleton_size]
                    dis = self.__calc_pose_distance(h1_tmp, h2_tmp)
                    if (min_dis > dis):
                        min_dis = dis
                        opt_id = j
            if min_dis > 500:
                h1_map += [n_new]
                n_new += 1
            else:
                h2_visited[opt_id] = True
                h1_map += [h2_map[opt_id]]

        return h1_map

    def __validate_skeletons(self, kp_preds_all, kp_scores_all, kp_maps_all, is_labeled=False):

        return kp_preds_all, kp_scores_all
        
        # get number of valid humans from list of frames
        num_valid_human = 1000
        for kp_preds in kp_preds_all:
            if kp_preds is None:
                return None, None
            num_valid_human = min(num_valid_human, len(kp_preds) // (2*self.skeleton_size))

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
                kp_preds_out += kp_preds[match_id*2*self.skeleton_size : (match_id+1)*2*self.skeleton_size]
                kp_scores_out += kp_scores[match_id*self.skeleton_size : (match_id+1)*self.skeleton_size]
            kp_preds_out_all.append(kp_preds_out)
            kp_scores_out_all.append(kp_scores_out)

        return kp_preds_out_all, kp_scores_out_all


    def get_valid_skeletons(self, skeleton_folder, is_labeled=False):

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
                new_map = [i for i in range(len(kp_scores)//self.skeleton_size)]
            else:
                new_map = self.__pose_match(kp_preds, pre_preds, pre_map)
            pre_map = new_map
            pre_preds = kp_preds
            pre_scores = kp_scores

            kp_maps_all.append(new_map)
            kp_preds_all.append(kp_preds)
            kp_scores_all.append(kp_scores)

        kp_preds_all, kp_scores_all = \
            self.__validate_skeletons(kp_preds_all, kp_scores_all, kp_maps_all, is_labeled)

        results = {}
        results['im_name_all'] = im_name_all
        results['kp_preds_all'] = kp_preds_all
        results['kp_scores_all'] = kp_scores_all
        with open(
                os.path.join(skeleton_folder, 'valid_skeleton.json'),
                'w') as f:
            json.dump(results, f)

    def __load_valid_skeleton_json(self, skeleton_folder):
        f = open(os.path.join(skeleton_folder, 'valid_skeleton.json'))
        data = json.load(f)
        im_name_all = data['im_name_all']
        kp_preds_all = data['kp_preds_all']
        kp_scores_all = data['kp_scores_all']
        f.close()
        return im_name_all, kp_preds_all, kp_scores_all

    def __get_hand_cors(self, kp_preds_all, kp_scores_all, target_kps):
        num_valid_human = len(kp_preds_all[0]) // (2*self.skeleton_size)
        assert(num_valid_human == len(kp_scores_all[0]) // self.skeleton_size)

        hand_cors = []
        for h in range(num_valid_human):
            x1, y1, x2, y2 = 10000, 10000, 0, 0
            for i, kp_preds in enumerate(kp_preds_all):
                kp_scores = kp_scores_all[i]

                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]
                for n in target_kps:
                    if float(kp_scores_h[n]) <= 0.05:
                        continue
                    x1 = min(x1, kp_preds_h[2*n]-20)
                    y1 = min(y1, kp_preds_h[2*n+1]-20)
                    x2 = max(x2, kp_preds_h[2*n]+20)
                    y2 = max(y2, kp_preds_h[2*n+1]+20)
            hand_cors.append([x1, y1, x2, y2])

        return hand_cors

    def vis_skeleton(self, in_clip_folder, skeleton_folder, result_labels=None, is_labeled=False, is_save=False):

        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        for i, im_name in enumerate(im_name_all):
            img = cv2.imread(os.path.join(in_clip_folder, im_name))
            img_out = self.__plot_skeleton(img, kp_preds_all[i], kp_scores_all[i], target_kps, result_labels)
            cv2.imshow('skeletons', img_out)
            cv2.waitKey(15)

            if is_save:

                if result_labels is None:
                    vis_out_folder = os.path.join(skeleton_folder, 'vis')
                else:
                    vis_out_folder = os.path.join(skeleton_folder, 'res')
                if not os.path.exists(vis_out_folder):
                    os.makedirs(vis_out_folder)

                cv2.imwrite(os.path.join(vis_out_folder, im_name), img_out)

    def get_hand_clip(self, in_clip_folder, skeleton_folder, out_clip_folder, is_labeled=False):

        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        hand_cors = self.__get_hand_cors(kp_preds_all, kp_scores_all, target_kps)

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

    def create_TrainTestlist(self, clip_folder, TrainTest_folder, sample_rate):

        Trainlist_name = TrainTest_folder + 'trainlist01.txt'
        Testlist_name = TrainTest_folder + 'testlist01.txt'

        label_map = {'others':'1', 'pick':'2', 'scratch':'3'}

        Trainlist = open(Trainlist_name, 'w')
        Testlist = open(Testlist_name, 'w')
        count = 0
        for sub in sorted(os.listdir(clip_folder)):
            sub_folder = clip_folder + sub
            for subsub in sorted(os.listdir(sub_folder)):
                contents = sub + '/' + subsub + ' ' + label_map[sub] + '\n'
                if count % sample_rate == 0: # test
                    Testlist.write(contents)
                else:
                    Trainlist.write(contents)
                count += 1

        Trainlist.close()
        Testlist.close()

    def create_Testlist(self, clip_folder, TrainTest_folder):

        Testlist_name = TrainTest_folder + '/testlist01.txt'

        Testlist = open(Testlist_name, 'w')
        print(clip_folder)
        for sub in sorted(os.listdir(clip_folder)):
            sub_folder = clip_folder + sub
            for subsub in sorted(os.listdir(sub_folder)):
                contents = sub + '/' + subsub + ' 0' + '\n'
                Testlist.write(contents)

        Testlist.close()
    

if __name__ == "__main__":

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
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
            out_clip_folder = base_out_clip_folder + act + '_' + sub

            # st.get_skeleton(in_clip_folder, skeleton_folder)
            # st.get_valid_skeletons(skeleton_folder, is_labeled)
            # st.vis_skeleton(in_clip_folder, skeleton_folder, None, is_labeled, is_save=False)
            st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder, is_labeled)

    clip_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand/'
    TrainTest_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/TrainTestlist/'
    st.create_TrainTestlist(clip_folder, TrainTest_folder, sample_rate=5)