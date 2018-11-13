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



def get_skeleton(video_name):
    args.video = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/' + video_name
    args.outputpath = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/' + video_name[:-4]
    args.save_img = True

    videofile = args.video
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    
    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load detection loader
    print('Loading YOLO model..')
    test_loader = VideoDetectionLoader(videofile).start()
    (fourcc,fps,frameSize) = test_loader.videoinfo()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_'+videofile.split('/')[-1].split('.')[0]+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }

    im_names_desc =  tqdm(range(test_loader.length()))
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            # Human Detection
            (inp, orig_img, boxes, scores) = test_loader.read()            
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name=str(i)+'.jpg')
                continue
            print("test loader:", test_loader.len())
            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)

            # print(boxes)

            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp, boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name=str(i)+'.jpg')
            print("writer:" , writer.len())
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        # TQDM
        im_names_desc.set_description(
            'det time: {dt:.4f} | pose time: {pt:.4f} | post process: {pn:.4f}'.format(
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
    write_json(final_result, args.outputpath, 'alphapose-results.json')

    out_file = open(os.path.join(args.outputpath, 'skeleton.txt'), 'w')
    for im_res in final_result:
        im_name = im_res['imgname']

        out_file.write(im_name)
        for human in im_res['result']:
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']

            for n in range(kp_scores.shape[0]):
                out_file.write(' ' + str(int(kp_preds[n, 0])))
                out_file.write(' ' + str(int(kp_preds[n, 1])))
                out_file.write(' ' + str(round(float(kp_scores[n]), 2)))
        out_file.write('\n')
    out_file.close()



def vis_frame(vis_out_folder, im_name, kp_preds, kp_scores, human_index, skeleton_size):

    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                (77,222,255), (255,156,127), 
                (0,127,255), (255,127,77), (0,77,255), (255,77,36)]

    img = np.zeros((720, 1280, 3), np.uint8)

    for h in range(len(kp_scores) // skeleton_size): # number of human
        kp_preds_h = kp_preds[h*2*skeleton_size : (h+1)*2*skeleton_size]
        kp_scores_h = kp_scores[h*skeleton_size : (h+1)*skeleton_size]

        kp_preds_h += [(int(kp_preds_h[10])+int(kp_preds_h[12]))/2, (int(kp_preds_h[11])+int(kp_preds_h[13]))/2]
        kp_scores_h += [(float(kp_scores_h[5]) + float(kp_scores_h[6]))/2]
        
        cor_x, cor_y = int(kp_preds_h[-2]), int(kp_preds_h[-1])
        cv2.putText(img, str(human_index[h]), (int(cor_x), int(cor_y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        part_line = {}
        for n in range(len(kp_scores_h)):
            # if float(kp_scores_h[n]) <= 0.05:
            #     continue

            cor_x, cor_y = int(kp_preds_h[2*n]), int(kp_preds_h[2*n+1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if i < 4 or i > 10:
                continue
                
            if start_p in part_line and end_p in part_line:
                # if not 7 in [start_p, end_p] and not 8 in [start_p, end_p]:
                    # continue
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))

    cv2.imwrite(vis_out_folder+'/'+im_name, img)

def post_distance(h1, h2):
    assert(len(h1) == len(h2))
    dis = 0
    # for k in range(len(h1)):
    for k in range(10, 20, 1):
        dis += abs(int(h1[k]) - int(h2[k]))
    return dis

# h1: new
# h2: previous
def pose_match(h1, h2, h2_map, skeleton_size):
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
                dis = post_distance(h1_tmp, h2_tmp)
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

def get_pred_score(kp):
    kp_preds = []
    kp_scores = []
    for i in range(len(kp)):
        if (i+1) % 3 == 0:
            kp_scores += [float(kp[i])]
        else:
            kp_preds += [int(kp[i])]
    return kp_preds, kp_scores


def vis_skeleton(video_name, is_vis=False):

    T = 40
    skeleton_size = 17

    base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/' + video_name[:-4]

    vis_out_folder = os.path.join(base_folder, 'viss')
    if not os.path.exists(vis_out_folder):
        os.makedirs(vis_out_folder)

    in_skeleton_file = os.path.join(base_folder, 'skeleton.txt')
    in_skeleton_list = [line.split() for line in open(in_skeleton_file, 'r')]

    demo_skeleton_file_name = os.path.join(base_folder, 'skeleton_demo.txt')
    demo_skeleton_file = open(demo_skeleton_file_name, 'a')

    his_kp = []
    pre_kp = []
    for line_id in range(len(in_skeleton_list)):

        line = in_skeleton_list[line_id]
        im_name = line[0]

        kp_preds, kp_scores = get_pred_score(line[1:])
        if len(kp_preds) == 0:
            his_kp += [None]
            continue

        if len(pre_kp) == 0:
            new_map = [i for i in range(len(kp_scores)//skeleton_size)]
        else:
            new_map = pose_match(kp_preds, pre_kp, pre_map, 2*skeleton_size)
        pre_kp = kp_preds
        pre_map = new_map

        his_kp += [new_map]

        if is_vis:
            print(im_name)
            vis_frame(vis_out_folder, im_name, kp_preds, kp_scores, new_map, skeleton_size)

    for line_id in range(T, 1000-T, 1):
        im_name = in_skeleton_list[line_id][0]
        for human_index in his_kp[line_id]:
            flag = True
            ROIs = []
            for t in range(-T//2, T//2):
                if human_index not in his_kp[line_id+t]:
                    flag = False
                    break
                else:
                    for match_id in range(len(his_kp[line_id+t])):
                        if his_kp[line_id+t][match_id] == human_index:
                            break
                    print(his_kp[line_id+t], human_index, match_id)

                    line = in_skeleton_list[line_id+t]
                    kp_preds, kp_scores = get_pred_score(line[1+match_id*3*skeleton_size : 1+(match_id+1)*3*skeleton_size])
                    ROIs += [kp_preds]
            if flag:
                for ROI in ROIs:
                    demo_skeleton_file.write(im_name + '_' + str(human_index))
                    for x in ROI:
                        demo_skeleton_file.write(' ' + str(x))
                    demo_skeleton_file.write('\n')

    demo_skeleton_file.close()


def write_ROI(target_h, pre_after, skeleton_size, T, demo_skeleton_file_name):
    if (len(pre_after) != T):
        return

    flag = True
    ROI = []
    for h2 in pre_after:
        min_dis = 10000000
        opt_id = 0
        n2 = len(h2) // skeleton_size
        # print(h2, n2)
        for j in range(n2):
            h2_tmp = h2[j*skeleton_size : (j+1)*skeleton_size]
            dis = post_distance(target_h, h2_tmp)
            if min_dis > dis:
                min_dis = dis
                opt_id = j
        if min_dis > 5000:
            flag = False
            break
        else:
            h2_target = h2[opt_id*skeleton_size : (opt_id+1)*skeleton_size]
            ROI += [h2_target]

    if flag:
        print(len(ROI))
        demo_skeleton_file = open(demo_skeleton_file_name, 'a')
        demo_skeleton_file.close()

def find_scratch(video_name):

    T = 40
    skeleton_size = 17

    base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/' + video_name[:-4]

    vis_out_folder = os.path.join(base_folder, 'viss')
    if not os.path.exists(vis_out_folder):
        os.makedirs(vis_out_folder)

    in_skeleton_file = os.path.join(base_folder, 'skeleton.txt')
    in_skeleton_list = [line.split() for line in open(in_skeleton_file, 'r')]

    demo_skeleton_file_name = os.path.join(base_folder, 'skeleton_demo.txt')

    pre_kp = []
    for line_id in range(len(in_skeleton_list)):

        line = in_skeleton_list[line_id]
        im_name = line[0]
        print(im_name)

        if line_id > T//2 and line_id < len(in_skeleton_list) - T//2:
            num_h = len(line[1:]) // (3*skeleton_size)
            for h in range(num_h):
                target_h = line[1+h*3*skeleton_size : 1+(h+1)*3*skeleton_size]
                target_h_kp_preds, _ = get_pred_score(target_h)

                pre_after = []
                for line in in_skeleton_list[line_id-T//2: line_id+T//2]:
                    pre_after_kp_preds, _ = get_pred_score(line[1:])
                    pre_after += [pre_after_kp_preds]

                write_ROI(target_h_kp_preds, pre_after, 2*skeleton_size, T, demo_skeleton_file_name)


def demo_video():
    
    video_name = '1.mp4'
    # video_name = 'scratch000333.mp4'
    # video_name = '1-2.mp4' 

    get_skeleton(video_name)
    # vis_skeleton(video_name)

    # find_scratch(video_name)


if __name__ == "__main__":

    demo_video()