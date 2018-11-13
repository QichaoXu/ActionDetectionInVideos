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


def vis_frame(vis_out_folder, subsubdir, im_name, im_res):

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

    vis_input_folder = '/media/qcxu/qcxuDisk/windows_datasets_skeleton_label/'
    frame = cv2.imread(os.path.join(vis_input_folder, subsubdir, im_name))

    img = frame
    height,width = img.shape[:2]
    img = cv2.resize(img,(int(width/2), int(height/2)))

    im_res += [(int(im_res[15])+int(im_res[18]))/2, (int(im_res[16])+int(im_res[19]))/2]
    im_res += [(float(im_res[17])+float(im_res[20]))/2]

    # kp_scores = [0.9 for n in range(len(im_res))]
    part_line = {}
    # Draw keypoints
    for n in range(len(im_res)//3):
        if float(im_res[3*n+2]) <= 0.05:
            continue

        cor_x, cor_y = int(im_res[3*n]), int(im_res[3*n+1])

        part_line[n] = (int(cor_x/2), int(cor_y/2))
        bg = img.copy()
        cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
        # Now create a mask of logo and create its inverse mask also
        transparency = max(0, min(1, float(im_res[3*n+2])))
        img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            bg = img.copy()

            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 1#(float(im_res[3*start_p+2]) + float(im_res[3*end_p+2])) + 1
            polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(bg, polygon, line_color[i])
            transparency = 0.9#max(0, min(1, 0.5*(float((im_res[3*start_p+2])) + float(im_res[3*end_p+2]))))
            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
    img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(vis_out_folder+'/'+im_name, img)

def windows_datasets_get_skeleton():
    args.save_img = True

    base_folder = '/media/qcxu/qcxuDisk/windows_datasets_skeleton_label'
    out_base_folder = '/media/qcxu/qcxuDisk/windows_datasets_skeleton'
    if not os.path.exists(out_base_folder):
        os.makedirs(out_base_folder)
    # for subdir in sorted(os.listdir(base_folder)):
    #     if not subdir endswith('.clip'):
    #         continue
    sub_folder = os.path.join(base_folder,  '')
    for subsubdir in sorted(os.listdir(sub_folder)):
        if not subsubdir.endswith('.clip'):
            continue
        if not '21' in subsubdir:
            continue

        sub_sub_folder = os.path.join(sub_folder, subsubdir)
        image_name_list = [img for img in sorted(os.listdir(sub_sub_folder)) if img.endswith(".jpg")]
        print(sub_sub_folder, len(image_name_list))
        if len(image_name_list) == 0:
            continue

        args.inputpath = sub_sub_folder
        args.outputpath = os.path.join(out_base_folder, subsubdir)

        inputpath = args.inputpath
        inputlist = args.inputlist
        outputpath = args.outputpath
        print(outputpath)
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
            # for root, dirs, files in os.walk(inputpath):
            #     im_names = files
        else:
            raise IOError('Error: must contain either --indir/--list')

        print(len(im_names))

        # Load input images
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
        write_json(final_result, outputpath, 'alphapose-results.json')

        out_file = open(os.path.join(outputpath, 'skeleton.txt'), 'w')
        i = 0
        for im_name_tmp in im_names:
            im_res = final_result[i]
            im_name = im_res['imgname']

            if im_name != im_name_tmp:
                out_file.write(im_name_tmp)
                out_file.write('\n')
                continue

            i += 1

            out_file.write(im_name)
            for human in im_res['result']:
                kp_preds = human['keypoints']
                kp_scores = human['kp_score']
                for n in range(kp_scores.shape[0]):
                    out_file.write(' ' + str(int(kp_preds[n, 0])))
                    out_file.write(' ' + str(int(kp_preds[n, 1])))
                    # out_file.write(' ' + str(float(kp_scores[n])))
            out_file.write('\n')
        out_file.close()


def process_skeleton():
    
    T = 40
    skeleton_size = 17*2

    base_folder = '/media/qcxu/qcxuDisk/windows_datasets_skeleton/'
    for subsubdir in sorted(os.listdir(base_folder)):
        if not '21' in subsubdir: # 17 18 21 43 102
            continue

        sub_folder = os.path.join(base_folder, subsubdir)
        vis_out_folder = os.path.join(sub_folder, 'viss')
        if not os.path.exists(vis_out_folder):
            os.makedirs(vis_out_folder)

        in_skeleton_file = os.path.join(sub_folder, 'skeleton.txt')
        in_skeleton_list = [line.split() for line in open(in_skeleton_file, 'r')]

        out_skeleton_file_name = os.path.join(sub_folder, 'skeleton_out.txt')
        out_skeleton_file = open(out_skeleton_file_name, 'w')

        counter = 0
        line_id = 0
        while line_id < len(in_skeleton_list)-T:
        # for line in in_skeleton_list:

            flag = True
            for t in range(T):
                line = in_skeleton_list[line_id+t]
                if len(line) != 1+skeleton_size:
                    flag = False
                    break

            # if flag:
            #     diff_sum = 0.0
            #     for t in range(T-1):
            #         line1 = in_skeleton_list[line_id+t]
            #         line2 = in_skeleton_list[line_id+t+1]
            #         diff = 0.0
            #         for j in range(skeleton_size//3):
            #             tmpx = abs(int(line1[1+3*j]) - int(line2[1+3*j]))
            #             tmpy = abs(int(line1[1+3*j+1]) - int(line2[1+3*j+1]))
            #             # print(subsubdir, line1[0], t, j, tmp)
            #             diff += tmpx
            #             diff += tmpy
            #         diff_sum += diff
            #     # print(diff_sum)

            if flag:
                for t in range(T):
                    line = in_skeleton_list[line_id+t]
                    out_skeleton_file.write(' '.join(line[:1+skeleton_size])+'\n')
                    # vis_frame(vis_out_folder, subsubdir, line[0], line[1:1+skeleton_size])
                counter += 1

            line_id += T
        out_skeleton_file.close()

        print(out_skeleton_file_name, counter)



if __name__ == "__main__":
    
    windows_datasets_get_skeleton()
    process_skeleton()
