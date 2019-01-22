
import numpy as np
import copy
import os
import cv2
import random
import json
import time

class skeleton_tools:

    def __init__(self):
        self.skeleton_size = 17

    def __plot_skeleton(self, img, kp_preds, kp_scores, target_kps, is_plot=True, result_labels=None, thres=0.75):

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
                    if i not in [4,5,6,7,8]:
                        continue
                        
                    if start_p in part_line and end_p in part_line:
                        start_xy = part_line[start_p]
                        end_xy = part_line[end_p]
                        cv2.line(img, start_xy, end_xy, line_color[i], 
                            int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))
            return img, None, None, None

        else:

            prob_out = []
            bbox_hand_out = []
            bbox_human_out = []
            for h in range(len(kp_scores) // self.skeleton_size): # number of human
                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]

                kp_preds_h += [(int(kp_preds_h[10])+int(kp_preds_h[12]))/2, (int(kp_preds_h[11])+int(kp_preds_h[13]))/2]
                kp_scores_h += [(float(kp_scores_h[5]) + float(kp_scores_h[6]))/2]

                cor_x, cor_y = int(kp_preds_h[-2]), int(kp_preds_h[-1])

                cls_map = ['others', 'pick', 'scratch']
                result_cls_id = int(result_labels[h][0])
                result_prob = result_labels[h][1][2]
                # if cls_map[result_cls_id] == 'scratch' and result_prob > thres:
                if result_prob > thres:

                    # get rectangle
                    x1, y1, x2, y2 = 1000, 1000, 0, 0
                    x1_h, y1_h, x2_h, y2_h = 1000, 1000, 0, 0
                    for i in range(self.skeleton_size):
                        x1_h = min(x1_h, int(kp_preds_h[2*i]))
                        y1_h = min(y1_h, int(kp_preds_h[2*i+1]))
                        x2_h = max(x2_h, int(kp_preds_h[2*i]))
                        y2_h = max(y2_h, int(kp_preds_h[2*i+1]))
                        if i in target_kps:
                            x1 = min(x1, int(kp_preds_h[2*i]))
                            y1 = min(y1, int(kp_preds_h[2*i+1]))
                            x2 = max(x2, int(kp_preds_h[2*i]))
                            y2 = max(y2, int(kp_preds_h[2*i+1]))
                    bbox_hand_out.append([x1, y1, x2, y2])
                    bbox_human_out.append([x1_h, y1_h, x2_h, y2_h])
                    prob_out.append(result_prob)

                    if is_plot:
                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

                        # put text
                        cv2.putText(img, '{:.2f}'.format(result_prob), 
                            (int(cor_x), int(cor_y)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

                        # Draw limbs
                        part_line = {}
                        for n in range(len(kp_scores_h)):
                            cor_x, cor_y = int(kp_preds_h[2*n]), int(kp_preds_h[2*n+1])
                            part_line[n] = (cor_x, cor_y)

                        for i, (start_p, end_p) in enumerate(l_pair):
                            if i not in [4,5,6,7,8]:
                                continue

                            if start_p in part_line and end_p in part_line:
                                start_xy = part_line[start_p]
                                end_xy = part_line[end_p]
                                cv2.line(img, start_xy, end_xy, line_color[i], 
                                    int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))

            return img, prob_out, bbox_hand_out, bbox_human_out

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

    # h1: previous, h2: new
    def __pose_match(self, kp_preds_new, kp_preds_pre, kp_scores_new, kp_scores_pre):

        kp_preds_out = []
        kp_scores_out = []
        
        n_pre = len(kp_preds_pre) // (2*self.skeleton_size)
        n_new = len(kp_preds_new) // (2*self.skeleton_size)
        # 
        visited_new = [False for i in range(n_new)]
        for i in range(n_pre):
            min_dis = 10000000
            opt_preds = None
            opt_scores = None
            for j in range(n_new):
                if not visited_new[j]:
                    s2_tmp = kp_scores_new[j*self.skeleton_size: (j+1)*self.skeleton_size]

                    h1_tmp = kp_preds_pre[i*2*self.skeleton_size: (i+1)*2*self.skeleton_size]
                    h2_tmp = kp_preds_new[j*2*self.skeleton_size: (j+1)*2*self.skeleton_size]
                    dis = self.__calc_pose_distance(h1_tmp, h2_tmp)
                    # print(i, j, dis)
                    if (min_dis > dis):
                        min_dis = dis
                        opt_preds = h2_tmp
                        opt_scores = s2_tmp

            if min_dis > 500 or opt_preds is None:
                opt_preds = kp_preds_pre[i*2*self.skeleton_size: (i+1)*2*self.skeleton_size]
                opt_scores = kp_scores_pre[i*self.skeleton_size: (i+1)*self.skeleton_size]

            kp_preds_out += opt_preds
            kp_scores_out += opt_scores

        return kp_preds_out, kp_scores_out

    def __validate_skeletons(self, kp_preds_all, kp_scores_all):

        kp_preds_out_all = [kp_preds_all[0]]
        kp_scores_out_all = [kp_scores_all[0]]
        for i, kp_preds in enumerate(kp_preds_all):
            if i == 0:
                continue

            kp_scores = kp_scores_all[i]

            kp_preds_pre = kp_preds_out_all[-1]
            kp_scores_pre = kp_scores_out_all[-1]

            kp_preds_out, kp_scores_out = self.__pose_match(kp_preds, kp_preds_pre, kp_scores, kp_scores_pre)
            kp_preds_out_all.append(kp_preds_out)
            kp_scores_out_all.append(kp_scores_out)

        return kp_preds_out_all, kp_scores_out_all


    def get_valid_skeletons(self, skeleton_folder, in_skeleton_list, is_savejson=True):

        if in_skeleton_list is None:
            in_skeleton_file = os.path.join(skeleton_folder, 'skeleton.txt')
            in_skeleton_list = [line.split() for line in open(in_skeleton_file, 'r')]

        im_name_all = []
        kp_preds_all = []
        kp_scores_all = []

        for line_id in range(len(in_skeleton_list)):
            line = in_skeleton_list[line_id]
            im_name_all.append(line[0])

            kp_preds, kp_scores = self.__get_pred_score(line[1:])
            kp_preds_all.append(kp_preds)
            kp_scores_all.append(kp_scores)

        # validate skeleton
        kp_preds_all, kp_scores_all = self.__validate_skeletons(kp_preds_all, kp_scores_all)

        if is_savejson:
            results = {}
            results['im_name_all'] = im_name_all
            results['kp_preds_all'] = kp_preds_all
            results['kp_scores_all'] = kp_scores_all
            with open(
                    os.path.join(skeleton_folder, 'valid_skeleton.json'),
                    'w') as f:
                json.dump(results, f)

        return im_name_all, kp_preds_all, kp_scores_all

    def __load_valid_skeleton_json(self, skeleton_folder, json_file_name):
        f = open(os.path.join(skeleton_folder, json_file_name))
        data = json.load(f)
        im_name_all = data['im_name_all']
        kp_preds_all = data['kp_preds_all']
        kp_scores_all = data['kp_scores_all']
        f.close()
        return im_name_all, kp_preds_all, kp_scores_all

    def vis_skeleton(self, in_clip_folder, skeleton_folder, json_file_name, 
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                result_labels=None, is_save=False, is_vis=False, is_plot=True,
                thres=0.75, waitTime=5):

        target_kps = [5, 6, 7, 8, 9, 10]

        if im_name_all is None:
            im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder, json_file_name)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        img_out_all = []
        prob_out_all = []
        bbox_hand_out_all = []
        bbox_human_out_all = []
        for i, im_name in enumerate(im_name_all):
            if imglist is None:
                # print(os.path.join(in_clip_folder, im_name))
                img = cv2.imread(os.path.join(in_clip_folder, im_name))
            else:
                img = imglist[i].copy()
            img_out, prob_out, bbox_hand_out, bbox_human_out = self.__plot_skeleton(img, kp_preds_all[i], kp_scores_all[i], 
                target_kps, is_plot, result_labels, thres)
            # print(img_out.shape)

            if is_vis:
                cv2.imshow('skeletons', img_out)
                cv2.waitKey(waitTime)

            if is_save:
                if result_labels is None:
                    if json_file_name == 'valid_skeleton.json':
                        vis_out_folder = os.path.join(skeleton_folder, 'vis_valid')
                    else:
                        vis_out_folder = os.path.join(skeleton_folder, 'vis_all')
                else:
                    vis_out_folder = os.path.join(skeleton_folder, 'res')
                if not os.path.exists(vis_out_folder):
                    os.makedirs(vis_out_folder)

                cv2.imwrite(os.path.join(vis_out_folder, im_name), img_out)
            else:
                img_out_all.append(img_out)
                bbox_hand_out_all.append(bbox_hand_out)
                bbox_human_out_all.append(bbox_human_out)
                prob_out_all.append(prob_out)

        return img_out_all, prob_out_all[0], bbox_hand_out_all[0], bbox_human_out_all[0]

    def __multi_moving_average(self, X, window_size, times):
        for t in range(times):
            X = self.__moving_average(X, window_size)
        return X

    def __moving_average(self, X, window_size):
        window = np.ones(int(window_size))/float(window_size)
        smoothed = np.convolve(X, window, 'same').astype(int)

        start = window_size//2
        end = len(smoothed) - window_size//2
        X_new = X.copy()
        X_new[start:end] = smoothed[start:end] # keep the two ends unchanged

        return X_new

    def __smooth_coordinate(self, kp_preds_all):

        hand_bbox_tmp = np.array(kp_preds_all).transpose()
        for i in range(hand_bbox_tmp.shape[0]):
            X = hand_bbox_tmp[i].copy()
            smoothed = self.__multi_moving_average(X, window_size=5, times=3)
            hand_bbox_tmp[i] = smoothed
        hand_bbox = np.array(hand_bbox_tmp).transpose()

        return hand_bbox.tolist()

    def __get_hand_bboxs(self, kp_preds_all, kp_scores_all, target_kps, is_static_BG=False):
        num_valid_human = len(kp_preds_all[0]) // (2*self.skeleton_size)
        assert(num_valid_human == len(kp_scores_all[0]) // self.skeleton_size)

        hand_bboxs = []
        for h in range(num_valid_human):
            x1, y1, x2, y2 = 10000, 10000, 0, 0
            box_w, box_h = 0, 0
            for i, kp_preds in enumerate(kp_preds_all):
                kp_scores = kp_scores_all[i]

                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]
                for n in target_kps:
                    if float(kp_scores_h[n]) <= 0.05:
                        continue
                    x1 = min(x1, kp_preds_h[2*n] - 20)
                    y1 = min(y1, kp_preds_h[2*n+1] - 20)
                    x2 = max(x2, kp_preds_h[2*n] + 20)
                    y2 = max(y2, kp_preds_h[2*n+1] + 20)
                    box_w = max(box_w, x2-x1)
                    box_h = max(box_h, y2-y1)

            if is_static_BG:
                hand_bboxs.append([x1, y1, x2, y2])
            else:
                hand_bboxs_h = []
                for kp_preds in kp_preds_all:

                    kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]

                    # get center
                    box_cx, box_cy = 0, 0
                    target_kps = [5,6,11,12]
                    for n in target_kps:
                        box_cx += kp_preds_h[2*n]
                        box_cy += kp_preds_h[2*n+1]
                    box_cx =  box_cx // len(target_kps)
                    box_cy =  box_cy // len(target_kps)

                    box_x1 = box_cx - (box_w//2)
                    box_y1 = box_cy - (box_h//2)
                    box_x2 = box_cx + (box_w//2)
                    box_y2 = box_cy + (box_h//2)
                    hand_bboxs_h.append([box_x1, box_y1, box_x2, box_y2])

                hand_bboxs.append(hand_bboxs_h)

        return hand_bboxs

    def __crop_moving_image(self, img, x1, y1, x2, y2):
        if x1 == x2:
            return None

        img_out = np.zeros(shape=(y2-y1, x2-x1, 3), dtype='uint8')
        out_x1 = max(0, -x1)
        out_y1 = max(0, -y1)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])
        try:
            img_out[out_y1:out_y1+y2-y1, out_x1:out_x1+x2-x1, :] = img[y1:y2, x1:x2, :]
            return img_out
        except:
            return None

    def __crop_image(self, img, x1, y1, x2, y2):
        if x1 == 10000 or x2 == 10000 or x2 == 0 or y2 == 0:
            return None

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])
        try:
            img_out = img[y1:y2, x1:x2, :]
            return img_out
        except:
            return None
    
    def __create_heatmap(self, joints, image_size, target_kps, sigma=5):

        target = np.zeros((self.skeleton_size,
                           image_size[0],
                           image_size[1]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        # for joint_id in range(self.skeleton_size):
        for joint_id in target_kps:
            feat_stride = [1, 1] #image_size / image_size
            mu_x = int(joints[2*joint_id] / feat_stride[0] + 0.5)
            mu_y = int(joints[2*joint_id+1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= image_size[1] or ul[1] >= image_size[0] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                # target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], image_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], image_size[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], image_size[1])
            img_y = max(0, ul[1]), min(br[1], image_size[0])

            v = 1 #target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        target = np.sum(target, axis=0)
        target[target > 1.0] = 1.0
        target *= 255
        target = target.astype(np.uint8)
        return target[:, :, np.newaxis]

    def get_hand_clip(self, in_clip_folder, out_clip_folder, skeleton_folder, json_file_name, 
                im_name_all, kp_preds_all, kp_scores_all, imglist,
                is_save=False, is_vis=False, is_static_BG=False, is_labeled=False, 
                is_heatmap=False, waitTime=5):

        if imglist is None and not os.path.exists(in_clip_folder):
            print('Error! imglist is None and no such in_clip_folder exists!\n\n')
            return
        if im_name_all is None and not os.path.exists(json_file_name):
            print('Error! im_name_all is None and no such json_file_name exists!\n\n')
            return

        target_kps = [5, 6, 7, 8, 9, 10]

        if im_name_all is None:
            im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder, json_file_name)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        ## smooth keypoints
        time1 = time.time()
        kp_preds_all = self.__smooth_coordinate(kp_preds_all)
        # print(time.time() - time1)

        time1 = time.time()
        hand_bboxs = self.__get_hand_bboxs(kp_preds_all, kp_scores_all, target_kps, is_static_BG)
        # print(time.time() - time1)
        ## if clip is already labeled, only one person is labeled in each clip
        if is_labeled:
            hand_bboxs = hand_bboxs[:1]

        time1 = time.time()
        img_out_all = []
        heatmap_out_all = []
        for h, hand_bbox in enumerate(hand_bboxs):

            img_out_h = []
            heatmap_out_h = []
            for i, im_name in enumerate(im_name_all):
                if imglist is None:
                    img = cv2.imread(os.path.join(in_clip_folder, im_name))
                else:
                    img = imglist[i].copy()

                # create heatmap
                if is_heatmap:
                    kp_preds_h = kp_preds_all[i][h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                    heatmap = self.__create_heatmap(kp_preds_h, img.shape, target_kps)                    

                    if is_static_BG:
                        x1, y1, x2, y2 = hand_bbox
                        heatmap_out = self.__crop_image(heatmap, x1, y1, x2, y2)
                    else:
                        x1, y1, x2, y2 = hand_bbox[i]
                        heatmap_out = self.__crop_moving_image(heatmap, x1, y1, x2, y2)

                    if heatmap_out is not None:
                        heatmap_out_h.append(heatmap_out)

                        if is_vis:
                            cv2.imshow('skeletons', heatmap_out)
                            cv2.waitKey(waitTime)

                        if is_save:
                            out_folder = out_clip_folder + '_' + str(h+1)
                            if not os.path.exists(out_folder):
                                os.makedirs(out_folder)
                            heatmap_name = im_name.split('.')[0] + '_heatmap.jpg'
                            cv2.imwrite(os.path.join(out_folder, heatmap_name), heatmap_out)

                if is_static_BG:
                    x1, y1, x2, y2 = hand_bbox
                    img_out = self.__crop_image(img, x1, y1, x2, y2)
                else:
                    x1, y1, x2, y2 = hand_bbox[i]
                    img_out = self.__crop_moving_image(img, x1, y1, x2, y2)

                if img_out is not None:
                    img_out_h.append(img_out)

                    if is_vis:
                        cv2.imshow('images', img_out)
                        cv2.waitKey(waitTime)

                    if is_save:
                        out_folder = out_clip_folder + '_' + str(h+1)
                        if not os.path.exists(out_folder):
                            os.makedirs(out_folder)
                        img_name = im_name.split('.')[0] + '.jpg'
                        cv2.imwrite(os.path.join(out_folder, img_name), img_out)

            img_out_all.append(img_out_h)
            heatmap_out_all.append(heatmap_out_h)
        # print(time.time() - time1)

        return img_out_all, heatmap_out_all


def create_clip():

    is_static_BG = True
    if is_static_BG:
        ske_folder = 'hand_static_BG'
    else:
        ske_folder = 'hand_nonStatic_BG'

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']
    bad_others_list = ['_'.join(line.split('_')[1:5]) for line in open(os.path.join(base_folder, 
        'bad_others', 'bad_others.txt'), 'r')]
    # print(bad_others_list)

    # base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
    # __action__ = ['normal', 'clean', 'pick', 'scratch']

    # st = skeleton_tools()

    for act in __action__:

        is_labeled = True
        if act == 'others':
            is_labeled = False

        if act != 'others':
            continue

        base_in_clip_folder = base_folder + act + '/skeletons/'
        base_skeleton_folder = base_folder + act + '/skeletons/'
        base_out_clip_folder = base_folder + ske_folder + '/' + act + '/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):

            # print(sub)
            if sub in bad_others_list:
                print('bad_others_list', sub)
                # print(os.path.join(base_in_clip_folder, sub), os.path.join(base_folder, 'bad_others', sub))
                os.rename(os.path.join(base_in_clip_folder, sub), os.path.join(base_folder, 'bad_others', sub))
                continue

            # # if act != 'pick' or sub[:8] != 'Video_12':
            # #     continue

            # # if act == 'others':# or sub != 'Video_11_1_1':
            # #     continue

            # if act == 'others' and sub_id % 4 != 0:
            #     continue

            # print(act, sub)

            # in_clip_folder = base_in_clip_folder + sub
            # skeleton_folder = base_skeleton_folder + sub
            # out_clip_folder = base_out_clip_folder + act + '_' + sub

            # im_name_all, kp_preds_all, kp_scores_all = st.get_valid_skeletons(
            #     skeleton_folder, in_skeleton_list=None, is_savejson=True)
            # # st.vis_skeleton(in_clip_folder, skeleton_folder, 'None.json',
            # #     im_name_all, kp_preds_all, kp_scores_all, imglist=None,
            # #     result_labels=None, is_save=True, is_vis=True, thres=0.3)
            # # st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder, 'None.json',
            # #     im_name_all, kp_preds_all, kp_scores_all, imglist=None,
            # #     is_save=True, is_vis=True, is_static_BG=is_static_BG, is_labeled=is_labeled, 
            # #     is_heatmap=False)
            # st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder, 'None.json',
            #     im_name_all, kp_preds_all, kp_scores_all, imglist=None,
            #     is_save=True, is_vis=False, is_static_BG=is_static_BG, is_labeled=is_labeled, 
            #     is_heatmap=True)



if __name__ == "__main__":
    
    None
    # create_clip()
