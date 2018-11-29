
import numpy as np
import copy
import os
import cv2
import random
import json



class skeleton_tools:

    def __init__(self):
        self.skeleton_size = 17

    def __plot_skeleton(self, img, kp_preds, kp_scores, target_kps, result_labels, thres):

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
                        cv2.line(img, start_xy, end_xy, line_color[i], 
                            int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))
            return img

        else:

            for h in range(len(kp_scores) // self.skeleton_size): # number of human
                kp_preds_h = kp_preds[h*2*self.skeleton_size : (h+1)*2*self.skeleton_size]
                kp_scores_h = kp_scores[h*self.skeleton_size : (h+1)*self.skeleton_size]

                kp_preds_h += [(int(kp_preds_h[10])+int(kp_preds_h[12]))/2, (int(kp_preds_h[11])+int(kp_preds_h[13]))/2]
                kp_scores_h += [(float(kp_scores_h[5]) + float(kp_scores_h[6]))/2]

                cor_x, cor_y = int(kp_preds_h[-2]), int(kp_preds_h[-1])

                cls_map = ['others', 'pick', 'scratch']
                result_cls_id = int(result_labels[h][0])
                result_prob = result_labels[h][1][2] #[result_cls_id]
                if cls_map[result_cls_id] == 'scratch' and result_prob > thres: 
                    cv2.putText(img, '{}:{:.3f}'.format(cls_map[result_cls_id], result_prob), 
                        (int(cor_x), int(cor_y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

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
                            cv2.line(img, start_xy, end_xy, line_color[i], 
                                int(2*(float(kp_scores_h[start_p]) + float(kp_scores_h[end_p])) + 1))
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

    def __validate_skeletons(self, kp_preds_all, kp_scores_all, is_labeled=False):

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


    def get_valid_skeletons(self, skeleton_folder, is_labeled=False):

        in_skeleton_file = os.path.join(skeleton_folder, 'skeleton.txt')
        print(in_skeleton_file)
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

        # original skeleton
        results = {}
        results['im_name_all'] = im_name_all
        results['kp_preds_all'] = kp_preds_all
        results['kp_scores_all'] = kp_scores_all
        with open(
                os.path.join(skeleton_folder, 'all_skeleton.json'),
                'w') as f:
            json.dump(results, f)

        # validate skeleton
        kp_preds_all, kp_scores_all = self.__validate_skeletons(kp_preds_all, kp_scores_all, is_labeled)

        results = {}
        results['im_name_all'] = im_name_all
        results['kp_preds_all'] = kp_preds_all
        results['kp_scores_all'] = kp_scores_all
        with open(
                os.path.join(skeleton_folder, 'valid_skeleton.json'),
                'w') as f:
            json.dump(results, f)

    def __load_valid_skeleton_json(self, skeleton_folder, json_file_name='valid_skeleton.json'):
        f = open(os.path.join(skeleton_folder, json_file_name))
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
            box_w, box_h = 0, 0
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
                    box_w = max(box_w, x2-x1)
                    box_h = max(box_h, y2-y1)
            # hand_cors.append([x1, y1, x2, y2])

            hand_cors_h = []
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
                hand_cors_h.append([box_x1, box_y1, box_x2, box_y2])
            hand_cors.append(hand_cors_h)

        return hand_cors

    def vis_skeleton(self, in_clip_folder, skeleton_folder, json_file_name='valid_skeleton.json', 
                result_labels=None, is_labeled=False, is_save=False, thres=0.75):

        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder, json_file_name)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        for i, im_name in enumerate(im_name_all):
            img = cv2.imread(os.path.join(in_clip_folder, im_name))
            img_out = self.__plot_skeleton(img, kp_preds_all[i], kp_scores_all[i], target_kps, result_labels, thres)
            cv2.imshow('skeletons', img_out)
            cv2.waitKey(15)

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

    def get_hand_clip(self, in_clip_folder, skeleton_folder, out_clip_folder, is_labeled=False):

        target_kps = [5, 6, 7, 8, 9, 10]

        im_name_all, kp_preds_all, kp_scores_all = self.__load_valid_skeleton_json(skeleton_folder)

        if kp_preds_all is None:
            print('**** No valid skeletons ****')
            print(skeleton_folder)
            return

        hand_cors = self.__get_hand_cors(kp_preds_all, kp_scores_all, target_kps)

        for human_id, hand_cor in enumerate(hand_cors):
            # x1, y1, x2, y2 = hand_cor

            ## smooth
            hand_cor_tmp = np.array(hand_cor).transpose()
            for i in range(hand_cor_tmp.shape[0]):
                X = hand_cor_tmp[i].copy()
                smoothed = self.__multi_moving_average(X, window_size=5, times=3)
                hand_cor_tmp[i] = smoothed
            hand_cor = np.array(hand_cor_tmp).transpose()

            out_folder = out_clip_folder + '_' + str(human_id+1)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            for i, im_name in enumerate(im_name_all):
                x1, y1, x2, y2 = hand_cor[i]
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

        label_map = {'others':'1', 'pick':'2', 'scratch':'3'}

        Trainlist = []
        Testlist = []
        count = 0
        for sub in sorted(os.listdir(clip_folder)):
            sub_folder = clip_folder + sub
            for subsub in sorted(os.listdir(sub_folder)):
                contents = sub + '/' + subsub + ' ' + label_map[sub] + '\n'
                if count % sample_rate == 0: # test
                    Testlist.append(contents)
                else: # train
                    Trainlist.append(contents)

                count += 1

        random.shuffle(Trainlist)
        random.shuffle(Testlist)

        Trainlist_name = TrainTest_folder + 'trainlist01.txt'
        Trainlist_file = open(Trainlist_name, 'w')
        for contents in Trainlist:
            Trainlist_file.write(contents)
        Trainlist_file.close()

        Testlist_name = TrainTest_folder + 'testlist01.txt'
        Testlist_file = open(Testlist_name, 'w')
        for contents in Testlist:
            Testlist_file.write(contents)
        Testlist_file.close()

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
    __action__ = ['others', 'pick', 'scratch']

    # base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
    # __action__ = ['normal', 'clean', 'pick', 'scratch']


    st = skeleton_tools()

    sample_rate = 4
    for act in __action__:

        if act == 'others':
            is_labeled = False
        else:
            is_labeled = True

        base_in_clip_folder = base_folder + act + '/clips/'
        base_skeleton_folder = base_folder + act + '/skeletons/'
        base_out_clip_folder = base_folder + 'hand/' + act + '/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):
            # if sub != 'Video_26_1_27':
            #     continue

            # if act == 'others' and sub_id % sample_rate != 0:
            #     continue

            in_clip_folder = base_in_clip_folder + sub
            skeleton_folder = base_skeleton_folder + sub
            out_clip_folder = base_out_clip_folder + act + '_' + sub

            st.get_valid_skeletons(skeleton_folder, is_labeled)
            st.vis_skeleton(
                in_clip_folder, skeleton_folder, json_file_name='valid_skeleton.json',
                result_labels=None, is_labeled=is_labeled, is_save=True, thres=0.3)
            st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder, is_labeled)


    # clip_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand/'
    # TrainTest_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/TrainTestlist/'
    # st.create_TrainTestlist(clip_folder, TrainTest_folder, sample_rate=sample_rate)