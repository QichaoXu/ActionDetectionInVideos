
from multiprocessing import Queue
from DetectionMultiThreads import DetectionMultiThreads

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime

def resize_fixed_ratio(img, target_height, target_width):
    img_out = np.ones((target_height, target_width, 3), np.uint8) * 255
    height, width = img.shape[:2]
    if height/width > target_height/target_width:
        ratio = height / target_height
        new_width = int(width/ratio)
        img = cv2.resize(img, (new_width, target_height))
        width_start = (target_width - new_width) // 2
        img_out[:, width_start:width_start+new_width, :] = img
    else:
        ratio = width / target_width
        new_height = int(height/ratio)
        img = cv2.resize(img, (target_width, new_height))
        height_start = (target_height - new_height) // 2
        img_out[height_start:height_start+new_height, :, :] = img
    return img_out

def demo_camera(is_save_avi=False, is_static_BG=True, is_heatmap=False):

    T = 30
    if is_heatmap:
        reg_model_file = 'results-scratch-18-static_BG-30-skeleton/save_200.pth'
        model_type = 'resnet_skeleton'
        thres = 0.5
    else:
        reg_model_file = 'results-scratch-18-static_BG-30-finetuned/save_20.pth'
        model_type = 'resnet'
        thres = 0.7

    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose' # 'TFOpenpose'

    video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_new/'
    video_name = 'Video44'
    video_exp = '.mp4'
    # video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    # video_name = '1_7'
    # video_exp = '.avi'

    input_video_name = 0
    # input_video_name = video_path + video_name + video_exp
    if input_video_name == 0:
        print('\nreading video from camera ...\n')
        stream = cv2.VideoCapture(input_video_name)
        print(stream.get(cv2.CAP_PROP_FPS))
        if not stream.isOpened():
            print('No camera detected !!!\n')
            return
    else:
        print('\nreading video from file ' + input_video_name + '\n')
        stream = cv2.VideoCapture(input_video_name)
        print(stream.get(cv2.CAP_PROP_FPS))
        if not os.path.exists(input_video_name):
            print('Error! No such video exists')
            return

    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_detected_default = np.ones((height, width//3, 3), np.uint8) * 255

    out = None
    if is_save_avi:
        out_folder = '/media/qcxu/qcxuDisk/ActionDetectionInVideos_out_video'
        out_video_name = os.path.join(out_folder, time.strftime("%Y%m%d%H%M%S", time.localtime())+'.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width+width//3, height))

    queue_imglist = Queue()
    queue_img_out_all = Queue()
    det = DetectionMultiThreads(queue_imglist, queue_img_out_all, reg_model_file, skeleton_opt, 
        cuda_id_list=[0,1], sample_duration=T, model_type=model_type, sample_rate=1, 
        is_static_BG=is_static_BG, is_heatmap=is_heatmap, thres=thres)
    det.start()

    cv2.namedWindow('result')

    time1 = time.time()

    ret = True
    frame_id = 0
    clip_id = 0
    clip_id_out = 0
    img_out_all = []
    img_detected_all = [img_detected_default for i in range(T)]
    img_detected_idx = 0
    bbox_hand_out_all = []
    imglist = []
    while ret:
        try:
            # time1 = time.time()
            (ret, frame) = stream.read()

            frame_out = frame.copy()
            frame_detected = np.ones((height, width+width//3, 3), np.uint8) * 255
            frame_detected[:, :width, :] = frame_out
            frame_detected[:, width:, :] = img_detected_all[img_detected_idx]
            img_detected_idx = (img_detected_idx+1) % T
            # print(frame_out.shape)
            # print(frame_detected.shape)
            cv2.imshow('result', frame_detected)
            cv2.waitKey(1)

            if out is not None:
                out.write(frame_detected)

            # print(time.time() - time1)

            frame_id += 1

            if input_video_name != 0:
                if queue_imglist.qsize() > 2:
                    time.sleep(0.1)
                else:
                    time.sleep(0.04)

            if frame_id <= T:
                imglist.append(frame)

                if queue_img_out_all.qsize() > 0:
                    img_out_all_pre, bbox_hand_out_all, bbox_human_out_all, clip_id_out = queue_img_out_all.get()
                    print('queue_img_out_all get', queue_img_out_all.qsize(), clip_id_out, 
                        datetime.now().strftime("%H:%M:%S.%f"))

                    if len(bbox_hand_out_all) > 0:
                        bbox_hand_out_all = bbox_hand_out_all[:2] # display at most 2 instances
                        img_detected_all = []
                        for img_out in img_out_all_pre:
                            img_detected = img_detected_default.copy()
                            for i, bbox_hand_out in enumerate(bbox_hand_out_all):

                                x1, y1, x2, y2 = bbox_hand_out
                                x1_h, y1_h, x2_h, y2_h = bbox_human_out_all[i]
                                x1_h = max(0, x1_h-10)
                                y1_h = max(0, y1_h-10)
                                x2_h = min(img_out.shape[1], x2_h+10)
                                y2_h = min(img_out.shape[0], y2_h+10)

                                cv2.rectangle(img_out, (x1, y1), (x2, y2), (0,0,255), 2)
                                detected_part = img_out[y1_h:y2_h, x1_h:x2_h, :]
                                detected_part = resize_fixed_ratio(detected_part, height//2-5, width//3)
                                img_detected[height//2*(i):height//2*(i+1)-5, :, :] = detected_part

                            img_detected_all.append(img_detected)

                #     for img_out in img_out_all_pre:
                #         img_out_all.append(img_out)
                #     if len(img_out_all) >= 35:
                #         img_out_all = img_out_all[len(img_out_all)-30:]

                # if len(img_out_all) > 0:
                #     cv2.imshow('result', img_out_all[0])
                #     cv2.waitKey(1)
                #     img_out_all = img_out_all[1:]


                if frame_id == T:
                    frame_id = 0
                    queue_imglist.put([imglist, clip_id])
                    print('\nqueue_imglist put', queue_imglist.qsize(), clip_id, 
                        datetime.now().strftime("%H:%M:%S.%f"))
                    imglist = []
                    clip_id += 1
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    out.release()
    queue_imglist.put(['quit', 'quit'])
    det.join()
    print('total time', time.time() - time1)


if __name__ == "__main__":

    demo_camera(is_save_avi=True, is_static_BG=True, is_heatmap=False)