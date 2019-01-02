
from multiprocessing import Queue
from DetectionMultiThreads import DetectionMultiThreads

import os
import sys
import cv2
import time
from datetime import datetime

def demo_camera(is_save_avi=False, is_static_BG=True, is_heatmap=False):

    T = 30
    if is_heatmap:
        reg_model_file = 'results-scratch-18-static_BG-30-skeleton/save_200.pth'
        thres = 0.5
    else:
        reg_model_file = 'results-scratch-18-static_BG-30/save_200.pth'
        thres = 0.7

    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose'

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

    out = None
    if is_save_avi:
        out_video_name = video_path + video_name + '_'+ skeleton_opt + '_' + str(is_static_BG) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width, height))

    queue_imglist = Queue()
    queue_img_out_all = Queue()
    det = DetectionMultiThreads(queue_imglist, queue_img_out_all, reg_model_file, skeleton_opt, cuda_id_list=[1,0],
        sample_duration=T, sample_rate=1, is_static_BG=is_static_BG, is_heatmap=is_heatmap, thres=thres, out=out)
    det.start()

    cv2.namedWindow('result')

    time1 = time.time()

    ret = True
    frame_id = 0
    clip_id = 0
    clip_id_out = 0
    img_out_all = []
    imglist = []
    while ret:
        (ret, frame) = stream.read()
        frame_id += 1

        if input_video_name != 0:
            if queue_imglist.qsize() > 2:
                time.sleep(0.1)
            else:
                time.sleep(0.04)

        if frame_id <= T:
            imglist.append(frame)
            
            if queue_img_out_all.qsize() > 0:
                img_out_all_pre, clip_id_out = queue_img_out_all.get()
                print('queue_img_out_all get', queue_img_out_all.qsize(), clip_id_out, datetime.now().strftime("%H:%M:%S.%f"))
                for img_out in img_out_all_pre:
                    img_out_all.append(img_out)
                if len(img_out_all) >= 35:
                    img_out_all = img_out_all[len(img_out_all)-30:]

            if len(img_out_all) > 0:
                cv2.imshow('result', img_out_all[0])
                cv2.waitKey(1)
                img_out_all = img_out_all[1:]
                # print(len(img_out_all))

            if frame_id == T:
                frame_id = 0
                queue_imglist.put([imglist, clip_id])
                print('\nput into queue_imglist', queue_imglist.qsize(), clip_id, datetime.now().strftime("%H:%M:%S.%f"))
                imglist = []
                clip_id += 1

    cv2.destroyAllWindows()
    queue_imglist.put(['quit', 'quit'])
    det.join()

    print('total time', time.time() - time1)


if __name__ == "__main__":

    demo_camera(is_save_avi=False, is_static_BG=True, is_heatmap=True)