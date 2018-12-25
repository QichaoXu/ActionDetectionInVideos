
from Detection import Detection
from DetectionMultiThreads import DetectionMultiThreads

import os
import sys
import cv2
import time

from multiprocessing import Queue

def demo_video(is_save_avi=False):
    T = 45

    reg_model_file = 'results-scratch-18-static_BG/save_200.pth'
    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose'
    det = Detection(reg_model_file, skeleton_opt, cuda_id_list=[1,0],
        sample_rate=15, is_vis=True, waitTime=5, is_static_BG=True, thres=0.7)

    # video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_new/'
    # video_name = 'Video44'
    # video_exp = '.mp4'
    video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    video_name = '1_7'
    video_exp = '.avi'

    input_video_name = video_path + video_name + video_exp
    print(input_video_name)
    if not os.path.exists(input_video_name):
        print('Error! No such video exists')
    stream = cv2.VideoCapture(input_video_name)

    if is_save_avi:
        out_video_name = video_path + video_name + '_'+ skeleton_opt + '_' + str(is_static_BG) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width, height))

    time1 = time.time()

    ret = True
    frame_id = 0
    img_out_all = None
    imglist = []
    while ret:
        (ret, frame) = stream.read()
        frame_id += 1

        # if ret:
        #     cv2.imshow('frame', frame)
        #     cv2.waitKey(5)

        if frame_id <= T:
            imglist.append(frame)
            if frame_id == T:
                frame_id = 0
                img_out_all = det.run(imglist)
                imglist = []
            else:
                if not is_save_avi or img_out_all is None:
                    continue
                else:
                    out.write(img_out_all[frame_id-1])

    det.print_runtime()

    if is_save_avi:
        out.release()
    cv2.destroyAllWindows()

    print('total time', time.time() - time1)


def demo_video_multiThreads(is_save_avi=False):

    T = 45

    reg_model_file = 'results-scratch-18-static_BG/save_200.pth'
    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose'
    queue_imglist = Queue()
    det = DetectionMultiThreads(queue_imglist, reg_model_file, skeleton_opt, cuda_id_list=[0,1],
        sample_rate=15, is_vis=True, waitTime=5, is_static_BG=True, thres=0.7)
    det.start()

    # video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_new/'
    # video_name = 'Video44'
    # video_exp = '.mp4'
    video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    video_name = '1_7'
    video_exp = '.avi'

    input_video_name = video_path + video_name + video_exp
    print(input_video_name)
    if not os.path.exists(input_video_name):
        print('Error! No such video exists')
    stream = cv2.VideoCapture(input_video_name)

    if is_save_avi:
        out_video_name = video_path + video_name + '_'+ skeleton_opt + '_' + str(is_static_BG) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width, height))

    time1 = time.time()

    ret = True
    frame_id = 0
    img_out_all = None
    imglist = []
    while ret:
        (ret, frame) = stream.read()
        frame_id += 1

        # if ret:
        #     cv2.imshow('frame', frame)
        #     cv2.waitKey(5)

        if frame_id <= T:
            imglist.append(frame)
            if frame_id == T:
                frame_id = 0
                # img_out_all = det.run(imglist)
                queue_imglist.put(imglist)
                imglist = []
            else:
                if not is_save_avi or img_out_all is None:
                    continue
                else:
                    out.write(img_out_all[frame_id-1])

    if is_save_avi:
        out.release()
    cv2.destroyAllWindows()

    queue_imglist.put('quit')
    det.join()

    print('total time', time.time() - time1)


if __name__ == "__main__":

    # demo_video()
    demo_video_multiThreads()

