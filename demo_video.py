
from Detection import Detection

import os
import sys
import cv2
import time


def demo_video(is_save_avi=False, is_static_BG=True):

    T = 45
    reg_model_file = 'results-scratch-18-static_BG-45/save_200.pth'
    skeleton_opt = 'Alphapose' #'Alphapose' #'Openpose'

    video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_new/'
    video_name = 'Video44'
    video_exp = '.mp4'
    # video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    # video_name = '1_7'
    # video_exp = '.avi'

    # input_video_name = 0
    input_video_name = video_path + video_name + video_exp
    if input_video_name == 0:
        print('\nreading video from camera ...\n')
    else:
        print('\nreading video from file ' + input_video_name + '\n')
        if not os.path.exists(input_video_name):
            print('Error! No such video exists')
            return
    stream = cv2.VideoCapture(input_video_name)

    if is_save_avi:
        out_video_name = video_path + video_name + '_'+ skeleton_opt + '_' + str(is_static_BG) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_video_name, fourcc, 25.0, (width, height))

    det = Detection(reg_model_file, skeleton_opt, cuda_id_list=[1,0],
        sample_duration=T, sample_rate=15, is_vis=True, waitTime=5, is_static_BG=True, thres=0.7)

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


if __name__ == "__main__":

    demo_video(is_save_avi=False, is_static_BG=True)

