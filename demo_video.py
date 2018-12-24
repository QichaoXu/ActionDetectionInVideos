
from detection import detection

import os
import sys
import cv2


def demo_camera(is_save_avi=False):
    T = 45
    reg_model_file = 'results-scratch-18-static_BG/save_200.pth'
    # reg_model_file = 'results-scratch-18/save_200.pth'

    skeleton_opt = 'Openpose' #'Alphapose' #'Openpose'
    is_static_BG = True
    video_name = 'Video44'
    video_exp = '.mp4'
    det = detection(reg_model_file, skeleton_opt=skeleton_opt, is_vis=True, waitTime=15, is_static_BG=is_static_BG, thres=0.7)

    # video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/'
    video_path = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/video_new/'
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

    if is_save_avi:
        out.release()

    cv2.destroyAllWindows()
    det.print_runtime()


if __name__ == "__main__":

    demo_camera()