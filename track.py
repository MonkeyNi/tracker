import cv2
import os
from os.path import join
import argparse
import numpy as np
import time
from utils import get_infos
from visualization import read_images, get_FPS, draw_box
from tracker import Tracker, Tracker_pool, Tracking


def create_video(imgs, video_out='ori_tracked.avi'):
    """
    Create detection video and tracked video for compariation.
    Left is basic detection result and right is tracked result
    """
    print(f'There are {len(imgs)} frames in this video.')
    assert (len(imgs) != 0)
    h, w, c = list(imgs.values())[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = FPS
    out = cv2.VideoWriter(join(out_folder, video_out), fourcc, fps, (2*w, h))  
    # for testing
    out_images = join(out_folder, video_out)
    out_images = out_images.replace('.avi', '')
    if not os.path.exists(out_images):
        os.makedirs(out_images)
    # detection results
    detect_infos = get_infos(list(imgs.keys()), join(infer_path, infer_txt))
    # tracked results
    track_infos = get_infos(list(imgs.keys()), join(out_folder, track_out))

    p_count = [0, 0]
    for im in imgs:
        ims = [imgs[im], imgs[im].copy()]
        d_info, t_info = detect_infos[im], track_infos[im]
        blank = np.zeros([h, 2*w, c], np.uint8)
        for i, infos in enumerate([d_info, t_info]):
            if set(infos[0][:4]) != {0}:
                img, flag = draw_box(ims[i], infos, GT_THRESHOLD, basic_box=basic_box)
                if flag: p_count[i] += 1
                blank[:, i*w:(i+1)*w, :] = img
            else:
                blank[:, i*w:(i+1)*w, :] = imgs[im]
        out.write(blank)
        cv2.imwrite(join(out_images, im), blank)
    print(f'After second threshold: {GT_THRESHOLD}, detected lesion increase from {p_count[0]} to {p_count[1]}.')
    out.release()


def Track():
    # save tracked output
    tract_out = open(join(out_folder, track_out), 'w')
    tracker_pool = Tracker_pool()
    track_time, end_to_end_time = [], []

    images = get_infos(list(basic_images.keys()), join(infer_path, infer_txt))
    for image in images:
        # print(f'There are {len(tracker_pool.trackers)} trackers.')
        frame = basic_images[image]
        infos = images[image]
        start = time.time()
        tracking = Tracking(image, frame, infos, tracker_pool, gt_threshold=GT_THRESHOLD, TrackerType=TrackerType)
        tracked_infos, tracker_pool, tracked_time = tracking.update()
        gap = time.time() - start
        end_to_end_time.append(gap)
        track_time.extend(tracked_time)
        # record tracked result to txt file
        for info in tracked_infos:
            tract_out.write(image + ',')
            for ele in info[:-1]:
                tract_out.write(str(ele) + ',')
            tract_out.write(str(info[-1]) + '\n')
    tract_out.close()
    print('Average tracker speed is {}'.format(sum(track_time)/len(track_time)))
    print('Average end to end time is {}. Max is {}.'.format(sum(end_to_end_time)/len(end_to_end_time), max(end_to_end_time)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help='vidoe used to test', type=str, default='')
    parser.add_argument("-d", "--detection_result", help='path to detection result', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # video is used to get FPS and basic image information
    video = args.video
    basic_images = read_images(video)  # image_name: img

    # detection result and where tracked result will be saved
    infer_path = args.detection_result
    infer_path, infer_txt = os.path.split(infer_path)


    basic_box = join(os.getcwd(), 'bbox.png')
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    TrackerType = trackerTypes[4]
    print(f'Current tracking algorithm is {TrackerType}.')
    EXTENSINO, GT_THRESHOLD, V = 'png', 0.3, '0.3'

    out_folder = join(infer_path, f'v{V}_{TrackerType}_{GT_THRESHOLD}')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # track_out = f'v{V}_{TrackerType}_{GT_THRESHOLD}_track_{infer_txt}'
    track_out = f'track_{infer_txt}'
    Track()

    FPS = get_FPS(video)  # get the FPS
    # video_out = (f'v{V}_{TrackerType}_{GT_THRESHOLD}_{infer_txt}').replace('txt', 'avi')
    video_out = (f'{infer_txt}').replace('txt', 'avi')
    create_video(basic_images, video_out=video_out)