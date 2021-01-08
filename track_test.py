import cv2
from utils import createTrackerByName, get_track_bboxes, generate_tracked_info
from utils import iou_batch, associate_detections_to_trackers, get_id
from utils import get_roi, get_infos
from matching import cosine_distance
import numpy as np
from tracker import Tracker, Tracker_pool
from visualization import read_images, get_FPS, draw_box
import os
from os.path import join
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help='vidoe used to test', type=str, default='')
    parser.add_argument("-d", "--detection_result", help='path to detection result', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    test_age = 5
    test_skip = 0
    args = parse_args()
    # create test tracker
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    TrackerType = trackerTypes[4]

    video = args.video
    basic_images = read_images(video)

    infer_path = args.detection_result
    infer_path, infer_txt = os.path.split(infer_path)
    out_folder = join(infer_path, f'2_test_{TrackerType}_{test_age}')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_keys = list(basic_images.keys())
    images = get_infos(images_keys, join(infer_path, infer_txt))
    for i, image in enumerate(images):
        frame = basic_images[image]
        h, w, c = frame.shape
        infos = images[image]
        if infos != [[0]*4 + [1, 0.1, 0]]:
            out_img = np.zeros([h, w*(test_age+1), c])
            test_t_pool = Tracker_pool()
            tmp_frame = frame.copy()
            for info in infos:
                if info[-2] >= 0.3:
                    box = info[:4]
                    tmp_tracker = Tracker([[image, tmp_frame], [box,info[4],info[5]]],
                                                TrackerType)
                    
                    # import pdb; pdb.set_trace()
                    test_t_pool.add(tmp_tracker)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(tmp_frame,
                                (int(float(x1)),int(float(y1))),
                                (int(float(x2)), int(float(y2))),
                                (255,0,255),
                                2)
            if len(test_t_pool.trackers) != 0:
                out_img[:,:w,:] = tmp_frame

                # track the follow test_age frames
                test_image = images_keys[i+1: i+1+test_age]
                for j, t_imgage in enumerate(test_image, start=1):
                    t_frame = basic_images[t_imgage]
                    tmp_t_frame = t_frame.copy()
                    tracked_boxes, _ = get_track_bboxes(tmp_t_frame, test_t_pool)
                    t_bbox = [t.bbox for t in test_t_pool.trackers]
                    # print(f'{image} tracker is: {t_bbox}')
                    # print(f'   Tracked box is {tracked_boxes}')
                    for t_bbox in tracked_boxes:
                        x1, y1, x2, y2 = t_bbox
                        cv2.rectangle(tmp_t_frame,
                                    (int(float(x1)), int(float(y1))),
                                    (int(float(x2)), int(float(y2))),
                                    (255,255,255),
                                    2)
                    out_img[:,j*w:(j+1)*w, :] = tmp_t_frame
                cv2.imwrite(join(out_folder, image), out_img)
