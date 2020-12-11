import cv2
import os
import collections
import numpy as np
import time


def createTrackerByName(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are: {}'.format(trackerTypes))
    return tracker


def cor_change(box):
    """
    From x,y,w, h to x,y,x,y
    """
    box = [int(x) for x in list(box)]
    x1, y1, w, h = box
    return [x1, y1, w, h]


def get_infos(images, infer_txt):
    """
    Initialize detection results from detection txt file
    For each line, it should have format:
        "x1, y1, x2, y2, cla_id, scores\n"

    Args:
        images: list of image names
        infer_txt: path to detection result txt files

    Returns:
        dir: {image_name: [[x1,y1,x2,y2,cla_id,score,detected]]} 
            detected:
                0: detected
                1: detected and tracked
                2: not detected but tracked
    """
    res = collections.defaultdict(list)
    
    with open(infer_txt) as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue
            infos = line.split(',')
            f_name = infos[0] if infos[0].endswith('png') else f'{infos[0]}.png'
            info = infos[1:]
            score = float(info[5])
            info = [int(float(x)) for x in info]
            info[5] = score
            if len(info) != 7:
                info = info + [0]  # detect status
            res[f_name].append(info)
        # add FN
        base = [0]*4 + [1, 0.1, 0]
        for im in images:
            if len(res[im]) == 0:
                res[im] = [base]
    return {im: res[im] for im in sorted(res.keys())}


def get_track_bboxes(frame, tracker_pool):
    """
    Get track prediction for a frame

    Args:
        frame (np.array): the frame
        tracker_pool: pool of Trackers

    Returns:
        tracked_boxes: [[tracked_box]]
        tracked_time: only useful during testing
    """
    tracked_boxes, tracked_time = [], []
    for tracker in tracker_pool.trackers:
        track = tracker.tracker
        # import pdb; pdb.set_trace()
        start = time.time()
        success, track_boxes = track.update(frame)
        end = time.time()
        tracked_time.append((end-start))
        if (np.array(track_boxes) >= 0).min():
            tracked_boxes.extend(track_boxes)
    tracked_boxes = [cor_change(list(box)) for box in tracked_boxes]
    return tracked_boxes, tracked_time


def linear_assignment(cost_matrix):
    """
    Data association
    TODO: Hungarian algorithm is used now. It does not have weight. 
          KM or feature related algorithm maybe used in the future
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(det_bboxes, track_bboxes):
    """
    Compute overlap between detection results and tracked results

    Args:
        det_bboxes ([[]])
        track_bboxes ([[]]])
    """
    det_bboxes = np.expand_dims(det_bboxes, axis=1)
    track_bboxes = np.expand_dims(track_bboxes, axis=0)

    x1 = np.maximum(det_bboxes[..., 0], track_bboxes[..., 0])
    y1 = np.maximum(det_bboxes[..., 1], track_bboxes[..., 1])
    x2 = np.minimum(det_bboxes[..., 2], track_bboxes[..., 2])
    y2 = np.minimum(det_bboxes[..., 3], track_bboxes[..., 3])
    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    wh = w * h
    overlap = wh / ((det_bboxes[..., 2] - det_bboxes[..., 0]) * (det_bboxes[..., 3] - det_bboxes[..., 1])
        + (track_bboxes[..., 2] - track_bboxes[..., 0]) * (track_bboxes[..., 3] - track_bboxes[..., 1]) - wh)
    return overlap


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0):
        return np.empty((0,2), dtype=int), np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # one for one
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))
    
    unmatched_detections = [i for i, _ in enumerate(detections) if not i in matched_indices[:, 0]]
    unmatched_trackers = [i for i, _ in enumerate(trackers) if not i in matched_indices[:, 1]]

    # match_2 used to update lesion tracker with the lasted detection result
    matches, matches_2 = [],[]
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            if iou_matrix[m[0], m[1]] >= min(iou_threshold*2, 0.5):
                matches_2.append(m.reshape(1, 2))
    matches = np.empty((0,2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    matches_2 = np.empty((0,2), dtype=int) if len(matches_2) == 0 else np.concatenate(matches_2, axis=0)
    return matches, matches_2, np.array(unmatched_detections), np.array(unmatched_trackers)


def generate_tracked_info(t_bboxes, GT_THRESHOLD):
    """
    Return tracked info which will be saved in output txt file
    """
    tracked_info = []
    for tb in t_bboxes:
        info = tb + [1, min((GT_THRESHOLD + 0.1), 0.99), 2]
        tracked_info.append(info)
    return tracked_info