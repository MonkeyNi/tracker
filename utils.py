import cv2
import os
import collections
import numpy as np
import time
from visualization import draw_box
from eco import ECOTracker


def createTrackerByName(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'ECO']

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
    elif trackerType == trackerTypes[8]:
        tracker = ECOTracker(True)
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are: {}'.format(trackerTypes))
    
    # update tracker parameters
    # params = cv2.FileStorage("params.yaml", cv2.FILE_STORAGE_READ)
    # # params.write('detect_thresh', 0.6)
    # tracker.read(params.root())
    # params.release()
    return tracker


def cor_change(box, h, w):
    """
    Coordinate clean
    """
    if box == [0, 0, 5, 5]:
        return box
    box = [int(x) for x in list(box)]
    x1, y1, width, height = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x1+width, w-1), min(y1+height, h-1)
    if x1 >= x2 or y1 >= y2:
        return [0, 0, 5, 5]
    return [x1, y1, x2, y2]


def get_infos(images, infer_txt):
    """
    Initialize detection results from detection txt file
    For each line, it should have format:
        "x1, y1, x2, y2, cla_id, scores\n"

    Args:
        images: list of image names
        infer_txt: path to detection result txt files

    Returns:
        dir: {image_name: [[x1,y1,x2,y2,cla_id,score,track_id]]} 
            
    """
    res = collections.defaultdict(list)
    
    with open(infer_txt) as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue
            infos = line.split(',')
            f_name = infos[0] if infos[0].endswith('png') else f'{infos[0]}.png'
            info = infos[1:]
            x1, y1, x2, y2 = info[:4]
            x1, y1 = max(0, float(x1)), max(0, float(y1))
            # x2, y2 = min(x2, w-1), min(y2, h-1)
            w, h = float(x2)-x1, float(y2)-y1
            
            # For ECO tracker, w, h should be larger than 5%
            # TODO: replace hard value with percnetage
            if w <= 36 or h <= 25:
                continue
            
            info[:4] = [x1, y1, x2, y2]
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
    h, w, _ = frame.shape
    tracked_boxes, tracked_time = [], []
    for i, tracker in enumerate(tracker_pool.trackers):
        track = tracker.tracker
        start = time.time()

        success, track_boxes = track.update(frame)

        end = time.time()
        tracked_time.append((end-start))
        
        # tracker_pool.trackers[i].live_pool.pop(0)
        if not success:
            track_boxes = [0, 0, 5, 5]
            tracker_pool.trackers[i].live_pool.append(1)
        # else:
            # tracker_pool.trackers[i].live_pool.append(0)
        tracked_boxes.extend([track_boxes])
    tracked_boxes = [cor_change(list(box), h, w) for box in tracked_boxes]
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
    Compute IoU between detection results and tracked results

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
    # compute IoU
    iou = wh / ((det_bboxes[..., 2] - det_bboxes[..., 0]) * (det_bboxes[..., 3] - det_bboxes[..., 1])
        + (track_bboxes[..., 2] - track_bboxes[..., 0]) * (track_bboxes[..., 3] - track_bboxes[..., 1]) - wh)
    # compute overlap
    # overlap_dets = wh / ((det_bboxes[..., 2] - det_bboxes[..., 0]) * (det_bboxes[..., 3] - det_bboxes[..., 1]))
    # overlap_trs = wh / ((track_bboxes[..., 2] - track_bboxes[..., 0]) * (track_bboxes[..., 3] - track_bboxes[..., 1]))
    # overlap = max(overlap_dets, overlap_trs)
    return iou


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0):
        return np.empty((0,2), dtype=int), np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    # iou_matrix, overlap_matrix = iou_batch(detections, trackers)
    # import pdb; pdb.set_trace()
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
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    matches = np.empty((0,2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def generate_tracked_info(t_bboxes, tracker_pool, GT_THRESHOLD):
    """
    Return tracked info which will be saved in output txt file
    """
    tracked_info = []
    for tb, t in zip(t_bboxes, tracker_pool.trackers):
        info = tb + [1, t.score, 2]
        tracked_info.append(info)
    return tracked_info


def nms(dets, thresh=0.3):
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_intersect(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect


def nms_2(infos, thresh=0.5):
    """
    This function is used to deal with small/big overlap problem
    """
    if len(infos) == 1:
        return []
    infos = np.array(infos)
    dets, clas, scores = infos[:,:4], infos[:,4], infos[:,5]
    areas = np.array([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in dets])
    inds = areas.argsort()
    dets = dets[inds]
    delete = []
    for i, box in enumerate(dets[:-1]):
        flag = False
        check = dets[i+1:]
        for j, che in enumerate(check, i+1):
            intersect = compute_intersect(box, che)
            if intersect == 0:
                continue
            if float(intersect)/areas[inds[i]] >= thresh and clas[inds[i]] == clas[inds[j]]:
                flag = True
                break
        if flag:
            delete.append(inds[i])
    return delete


def save_key_frames(key_frames, out_folder, threshold=5, basic_box=False):
    """
    Save key frames

    Args:
        key_frames ([list]): [[[name, frame], [[box], score, cat_id], start, track_id], end]
        threhsold: if there are more than 'threshold' frames (e.g 1 second) have been detected, it it key frame
    """
    if len(key_frames) == 0:
        return
    end = key_frames[-1]
    for key_f in key_frames[:-1]:
        img_name, img = key_f[0]
        img_2 = img.copy()
        start, track_id = key_f[2], key_f[3]
        infos = key_f[1][0] + [key_f[1][2]] + [key_f[1][1]] + [track_id]
        if int(end) - int(start) >= threshold:
            img_2, _ = draw_box(img_2, [infos], 0, basic_box=basic_box)
            img_name = f'{key_f[1][2]}_{start}_{end}_{img_name}'
            cv2.imwrite(os.path.join(out_folder, img_name), img_2)


def get_id(name):
    return name[name.index('_')+1:name.rfind('.')]


def get_roi(box, frame):
    assert (len(box) == 4)
    return frame[box[1]:max(box[3], 5), box[0]:max(box[2], 5), :]
