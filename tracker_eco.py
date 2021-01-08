import cv2
from utils import createTrackerByName, get_track_bboxes, generate_tracked_info
from utils import iou_batch, associate_detections_to_trackers, nms, nms_2, get_id
from utils import get_roi
from matching import cosine_distance
import numpy as np
from interval import Interval


class Tracker():
    def __init__(self, infos, trackerType, track_id=0, age=10):
        """
        Initialize a tracker with frame and detection information.
        Args:
                infos ([list]): [frame ([image_name, np.array]), [boxes, scores, cats]].
                trackerType: algorithm used to track
                age (int, optional): to record the tracker status. Defaults to 2.
        """
        self.frame_name = infos[0][0]
        self.frame = infos[0][1]
        self.bbox = infos[1][0]
        self.score = infos[1][1]
        self.cat = infos[1][2]
        self.trackerType = trackerType
        self.age = age
        # init single tracker
        self.tracker = createTrackerByName(self.trackerType)
        self.w, self.h = self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]
        self.bbox_wh = [self.bbox[0], self.bbox[1], self.w, self.h]
        self.tracker.init(self.frame, self.bbox_wh)
        self.tracked_pool = [infos[1][:-1]]
        self.tracked_frame = [self.frame_name, self.frame_name]
        # key frame
        start = get_id(self.frame_name)
        self.key_frame = infos
        self.key_f_score = self.key_frame[1][1]
        self.key_start = start
        # track id
        self.track_id = track_id
        # tracker live, if sum(live_poo) >= 3, remove the tracker
        self.live_pool = [0]*5
        self.latest_gt = self.bbox
    
    def get_smoothed_cors(self, n, extra_box=False):
        """
        Average previous 5 dets result as showed bounding box
        """
        tracked_pool = self.tracked_pool
        if extra_box:
            tracked_pool.append(extra_box)
        avg_bboxes = np.asarray([b for b, _ in tracked_pool[-n:]])
        return list(np.average(avg_bboxes, axis=0).astype(np.int64))
    
    def update(self, infos, t_box, reinit=False):
        """
        Just update key frame if needed
        Args:
            infos: infos ([list]): [frame ([image_name, np.array]), [boxes, scores, cats]];
            t_box: if box shape change, tracker needs to be reinitialized;
        """
        bbox = infos[1][0]
        # update latest previous gt
        self.latest_gt = bbox
        # update key frame
        if self.key_f_score + 0.1 < infos[1][1]:
            self.key_frame = infos
            self.key_f_score = infos[1][1]
        # if box shape has changed, reinitialized tracker
        tw, th = t_box[2] - t_box[0], t_box[3] - t_box[1]
        dw, dh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rw, rh = tw/dw, th/dh
        
        if reinit:
            frame = infos[0][1]
            bbox_wh = [bbox[0], bbox[1], dw, dh]
            self.tracker.init(frame, bbox_wh)
            # print(f'Reinitiate {infos[0][0]}')
        
        shape_threshold = Interval(0.5, 1.5)
        if not rw in shape_threshold or not rh in shape_threshold:
            if not reinit:
                frame = infos[0][1]
                bbox_wh = [bbox[0], bbox[1], dw, dh]
                self.tracker.init(frame, bbox_wh)
                # print(f'Reinitiate {infos[0][0]}')


class Tracker_pool():
    """
    List of trackers. 
    """
    def __init__(self):
        self.trackers = []

    def add(self, tracker):
        self.trackers.append(tracker)

    def delete(self, i):
        self.trackers.pop(i)

    def update(self, i, state=0):
        if state >= 1:
            self.trackers[i].age = 10  # reinitialize age
        else:
            self.trackers[i].age -= 1

    def update_all(self, maximum_trackers=5):
        # get key frames
        key_trackers = [t for t in self.trackers if t.age < 0 or sum(t.live_pool) >= 3]
        # key_trackers = [t for t in self.trackers if t.age < 0]
        key_frames = [t.key_frame + [t.key_start, t.track_id] for t in key_trackers]
        
        self.trackers = [t for t in self.trackers if t.age >= 0 and sum(t.live_pool) < 3]
        # self.trackers = [t for t in self.trackers if t.age >= 0]
        self.trackers = self.trackers[-maximum_trackers:]
        return key_frames

    def update_miss(self):
        if self.empty():
            return
        for i in range(len(self.trackers)):
            self.trackers[i].age -= 1

    def empty(self):
        return True if len(self.trackers) == 0 else False
    
    def update_match(self, i, infos, max_length=5):
        """
        Args:
            infos ([[]]]): [[box], score, catid]
        """
        frame, info = infos
        info = info[:-1]
        if info not in self.trackers[i].tracked_pool:
            self.trackers[i].tracked_pool.append(info)
            self.trackers[i].tracked_frame[-1] = frame[0]
            self.trackers[i].tracked_pool = self.trackers[i].tracked_pool[-max_length:]


class Tracking():
    def __init__(self,
                 image,
                 frame,
                 infos,
                 tracker_pool,
                 tracked_threshold=0.25,
                 gt_threshold=0.3,
                 TrackerType='MEDIANFLOW',
                 track_id=0):
        """
        Do tracking for single frame

        Args:
            image (str): frame name
            frame (np.array): current frame
            infos ([[]]): [[x1,y1,x2,y2,cla_id,score,track_id]]
            tracker_pool: Trackers
            tracked_threshold: threshold determine if tracked

        Returns:
            tracked_info: [[x1,y1,x2,y2,cla_id,score,track_id]]
            tracer_pool: updated tracker pool
            tracked_time: []
        """
        self.image = image
        self.frame = frame
        self.infos = infos
        self.tracker_pool = tracker_pool
        self.tracked_threshold = tracked_threshold
        self.tracked_time = []
        self.gt_threshold = gt_threshold
        self.t_Type = TrackerType
        self.bboxes, self.scores, self.cats = [], [], []
        self.track_id = track_id

    def update(self):
        infos, tracker_pool, frame = self.infos, self.tracker_pool, self.frame
        # tracked_info = [[t.bbox, t.age, t.track_id] for t in tracker_pool.trackers]  # for test
        # print(tracked_info)
        GT_THRESHOLD = self.gt_threshold
        only_tracked = 0  # for test
        if not infos[0][:4] == [0]*4:
            self.bboxes = [i[:4] for i in infos]
            self.cats = [i[4] for i in infos]
            self.scores = [i[5] for i in infos]
            ## 1 if track is null, just add frame and bbox to tracker
            if tracker_pool.empty():
                for i, box in enumerate(self.bboxes):
                    score, catid = float(infos[i][-2]), infos[i][-3]
                    if score >= GT_THRESHOLD:
                        _track_id = next(self.track_id)
                        tracker_pool.add(Tracker([[self.image, frame], [box, score, catid]],
                                                 self.t_Type,
                                                 track_id=_track_id))
                        infos[i][-1] = _track_id
            ## 2 compare det results first (dets and box in tracker pool), which can save some track time
            else:
                # get tracked bbox
                tracked_boxes, self.tracked_time = get_track_bboxes(frame, tracker_pool)
                matched_dets, unmatched_dets, unmatched_trs = associate_detections_to_trackers(
                        self.bboxes,
                        tracked_boxes,
                        iou_threshold=self.tracked_threshold
                    )
                # match situation
                if not matched_dets.shape[0] == 0:
                    self.match(matched_dets, infos, tracker_pool, tracked_boxes, self.bboxes, self.scores, self.cats)
                # unmatched detection
                if not unmatched_dets.shape[0] == 0:
                    self.unmatched_dets(unmatched_dets, tracker_pool, self.bboxes, self.scores, self.cats)
                # unmatched tracker
                if not unmatched_trs.shape[0] == 0:
                    update_infos = self.unmatched_ts(unmatched_trs, tracker_pool, tracked_boxes)
                    infos.extend(update_infos)

        ## 3 If there is no detection, update all tracker status and show tracker prediction
        else:
            infos, only_tracked = self.track_only_update(self.tracker_pool, infos=self.infos)

        # NMS
        keep = nms([info[:4]+[info[5]] for info in infos])
        delete = nms_2([info[:-1] for info in infos])
        for i in range(len(infos)):
            if i not in keep or i in delete:
                infos[i][:4] = [0]*4
        key_frames = tracker_pool.update_all()
        if len(key_frames) != 0:
            key_frames.append(get_id(self.image))
        return infos, tracker_pool, self.tracked_time, key_frames, only_tracked

    def track_only_update(self, tracker_pool, infos=[], tracked_boxes=[]):
        """
        This function is used to deal with 'no detection but tracked' situation

        Args:
            infos ([type]): [description]
            tracker_pool ([type]): [description]
            tracked_box: tracker prediction

        Returns: updated infos
        """
        only_tracked = 0  # for test
        if tracked_boxes == []:
            tracked_boxes, self.tracked_time = get_track_bboxes(self.frame, tracker_pool)
        if len(tracked_boxes) == 1 and set(tracked_boxes[0]) == {0}:
            pass
        elif len(tracked_boxes) != 0:
            for i in range(len(tracked_boxes)):
                tracker_pool.update(i, state=1)

            infos = generate_tracked_info(tracked_boxes, tracker_pool, self.gt_threshold)
            for i, info in enumerate(infos):
                infos[i][-1] = tracker_pool.trackers[i].track_id

            # count tracked_only
            only_tracked = 1
        return infos, only_tracked

    def match(self, matched, infos, tracker_pool, tracked_box, box, scores, cats):
        """
        Tracker prediction match dets
        """
        for m in matched:
            infos[m[0]][-2] = min((infos[m[0]][-2] + self.gt_threshold + 0.01), 0.99)
            infos[m[0]][-1] = tracker_pool.trackers[m[1]].track_id
            tracker_pool.update(m[1], state=1)
            
            # just show tracded result
            infos[m[0]][:4] = tracked_box[m[1]]

            # update trackers
            if scores[m[0]] >= self.gt_threshold:
                _info = self.tmp_infos(m[0], box, scores, cats)
                tracker_pool.trackers[m[1]].update(_info, tracked_box[m[1]])

    def unmatched_dets(self, unmatched_detections, tracker_pool, box, scores, cats):
        """
        Deal with unmatched detection results
        Check dets with previous dets; if overlap, no new tracker; else, add new tracker
        """
        tracker_box = [t.latest_gt for t in tracker_pool.trackers]
        unmatched_bbox = [box[i] for i in unmatched_detections]
        matched_dets, unmatched_dets, unmatched_trs = associate_detections_to_trackers(
                        unmatched_bbox,
                        tracker_box,
                        iou_threshold=self.tracked_threshold
                    )
        # just show det result
        if not matched_dets.shape[0] == 0:
            for m in matched_dets:
                self.infos[m[0]][-1] = tracker_pool.trackers[m[1]].track_id
                tracker_pool.trackers[m[1]].live_pool[-1] = 0
                # there are dets but no trs, reinit
                if scores[m[0]] >= self.gt_threshold:
                    _info = self.tmp_infos(m[0], box, scores, cats)
                    tracker_pool.trackers[m[1]].update(_info, tracker_box[m[1]], reinit=True)
        if not unmatched_dets.shape[0] == 0:
            for n in unmatched_dets:
                m = box.index(unmatched_bbox[n])
                if scores[m] >= self.gt_threshold:
                    _info = self.tmp_infos(m, box, scores, cats)
                    _track_id = next(self.track_id)
                    tracker_pool.add(Tracker(_info, self.t_Type, track_id=_track_id))
                    self.infos[m][-1] = _track_id

    def unmatched_ts(self, unmatched_trackers, tracker_pool, tracked_boxes):
        """
        Deal with unmatched trackers
        """
        _t_pool = Tracker_pool()
        _t_pool.trackers = [tracker_pool.trackers[m] for m in unmatched_trackers]
        _ted_box = [tracked_boxes[m] for m in unmatched_trackers]
        update_infos, _ = self.track_only_update(_t_pool, tracked_boxes=_ted_box)
        for i, m in enumerate(unmatched_trackers):
            tracker_pool.trackers[m] = _t_pool.trackers[i]
        return update_infos

    def tmp_infos(self, ind, bboxes, scores, cats):
        """
        Help function
        """
        return [[self.image, self.frame], [bboxes[ind], scores[ind], cats[ind]]]
