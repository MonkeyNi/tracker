import cv2
from utils import createTrackerByName, get_track_bboxes, generate_tracked_info
from utils import iou_batch, associate_detections_to_trackers, nms, nms_2, get_id
from utils import get_roi
from matching import cosine_distance
import numpy as np
from interval import Interval


class Tracker():
    def __init__(self, infos, trackerType, track_id=0, age=5):
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
        
        # compute smoothed box (max length=5)
        self.tracked_pool = [self.bbox]
        
        self.tracked_frame = [self.frame_name, self.frame_name]
        # key frame
        start = get_id(self.frame_name)
        self.key_frame = infos
        self.key_f_score = self.key_frame[1][1]
        self.key_start = start
        # track id
        self.track_id = track_id
        # record wrong prediction number, more than 7 wrong prediction out of 12, the tracker will be removed
        self.live_pool = [0]*12
        self.latest_gt = self.bbox
    
    def get_smoothed_cors(self, extra_box=False):
        """
        Average previous 5 dets result as showed bounding box
        """
        self.tracked_pool = self.tracked_pool[-5:]
        if extra_box:
            self.tracked_pool.append(extra_box)
        avg_bboxes = np.asarray([b for b in self.tracked_pool])
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
        # rw, rh = tw/dw, th/dh
        ro, rn = tw/th, dw/dh
        so, sn = tw*th, (max(bbox[2], t_box[2])-min(bbox[0], t_box[0]))*(max(bbox[3], t_box[3])-min(bbox[1], t_box[1]))
        
        # box shape or size has changed
        if rn > ro*1.5 or rn < ro*0.5 or float(so)/sn <= 0.25:  # 1/2**2
            frame = infos[0][1]
            bbox_wh = [bbox[0], bbox[1], dw, dh]
            self.tracker.init(frame, bbox_wh)
            self.live_pool = [0]*12
            self.tracked_pool = [bbox]
            # print(f' Reinit: {ro, rn, so, sn}')


class Tracker_pool():
    """
    List of trackers. 
    """
    def __init__(self):
        self.trackers = []
        self.relative_age = 7
        self.age = 5

    def add(self, tracker):
        self.trackers.append(tracker)
        
    def update(self, i, state=0):
        if state >= 1:
            self.trackers[i].age = self.age  # reinitialize age
        else:
            self.trackers[i].age -= 1

    def update_all(self, maximum_trackers=5):
        # get key frames
        key_trackers = [t for t in self.trackers if t.age < 0 or sum(t.live_pool) >= self.relative_age]
        key_frames = [t.key_frame + [t.key_start, t.track_id] for t in key_trackers]
        
        self.trackers = [t for t in self.trackers if t.age >= 0 and sum(t.live_pool) < self.relative_age]
        self.trackers = self.trackers[-maximum_trackers:]
        return key_frames

    def empty(self):
        return True if len(self.trackers) == 0 else False
    

class Tracking():
    def __init__(self,
                 image,
                 frame,
                 infos,
                 tracker_pool,
                 tracked_threshold=0.6,
                 gt_threshold=0.2,
                 TrackerType='MEDIANFLOW',
                 track_id=0,
                 init_flag=[]):
        """
        Do tracking for single frame

        Args:
            image (str): frame name
            frame (np.array): current frame
            infos ([[]]): [[x1,y1,x2,y2,cla_id,score,track_id]]
            tracker_pool: Trackers
            tracked_threshold: threshold determine if tracked
            init_flag: determine if the dets can be used for tracker initialization

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
        self.init_flag = init_flag

    def update(self):
        infos, tracker_pool, frame = self.infos, self.tracker_pool, self.frame
        GT_THRESHOLD = self.gt_threshold
        only_tracked = 0  # for test
        tinfo = [t.tracked_pool for t in tracker_pool.trackers]
        
        if not infos[0][:4] == [0]*4:
            self.bboxes = [i[:4] for i in infos]
            self.cats = [i[4] for i in infos]
            self.scores = [i[5] for i in infos]
                        
            ## 1 if track is null, just add frame and bbox to tracker
            if tracker_pool.empty():
                for i, box in enumerate(self.bboxes):
                    score, catid = float(infos[i][-2]), infos[i][-3]
                    if self.init_flag[i]:
                        _track_id = next(self.track_id)
                        tracker_pool.add(Tracker([[self.image, frame], [box, score, catid]],
                                                 self.t_Type,
                                                 track_id=_track_id))
                        infos[i][-1] = _track_id
           
            ## 2 associate dets and trs
            else:
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
        tracker_id = min([i[-1] for i in infos])
        keep = nms([info[:4]+[info[5]] for info in infos])
        delete = nms_2([info[:-1] for info in infos])
        for i in range(len(infos)):
            if i not in keep or i in delete:
                infos[i][:4] = [0]*4
            else:
                infos[i][-1] = tracker_id
        key_frames = tracker_pool.update_all()
        if len(key_frames) != 0:
            key_frames.append(get_id(self.image))

        return infos, tracker_pool, self.tracked_time, key_frames, only_tracked

    def match(self, matched, infos, tracker_pool, tracked_box, box, scores, cats):
        """
        Tracker prediction match dets
        1). show tracker prediction
        2). add tracker id and update age
        3). check if tracker need to be reinitialized
            3.1). box shape changes a lot. w/h
            3.2). box size changes a lot
        """
        for m in matched:
            infos[m[0]][-2] = min((infos[m[0]][-2] + self.gt_threshold + 0.01), 0.99)
            infos[m[0]][-1] = tracker_pool.trackers[m[1]].track_id
            tracker_pool.update(m[1], state=1)
        
            # # just show tracded result
            # infos[m[0]][:4] = tracked_box[m[1]]
            
            # merge first
            extra_box = self.merge_box(infos[m[0]][:4], tracked_box[m[1]])
            # show smoothed result
            infos[m[0]][:4] = tracker_pool.trackers[m[1]].get_smoothed_cors(extra_box=extra_box)

            # update trackers
            if scores[m[0]] >= self.gt_threshold:
                _info = self.tmp_infos(m[0], box, scores, cats)
                tracker_pool.trackers[m[1]].update(_info, tracked_box[m[1]])

    def unmatched_dets(self, unmatched_detections, tracker_pool, box, scores, cats):
        """
        Deal with unmatched detection results
        1). first check dets with previous dets;
        2). if overlap, show dets;
        3). no overlap, check if we need to add new tracker;
        """
        # ground truth tracker box
        tgt_box = [t.bbox for t in tracker_pool.trackers]
        # previous ground truth tracker box
        tracker_box = [t.latest_gt for t in tracker_pool.trackers]
        unmatched_bbox = [box[i] for i in unmatched_detections]
        matched_dets, unmatched_dets, unmatched_trs = associate_detections_to_trackers(
                        unmatched_bbox,
                        tracker_box,
                        iou_threshold=self.tracked_threshold
                    )
        # Show det result
        if matched_dets.shape[0] != 0:
            for m in matched_dets:
                _ind = box.index(unmatched_bbox[m[0]])
                # update track id
                self.infos[_ind][-1] = tracker_pool.trackers[m[1]].track_id
                # update previous ground truth
                tracker_pool.trackers[m[1]].latest_gt = self.infos[_ind][:4]
                
                # merge first
                extra_box = self.merge_box(self.infos[_ind][:4], tracker_box[m[1]])
                # show smoothed result
                self.infos[_ind][:4] = tracker_pool.trackers[m[1]].get_smoothed_cors(extra_box=extra_box)
            
                tracker_pool.trackers[m[1]].live_pool[-1] = 0
                tracker_pool.update(m[1], state=1)
                # there are dets but no trs, reinit
                if scores[m[0]] >= 0.05:
                    
                    ## If there are dets and no trs, it can be blur or small dets. It means that
                    ## detection works better than tracker. We don't need to reinitiazlied tracker
                    # _info = self.tmp_infos(m[0], box, scores, cats)
                    # tracker_pool.trackers[m[1]].update(_info, tracker_box[m[1]], reinit=True)
                    
                    tracker_pool.trackers[m[1]].live_pool.append(-1)
        
        # decide if we need to initialized this unmathced det as a new tracker
        if unmatched_dets.shape[0] != 0:
            for n in unmatched_dets:
                m = box.index(unmatched_bbox[n])
                if self.init_flag[m]:
                    _info = self.tmp_infos(m, box, scores, cats)
                    # check if there are duplicate tracker boxes
                    tmp_box = [_info[1][0]]
                    ms, _, _ = associate_detections_to_trackers(
                        tmp_box,
                        tgt_box,
                        iou_threshold=self.tracked_threshold
                    )
                    if len(ms) == 0:
                        _track_id = next(self.track_id)
                        tracker_pool.add(Tracker(_info, self.t_Type, track_id=_track_id))
                        self.infos[m][-1] = _track_id

    def unmatched_ts(self, unmatched_trackers, tracker_pool, tracked_boxes):
        """
        Deal with unmatched trackers. Generate, filter and show tracker prediction
        """
        _t_pool = Tracker_pool()
        _t_pool.trackers = [tracker_pool.trackers[m] for m in unmatched_trackers]
        _ted_box = [tracked_boxes[m] for m in unmatched_trackers]
        update_infos, _ = self.track_only_update(_t_pool, tracked_boxes=_ted_box)
        for i, m in enumerate(unmatched_trackers):
            tracker_pool.trackers[m] = _t_pool.trackers[i]
        return update_infos

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
        
        if len(tracked_boxes) != 0:
            for i, ele in enumerate(tracked_boxes):
                if ele == [0, 0, 5, 5]:
                    tracker_pool.update(i, state=0)
                    tracker_pool.trackers[i].tracked_pool =tracker_pool.trackers[i].tracked_pool[1:]
                else:
                    tracker_pool.update(i, state=1)
                    # get smoothed boxes
                    tracked_boxes[i] = tracker_pool.trackers[i].get_smoothed_cors(extra_box=ele)
                
            infos = generate_tracked_info(tracked_boxes, tracker_pool, self.gt_threshold)
            for i, info in enumerate(infos):
                infos[i][-1] = tracker_pool.trackers[i].track_id

            # count tracked_only
            only_tracked = 1
        return infos, only_tracked
    
    def tmp_infos(self, ind, bboxes, scores, cats):
        """
        Help function
        """
        return [[self.image, self.frame], [bboxes[ind], scores[ind], cats[ind]]]
    
    def merge_box(self, box1, box2):
        if box1 == [0,0,5,5]:
            return box2
        if box2 == [0,0,5,5]:
            return box1
        return [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]