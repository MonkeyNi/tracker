## Tracker for object detection


### Install
cd tracker/eco/features/
python setup.py build_ext --inplace

### Test example
```
python track.py --video test/test2.mp4 --detection_result test/results2_0.1.txt

Detection result should has format:
    frame_name, x1, y1, x2, y2, categort_id, score
```
There will be an video, tracked result txt fie, and key frames in output. Output will be saved at the same folder as 'detection result'.

### ECO Setting
* Age (default age / maximum wrong prediction. Smaller age will cause duplicate key frames): 5/ less than 7 amony 12 frames
* Score: 0.45
* HOG (dimnesion / cell size): 11/4
* CN (dimnesion / cell size): 2/4
* IC (dimnesion / cell size): 1/3
* CG iterations for initialization: 2
* CG iterations for update: 2
* GN iterations for optimize projection matric: 1
* Minimum input side: 100
* Sample size (min/max): 10/40
* Search scale: 2
* LR: 0.009
* For better use experience, dets and trs will be merged first and then smoothed (with previous 5) results will be showed. In this way, we can get smoother result. 
* Use new 'overlap' for track: overlap = intersection / max(areaA, areaB)

### ECO Result (zsm2)
* Sensitivity: 0.6186
* Specificity: 0.9523
* For easy case (test1, test2), it works very well.
* CG iterations cannot be 1. It will have huge effect on accuracy. 
* HOG is the main feature. Its cell size cannot be smaller.
* In theory, seach scale should be decrease with decrease of input size. With lots of experiments, 2 is always a good choice.
* Default LR is good.

### Key points:<br>
1. The workflow is based on SORT;<br>
2. Tracking algorithm is ECO (python);<br>
3. Data association algorithm is Hungarian algorithm (which can be further replaced by more advanced algorithm, e.g. KM or feature releated algorithm);<br>

### OpenCV Setting (MedianFlow)
1. Detection threshold: 0.1
2. Ground truth threshold: 0.3
3. Tracked IoU threshold: 0.1
4. Tracked only (no detection) IoU threshold (no need for eco): 0.1
5. Cosine similarity threshold (no need for eco): 0.95
6. Tracker age: 10

### Dependency
1. OpenCV
2. Numpy

### Update
1. 20201215<br>
    * NMS<br>
    * Key frame extraction
2. 20201223
    * Consine similarity on RoI image
3. 20201221
    * Add track id
4. 20210108
    * Add new tracker: ECO
5. 20210203
    * Update OpenCV tracker code
    * Strict initialization for ECO: for each det, if it has 4 consecutive pre-dets, then it can be initialized
6. 20210208
    * Use new overlap: overlap = intersection / max(areaA, areaB); it can deal with 'A is much larger than B situation';
    * Fix some bugs


### ECO Code Reference
1. Official Matlab: https://github.com/martin-danelljan/ECO
2. Python: https://github.com/StrangerZhang/pyECO