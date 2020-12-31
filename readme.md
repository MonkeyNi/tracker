## Tracker for object detection

### Test example
```
python track.py --video test/test2.mp4 --detection_result test/results2_0.1.txt

Detection result should has format:
    frame_name, x1, y1, x2, y2, categort_id, score
```
There will be an video, tracked result txt fie, and key frames in output. Output will be saved at the same folder as 'detection result'.

### Default Setting
1. Detection threshold: 0.1
2. Ground truth threshold: 0.3
3. Tracked IoU threshold: 0.1
4. Tracked only (no detection) IoU threshold: 0.1
5. Cosine similarity threshold: 0.95
6. Tracker age: 3 

### Key points:<br>
1. The workflow is based on SORT;<br>
2. Tracking algorithm is from openCV API, default is 'KCF';<br>
3. Data association algorithm is Hungarian algorithm (which can be further replaced by more advanced algorithm, e.g. KM or feature releated algorithm);<br>

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

### TODO
1. Consine similarty on RoI features