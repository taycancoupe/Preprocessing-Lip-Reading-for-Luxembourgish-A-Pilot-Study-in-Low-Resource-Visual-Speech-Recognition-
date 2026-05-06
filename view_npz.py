import numpy as np
import cv2

data = np.load("outputs/test_face_crops.npz")
frames = data["frames"]

for i in range(0, len(frames), 30):  # every 30th frame
    frame = frames[i]
    cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)

cv2.destroyAllWindows()