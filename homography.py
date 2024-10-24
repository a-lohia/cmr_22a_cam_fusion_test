import numpy as np
import cv2
from helper import plotMatches
from matchPics import matchPics
from Calibration.calibration import undistort_image

# runs the cv2 calibration matrix on the image
# in this case it will calculate the matrix based on the data every time and then apply, 
# but can be optimized to only run calibration onc

left = undistort_image(cv2.imread("Left.jpg"))
right = undistort_image(cv2.flip(cv2.imread("Right.jpg"), 1))

# # No calibration of the camera frames before combining
# left = cv2.imread("Left.jpg")
# right = cv2.flip(cv2.imread("Right.jpg"), 1)

cv2.imshow("right", right)
cv2.imshow("left", left)


matches, locs1, locs2 = matchPics(left, right)


x1 = np.fliplr(locs1[matches[:, 0]])
x2 = np.fliplr(locs2[matches[:, 1]])

# plotMatches(left, right, matches, locs1, locs2)

# taking Right to Left
H, _ = cv2.findHomography(x2, x1, method=cv2.RANSAC)
print(H)

warped_cover = cv2.warpPerspective(left, np.linalg.inv(H), (right.shape[1], right.shape[0]))

print(warped_cover.shape)

# hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
# combined = compositeH(H, hp_cover, cv_desk)

warped_cover[warped_cover!=0] = (warped_cover[warped_cover!=0] + right[warped_cover!=0]) / 2

warped_cover[warped_cover==0] = right[warped_cover==0]



cv2.imshow(" ", warped_cover)
cv2.waitKey(0)
cv2.destroyAllWindows()