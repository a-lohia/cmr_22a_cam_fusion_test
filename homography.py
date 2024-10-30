import numpy as np
import cv2
from helper import plotMatches
from matchPics import matchPics
from Calibration.calibration import undistort_image

# runs the cv2 calibration matrix on the image
# in this case it will calculate the matrix based on the data every time and then apply, 
# but can be optimized to only run calibration once and use same H matrix

# code to flip an image and save it if neeeded
# cv2.imwrite("r1.jpg", cv2.flip(cv2.imread("r1.jpg"), 1))

left = undistort_image(cv2.imread("Left.jpg"))
right = undistort_image(cv2.imread("Right.jpg"))

left[:, :1600] = [0, 0, 0]
right[:, 220:] = [0, 0, 0]

cv2.imshow("right", right)
cv2.imshow("left", left)


matches, locs1, locs2 = matchPics(left, right)


x1 = np.fliplr(locs1[matches[:, 0]])
x2 = np.fliplr(locs2[matches[:, 1]])

plotMatches(left, right, matches, locs1, locs2)

# taking Right to Left
H, _ = cv2.findHomography(x2, x1, method=cv2.RANSAC)
print(H)

left = undistort_image(cv2.imread("Left.jpg"))
right = undistort_image(cv2.imread("Right.jpg"))

# takes the right image and transforms it via H into left space (modifies "right" variable)
warped_cover = cv2.warpPerspective(right, H, (2*right.shape[1], right.shape[0] + 1000))

# now paste them together
warped_cover[0:right.shape[0], 0:right.shape[1]] = left

print(warped_cover.shape)

# warped_cover[warped_cover!=0] = (warped_cover[warped_cover!=0] + left[warped_cover!=0]) / 2
# warped_cover[warped_cover==0] = left[warped_cover==0]


cv2.imwrite("homography_result.jpg", warped_cover)
cv2.imshow(" ", warped_cover)
cv2.waitKey(0)
cv2.destroyAllWindows()