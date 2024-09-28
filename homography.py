import numpy as np
import skimage
import cv2
from helper import plotMatches
from matchPics import matchPics

left = cv2.imread("Left.jpg")
right = cv2.flip(cv2.imread("Right.jpg"), -1)



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