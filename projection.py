import numpy as np
from PIL import Image
import cv2 as cv

PHI = 75.4
X_DIST = 44.48
Y_DIST = 57.55
theta = PHI/2


left, right = Image.open("Left.jpg"), Image.open("Right.jpg")
x, y = np.array(left), np.array(right)
y = np.flip(y, axis=(0, 1))

print(x.shape)
# New Idea. project R2 (x, y) down to R1 and add z axis as second dimension

def proj_transform(x, y, image, origin: tuple, angle: float=theta) -> np.ndarray:
    """
    angle: degrees
    """
    angle = np.radians(angle)
    # c, s = np.cos(angle), np.sin(angle)
    # T_l = np.array([[c, -s, X_DIST], [s, c, Y_DIST]]) 
    T_l = np.array([[1, 0], [0, np.sin(angle)]])  # projects x in camera space to x' in projected space

    image_data = image[x.flatten(), y.flatten()]
    # print(f"data {image_data.shape}")

    z = np.array([(x-origin[0]).flatten(), (y-origin[1]).flatten()])

    # print(np.vstack([(T_l @ z), image_data.T])[:2])

    assert np.all(np.abs((y-origin[1]).flatten()) >= np.abs(np.vstack([(T_l @ z), image_data.T])[1]))  # all transformed x coordinates are <= the original. They have been project correctly

    return np.vstack([(T_l @ z), image_data.T])

# w = [255, 255, 255]

# x[616, 1059] = w # image left
# y[587, 187] = w  # image right

# cv.imshow("left", x)
# cv.imshow("right", y)
Image.fromarray(x).show()
Image.fromarray(y).show()

X = 720
Y = 1200

# add origins to each image.
X_1, Y_1 = np.meshgrid(np.arange(X), np.arange(Y))
X_2, Y_2 = np.meshgrid(np.arange(X), np.arange(Y))

# project image planes onto combined plane
x_1 = proj_transform(X_1, Y_1, x, (616, 1059))
# x_1 = x_1[:, np.argsort(x_1[1])]
y_1 = proj_transform(X_2, Y_2, y, (587, 187))
# y_1 = y_1[:, np.argsort(y_1[1])]

# import pdb
# pdb.set_trace()

def horiz_pad(arr: np.ndarray, u) -> np.ndarray:
    """takes an RGB image array, arr, and adds white pixel padding till it's size: x, y"""

    x, y, c = arr.shape
    pad = 255* np.ones(shape=(u, y, c))

    print(pad.shape, arr.shape)
    return np.hstack([arr, pad])


def round_coordinates(a) -> np.ndarray:
    """
    takes input array a shape (5, N) where the rows represent the following information:
    1. X value
    2. Y value
    3. R Value
    4. G
    5. B

    returns output integer array of shape (X, Y, 3), where it rounds the x and y values to achieve this.
    """

    min_x, max_x = int(min(a[0])), int(max(a[0]))
    min_y, max_y = int(min(a[1])), int(max(a[1]))

    print(min_x, max_x)
    print(min_y, max_y)
    print(f"final size {max_x + 1 - min_x}, {max_y + 1 - min_y}, 3")
    out = np.zeros(shape=((max_x + 1 - min_x), (max_y + 1 - min_y), 3))

    # print(a.shape[-1]-1)
    
    for i in range(a.shape[-1]):

        # round x, y to nearest integer
        x = int(a[0][i]) - min_x
        y = int(a[1][i]) - min_y

        # if i > a.shape[-1] - 1000 and i % 10 == 0:
        #     print(f"y: {int(np.round(a[1][i]))} - {min_y} = {int(np.round(a[1][i])) - min_y}")

        # skip if out of bounds
        if x == max_x + 1 - min_x or y == max_y + 1 - min_y:
            continue

        check_output = False



        # if not np.any(out[x][y], where=0):
        #     print(f"data current: {out[x][y]}")
        #     check_output = True

        out[x][y] = (out[x][y] + a[2:,i]) / 2 if np.any(out[x][y]) else a[2:,i]
        # print(f"data now set to {out[x][y]}")

        if i == 0 or check_output:
            print(f"x: {x}, y: {y}")
            print(f"sample r,g,b list: {a[2:, 0]}")
            print(f"data now set to {out[x][y]}")


    return np.round(out).astype(np.uint8)

combined = round_coordinates(np.hstack((x_1, y_1)))


x = cv.resize(x, (int(720/2), int(1200/2)), interpolation=cv.INTER_CUBIC)
y = cv.resize(y, (int(720/2), int(1200/2)), interpolation=cv.INTER_CUBIC)

# top = np.hstack([x, y])

# print(top.shape)

# pad = np.abs(top.shape[0] - combined.shape[0])

# im = np.vstack([top, horiz_pad(combined, pad)])

print(combined.shape)
cv.imshow("combined", combined)

cv.waitKey(0) # wait for ay key to exit window
cv.destroyAllWindows() # close all windows