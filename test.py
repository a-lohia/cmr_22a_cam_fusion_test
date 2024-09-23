import numpy as np
from PIL import Image

PHI = 75.4
X_DIST = 44.48
Y_DIST = 57.55
theta = PHI/2

left, right = Image.open("Left.jpg"), Image.open("Right.jpg")
x, y = np.array(left), np.array(right)
y = np.flip(y, axis=(0, 1))

print(x.shape)

def left_transform(x, y, image, angle: float=theta) -> np.ndarray:
    """
    angle: degrees
    """
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    T_l = np.array([[c, -s, X_DIST], [s, c, Y_DIST]]) 

    image_data = image[x.flatten(), y.flatten()]
    print(f"data {image_data.shape}")

    z = np.array([x.flatten(), y.flatten(), np.ones(X*Y)])

    return np.vstack([(T_l @ z), image_data.T])

def right_transform(x, y, image, angle: float=theta) -> np.ndarray:
    """
    angle: degrees
    """

    angle = -1 *np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    T_r = np.array([[c, s, X_DIST], [-s, c, Y_DIST]]) 

    image_data = image[x.flatten(), y.flatten()]
    print(f"data {image_data.shape}")

    z = np.array([x.flatten(), y.flatten(), np.ones(X*Y)])

    return np.vstack([(T_r @ z), image_data.T])



# import pdb
# pdb.set_trace()

w = [255, 255, 255]

# y[616, 1059] = w # image left
# y[587, 187] = w  # image right
Image.fromarray(x).show()
Image.fromarray(y).show()


X = 720
Y = 1200
X_1, Y_1 = np.meshgrid(np.arange(X) - 616, np.arange(Y) - 1059)
X_2, Y_2 = np.meshgrid(np.arange(X) - 587, np.arange(Y) - 187)
x_1 = left_transform(X_1, Y_1, x)
y_1 = right_transform(X_2, Y_2, y)

# print(min(x_1[0]), max(x_1[0])) x goes from (-688 to 613)
# print(min(x_1[1]), max(x_1[1])) y goes from (58 to 1446)

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
    out = np.zeros(shape=((max_x + 1 - min_x), (max_y + 1 - min_y), 3))
    
    for i in range(a.shape[-1]-1):

        # round x, y to nearest integer
        x = int(np.round(a[0][i])) - min_x
        y = int(np.round(a[1][i])) - min_y

        if x == max_x + 1 - min_x or y == max_y + 1 - min_y:
            continue

        out[x][y] = int(out[x][y] + a[2:,i]) / 2 if not np.all(out[x][y], where=0) else a[2:,i]
        # print(f"data now set to {out[x][y]}")

        if i == 0:
            print(f"x: {x}, y: {y}")
            print(f"sample r,g,b list: {a[2:, 0]}")
            print(f"data now set to {out[x][y]}")


    return np.round(out).astype(np.uint8)

# import pdb
# pdb.set_trace()

# Image.fromarray(y_1).show(title="right")
# print(np.max(x_1[0]), np.max(x_1[1]))
# y_1[0] -= (np.max(x_1[0])-80)
# y_1[1] += (np.max(x_1[1])-750)
combined = round_coordinates(np.hstack((x_1, y_1)))

# combined = round_coordinates(x_1)

print(combined.shape)
Image.fromarray(combined).show()

