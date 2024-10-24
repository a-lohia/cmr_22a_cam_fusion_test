# Implements point to pixel algorithm as described in https://arxiv.org/pdf/1911.10150
# Written by Arya Lohia for CMR

import numpy as np

# ---------
# ALGORITHM
# ---------

def point_to_pixel(L: np.ndarray, S: np.ndarray, T: np.ndarray, M: np.ndarray) -> np.ndarray:
    '''
    INPUT: 

    input L: Set of Lidar points
    input S: Segmentation scores for camera pixels (H, W, Classes)
    input T: Homogenous transform matrix T (takes lidar to camera space)

    OUTPUT:
    
    output P: Set of painted pixels in lidar space (u, v, w, r, color)
    '''

    P = []

    # iterate through point vectors in the set of lidar points
    for l in range(L.shape[0]):

        # use transform matrix to transform lidar point into camera space
        l_transform = T@l
        l_image = M@l_transform # project transformed point onto camera plane
        assert len(l_image.shape) == 2  # make sure it is 2D

        # get the class (cone color) at the pixel given by the transformed lidar point.
        s = S[l_image[0], l_image[1], :]

        # TODO: confirm that this should be hstack
        l_painted = np.hstack(l, s)

        P.append(l_painted)

    return np.stack(P)


