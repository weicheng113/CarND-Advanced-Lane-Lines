import numpy as np
import cv2
import matplotlib.image as mpimg
from calibration import undistort
from chessboard import chessboard_size
from image_display import side_by_side_plot

def corners_unwarp(image, chessboard_size):
    """
    chessboard_size = (nx, ny) nx by ny size chessboard.
    """
    undist = undistort(image)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    nx, _ = chessboard_size

    if ret:
        cv2.drawChessboardCorners(undist, chessboard_size, corners, ret)
        width, height = image.shape[1], image.shape[0]

        src_corners = np.float32([corners[0], corners[nx-1], corners[-nx], corners[-1]])
        offset = 100 # offset for dst points
        dst_corners = np.float32([[offset, offset], [width-offset, offset],
                                  [offset, height-offset], [width-offset, height-offset]])

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped = cv2.warpPerspective(undist, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, M
    else:
        raise Exception("Cannot find corners", UserWarning)

def warp(image):
    width, height = image.shape[1], image.shape[0]

    middle_x = width//2
    top_y = 2*height//3
    top_half_lane = 93
    bottom_half_lane = 450

    src = np.float32([
        (middle_x-top_half_lane, top_y),
        (middle_x+top_half_lane, top_y),
        (middle_x+bottom_half_lane, height),
        (middle_x-bottom_half_lane, height)
    ])
    dst = np.float32([
        (middle_x-bottom_half_lane, 0),
        (middle_x+bottom_half_lane, 0),
        (middle_x+bottom_half_lane, height),
        (middle_x-bottom_half_lane, height)
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped, M, M_inverse

def demo_chessboard():
    sample_image_file = "camera_cal/calibration8.jpg"
    image = cv2.imread(sample_image_file)
    warped, M, M_inverse = corners_unwarp(image, chessboard_size)
    #cv2.imshow("warped", warped)
    #cv2.waitKey(0)
    cv2.imwrite('output_images/calibration8_warped.jpg',warped)

def demo():
    image = mpimg.imread('test_images/straight_lines2.jpg')
    undist_image = undistort(image)

    warped, M, M_inverse = warp(undist_image)
    side_by_side_plot(image, warped, im1_title="Image", im2_title="Warped Image")

#demo_chessboard()
#demo()
