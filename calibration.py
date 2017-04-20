import glob
import numpy as np
import cv2
import pickle
import warnings
import matplotlib.image as mpimg
from chessboard import chessboard_size
from image_display import side_by_side_plot

def calibrate(chessboard_image_files, chessboard_size):
    object_points = []
    image_points = []

    nx, ny = chessboard_size
    object_point = np.zeros((ny*nx,3), np.float32)
    object_point[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for image_filename in chessboard_image_files:
        img = cv2.imread(image_filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points
        if ret == True:
            object_points.append(object_point)
            image_points.append(corners)
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            warnings.warn("Cannot find corners for file '"+image_filename+"'", UserWarning)

    image_size = gray.shape[:2]
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints = object_points, imagePoints = image_points,
        imageSize = image_size, cameraMatrix = None, distCoeffs = None)
    return (camera_matrix, dist_coeffs)

def write_calibration(filename, chessboard_size):
    image_filenames = glob.glob("camera_cal/calibration*.jpg")
    camera_matrix, dist_coeffs = calibrate(image_filenames, chessboard_size)

    output = open(filename, 'wb')
    pickle.dump({"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}, output)
    output.close()

def read_calibration(filename):
    calibration_points_p = pickle.load(open(filename, "rb"))
    camera_matrix_p = calibration_points_p["camera_matrix"]
    dist_coeffs_p = calibration_points_p["dist_coeffs"]
    return (camera_matrix_p, dist_coeffs_p)

def undistort(image):
    camera_matrix, dist_coeffs = read_calibration("calibration.p")
    return cv2.undistort(src = image, cameraMatrix = camera_matrix, distCoeffs = dist_coeffs)

def demo_calibration():
    calibration_file = 'calibration.p'
    write_calibration(calibration_file, chessboard_size)

def demo_undist():
    image = mpimg.imread('test_images/straight_lines1.jpg')
    #image = mpimg.imread('camera_cal/calibration1.jpg')
    undist_image = undistort(image)
    side_by_side_plot(im1=image, im2=undist_image, im1_title="Original Image", im2_title="Undistorted Image")

#demo_calibration()
#demo_undist()
