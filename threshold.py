import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from perspective_transform import warp
from calibration import undistort
from image_display import side_by_side_plot
# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_threshold(rgb_image, orientation='x', sobel_kernel=3, threshold=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    if orientation == 'x':
        sobel_xy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_xy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.absolute(sobel_xy)

    sobel_scaled = np.uint8(255 * sobel_abs/np.max(sobel_abs))

    min, max = threshold
    binary_output = np.zeros_like(sobel_scaled)
    binary_output[(sobel_scaled > min) & (sobel_scaled < max)] = 1
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def magnitude_threshold(rgb_image, sobel_kernel=3, threshold=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    scaled = np.uint8(255 * gradient_magnitude/np.max(gradient_magnitude))

    min, max = threshold
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= min) & (scaled <= max)] = 1
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def direction_threshold(rgb_image, sobel_kernel=3, threshold=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    abs_gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    min, max = threshold
    binary_output = np.zeros_like(abs_gradient_direction)
    binary_output[(abs_gradient_direction > min) & (abs_gradient_direction < max)] = 1
    return binary_output

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(rgb_image, channel='S', threshold=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)

    if channel == 'S':
        selected = hls[:,:,2]
    elif channel == 'L':
        selected = hls[:,:,1]
    else:
        selected = hls[:,:,0] # Hue channel

    min, max = threshold
    binary_output = np.zeros_like(selected)
    binary_output[(selected > min) & (selected <= max)] = 1

    return binary_output

def rgb_select(rgb_image, channel='R', threshold=(0, 255)):
    if channel == 'R':
        selected = rgb_image[:,:,0]
    elif channel == 'G':
        selected = rgb_image[:,:,1]
    else:
        selected = rgb_image[:,:,2] # Blue channel

    min, max = threshold
    binary_output = np.zeros_like(selected)
    binary_output[(selected > min) & (selected <= max)] = 1

    return binary_output

def gradient_combined_demo():
    # Choose a Sobel kernel size
    ksize = 15 # Choose a larger odd number to smooth gradient measurements

    image = mpimg.imread('test_images/test5.jpg')
    #image = mpimg.imread('signs_vehicles_xygrad.png')
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(rgb_image=image, orientation='x', sobel_kernel=3, threshold=(20, 255))
    plt.imshow(gradx, cmap = 'gray')
    grady = abs_sobel_threshold(rgb_image=image, orientation='y', sobel_kernel=ksize, threshold=(20, 100))
    #plt.imshow(grady, cmap = 'gray')
    magnitude_binary = magnitude_threshold(rgb_image=image, sobel_kernel=ksize, threshold=(30, 100))
    #plt.imshow(magnitude_binary, cmap = 'gray')
    direction_binary = direction_threshold(rgb_image=image, sobel_kernel=15, threshold=(0.7, 1.3))
    #plt.imshow(direction_binary, cmap = 'gray')

    combined = np.zeros_like(direction_binary)
    #combined[(gradx == 1) & (grady == 1)] = 1
    #combined[(magnitude_binary == 1) & (direction_binary == 1)] = 1
    combined[((gradx == 1) & (grady == 1)) | ((magnitude_binary == 1) & (direction_binary == 1))] = 1
    #plt.imshow(combined, cmap = 'gray')
    plt.show()

def threshold_image(rgb_image):
    # Step 2: S channel or Sobel X threshold.
    s_binary = hls_select(rgb_image=rgb_image, channel='S', threshold=(170, 255))
    #l_binary = hls_select(rgb_image=undist_image, channel='H', threshold=(170, 255))
    r_binary = rgb_select(rgb_image=rgb_image, channel='R', threshold=(220, 255))
    #sobelx_binary = abs_sobel_threshold(rgb_image=rgb_image, orientation='x', threshold=(20, 255), sobel_kernel=3)

    combined_binary = np.zeros_like(s_binary)
    #combined_binary[(s_binary == 1) | (sobelx_binary == 1)] = 1
    #combined_binary[(s_binary == 1) & ((r_binary == 1) | (sobelx_binary == 1))] = 1
    combined_binary[(s_binary == 1) | (r_binary == 1)] = 1
    #plt.imshow(sobelx_binary, cmap = 'gray')
    #plt.show()
    return combined_binary

def demo_threshold_image():
    image = mpimg.imread('test_images/straight_lines1.jpg')
    undist = undistort(image)
    warped, M, M_inverse = warp(undist)

    binary_warped = threshold_image(warped)
    side_by_side_plot(image, binary_warped, im1_title='Image', im2_title='Binary Warped', im2_cmap='gray')
    #plt.imshow(binary_image, cmap = 'gray')
    #plt.show()

#demo_threshold_image()