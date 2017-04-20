import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import warnings
from perspective_transform import warp
from calibration import undistort
from threshold import threshold_image
from image_display import side_by_side_plot

class Lane():
    def __init__(self, leftx, rightx, y, confidence):
        self.leftx = leftx
        self.rightx = rightx
        self.y = y
        self.confidence = confidence
        self.left_fit = None
        self.right_fit = None
    def firstx(self):
        return (self.leftx[0], self.rightx[0])

    def midx(self):
        mid = int(len(self.leftx)/2)
        return (self.leftx[mid], self.rightx[mid])

    def polyfit(self):
        if self.left_fit is None:
            self.left_fit = np.polyfit(self.y, self.leftx, 2)
            self.right_fit = np.polyfit(self.y, self.rightx, 2)
        return (self.left_fit, self.right_fit)

    def measure_curvature(self, x_meter_per_pixel, y_meter_per_pixel):
        # Fit new polynomials to x,y in world space
        try:
            float_y = np.array(self.y).astype(float)
            float_leftx = np.array(self.leftx).astype(float)
            float_rightx = np.array(self.rightx).astype(float)
            left_fit_cr = np.polyfit(float_y * y_meter_per_pixel, float_leftx * x_meter_per_pixel, 2)
            right_fit_cr = np.polyfit(float_y * y_meter_per_pixel, float_rightx * x_meter_per_pixel, 2)
        except TypeError as e:
            raise e
        # Calculate the new radii of curvature
        y_eval = np.max(self.y)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*y_meter_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*y_meter_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # Example values: 632.1 m    626.2 m
        return (left_curverad, right_curverad)
    def ploty(self, y):
        left_fit, right_fit = self.polyfit()
        leftx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        rightx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        return (leftx, rightx)
    def good_confidence(self):
        return self.confidence > 0.2

class Convolution():
    def __init__(self, width, height, window_width, window_height, margin, rough_lane_width):
        self.width = width
        self.height = height
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.half_width = int(self.width/2)
        self.window = np.ones(self.window_width) # Create our window template that we will use for convolutions
        self.n_slices = int(self.height/self.window_height)
        self.min_lane_width = rough_lane_width - 100
        self.max_lane_width = rough_lane_width + 100

    def find_lane(self, binary_warped):

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template
        # Sum quarter bottom of image to get slice, could use a different ratio
        image_layer = np.sum(binary_warped[int(3*self.height/4):,:], axis=0)
        conv_signal = np.convolve(self.window, image_layer, mode='same')
        l_center = np.argmax(conv_signal[:self.half_width])
        r_center = np.argmax(conv_signal[self.half_width:]) + self.half_width
        l_center_signal = conv_signal[l_center]
        r_center_signal = conv_signal[r_center]

        if self.one_weak_signal(l_center_signal, r_center_signal) and (not self.is_lane_width_acceptable(l_center, r_center)):
            warnings.warn('Invalid signal: (l_center: {}, l_center_signal: {}, r_center:{}, r_center_signal: {})'.format(
                l_center, l_center_signal, r_center, r_center_signal), UserWarning)
            return None
        # if self.strong_signal(conv_signal[l_center]) and self.weak_signal(conv_signal[r_center]):
        #     r_center_mid = self.width - l_center
        #     r_center = self.find_center(conv_signal, r_center_mid)
        # elif self.weak_signal(conv_signal[l_center]) and self.strong_signal(conv_signal[r_center]):
        #     l_center_mid = self.width - r_center
        #     l_center = self.find_center(conv_signal, l_center_mid)

        return self.find_lane_with_hint((l_center, r_center), binary_warped)

    def find_center(self, conv_signal, center_reference):
        min_index = int(max(center_reference-self.margin,0))
        max_index = int(min(center_reference+self.margin,self.width))
        return np.argmax(conv_signal[min_index:max_index])+min_index

    def weak_signal(self, signal):
        return signal < 500

    def one_weak_signal(self, *signals):
        for signal in signals:
            if self.weak_signal(signal):
                return True
        return False

    def strong_signal(self, signal):
        return not self.weak_signal(signal)

    def strong_signals(self, *signals):
        for signal in signals:
            if self.weak_signal(signal):
                return False
        return True

    def is_lane_width_acceptable(self, l_center, r_center):
        lane_width = r_center - l_center
        if (lane_width > self.min_lane_width) and (lane_width < self.max_lane_width):
            return True
        else:
            warnings.warn('Un-acceptable lane width: {} - (min: {}, max: {})'.format(lane_width, self.min_lane_width, self.max_lane_width), UserWarning)
            return False

    def find_lane_with_hint(self, left_right_center, binary_warped):
        l_center, r_center = left_right_center
        n_confident_slices = 0
        # Go through each layer looking for max pixel locations
        centroids = [] # Store the (left,right) window centroid positions per level
        for level in range(0, self.n_slices):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(binary_warped[int(self.height-(level+1)*self.window_height):int(self.height-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(self.window, image_layer, mode='same')
            # Find the best left centroid by using past left center as a reference
            new_l_center = self.find_center(conv_signal, l_center)
            # Find the best right centroid by using past right center as a reference
            new_r_center = self.find_center(conv_signal, r_center)

            l_center_signal = conv_signal[new_l_center]
            r_center_signal = conv_signal[new_r_center]
            if self.is_lane_width_acceptable(new_l_center, new_r_center):
                if self.strong_signal(l_center_signal):
                    l_center = new_l_center
                if self.strong_signal(r_center_signal):
                    r_center = new_r_center
                if self.strong_signals(l_center_signal, r_center_signal):
                    n_confident_slices += 1

            y = level * self.window_height
            centroids.append((l_center,r_center, y))
        confidence = float(n_confident_slices)/self.n_slices
        centroids = np.array(centroids)
        lane = Lane(leftx=centroids[:,0], rightx=centroids[:,1], y=centroids[:,2], confidence=confidence)
        return lane

    def draw_lane(self, lane, M_inverse):
        # Create an image to draw the lines on
        warp_zero = np.zeros((self.height, self.width)).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        y = np.linspace(0, self.height-1, 72)
        leftx, rightx = lane.ploty(y)
        pts_left = np.array([np.transpose(np.vstack([leftx, y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inverse, (self.width, self.height))
        return newwarp

    def center_offset(self, lane, x_meter_per_pixel):
        l_center, r_center = lane.firstx()
        image_center = self.width/2.0
        lane_center = (l_center + r_center)/2.0
        pixel_offset = image_center - lane_center

        return (pixel_offset * x_meter_per_pixel)

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def demo_mask():
    # Read in a thresholded image
    #warped = mpimg.imread('warped_example.jpg')
    image = mpimg.imread('test_images/test1.jpg')
    undist = undistort(image)
    warped, M, M_inverse = warp(undist)
    binary_warped = threshold_image(warped)
    #plt.imshow(binary_warped)
    #plt.show()
    #warped = mpimg.imread('output_images/straight_lines1_warped.jpg')
    # window settings
    convolution = createConvolution()
    lane = convolution.find_lane(binary_warped)
    window_width = convolution.window_width
    window_height = convolution.window_height
    # If we found any window centers
    if lane is not None:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Go through each level and draw the windows
        for level in range(0,len(lane.leftx)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,binary_warped,lane.leftx[level],level)
            r_mask = window_mask(window_width,window_height,binary_warped,lane.rightx[level],level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)

    # Display the final results
    side_by_side_plot(binary_warped, output, im1_title="Binary Warped", im2_title="Conv Search", im1_cmap='gray', im2_cmap='gray')

def createConvolution():
    height = 720
    width = 1280
    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching    print("warped shape: ", warped.shape)
    rough_lane_width = 897
    return Convolution(width = width, height = height,
                       window_width = window_width, window_height = window_height,
                       margin = margin, rough_lane_width = rough_lane_width)

def to_binary_image(image):
    undist = undistort(image)
    binary_warped, M, M_inverse = warp(undist)

    return threshold_image(binary_warped)

def demo_convolution():
    # Read in a thresholded image
    image = mpimg.imread('test_images/straight_lines2.jpg')
    undist = undistort(image)
    warped, M, M_inverse = warp(undist)
    binary_warped = threshold_image(warped)
    #plt.imshow(binary_warped)
    #plt.show()
    #warped = mpimg.imread('output_images/straight_lines1_warped.jpg')
    # window settings
    height = binary_warped.shape[0]
    convolution = createConvolution()
    lane = convolution.find_lane(binary_warped)

    left_fit, right_fit = lane.polyfit()
    y = np.linspace(0, height-1, 72)
    leftx, rightx = lane.ploty(y)

    y_meter_per_pixel = 30/720 # meters per pixel in y dimension
    x_meter_per_pixel = 3.7/700 # meters per pixel in x dimension
    left_curverad, right_curverad = lane.measure_curvature(x_meter_per_pixel=x_meter_per_pixel, y_meter_per_pixel=y_meter_per_pixel)

    center_offset = convolution.center_offset(lane, x_meter_per_pixel)

    lane_drawn = convolution.draw_lane(lane, M_inverse)
    result = cv2.addWeighted(undist, 1, lane_drawn, 0.3, 0)

    plt.imshow(result)
    plt.plot(leftx, y, color='yellow')
    plt.plot(rightx, y, color='yellow')
    plt.text(5, 30, 'Left, Right Curvature: {:.2f}m, {:.2f}m'.format(left_curverad, right_curverad), fontsize=10, color='red')
    plt.text(5, 50, 'Center Lane Offset: {:.2f}m, Confidence: {:.2f}%'.format(center_offset, lane.confidence*100), fontsize=10, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

#demo_mask()
demo_convolution()