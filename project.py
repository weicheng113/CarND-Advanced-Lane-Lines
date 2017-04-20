import cv2
from moviepy.editor import VideoFileClip
import os
from threshold import threshold_image
from convolution import Convolution, Lane
from calibration import undistort
from perspective_transform import warp
from image_display import side_by_side_plot
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.convolution = self.createConvolution()
        self.y_meter_per_pixel = 30/720 # meters per pixel in y dimension
        self.x_meter_per_pixel = 3.7/700 # meters per pixel in x dimension

        self.recent_5_lanes = self.zero_lanes(5)
        #lane of polynomial coefficients averaged over the last n iterations
        self.best_lane = None
        self.last_good_lane = None

        self.n_best_lane_used = 0
        self.n_lane_used = 0
        self.n_last_good_lane_used = 0
        self.n_no_lane_used = 0

    def zero_lanes(self, n):
        return [None for i in range(n)]
    def createConvolution(self):
        width = 1280
        height = 720
        window_width = 50
        window_height = 40 # Break image into 18 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching    print("warped shape: ", warped.shape)
        rough_lane_width = 897
        return Convolution(width = width, height = height, window_width = window_width,
                           window_height = window_height, margin = margin,
                           rough_lane_width = rough_lane_width)
    def detect(self, rgb_image):
        undist_image = undistort(rgb_image)
        warped, M, M_inverse = warp(undist_image)
        binary_warped = threshold_image(warped)

        lane = None
        if self.best_lane is not None:
            lane = self.convolution.find_lane_with_hint(self.best_lane.midx(), binary_warped)
        if lane is None:
            lane = self.convolution.find_lane(binary_warped)
        self.add_lane(lane)

        lane_selected = None
        if self.best_lane is not None:
            lane_selected = self.best_lane
            self.n_best_lane_used += 1
        elif (lane is not None) and lane.good_confidence():
            lane_selected = lane
            self.n_lane_used += 1
        else:
            lane_selected = self.last_good_lane
            self.n_last_good_lane_used += 1

        if (lane is not None) and lane.good_confidence():
            self.last_good_lane = lane

        if lane_selected is not None:
            left_curverad, right_curverad = self.measure_curvature(lane_selected)
            center_offset = self.center_offset(lane_selected)
            lane_drawn = self.convolution.draw_lane(lane_selected, M_inverse = M_inverse)
            out_image = cv2.addWeighted(undist_image, 1, lane_drawn, 0.3, 0)
            cv2.putText(out_image, 'Left, Right Curvature: {:.2f}m, {:.2f}m'.format(left_curverad, right_curverad),
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(out_image, 'Center Lane Offset: {:.2f}m'.format(center_offset),
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return out_image
        else:
            if lane is not None:
                print(lane.confidence)
            side_by_side_plot(undist_image, binary_warped)
            self.n_no_lane_used += 1
            return undist_image
    def add_lane(self, lane):
        self.recent_5_lanes.append(lane)
        self.recent_5_lanes.pop(0)
        assert len(self.recent_5_lanes) == 5

        leftx = []
        rightx = []
        y = []
        total_confidence = 0.0
        n_valid_lanes = 0
        for lane in self.recent_5_lanes:
            if (lane is not None) and lane.good_confidence():
                leftx.extend(lane.leftx)
                rightx.extend(lane.rightx)
                y.extend(lane.y)
                total_confidence += lane.confidence
                n_valid_lanes += 1

        if n_valid_lanes>1:
            self.best_lane = Lane(leftx, rightx, y, total_confidence/n_valid_lanes)
        else:
            self.best_lane = None
    def measure_curvature(self, lane):
        return lane.measure_curvature(x_meter_per_pixel=self.x_meter_per_pixel, y_meter_per_pixel=self.y_meter_per_pixel)
    def center_offset(self, lane):
        return self.convolution.center_offset(lane=lane, x_meter_per_pixel = self.x_meter_per_pixel)
    def stats(self):
        return 'Lane selection in times(best lane: {}, lane: {}, last good lane: {}, no lane: {})'.format(
            self.n_best_lane_used, self.n_lane_used, self.n_last_good_lane_used, self.n_no_lane_used)

def process_video(file_path, out_dir):
    filename = file_path.replace('\\', '/').split("/")[-1]
    output_file_path = os.path.join(out_dir, filename.replace(".mp4",'_processed.mp4'))

    clip1 = VideoFileClip(file_path)
    line = Line()
    video_clip = clip1.fl_image(line.detect)
    video_clip.write_videofile(output_file_path, audio=False)
    print(line.stats())

process_video("project_video.mp4", ".")