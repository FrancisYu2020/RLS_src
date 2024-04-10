# from pynput import keyboard
import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import time

class Annotator:
    def __init__(self, global_start_timestamp, start_idx, screenshot_width=None, \
                 top_margin=370, left_margin=107, right_margin=10, bottom=413, \
                 interval_length=30, gap_threshold=10, screen_change_threshold=0.01, \
                 csv_file_path=None, roi=(689, 81, 734, 99), \
                 snapshots_path=None, show_debug=False) -> None:
        '''
        global_start_timestamp: the timestamp where the recordings started to play
        screenshot_width: the width of the screenshot, will be initialized if set to None
        top_margin: the top most row index of the leg movement EMG region
        left_margin: the left most column index of the leg movement EMG region
        right_margin: the right most column index of the leg movement EMG region
        bottom: the bottom most row index of the leg movement EMG region
        interval_length: the recording time span in each frame, default 60 s
        gap_threshold: TODO finish this comment
        screen_change_threshold: the ratio of pixels changed to judge if the screen has changed or not
        csv_file_path: the path to the label file
        roi: region of interest for the screen changing, (left margin, top margin, right margin, bottom coordinate )
        snapshots_dir: path to save snapshots for each frame to check the labeling quality
        '''
        # initialize the object with the preset attributes and parameters
        self.global_start_timestamp = global_start_timestamp
        self.screenshot_width = screenshot_width
        self.top_margin = top_margin
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.bottom = bottom
        self.interval_length = interval_length
        self.gap_threshold = gap_threshold
        self.screen_change_threshold = screen_change_threshold
        if not csv_file_path:
            csv_file_path = snapshots_path.split("/")[-1]
        elif not csv_file_path.endswith(".csv"):
            csv_file_path += ".csv"
        self.csv_file_path = os.path.join(snapshots_path, csv_file_path)
        self.roi = roi
        self.snapshots_path = snapshots_path
        self.start_idx = start_idx
        self.show_debug = show_debug

        # initialize the screenshot width by taking a screenshot and check if screenshot width is not specified
        if self.screenshot_width is None:
            self.screenshot_width = np.array(pyautogui.screenshot()).shape[1]

    def get_current_timestamp(self, start_timestamp, x_coordinate):
        '''
        function used to interpolate the current timestamp using the start timestamp and pixel's x coordinate
        
        start_timestamp: the start timestamp at the current screenshot frame, type - tuple (start_h, start_min, start_s)
        left_margin: pixels need to be ignored on the left margin
        right_offset: pixels need to be ignored on the right margin
        screenshot_width: the total pixel width of the screenshot image
        x_coordinate: the x pixel coordinate of the point of interest 
        
        return: h, min, s
        '''
        start_h, start_min, start_s = start_timestamp
        interval_ratio = (x_coordinate - self.left_margin) / (self.screenshot_width - self.left_margin - self.right_margin)
        global_second = start_h * 3600 + start_min * 60 + start_s + self.interval_length * interval_ratio
        h = global_second // 3600
        global_second -= h * 3600
        min = global_second // 60
        s = global_second - min * 60
        return [h, min, s]

    def get_screenshot_start_timestamp(self, screenshot_idx):
        '''
        function to get the current screenshot's start timestamp

        global_start_timestamp: the start timestamp at the very beginning, type - tuple (start_h, start_min, start_s)
        screenshot_idx: the frame idx of currently played recording, 0-indexed
        intervals: the time interval used in playing the recording

        return: start_h, start_min, start_s
        '''
        start_h, start_min, start_s = self.global_start_timestamp
        global_second = start_h * 3600 + start_min * 60 + start_s + self.interval_length * screenshot_idx
        start_h = global_second // 3600
        global_second -= start_h * 3600
        start_min = global_second // 60
        start_s = global_second - start_min * 60
        return [start_h, start_min, start_s]

    def compress_intervals(self, arr):
        '''
        function used to compress the consecutive intervals

        arr: the input index arr of the positive regions, e.g. [2,2,2,3,4,5,10,11,12]
        gap_threshold: the maximum gap between two intervals, if smaller than this gap, the two intervals are merged

        return: list of intervals, e.g. [[2, 5], [10, 12]]
        '''
        # if no red pixels, we directly return empty list
        if len(arr) == 0:
            return []
        arr.sort()
        # Ensure the array is unique and sorted
        arr = np.unique(arr)
        # Find the differences between consecutive elements
        diff = np.diff(arr)
        # Indices where the difference is greater than 1
        split_indices = np.where(diff > self.gap_threshold)[0] + 1
        # Split the array into sub-arrays where the difference between numbers is more than 1
        sub_arrays = np.split(arr, split_indices)
        # Convert sub-arrays into intervals [min, max]
        intervals = [ [sub_arr[0], sub_arr[-1]] for sub_arr in sub_arrays ]
        return intervals

    def intervals_to_timestamps(self, intervals, screenshot_idx):
        '''
        function to convert the intervals got from the screen to the actual timestamps

        intervals: list of pixel intervals, type -> List[List]

        return: np.ndarray (len(intervals), 6)
        '''
        start_timestamp = self.get_screenshot_start_timestamp(screenshot_idx)
        print(f'start timestamp for frame {screenshot_idx * (self.interval_length // 30) + self.start_idx} is {start_timestamp}')
        for i in range(len(intervals)):
            start_x, end_x = intervals[i]
            curr_start = self.get_current_timestamp(start_timestamp, start_x)
            curr_end = self.get_current_timestamp(start_timestamp, end_x)
            intervals[i] = curr_start + curr_end
        return np.array(intervals) if intervals else None
        

    def screen_changed(self, prev_screen, new_screen):
        '''
        function to check if the new screen is different from the previous screen

        prev_screen: cv2 bgr array of the old screen
        new_screen: cv2 bgr array of the current screen
        threshold: the lower bound of the percentage of pixel change for the frame change

        return: boolean value of whether the screen has changed
        '''
        # Calculate the difference
        l, u, r, b = self.roi
        diff = cv2.absdiff(prev_screen, new_screen)[u : b + 1, l : r + 1, :] > 0

        # Convert difference to a percentage change
        percent_changed = np.sum(diff) / np.prod(diff.shape)
        return percent_changed > self.screen_change_threshold

    def process_screenshot(self, screenshot, screenshot_idx):
        '''
        main function to get the positive sample time intervals

        screenshot: the cv2 RGB image array input of the currently processed screenshot
        csv_file_path: path the the database where we save the label data
        top_margin: the length of the top margin to focus on the region of interest
        left_margin: the length of the left margin to focus on the region of interest
        right_margin: the length of the right margin to focus on the region of interest

        return: None
        ''' 
        # Find red pixels
        red_pixels = screenshot[self.top_margin : self.bottom, self.left_margin : -self.right_margin, ...]
        red_pixels = np.where((red_pixels[:, :, 2] > 200) & (red_pixels[:, :, 1] < 100) & (red_pixels[:, :, 0] < 100))
        coordinates = red_pixels[1] + self.left_margin
        intervals = self.compress_intervals(coordinates)
        print(intervals)

        # visualization
        for start, end in intervals:
            cv2.line(screenshot, (start, 5 + self.top_margin), (end, 5 + self.top_margin), (255, 0, 0), 2)
        
        # Display the modified screenshot
        if self.snapshots_path:
            if not os.path.exists(self.snapshots_path):
                os.mkdir(self.snapshots_path)
            cv2.imwrite(os.path.join(self.snapshots_path, f'{screenshot_idx}.png'), screenshot)

        # Save to CSV
        df = pd.DataFrame(self.intervals_to_timestamps(intervals, screenshot_idx), columns=['start_h', 'start_min', 'start_s', 'end_h', 'end_min', 'end_s'])
        df.to_csv(self.csv_file_path, mode='a', index=False, header=not os.path.exists(self.csv_file_path))

        if self.show_debug:
            self.debug_draw(screenshot, self.left_margin, self.top_margin, self.screenshot_width - self.right_margin, self.bottom)
            # l, u, r, d = self.roi
            # self.debug_draw(screenshot, l, u, r, d)
    
    def debug_draw(self, img, l, u, r, d, color=(255, 0, 0), thickness=2):
        cv2.line(img, (l, u), (l, d), color, thickness)
        cv2.line(img, (l, u), (r, u), color, thickness)
        cv2.line(img, (r, u), (r, d), color, thickness)
        cv2.line(img, (l, d), (r, d), color, thickness)

