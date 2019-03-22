#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:07:06 2019

@author: govardhan
"""

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
import cv2
from moviepy.editor import VideoFileClip
from lanetracker.tracker import LaneTracker
from lanetracker.camera import CameraCalibration

calibrate = CameraCalibration(glob.glob('camera_cal/calibration*.jpg'), retain_calibration_images=True)

video_output_name = 'video/project_video_annotated.mp4'
video = VideoFileClip("video/project_video.mp4")
tracker = LaneTracker(calibrate(video.get_frame(0)))
video_output = video.fl_image(tracker.process)
get_ipython().magic('time video_output.write_videofile(video_output_name, audio=False)')
