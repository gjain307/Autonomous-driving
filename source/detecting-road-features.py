# coding: utf-8
# # Lane Finding
# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
import cv2
# Camera calibration
# In[2]:
from lanetracker.camera import CameraCalibration

calibrate = CameraCalibration(glob.glob('camera_cal/calibration*.jpg'), retain_calibration_images=True)

print('Correction images (successfully detected corners):')
plt.figure(figsize = (11.5, 9))
gridspec.GridSpec(5, 4)
# Step through the list and search for chessboard corners
for i, image in enumerate(calibrate.calibration_images_success):
    plt.subplot2grid((5, 4), (i // 4, i % 4), colspan=1, rowspan=1)
    plt.imshow(image)
    plt.axis('off')
plt.show()

print('\nTest images (failed to detect corners):')
for i, image in enumerate(calibrate.calibration_images_error):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Original', fontsize=10)
    ax2.axis('off')
    ax2.imshow(calibrate(image))
    ax2.set_title('Calibrated', fontsize=10)


# Color & gradient threshold pipeline.

# In[10]:


from lanetracker.gradients import get_edges

image = mpimg.imread('test_images/test6.jpg')
result = get_edges(image, separate_channels=True)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Original', fontsize=18)
ax2.axis('off')
ax2.imshow(result)
ax2.set_title('Edges', fontsize=18)


# Perspective transform

# In[11]:


from lanetracker.perspective import flatten_perspective

image = mpimg.imread('test_images/test6.jpg')
result, _ = flatten_perspective(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Original', fontsize=18)
ax2.axis('off')
ax2.imshow(result)
ax2.set_title('Bird\'s eye view', fontsize=18)


# Finding the lines

# In[3]:


from lanetracker.tracker import LaneTracker

for image_name in glob.glob('test_images/*.jpg'):
    calibrated = calibrate(mpimg.imread(image_name))
    lane_tracker = LaneTracker(calibrated)
    overlay_frame = lane_tracker.process(calibrated, draw_lane=True, draw_statistics=True)
    mpimg.imsave(image_name.replace('test_images', 'output_images'), overlay_frame)
    plt.imshow(overlay_frame)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Applying pipeline to video

# In[ ]:


from moviepy.editor import VideoFileClip

video_output_name = 'video/project_video_annotated1.mp4'
video = VideoFileClip("video/project_video2.mp4")
tracker = LaneTracker(calibrate(video.get_frame(0)))
video_output = video.fl_image(tracker.process)
image.setflags(write=1)
get_ipython().magic('time video_output.write_videofile(video_output_name, audio=False)')
