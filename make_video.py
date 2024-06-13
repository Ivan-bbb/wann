import cv2
import os
import glob
from natsort import natsorted

task_name = "Cancer"
folder_dir = f"./topologies/{task_name}"

filenames = glob.glob(f"{folder_dir}/*.png")
print(filenames)
# filenames.sort()
# print(filenames)
filenames = natsorted(filenames)
print(filenames)
images = [cv2.imread(img) for img in filenames]

fps = 1
video_dim = (images[0].shape[1], images[0].shape[0])
print(video_dim)
vidwriter = cv2.VideoWriter(f"./videos/{task_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim)
for img in images:
    vidwriter.write(img)
vidwriter.release()