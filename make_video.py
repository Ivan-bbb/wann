import cv2
import glob
from natsort import natsorted
import imageio
from config import config

task_name = "Cancer"
folder_dir = f"./topologies/{task_name}"

filenames = glob.glob(f"{folder_dir}/*.png")
filenames = natsorted(filenames)
images = [cv2.imread(img) for img in filenames]

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (images[0].shape[1] // 2, 50) 
fontScale = 2
color = (255, 0, 0) 
thickness = 2

with imageio.get_writer(f"./videos/{task_name}.gif", mode="I", fps=1) as writer:
    for idx, frame in enumerate(images):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.putText(rgb_frame, f"Generation {idx*config.save_interval}", org, font, fontScale, color, thickness, cv2.LINE_AA)
        writer.append_data(rgb_frame)
