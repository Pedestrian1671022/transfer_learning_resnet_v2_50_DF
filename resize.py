import os
import cv2
import numpy as np
from PIL import Image

dataset_dir = "test"
flowers = ["true/IR", "true/RD"]
photo_files = []

for flower in flowers:
    for image in os.listdir(os.path.join(dataset_dir, flower)):
        photo_files.append(os.path.join(dataset_dir, flower, image))

for photo_file in photo_files:
  print(photo_file)
  img = Image.open(photo_file)
  img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2
  img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)

  cv2.imwrite(photo_file.replace("true","true_").replace("bmp", ".jpg"), img)