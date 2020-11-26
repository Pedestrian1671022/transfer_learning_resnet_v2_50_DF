import os
import cv2

dataset_dir = "test"
flowers = ["false/IR", "false/RD"]
photo_files = []

for flower in flowers:
    for image in os.listdir(os.path.join(dataset_dir, flower)):
        photo_files.append(os.path.join(dataset_dir, flower, image))

for photo_file in photo_files:
  print(photo_file)
  img = cv2.imread(photo_file, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
  cv2.imwrite(photo_file.replace("false","false_").replace("bmp", ".jpg"), img)
