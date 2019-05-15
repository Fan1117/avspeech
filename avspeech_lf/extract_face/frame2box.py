
import os
import io
import cv2
from PIL import Image

filepath = 'output.txt'
with open(filepath) as fp:
   line = fp.readline()
   line = line.replace('[','')
   line = line.replace(']','')
   line = line.replace('(','')
   line = line.replace(')','')
   line = line.replace(' ','')
   boxes=[line.strip()]
   #cnt = 1
   while line:
      line = fp.readline()
      line = line.strip()
      line = line.replace('[','')
      line = line.replace(']','')
      line = line.replace('(','')
      line = line.replace(')','')
      line = line.replace(' ','')
      boxes.append(line)
boxes.pop()
for i in range(75):
    boxes[i] = boxes[i].split(',')
    #print(boxes[i])
#print(type(boxes[30][0]))
#boxes = open('output.txt', 'r')
#print(boxes[30])
#print(int(boxes[30][0]),int(boxes[30][1]), int(boxes[30][2]), int(boxes[30][5]))


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()


if __name__ == '__main__':
   for i in range(75):
      image = 'framebox'+str(i)+'.jpg'
      crop(image, (int(boxes[i][0]),int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][5])), 'cropped'+str(i)+'.jpg') 




