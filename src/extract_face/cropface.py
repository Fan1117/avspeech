import os 
import io
import cv2
from PIL import Image

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

paths = ['ymD5uLlLc0g_036.033000-040.900000','Swss72CHSWg_090.023267-097.297200',
        'Swss72CHSWg_090.023267-097.297200','AvWWVOgaMlk_090.000000-093.566667',
         'akwvpAiLFk0_144.680000-150.000000']

for path in paths:
 filepath = '../../../download/separated_data/faces/'+path+'/box.txt'
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

 if __name__ == '__main__':
   for i in range(75):
      image = '../../../download/separated_data/faces/'+path+'/out'+str(i)+'.jpg'
      crop(image, (int(boxes[i][0]),int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][5])), '../../../download/separated_data/faces1/'+path+'/cropped'+str(i)+'.jpg')

