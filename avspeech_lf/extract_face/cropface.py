import os 

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

#paths = ['019QoF6jwBU_150.633000-154.466000','6CUNIOtQ9L4_139.360000-144.440000',
#         'sl08afxcx4_115.515400-119.986533','bLEddi92aFI_171.480000-177.400000']
    
faces_folder = '../../../lf/separated_data/faces'

paths = []
for (dirpath, dirnames, filenames) in os.walk(faces_folder):
    paths.extend(dirnames)
    break
count = 0
for path in paths:
 print("{0}/{1}".format(count, len(paths)), path)
 count += 1
 filepath = '../../../lf/separated_data/faces/'+path+'/box.txt'
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
 if len(boxes) < 75:
     continue
 for i in range(75):
    boxes[i] = boxes[i].split(',')
    #print(boxes[i])
 #print(type(boxes[30][0]))
 #boxes = open('output.txt', 'r')
 #print(boxes[30])
 #print(int(boxes[30][0]),int(boxes[30][1]), int(boxes[30][2]), int(boxes[30][5]))

 
 for i in range(75):
    image = '../../../lf/separated_data/faces/'+path+'/out'+str(i)+'.jpg'
    try:
        os.makedirs('../../../lf/separated_data/faces1/'+path)
    except FileExistsError:
        pass
    crop(image, (int(boxes[i][0]),int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][5])), '../../../lf/separated_data/faces1/'+path+'/cropped'+str(i)+'.jpg')

