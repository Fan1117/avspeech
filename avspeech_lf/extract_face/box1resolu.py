import cv2
from PIL import Image
import glob, os
for i in range(75):

   im = cv2.imread('cropped'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED)
   print(im.shape)
   size = (int(64), int(64))

   #for infile in glob.glob('cropped'+str(i)+'.jpg'):
      #file, ext = os.path.splitext(infile)
      #im = Image.open(infile)
      #im.thumbnail(size)
   resized = cv2.resize(im, size, interpolation = cv2.INTER_AREA)
   cv2.imwrite( 'face'+str(i)+'.jpg', resized);
  # waitKey(0);
   # resized.save(file +".thumbnail", "JPEG")
   print('Resized Dimensions : ',resized.shape)
