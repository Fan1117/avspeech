import cv2
from PIL import Image
import glob, os

paths = ['ymD5uLlLc0g_036.033000-040.900000','Swss72CHSWg_090.023267-097.297200',
        'Swss72CHSWg_090.023267-097.297200','AvWWVOgaMlk_090.000000-093.566667',
         'akwvpAiLFk0_144.680000-150.000000']

for path in paths:  
  for i in range(75):
   im = cv2.imread('../../../download/separated_data/faces1/'+path+'/cropped'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED)
   print(im.shape)
   size = (int(64), int(64))

   #for infile in glob.glob('cropped'+str(i)+'.jpg'):
      #file, ext = os.path.splitext(infile)
      #im = Image.open(infile)
      #im.thumbnail(size)
   resized = cv2.resize(im, size, interpolation = cv2.INTER_AREA)
   #gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
   cv2.imwrite( '../../../download/separated_data/faces2/'+path+'/face'+str(i)+'.jpg', resized);
  # waitKey(0);
   # resized.save(file +".thumbnail", "JPEG")
   print('Resized Dimensions : ',resized.shape)
