import cv2
import os

#paths = ['019QoF6jwBU_150.633000-154.466000','6CUNIOtQ9L4_139.360000-144.440000',
#         'sl08afxcx4_115.515400-119.986533','bLEddi92aFI_171.480000-177.400000'] 


faces1_folder = '../../../100_download/separated_data/faces1'

paths = []
for (dirpath, dirnames, filenames) in os.walk(faces1_folder):
    paths.extend(dirnames)
    break


count = 0
for path in paths:  
  print("{0}/{1}".format(count, len(paths)))
  count += 1
  for i in range(75):
   im = cv2.imread('../../../100_download/separated_data/faces1/'+path+'/cropped'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED)
   print(im.shape)
   size = (int(64), int(64))

   #for infile in glob.glob('cropped'+str(i)+'.jpg'):
      #file, ext = os.path.splitext(infile)
      #im = Image.open(infile)
      #im.thumbnail(size)
   resized = cv2.resize(im, size, interpolation = cv2.INTER_AREA)
   #gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
   try:
       os.makedirs('../../../100_download/separated_data/faces2/'+path)
   except FileExistsError:
       pass
   
   
   
   cv2.imwrite( '../../../100_download/separated_data/faces2/'+path+'/face'+str(i)+'.jpg', resized);
  # waitKey(0);
   # resized.save(file +".thumbnail", "JPEG")
   print('Resized Dimensions : ',resized.shape)
