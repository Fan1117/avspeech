from keras_facenet import FaceNet
embedder = FaceNet()

# images is a list of images, each as an
# np.ndarray of shape (H, W, 3).
import cv2
import numpy as np
import os

#paths = ['019QoF6jwBU_150.633000-154.466000','6CUNIOtQ9L4_139.360000-144.440000',
#         'sl08afxcx4_115.515400-119.986533','bLEddi92aFI_171.480000-177.400000'] 

faces2_folder = '../../../100_download/separated_data/faces2'

paths = []
for (dirpath, dirnames, filenames) in os.walk(faces2_folder):
    paths.extend(dirnames)
    break

count = 0
for path in paths:
  print("{0}/{1}".format(count, len(paths)))
  count += 1
  a = np.zeros((75, 512))
  for i in range(75):
    image = cv2.imread('../../../100_download/separated_data/faces2/'+path+'/face'+str(i)+'.jpg')
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    print(image.shape)
    tmp = []
    tmp.append(image)
    a[i,:] = embedder.embeddings(tmp)
  try:
      os.makedirs('../../../100_download/separated_data/nparray')
  except FileExistsError:
      pass
  np.save('../../../100_download/separated_data/nparray/'+path+'_facenet.npy',a)
#    with open('../../../download/separated_data/faces2/'+path+'/facenet_out.csv', mode='a') as f:
#      facenet_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#      facenet_writer.writerow([path,str(i),embeddings])
