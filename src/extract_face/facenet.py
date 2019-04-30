from keras_facenet import FaceNet
embedder = FaceNet()

# images is a list of images, each as an
# np.ndarray of shape (H, W, 3).
import cv2
import numpy as np
import csv

paths = ['ymD5uLlLc0g_036.033000-040.900000','Swss72CHSWg_090.023267-097.297200',
        'Swss72CHSWg_090.023267-097.297200','AvWWVOgaMlk_090.000000-093.566667',
         'akwvpAiLFk0_144.680000-150.000000']
for path in paths:
  for i in range(75):
    image = cv2.imread('../../../download/separated_data/faces2/'+path+'/face'+str(i)+'jpg')
    image = np.array(image)
    tmp = []
    tmp.append(image)
    embeddings = embedder.embeddings(tmp)

    with open('../../../download/separated_data/faces2/'+path+'/facenet_out.csv', mode='a') as f:
      facenet_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      facenet_writer.writerow([path,str(i),embeddings])
