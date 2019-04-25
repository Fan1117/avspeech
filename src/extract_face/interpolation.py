
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift
import cv2

def inter(images,t):
#input:
# images: list of arrays/frames ordered according to motion
# t: parameter ranging from 0 to 1 corresponding to first and last frame
#returns: interpolated image

#direction of movement, assumed to be approx. linear
  a=np.array(center_of_mass(images[0]))
  b=np.array(center_of_mass(images[-1]))
 # print(a,b)
 #find index of two nearest frames
  v=a+t*(b-a)
 # print(v)
  tstar=np.linalg.norm(v-a)/np.linalg.norm(b-a)
  im1_shift=shift(images[0],(b-a)*(1-tstar))
  im2_shift=shift(images[1],(b-a)*(tstar))
 # print(im1_shift,im2_shift)
 # print((b-a)*(tstar))
  return im1_shift+im2_shift

from PIL import Image
image1 = Image.open(r'face'+str(1)+'.jpg')
image2 = Image.open(r'face'+str(70)+'.jpg')
images = [np.array(image1),np.array(image2)]
tmp = inter(images,0.25)
cv2.imwrite('interface1.jpg',tmp)
tmp = inter(images,0.5)
cv2.imwrite('interface2.jpg',tmp)
tmp = inter(images,0.75)
cv2.imwrite('interface3.jpg',tmp)  # HERE WE ONLY TEST 3 INTERPOLATION IMAGES
