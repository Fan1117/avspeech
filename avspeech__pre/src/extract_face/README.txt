The order to implement our extract_face algorithm:

1. video2frame.py is used to convert video file to 75 frames.
2. frame1box.py uses google-cloud-vision API to extract face in each frame and outputs box coordinates of obtained faces
   into an output.txt file.
3. frame2box.py crops face area of each frame using box coordinates and outputs cropped face images. OUTPUT: croppedxx.jpg
4. box1resolu.py downsamples face images to 64 * 64. OUTPUT: facexx.jpg
5. We have not decide if we use open-cv interpolation or dilated convolution network for image interpolation. TBD
