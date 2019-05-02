"""Draws squares around detected faces in the given image."""


# [START vision_face_detection_tutorial_imports]
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw

# [END vision_face_detection_tutorial_imports]


# [START vision_face_detection_tutorial_send_request]
def detect_face(face_file, max_results=1):
    """Uses the Vision API to detect faces in the given file.
    Args:
        face_file: A file-like object containing an image with faces.
    Returns:
        An array of Face objects with information about the picture.
    """
    # [START vision_face_detection_tutorial_client]
    #client = vision.ImageAnnotatorClient()
    # [END vision_face_detection_tutorial_client]

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations
# [END vision_face_detection_tutorial_send_request]


# [START vision_face_detection_tutorial_process_response]
def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.
    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    
    box = [(63, 63), (300, 63), (300, 200), (63, 200)]
    # Sepecify the font-family and the font-size
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save(output_filename)
    
    return box
# [END vision_face_detection_tutorial_process_response]

# [START vision_face_detection_tutorial_run_application]


def main(input_filename, output_filename, max_results,path):
       
    for i in range(len(input_filename)):
        with open(input_filename[i], 'rb') as image:
            faces = detect_face(image, max_results)
            #print('Found {} face{}'.format(len(faces), '' if len(faces) == 1 else 's'))
            if len(faces) < 1:
                continue
            if len(faces) > 1:
                continue
            
            try:
                os.makedirs('../../../../pre_data/separated_data/faces/'+path)
            except FileExistsError:
                pass
            #print('Writing to file {}'.format(output_filename[i]))
            # Reset the file pointer, so we can read the file again
            image.seek(0)
            box = highlight_faces(image, faces, output_filename[i])
        with open("../../../../pre_data/separated_data/faces/"+path+"/box.txt","a") as f:#append mode
           f.write(str(box)+'\n') 
# [END vision_face_detection_tutorial_run_application]

#paths = ['019QoF6jwBU_150.633000-154.466000','6CUNIOtQ9L4_139.360000-144.440000',
#        'sl08afxcx4_115.515400-119.986533','bLEddi92aFI_171.480000-177.400000'] 


import os

client = vision.ImageAnnotatorClient()

frame_folder = '../../../../pre_data/separated_data/frame'

paths = []
for (dirpath, dirnames, filenames) in os.walk(frame_folder):
    paths.extend(dirnames)
    break

count = 0
for path in paths:
    print("{0}/{1}".format(count, len(paths)))
    count += 1

    input_image = []
    output = []
    for i in range(75):
       input_image.append('../../../../pre_data/separated_data/frame/'+path+'/'+str(i)+'.jpg')
       output.append('../../../../pre_data/separated_data/faces/'+path+'/out'+str(i)+'.jpg')
    max_results = 1
    main(input_image,output,max_results, path)

