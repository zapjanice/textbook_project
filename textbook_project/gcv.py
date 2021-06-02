import io
import os
import cv2

# Imports the Google Cloud client library
from google.cloud import vision


def detect_text(img_dir, ant_ind, entity):
    """Detects text in the file."""
    if ant_ind < 10:
        ant_ind = f"0{ant_ind}"
    path = f"{img_dir}/{ant_ind}-{entity}.png"

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        vertices = ([
            '({},{})'.format(vertex.x, vertex.y)
            for vertex in text.bounding_poly.vertices
        ])

        return '\n"{}"'.format(text.description)

    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(
                            response.error.message))
    return 'error'


# Instantiates a client


def crop_image(img_dir, coord, entity, ant_ind):
    # Apply Datafarme of left, top, wideth, and height with original image pass
    # df = df[[left, top, wideth, height]]
    # df[transc] = df.apply(crop_image, axis=1)

    X = int(coord.left)
    Y = int(coord.top)
    W = int(coord.width)
    H = int(coord.height)

    cv2_img = cv2.imread(f"{img_dir}/original.png")
    cropped_image = cv2_img[Y:Y + H, X:X + W]

    if ant_ind < 10:
        ant_ind = f"0{ant_ind}"
    cv2.imwrite(f'{img_dir}/{ant_ind}-{entity}.png', cropped_image)



if __name__ == '__main__':
    # The name of the image file to annotate
    file_name = os.path.abspath('resources/contour1.png')
    detect_text(file_name)
