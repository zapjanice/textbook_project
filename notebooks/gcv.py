import io
import os

# Imports the Google Cloud client library
from google.cloud import vision


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))
        vertices = ([
            '({},{})'.format(vertex.x, vertex.y)
            for vertex in text.bounding_poly.vertices
        ])

        print('bounds: {}'.format(','.join(vertices)))

        return text

    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(
                            response.error.message))
    return 'error'
# Instantiates a client

if __name__ == '__main__':
    # The name of the image file to annotate
    file_name = os.path.abspath('resources/contour1.png')
    detect_text(file_name)
