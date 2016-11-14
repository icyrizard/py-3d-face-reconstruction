import os
import json
import traceback
import os.path
import base64
from glob import glob

import cv2
import numpy as np
from tornado import websocket, web, ioloop, autoreload

import pca
from datasets import imm
from reconstruction import reconstruction
from settings import logger
from utility import import_dataset_module


FILES_DIR = '/data/'
FACE_DB_NAME = 'imm_face_db'
FACE_DB = '{}{}'.format(FILES_DIR, FACE_DB_NAME)
DATASET = os.environ.get('DATASET', 'ibug')  # see src/datasets for options


class ImageWebSocketHandler(websocket.WebSocketHandler):
    handlers = {
        'filename': 'handle_return_image',
        'reconstruction': 'handle_return_reconstruction'
    }

    def __init__(self, *args, **kwargs):
        self.images = glob('{}/*.jpg'.format(FACE_DB))
        self.asf = glob('{}/*.asf'.format(FACE_DB))

        self.images.sort()
        self.asf.sort()

        # todo get from settings
        model_texture_file = '{}/pca_{}_texture_model.npy'.format(
            FILES_DIR, DATASET)
        model_shape_file = '{}/pca_{}_shape_model.npy'.format(
            FILES_DIR, DATASET)

        self.shape_model = pca.PCAModel(model_shape_file)
        self.texture_model = pca.PCAModel(model_texture_file)

        websocket.WebSocketHandler.__init__(self, *args, **kwargs)

    def __get_base64_image(self, filename):
        image = None

        with open(filename, "rb") as f:
            image = base64.b64encode(f.read())

        return image

    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("Websocket opened")

    def __return_error(self, message):
        self.write_message(json.dumps(
            {'error': message}
        ))

    def handle_return_reconstruction(self, message):
        """ Return the reconstruction of the given image """
        image_index = message['reconstruction_index']
        image_as_background = message.get('background_image', True)
        shape_components = message.get('shape_components', 58)
        shape_eigenvalues_multiplier = message.get('shape_eigenvalues')
        #image = message.get('image')
        #input_image = base64.b64decode(image)

        logger.info('using %s shape_components', shape_components)

        if DATASET == 'imm':
            image_filename = self.asf[image_index]
        else:
            image_filename = self.images[image_index]

        dst_image = reconstruction.reconstruct_shape_texture(
            DATASET,
            self.shape_model,
            self.texture_model,
            image_filename,
            shape_components,
            shape_eigenvalues_multiplier,
            image_as_background=image_as_background
        )

        _, reconstructed = cv2.imencode('.jpg', dst_image)
        reconstructed = base64.b64encode(reconstructed)

        self.write_message(json.dumps({'reconstructed': reconstructed}))

    def handle_return_image(self, message):
        filename = message['filename']
        image = self.__get_base64_image(filename)

        self.write_message(json.dumps({'image': image}))

    def on_message(self, message):
        message = json.loads(message)

        for m in message.keys():
            try:
                handler = getattr(self, self.handlers[m])
                print handler
                handler(message[m])
            except (AttributeError, KeyError) as e:
                msg = 'no handler for {}'.format(m)
                print(msg, e)
                self.__return_error(msg)
            except Exception as e:
                msg = 'no handler for {}'.format(m)
                print(msg, e)
                self.__return_error(msg)
                traceback.print_exc()

    def on_close(self):
        print("WebSocket closed")


class ApiHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        self.images = glob('{}/*.jpg'.format(FACE_DB))
        self.asf_files = glob('{}/*.asf'.format(FACE_DB))
        self.images.sort()
        self.asf_files.sort()

        web.RequestHandler.__init__(self, *args, **kwargs)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'application/vnd.api+json')


class FaceHandler(ApiHandler):
    @web.asynchronous
    def get(self, *args):
        """
        Get's all faces in the given FACE_DB and returns the url to be able
        to do requests with it in the webapplication.
        """
        data = []

        for id, filename in enumerate(self.asf_files):
            data.append({
                'type': 'faces',
                'id': id,
                'attributes': {
                    'filename': '{}/{}'.format(
                        FACE_DB_NAME, os.path.basename(self.images[id])
                    ),
                }
            })

        result = {
            'data': data
        }

        self.write(json.dumps(result))
        self.finish()

app = web.Application([
    (r'/reconstruction[\/0-9]?', ImageWebSocketHandler),
    (r'/api/v1/faces[\/0-9]?', FaceHandler),
    (r'/data/(.*)', web.StaticFileHandler, {'path': '../data'}),
    (r'/docs/(.*)', web.StaticFileHandler, {'path': 'docs/build/html'}),
])


if __name__ == '__main__':
    app.listen(8888, xheaders=True)
    ioloop = ioloop.IOLoop.instance()
    autoreload.start(ioloop)
    ioloop.start()
