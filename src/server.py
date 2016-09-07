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

BASE = '../viewer/app'
FILES_DIR = '../data/'
FACE_DB_NAME = 'imm_face_db'
FACE_DB = '{}{}'.format(FILES_DIR, FACE_DB_NAME)


class ImageWebSocketHandler(websocket.WebSocketHandler):
    handlers = {
        'filename': 'handle_return_image',
        'reconstruction': 'handle_return_reconstruction'
    }

    def __init__(self, *args, **kwargs):
        self.images = glob('{}/*.jpg'.format(FACE_DB))
        self.asf = glob('{}/*.asf'.format(FACE_DB))

        # todo get from settings
        model_texture_file = '{}/pca_ibug_texture_model.npy'.format(FILES_DIR)
        model_shape_file = '{}/pca_ibug_shape_model.npy'.format(FILES_DIR)

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
        print("WebSocket opened")

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

        shape_eigenvalues_multiplier = np.asarray(
            shape_eigenvalues_multiplier, dtype=np.float32
        )

        logger.info('using %s shape_components', shape_components)

        image_filename = self.images[image_index]

        dataset_module = import_dataset_module('ibug')
        input_points = dataset_module.factory(filename=image_filename)
        input_image = input_points.get_image()

        mean_points = dataset_module.factory(points_list=self.shape_model.mean_values)
        mean_points.get_scaled_points(input_image.shape)

        # set dst image to an empty image if value is None
        if image_as_background is False:
            h, w, _ = input_image.shape
            dst_image = np.full((h, w, 3), fill_value=0, dtype=np.uint8)
        else:
            dst_image = input_image

        output_points = dataset_module.factory(
            points_list=input_points.get_points()
        )

        shape_Vt = self.shape_model.Vt
        shape_Vt = reconstruction.scale_eigenvalues(
            shape_Vt, shape_eigenvalues_multiplier
        )

        # recontruct the shape
        reconstruction.reconstruct_shape(
            output_points,
            self.shape_model,
            shape_Vt=shape_Vt  # overwrite by scaled Vt
        )

        reconstruction.reconstruct_texture(
            input_image,  # src image
            dst_image,  # dst image
            self.texture_model,
            input_points,  # shape points input
            mean_points,   # shape points mean
            output_points
        )

        output_points.draw_triangles(image=dst_image, show_points=False)

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
        web.RequestHandler.__init__(self, *args, **kwargs)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'application/vnd.api+json')


class FaceHandler(ApiHandler):
    @web.asynchronous
    def get(self, *args):
        data = []

        for id, filename in enumerate(self.asf_files):
            Points = imm.IMMPoints(filename)
            data.append({
                'type': 'faces',
                'id': id,
                'attributes': {
                    'filename': '{}/{}'.format(FACE_DB_NAME, os.path.basename(self.images[id])),
                    'shape': Points.get_scaled_points((480, 640)).tolist()
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
    app.listen(8888)
    ioloop = ioloop.IOLoop.instance()
    autoreload.start(ioloop)
    ioloop.start()
