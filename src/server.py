import json
import base64
from glob import glob

from tornado import websocket, web, ioloop

import imm_points as imm

BASE = '../viewer/app'
FILES_DIR = '../data/'
FACE_DB = '{}{}'.format(FILES_DIR, 'imm_face_db')


class ImageWebSocketHandler(websocket.WebSocketHandler):
    handlers = {
        'filename': 'handle_return_image',
        'reconstruction_index': 'handle_return_reconstruction'
    }

    def __init__(self, *args, **kwargs):
        self.images = glob('{}/*.jpg'.format(FACE_DB))
        self.asf = glob('{}/*.asf'.format(FACE_DB))

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
        #self.write_message(
        #    json.dumps({
        #        'n_images': len(self.images),
        #        'image': self.__get_base64_image(self.images[0])
        #    }
        #))

        #self.write_message(json.dumps({'n_images': len(self.images)}))

    def __return_error(self, message):
        self.write_message(json.dumps(
            {'error': message}
        ))

    def handle_return_reconstruction(self, message):
        image_index = message['reconstruction_index']
        filename = self.images[image_index]
        image = self.__get_base64_image(filename)

        self.write_message(json.dumps({'reconstructed': image}))

    def handle_return_image(self, message):
        filename = message['filename']
        #filename = self.images[image_index]
        image = self.__get_base64_image(filename)

        self.write_message(json.dumps({'image': image}))

    def on_message(self, message):
        message = json.loads(message)

        for m in message.keys():
            try:
                handler = getattr(self, self.handlers[m])
            except (AttributeError, KeyError) as e:
                msg = 'no handler for {}'.format(m)
                print(msg, e)
                self.__return_error(msg)

            handler(message)

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
                    'filename': filename,
                    'shape': Points.get_scaled_points(shape=(480, 640)).tolist()
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
])


if __name__ == '__main__':
    app.listen(8888)
    ioloop.IOLoop.instance().start()
