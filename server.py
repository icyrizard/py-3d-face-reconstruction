import json
import base64

from glob import glob

from tornado import websocket, web, ioloop

BASE = 'viewer/app'
FILES_DIR = 'data/'
FACE_DB = '{}{}'.format(FILES_DIR, 'imm_face_db')


class SocketHandler(websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.images = glob('{}/*.jpg'.format(FACE_DB))
        websocket.WebSocketHandler.__init__(self, *args, **kwargs)

    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")
        self.write_message(json.dumps({'n_images': len(self.images)}))

    def on_message(self, message):
        message = json.loads(message)
        image_index = message['image_index']
        filename = self.images[image_index]

        image = None

        with open(filename, "rb") as f:
            image = base64.b64encode(f.read())

        self.write_message(json.dumps({'image': image}))

    def on_close(self):
        print("WebSocket closed")


#class ApiHandler(web.RequestHandler):
#    def __init__(self, *args, **kwargs):
#        self.images = glob('{}/*.jpg'.format(FACE_DB))
#        websocket.RequestHandler.__init__(self, *args, **kwargs)
#
#    @web.asynchronous
#    def get(self, *args):
#        self.finish()
#
#        if self.get_argument('images'):
#            self.get_images()
#
#        #id = self.get_argument("id")
#        #value = self.get_argument("value")
#        #data = {"id": id, "value" : value}
#        #data = json.dumps(data)
#
#        #self.write_message(data)
#
#    @web.asynchronous
#    def post(self):
#        pass

app = web.Application([
    (r'/reconstruction', SocketHandler),
])


if __name__ == '__main__':
    app.listen(8888)
    ioloop.IOLoop.instance().start()
