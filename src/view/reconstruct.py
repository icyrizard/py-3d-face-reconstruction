import kivy
kivy.require('1.0.7')

import numpy as np
import cv2

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Mesh, Line, Triangle
from kivy.graphics.texture import Texture
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics.context_instructions import Color
from functools import partial
from math import cos, sin, pi

import imm
from utils import utils
from utils.texture import fill_triangle
#import IMMPoints, build_feature_vectors, \
#    flatten_feature_vectors
import pca
import aam


class ReconstructCanvas(Widget):
    def __init__(self, **kwargs):
        super(ReconstructCanvas, self).__init__(**kwargs)
        self.canvas.clear()

        with self.canvas:
            self.texture = InstructionGroup()

    def build_texture(self, filename, Vt_texture, input_points, mean_value_points,
                    mean_values_texture, triangles, n_texture_components):
        self.texture.clear()
        #image_width, image_height = self.get_rendered_size()
        #InputPoints = imm.IMMPoints(filename=filename)
        #input_image = InputPoints.get_image()

        ##buf = np.zeros(input_image.shape, dtype=np.uint8)
        ##buf = np.repeat(np.array([255, 0, 0]), input_image.shape[0] * input_image.shape[1])

        buf = cv2.imread('data/test_data/red_500x500.png')
        h, w, c = buf.shape

        #InputPoints.get_scaled_points((input_image.shape))
        #offset_x, offset_y, w, h = InputPoints.get_bounding_box()

        #MeanPoints = imm.IMMPoints(points_list=mean_value_points)

        #utils.reconstruct_texture(
        #    input_image,  # src image
        #    buf,          # dst image
        #    Vt_texture,   # Vt
        #    InputPoints,  # shape points input
        #    MeanPoints,   # shape points mean
        #    mean_values_texture,  # mean texture
        #    triangles,    # triangles
        #    n_texture_components  # learned n_texture_components
        #)

        #ratio_x = image_width / w
        #ratio_y = image_height / h

        #buf = cv2.resize(np.asarray(buf, np.uint8), (int(image_width), int(image_height)))
        #buf = buf[offset_y: offset_y + h, offset_x: offset_x + w]
        #size = 64 * 64 * 3
        #buf = [int(x * 255 / size) for x in range(size)]
        #print buf

        #cv2.imshow('original', buf)
        #buf = cv2.flip(buf, 0)
        # buf = buf.tostring()
        #buf = b''.join(map(chr, buf))

        #texture = Texture.create(size=self.size, colorfmt='rgb')
        #texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        #pos_x = self.center[0] - self.size[0] / 2.0
        #pos_y = self.center[1] - self.size[1] / 2.0

        texture = Texture.create(size=(w, h), colorfmt="bgr")
        #arr = np.ndarray(shape=[16, 16, 3], dtype=np.uint8)

        # fill your numpy array here
        data = buf.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")

        pos_x = self.center[0] - self.size[0] / 2.0
        pos_y = self.center[1] - self.size[1] / 2.0

        self.texture.add(
            Rectangle(texture=texture, size=(w, h), pos=(pos_x, pos_y))
        )

        self.canvas.add(self.texture)
        self.canvas.ask_update()


class ImageCanvas(Widget):
    def __init__(self, **kwargs):
        super(ImageCanvas, self).__init__(**kwargs)

        # TODO: don't init with a picture, we shouldnt have that knowlegde
        self.filename_image = 'data/imm_face_db/40-1m.jpg'
        self.canvas.clear()

        with self.canvas:
            self.image = Image(pos=self.pos, size=self.size, source=self.filename_image)
            self.triangles = InstructionGroup()
            self.outline = InstructionGroup()

        self.bind(pos=self.update_rect, size=self.update_rect)

    def get_rendered_size(self):
        """
        get the rendered size of the image
        Returns:
            (tuple) width, height in pixels
        """
        return self.image.get_norm_image_size()

    def get_image_left(self, image_width):
        """
        return the location of the image, calculated from the center of the
        canvas, using the image width
        """
        return self.center[0] - image_width / 2.0

    def get_image_bottom(self, image_height):
        """
        return the location of the image, calculated from the center of the
        canvas, using the image width
        """
        return self.center[1] - image_height / 2.0

    def update_rect(self, *args):
        self.image.pos = self.pos
        self.image.size = self.size
        self.image.source = self.filename_image

    def update_image(self, filename):
        self.filename_image = filename
        self.image.source = self.filename_image
        self.canvas.ask_update()

    def build_line_grid(self, r_shape, triangles):
        self.triangles.clear()

        image_width, image_height = self.get_rendered_size()

        for tri in triangles:
            self.triangles.add(Color(0, 0, 1, 1))
            points = r_shape[tri]
            x = points[:, 0] * image_width + self.get_image_left(image_width)
            y = (1.0 - points[:, 1]) * image_height + self.get_image_bottom(image_height)

            # draw lines between three points
            self.triangles.add(Line(points=[
                x[0], y[0], x[1], y[1], x[2], y[2], x[0], y[0]])
            )

            self.triangles.add(Color(0, 1, 0, 0.5))
            self.triangles.add(Line(circle=(x[0], y[0], 3)))
            self.triangles.add(Line(circle=(x[1], y[1], 3)))
            self.triangles.add(Line(circle=(x[2], y[2], 3)))

        self.canvas.add(self.triangles)
        self.canvas.ask_update()

    def build_mesh(self, r_shape):
        vertices = []
        xy_vertices = []

        for i in range(58):
            x = r_shape[i][0] * (self.center[0] + self.image.size[0] / 2.)
            y = (1.0 - r_shape[i][1]) * self.center[1] + self.center[1] / 2.

            vertices.extend([x, y, 0, 0])
            xy_vertices.append([x, y])

        xy_vertices = np.array(xy_vertices)

        indices = []
        indices = aam.get_triangles(xy_vertices[:, 0], xy_vertices[:, 1])
        indices = np.ndarray.flatten(indices)

        self.mesh.vertices = vertices
        self.mesh.indices = indices


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(RootWidget, self).__init__(**kwargs)

        self.files = kwargs['args'].files
        self.mean_value_points = kwargs['mean_value_points']
        self.mean_values_texture = kwargs['mean_values_texture']
        self.eigenv_shape = kwargs['eigenv_shape']
        self.eigenv_texture = kwargs['eigenv_texture']
        self.triangles = kwargs['triangles']
        self.n_shape_components = kwargs['args'].n_components
        self.n_texture_components = kwargs['n_texture_components']
        self.multipliers = np.ones(self.eigenv_shape.shape[0])
        self.texture_multipliers = np.ones(self.eigenv_texture.shape[0])

        # slider index
        self.index = 0
        self.filename = ''

        self.shape_list = aam.build_shape_feature_vectors(
            self.files, imm.get_imm_points, flattened=True
        )

        # sliders
        self.add_shape_sliders()
        self.add_texture_sliders()
        self.add_image_slider()
        self.add_components_slider()
        # ======= #

        # event binders
        self.ids['image_viewer'].bind(size=self.on_resize)

    def add_components_slider(self):
        n_components_slider = self.ids['n_shape_components']
        n_components_slider.value = self.n_shape_components
        n_components_slider.bind(value=self.update_n_components)

    def add_image_slider(self):
        image_slider = self.ids['image_slider']
        image_slider.max = len(self.files) - 1
        image_slider.bind(value=self.update_image)

    def add_texture_sliders(self):
        box_layout = self.ids['texture_eigenvalues']

        for c in range(10):
            slider = Slider(min=-10, max=10, value=0, id=str(c))
            box_layout.add_widget(slider)
            slider.bind(value=self.update_texture_eigenvalues)


    def add_shape_sliders(self):
        box_layout = self.ids['eigenvalues']

        for c in range(self.n_shape_components):
            slider = Slider(min=-10, max=10, value=0, id=str(c))
            box_layout.add_widget(slider)
            slider.bind(value=self.update_eigenvalues)

    def reset_sliders(self):
        self.multipliers = np.ones(self.eigenv_shape.shape[0])
        self.texture_multipliers = np.ones(self.texture_multipliers.shape[0])

        box_layout = self.ids['eigenvalues']

        for c in box_layout.children:
            c.value = 0

        texture_eigenvalues = self.ids['texture_eigenvalues']

        for c in texture_eigenvalues.children:
            c.value = 0

    def update_image_viewer(self):
        self.filename = self.files[self.index].split('.')[0] + '.jpg'
        Vt_shape = np.dot(np.diag(self.multipliers), self.eigenv_shape)
        Vt_texture = np.dot(np.diag(self.texture_multipliers), self.eigenv_texture)

        input_points = self.shape_list[self.index]

        r_shape = pca.reconstruct(
            input_points, Vt_shape, self.mean_value_points,
            n_components=self.n_shape_components
        ).reshape((-1, 2))

        self.ids['image_viewer'].update_rect()
        self.ids['image_viewer'].update_image(self.filename)
        self.ids['image_viewer'].build_line_grid(r_shape, self.triangles)
        self.ids['reconstruct_viewer'].build_texture(
                self.files[self.index],
                self.eigenv_texture,
                input_points,
                self.mean_value_points,
                self.mean_values_texture,
                self.triangles,
                self.n_texture_components
        )

    def on_resize(self, *args):
        self.update_image_viewer()

    def update_n_components(self, slider, index):
        self.n_shape_components = int(index)
        self.update_image_viewer()

    def update_image(self, slider, index):
        self.index = int(index)
        self.reset_sliders()
        self.update_image_viewer()

    def update_texture_eigenvalues(self, slider, value):
        multiplier_index = int(slider.id)
        self.texture_multipliers[multiplier_index] = value
        self.update_image_viewer()

    def update_eigenvalues(self, slider, value):
        multiplier_index = int(slider.id)
        self.multipliers[multiplier_index] = value
        self.update_image_viewer()


class ReconstructApp(App):
    kv_directory = 'src/view/templates'

    def __init__(self, **kwargs):
        super(ReconstructApp, self).__init__(**kwargs)

    def set_values(self, **kwargs):
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])

    def build(self):
        return RootWidget(
            args=self.args,
            eigenv_shape=self.eigenv_shape,
            eigenv_texture=self.eigenv_texture,
            mean_value_points=self.mean_value_points,
            n_shape_components=self.n_shape_components,
            mean_values_texture=self.mean_values_texture,
            n_texture_components=self.n_texture_components,
            triangles=self.triangles
        )

if __name__ == '__main__':
    ReconstructApp().run()
