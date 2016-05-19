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

import imm_points as imm
#import IMMPoints, build_feature_vectors, \
#    flatten_feature_vectors
import pca
import aam


class ImageCanvas(Widget):
    def __init__(self, **kwargs):
        super(ImageCanvas, self).__init__(**kwargs)

        # TODO: don't init with a picture, we shouldnt have that knowlegde
        self.filename_image = 'data/imm_face_db/40-1m.jpg'
        self.canvas.clear()

        with self.canvas:
            self.image = Image(pos=self.pos, size=self.size, source=self.filename_image)
            self.mesh = Mesh(mode='triangle_fan')
            self.triangles = InstructionGroup()
            self.texture = InstructionGroup()
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

    def build_texture(self, r_shape, r_texture, triangles):
        self.texture.clear()

        image_width, image_height = self.get_rendered_size()

        bary_centric_range = np.linspace(0, 1, num=20)
        texture = Texture.create(size=(image_width, image_height), colorfmt='bgr')
        buf = np.zeros((image_width, image_height, 3), dtype=np.uint8)

        for tri in triangles[:1]:
            points = r_shape[tri]
            pixels = r_texture[tri].reshape((-1, 3))

            x = points[:, 0] * image_width + self.get_image_left(image_width)
            y = (1.0 - points[:, 1]) * image_height + self.get_image_bottom(image_height)

            p1 = [x[0], y[0]]
            p2 = [x[1], y[1]]
            p3 = [x[2], y[2]]

            L = np.zeros((3, 1))

            for s_i, s in enumerate(bary_centric_range):
                for t_i, t in enumerate(bary_centric_range):
                    if s + t <= 1:
                        # build lambda's
                        L[0] = s
                        L[1] = t
                        L[2] = 1 - s - t

                        cart_x, cart_y, _ = aam.barycentric2cartesian(p1, p2, p3, L)
                        buf[s_i, t_i, :] = pixels[s_i * 20 + t_i, :]

        #buf = b''.join(map(chr, buf))
        cv2.imshow('image', buf)
        cv2.waitKey(0)

        #texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #self.texture.add(Rectangle(texture=texture, pos=(0, 100),
        #                 size=(image_width, image_height)))

        self.canvas.add(self.texture)
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
        self.mean_values_shape = kwargs['mean_values_shape']
        self.mean_values_texture = kwargs['mean_values_texture']
        self.eigenv_shape = kwargs['eigenv_shape']
        self.eigenv_texture = kwargs['eigenv_texture']
        self.triangles = kwargs['triangles']
        self.n_components = kwargs['args'].n_components
        self.multipliers = np.ones(self.eigenv_shape.shape[1])

        # slider index
        self.index = 0
        self.filename = ''

        image_slider = self.ids['image_slider']
        image_slider.max = len(self.files) - 1
        image_slider.bind(value=self.update_image)

        n_components_slider = self.ids['n_components']
        n_components_slider.value = self.n_components
        n_components_slider.bind(value=self.update_n_components)

        self.ids['image_viewer'].bind(size=self.on_resize)
        box_layout = self.ids['eigenvalues']

        self.shape_list = aam.build_shape_feature_vectors(
            self.files, imm.get_imm_points, flattened=True)

        for c in range(self.n_components):
            slider = Slider(min=-10, max=10, value=0, id=str(c))
            box_layout.add_widget(slider)
            slider.bind(value=self.update_eigenvalues)

    def reset_sliders(self):
        self.multipliers = np.ones(self.eigenv_shape.shape[1])
        box_layout = self.ids['eigenvalues']

        for c in box_layout.children:
            c.value = 0

    def update_image_viewer(self):
        self.filename = self.files[self.index].split('.')[0] + '.jpg'
        Vt_shape = np.dot(np.diag(self.multipliers), self.eigenv_shape)
        # Vt_texture = np.dot(np.diag(self.multipliers), self.eigenv_texture)

        r_shape = pca.reconstruct(
            self.shape_list[self.index], Vt_shape, self.mean_values_shape,
            n_components=self.n_components
        ).reshape((-1, 2))

        # image = cv2.imread(self.filename)
        # pixels = aam.sample_from_triangles(image, r_shape, self.triangles)
        # pixels = np.ndarray.flatten(pixels)

        # r_texture = pca.reconstruct(
        #     pixels, self.eigenv_texture, self.mean_values_texture,
        #     n_components=50000).reshape((95, -1))
        # self.ids['image_viewer'].build_texture(r_shape, r_texture, self.triangles)

        self.ids['image_viewer'].update_rect()
        self.ids['image_viewer'].update_image(self.filename)
        self.ids['image_viewer'].build_line_grid(r_shape, self.triangles)

    def on_resize(self, *args):
        self.update_image_viewer()

    def update_n_components(self, slider, index):
        self.n_components = int(index)
        self.update_image_viewer()

    def update_image(self, slider, index):
        self.index = int(index)
        self.reset_sliders()
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
            mean_values_shape=self.mean_values_shape,
            mean_values_texture=self.mean_values_texture,
            triangles=self.triangles
        )

if __name__ == '__main__':
    ReconstructApp().run()
