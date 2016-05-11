import kivy
kivy.require('1.0.7')

import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Mesh, Line, Triangle
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

        self.filename_image = 'data/imm_face_db/40-1m.jpg'
        self.canvas.clear()

        with self.canvas:
            self.image = Image(pos=self.pos, size=self.size, source=self.filename_image)
            self.mesh = Mesh(mode='triangle_fan')
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

    def build_line_grid(self, reconstructed):
        self.triangles.clear()

        image_width, image_height = self.get_rendered_size()
        triangles = aam.get_triangles(reconstructed[:, 0], reconstructed[:, 1])

        for tri in triangles:
            self.triangles.add(Color(0, 0, 1, 1))
            points = reconstructed[tri]
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

    def build_mesh(self, reconstructed):
        vertices = []
        xy_vertices = []

        for i in range(58):
            x = reconstructed[i][0] * (self.center[0] + self.image.size[0] / 2.)
            y = (1.0 - reconstructed[i][1]) * self.center[1] + self.center[1] / 2.

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

        self.images = kwargs['args'].asf
        self.mean_values = kwargs['mean_values']
        self.Vt = kwargs['eigen_vectors']
        self.n_components = kwargs['args'].n_components
        self.multipliers = np.ones(self.Vt.shape[1])

        # slider index
        self.index = 0
        self.filename = ''

        image_slider = self.ids['image_slider']
        image_slider.max = len(self.images) - 1
        image_slider.bind(value=self.update_image)

        n_components_slider = self.ids['n_components']
        n_components_slider.value = self.n_components
        n_components_slider.bind(value=self.update_n_components)

        self.ids['image_viewer'].bind(size=self.on_resize)
        box_layout = self.ids['eigenvalues']

        self.landmark_list = aam.build_feature_vectors(self.images,
                imm.get_imm_landmarks, flattened=True)

        for c in range(self.n_components):
            slider = Slider(min=-10, max=10, value=0, id=str(c))
            box_layout.add_widget(slider)
            slider.bind(value=self.update_eigenvalues)

    def reset_sliders(self):
        self.multipliers = np.ones(self.Vt.shape[1])
        box_layout = self.ids['eigenvalues']

        for c in box_layout.children:
            c.value = 0

    def update_image_viewer(self):
        self.filename = self.images[self.index].split('.')[0] + '.jpg'
        Vt = np.dot(np.diag(self.multipliers), self.Vt)

        reconstruction = pca.reconstruct(
            self.landmark_list[self.index], Vt, self.mean_values,
            n_components=self.n_components
        )

        reconstruction = reconstruction.reshape((-1, 2))

        self.ids['image_viewer'].update_rect()
        self.ids['image_viewer'].update_image(self.filename)
        self.ids['image_viewer'].build_line_grid(reconstruction)

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
        self.eigen_vectors = kwargs['eigen_vectors']
        self.mean_values = kwargs['mean_values']
        self.args = kwargs['args']

        super(ReconstructApp, self).__init__(**kwargs)

    def build(self):
        return RootWidget(
            args=self.args, eigen_vectors=self.eigen_vectors,
            mean_values=self.mean_values
        )

if __name__ == '__main__':
    ReconstructApp().run()
