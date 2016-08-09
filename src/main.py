#!/usr/local/bin/python
# python std
import argparse
import importlib
import copy

# installed packages
import cv2

# local imports
import pca
import aam
import numpy as np
# import imm

from reconstruction import reconstruction

from settings import logger


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')
    pca_group = parser.add_argument_group('show_reconstruction')

    pca_group.add_argument(
        '--reconstruct', action='store_true',
        help='Reconstruct one face with a given pca model'
    )

    pca_group.add_argument(
        '--generate_call_graph', action='store_true',
        help='Generate call graph from the reconstruction'
    )

    pca_group.add_argument(
        '--save_pca_shape', action='store_true',
        help='save the pca shape model'
    )

    pca_group.add_argument(
        '--save_pca_texture', action='store_true',
        help='save the pca texture model'
    )

    pca_group.add_argument(
        '--files', nargs='+', help='files to process'
    )

    pca_group.add_argument(
        '--n_components', default=10, type=int,
        help='number of principle components to keep and are able to manipulate'
    )

    pca_group.add_argument(
        '--model_shape_file', type=str,
        help='pca model file that contains or is going to contain the pca shape model'
    )

    pca_group.add_argument(
        '--shape_type', type=str, choices=['imm'],
        help='type of shape, annotated dataset'
    )

    pca_group.add_argument(
        '--model_texture_file', type=str,
        help='pca model file that contains or is going to contain the pca texture model'
    )

    return parser


def import_dataset_module(shape_type):
    """
    Includes the right implementation for the right dataset implementation for
    the given shape type, see --help for the available options.

    Args:
        shape_type(string): Name of the python file inside the
        `src/datasets` folder.
    """
    return importlib.import_module('datasets.{}'.format(shape_type))


def save_pca_model_texture(args):
    """
    save the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is saved in the following way:

        np.load(filename, np.assary([Vt, mean_values])

        And accessed by:

        Vtm = np.load(args.model_file_texture)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

    """
    assert args.files, '--files should be given'
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.shape_type, '--shape_type the type of dataset, see datasets module'

    dataset_module = import_dataset_module(args.shape_type)
    shape_model = pca.PCAModel(args.model_shape_file)
    mean_points = dataset_module.IMMPoints(points_list=shape_model.mean_values)

    textures = aam.build_texture_feature_vectors(
        args.files,
        dataset_module.get_imm_image_with_landmarks,  # function
        mean_points,
        shape_model.triangles
    )

    mean_texture = aam.get_mean(textures)
    _, s, Vt, n_components = pca.pca(textures, mean_texture)

    pca.save(Vt, s, n_components, mean_texture, shape_model.triangles, args.model_texture_file)

    logger.info('texture pca model saved in %s', args.model_texture_file)


def save_pca_model_shape(args):
    """
    save the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is saved in the following way:

        np.load(filename, np.assary([Vt, mean_values])

        And accessed by:

        Vtm = np.load(args.model_shape_file)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

    """
    assert args.files, '--files should be given'
    assert args.model_shape_file, '--model_shape_file needs to be provided to save the pca model'
    assert args.shape_type, '--shape_type the type of dataset, see datasets module'

    dataset_module = import_dataset_module(args.shape_type)

    points = aam.build_shape_feature_vectors(
        args.files, dataset_module.get_imm_points, flattened=True
    )

    mean_values = aam.get_mean(points)

    _, s, Vt, n_components = pca.pca(points, mean_values)

    mean_xy = mean_values.reshape((-1, 2))
    triangles = aam.get_triangles(mean_xy[:, 0], mean_xy[:, 1])

    pca.save(Vt, s, n_components, mean_values, triangles, args.model_shape_file)
    logger.info('shape pca model saved in %s', args.model_shape_file + '_shape')


def generate_call_graph(args):
    """Performance debug function, will be (re)moved later. """
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.shape_type, '--shape_type the type of dataset, see datasets module'

    dataset_module = import_dataset_module(args.shape_type)

    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    graphviz = GraphvizOutput(output_file='filter_none.png')

    with PyCallGraph(output=graphviz):
        shape_model = pca.PCAModel(args.model_shape_file)
        texture_model = pca.PCAModel(args.model_texture_file)

        input_points = dataset_module.IMMPoints(filename='data/imm_face_db/40-3m.asf')
        input_image = input_points.get_image()

        mean_points = dataset_module.IMMPoints(points_list=shape_model.mean_values)
        mean_points.get_scaled_points(input_image.shape)

        reconstruction.reconstruct_texture(
            input_image,  # src image
            input_image,  # dst image
            texture_model,
            input_points,  # shape points input
            mean_points,   # shape points mean
        )

def show_reconstruction(args):
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.shape_type, '--shape_type the type of dataset, see datasets module'

    dataset_module = import_dataset_module(args.shape_type)

    shape_model = pca.PCAModel(args.model_shape_file)
    texture_model = pca.PCAModel(args.model_texture_file)

    input_points = dataset_module.IMMPoints(
        filename='data/imm_face_db/01-1m.asf'
    )

    input_image = input_points.get_image()

    mean_points = dataset_module.IMMPoints(points_list=shape_model.mean_values)
    mean_points.get_scaled_points(input_image.shape)

    n_components = 58
    count = 0
    shape_eigenvalues_multiplier = np.ones(15, dtype=np.float32)

    while True:
        input_image_copy = input_image.copy()
        input_points_copy = copy.deepcopy(input_points)

        output_points = dataset_module.IMMPoints(
            points_list=input_points.get_points()
        )

        # scale by scaling the Vt matrix
        shape_Vt = shape_model.Vt
        shape_Vt = reconstruction.scale_eigenvalues(
            shape_Vt, shape_eigenvalues_multiplier
        )

        # recontruct the shape
        reconstruction.reconstruct_shape(
            output_points,
            shape_model,
            shape_Vt=shape_Vt,  # overwrite by scaled Vt
            n_components=n_components - count
        )

        # use the new shape ane mean points to reconstruct
        reconstruction.reconstruct_texture(
            input_image_copy,  # src image
            input_image_copy,  # dst image
            texture_model,
            input_points_copy,  # shape points input
            mean_points,    # shape points mean
            output_points
        )

        output_points.get_scaled_points(input_image.shape)
        output_points.draw_triangles(image=input_image_copy, show_points=False)

        dst = reconstruction.get_texture(
            mean_points, texture_model.mean_values
        )

        #count += 2
        shape_eigenvalues_multiplier[0] += 0.1

    cv2.destroyAllWindows()


def main():
    """main"""
    parser = add_parser_options()
    args = parser.parse_args()

    if args.save_pca_shape:
        save_pca_model_shape(args)
    elif args.save_pca_texture:
        save_pca_model_texture(args)
    elif args.reconstruct:
        show_reconstruction(args)
    elif args.generate_call_graph:
        generate_call_graph(args)

if __name__ == '__main__':
    main()
