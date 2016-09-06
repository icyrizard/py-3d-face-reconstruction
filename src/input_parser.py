import argparse

SUPPORTED_DATASETS = ['imm', 'ibug']


def get_argument_parser():
    parser = argparse.ArgumentParser(description='AAM tool')
    pca_group = parser.add_argument_group('show_reconstruction')

    pca_group.add_argument(
        '--reconstruct', action='store_true',
        help='Reconstruct one face with a given pca model'
    )

    pca_group.add_argument(
        '--test_landmarks', action='store_true',
        help='Test landmark detection of dlib using a test image'
    )

    pca_group.add_argument(
        '--image', type=str,
        help='Use this file as an image, can be used with different commands'
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
        '--shape_type', type=str, choices=SUPPORTED_DATASETS,
        help='type of shape, annotated dataset'
    )

    pca_group.add_argument(
        '--model_texture_file', type=str,
        help='pca model file that contains or is going to contain the pca texture model'
    )

    return parser
