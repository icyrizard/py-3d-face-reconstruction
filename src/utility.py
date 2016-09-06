import importlib


def import_dataset_module(shape_type):
    """
    Includes the right implementation for the right dataset implementation for
    the given shape type, see --help for the available options.

    Args:
        shape_type(string): Name of the python file inside the
        `src/datasets` folder.
    """
    return importlib.import_module('datasets.{}'.format(shape_type))
