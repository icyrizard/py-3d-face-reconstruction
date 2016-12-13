import numpy as np
from bfm_to_obj import load_bfm_model, save_to_obj, load_bfm_attributes


def gen_random_shape(mean, Vt, eigenvalues, alpha):
    # generate influences eigenvalues, with a random sample
    # of a normal distribution.
    infl_eigenvalues = alpha * eigenvalues
    random_face = mean + np.dot(Vt, infl_eigenvalues)

    return random_face


def gen_random_texture(mean, Vt, eigenvalues, betha):
    # generate influences eigenvalues, with a random sample
    # of a normal distribution.
    infl_eigenvalues = betha * eigenvalues
    random_face = mean + np.dot(Vt, infl_eigenvalues)

    return random_face


def gender_face_tex(attr, betha, n=5):
    """
    Influence gender.

    Args:
        attr(attributes model) - see BFM documentation
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape
        n(integer) - amount of influence

    Notes:
        Higher n is more male.
        Lower n is more female.
        Minus **n** is allowed.
    """
    print 'tex', len(betha)
    return betha + n * attr['gender_tex'][:len(betha)]


def age_face_tex(attr, betha, n=50):
    """
    Influence age.

    Args:
        attr(attributes model) - see BFM documentation.
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape.
        n(integer) - amount of influence.

    Notes:
        Higher n is older.
        Lower n is younger.
        Minus **n** is allowed.
    """
    return betha + n * attr['age_tex'][:199]


def weight_face_tex(attr, betha, n=30):
    """
    Influence weight.

    Args:
        attr(attributes model) - see BFM documentation.
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape.
        n(integer) - amount of influence

    Notes:
        Higher n is fatter.
        Lower n is thinner.
        Minus **n** is allowed.
    """
    return betha + n * attr['weight_tex'][:199]


def gender_face_shape(attr, alpha, n=5):
    """
    Influence gender.

    Args:
        attr(attributes model) - see BFM documentation
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape
        n(integer) - amount of influence

    Notes:
        Higher n is more male.
        Lower n is more female.
        Minus **n** is allowed.
    """
    return alpha + n * attr['gender_shape'][:199]


def age_face_shape(attr, alpha, n=50):
    """
    Influence age.

    Args:
        attr(attributes model) - see BFM documentation.
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape.
        n(integer) - amount of influence.

    Notes:
        Higher n is older.
        Lower n is younger.
        Minus **n** is allowed.
    """
    return alpha + n * attr['age_shape'][:199]


def weight_face_shape(attr, alpha, n=30):
    """
    Influence weight.

    Args:
        attr(attributes model) - see BFM documentation.
        alpha(ndarray) - random range of floats generates with np.randn for
            example with dim of the eigenvalues of the shape.
        n(integer) - amount of influence

    Notes:
        Higher n is fatter.
        Lower n is thinner.
        Minus **n** is allowed.
    """
    return alpha + n * attr['weight_shape'][:199]


def gen_random_shape_coef(attr, dim):
    alpha = np.random.randn(dim, 1)
    alpha = age_face_shape(attr, alpha, 1)
    alpha = gender_face_shape(attr, alpha, -5)
    alpha = weight_face_shape(attr, alpha, -10)

    return alpha


def gen_random_tex_coef(attr, dim):
    betha = np.random.randn(dim, 1)
    betha = age_face_tex(attr, betha, 1)
    betha = gender_face_tex(attr, betha, -5)
    betha = weight_face_tex(attr, betha, -10)

    return betha


def main():
    dmm = load_bfm_model()
    attr = load_bfm_attributes()

    Vt_shape = dmm['shapePC']
    mean_shape = dmm['shapeMU']
    eigenv_shape = dmm['shapeEV']

    alpha = gen_random_shape_coef(attr, eigenv_shape.shape[0])

    random_face_shape = gen_random_shape(
        mean_shape, Vt_shape, eigenv_shape, alpha
    )

    Vt_tex = dmm['texPC']
    mean_tex = dmm['texMU']
    eigenv_tex = dmm['texEV']

    betha = gen_random_tex_coef(attr, eigenv_tex.shape[0])

    random_face_texture = gen_random_texture(
        mean_tex, Vt_tex, eigenv_tex, betha
    )

    save_to_obj(
        random_face_shape,
        random_face_texture,
        dmm['tl'], 'random_out_1.obj'
    )


if __name__ == '__main__':
    main()
