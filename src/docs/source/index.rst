.. 3D Face Reconstruction documentation master file, created by
   sphinx-quickstart on Mon Aug  1 16:41:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to 3D Face Reconstruction's documentation!
==================================================

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents
   :name: mastertoc

   datasets
   aam
   pca
   reconstruction/reconstruction
   reconstruction/texture

!!!Work in progress!!!
======================

PCA reconstruction
==================

Principle Component Analysis is one of the most used methods in the field of statistics, it is used for dimension reduction of data and is capable of removing outliers which ultimately improves learning algorithms. In this case we use PCA for both shape and texture reconstruction. Given an image of person's face we would be able to reconstruct it using a PCA Model. The motivation for using PCA is that we can fill in missing data and remove outliers given one image of person. If for some reason the image is very cluttered, we would still be able to 'predict' how this person would look like, given all the faces we have used to train the PCA Model.

For the PCA reconstruction method has a couple of prerequisites are required. First off, the PCA Model itself. For those who are familiar with PCA know that we need to have a flattened feature vector. Both the dimensions and the content of this feature vector may be arbitrary, but have to be exactly the same from subject to subject, (i.e., there can be no difference in the number of annotated landmarks or order, landmark 1 in subject A, is landmark 1 in subject B). In this case we use it for the shape and texture. The shape feature vector contains the following data:

```
[[x_1, y_1], [x_2, y_2], ..., [x_n, y_n]]  -> (flattened) [x_1, y_1, x_2, y_2, x_n, y_n]
```

The x,y values are the location of landmarks in an image. Such a cluster of annotated locations in an image construct a shape we call Active Appearance Model(AAM)[1]. For a serie of annotated pictures with landmark location we can build mean AAM. For this particular implementation we started with supporting the Imm Dataset[^imm_dataset], for the simple reason that it is open for usage without any license agreement before hand (make sure we are correct about this). This is what we call the mean face, which is very important for the construction of the PCA Model, any PCA Model for that matter.

The texture PCA data is somewhat more difficult and depends on a given shape. In our case this given shape is the mean AAM that we have built previously. We need to add extra information to this AAM mean shape, namely a unique set of triangles that can be constructed from the set of landmarks. For this we use the Delaunay algorithm which does exactly this. The triangles help us find corresponding pixels in shape A and B. This solves the problem of pixel correspondences and is important for constructing a mean texture for the reasons explained previously about how a feature vector should look like. Pixel 1 in triangle 1 in subject A needs to correspond to exactly the same pixel (relatively) to pixel 1 in triangle 1 in subject B. This of course is sensitive to noise, but the pixels in the nose region must correspond from subject to subject, this prevents that we reconstruct an eye with a nose for instance (Note: remove this last sentence in a serious text).

References
==========

[1]: Cootes, T. F., Edwards, G. J., & Taylor, C. J. (1998, June). Active appearance models. In European conference on computer vision (pp. 484-498). Springer Berlin Heidelberg.

Links
=====

[^imm_dataset]: http://www.imm.dtu.dk/~aam/datasets/datasets.html "Imm dataset"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

