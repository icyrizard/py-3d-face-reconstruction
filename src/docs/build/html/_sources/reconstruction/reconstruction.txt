Reconstruction Module
=====================

As explained in [PCA Reconstruction](home) we need a flattened feature vector to able to build a PCA Model. This  holds for both shape and texture model. Currently we implement the independent AAM model where we keep the feature vector separate. Note that we could also choose to combine the shape and appearance in a single flattened feature vector (TODO: elaborate our choice more about this, if possible).

We use the imm dataset[^imm_dataset] for this. We first need to build the mean shape of the all the images. The dataset has a .asf file and an equally named .jpg file. The .asf file contains the locations of the landmars (normalized by the width and height of the image). In `src/imm_points.py` we find the ImmPoints class that implements all functions needed to read this file. 

[^imm_dataset]: http://www.imm.dtu.dk/~aam/datasets/datasets.html "Imm dataset"


.. automodule:: reconstruction.reconstruction
    :members:
