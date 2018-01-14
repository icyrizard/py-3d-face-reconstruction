# About this tool
This tool is meant to as a pipeline to show 2D and 3D reconstructions using An
Active Appearance Model and a 2D or 3D model to rebuild the face. Face
reconstruction is a difficult subject but like with everything, if you
understand the steps, it's actually ok. This small library, can give you a
feeling what is needed to solve this problem, but also some quick-and-dirty
tricks are used. Like using dlib to solve landmark detection, instead of
estimating them using a more traditional way by as done by [[Coots|coots]].
Instead, dlib uses a sophistaticated approach to estimate 2D landmarks. This
capability is re-used to find the PCA parameters needed to rebuild a person's
face.

## Prerequisites
- Docker

# Run
- $ make
> *Note*: this will build the docker image, retrieves the imm dataset and dlib
> trained landmark file.
> $ make server
- Use https://github.com/icyrizard/py-3d-face-reconstruction-viewer to run a
viewer which uses the server to make the reconstruction insightful.

# Imm dataset
# IBUG + dlib
For the IBUG dataset we use dlib to detect landmarks. You will need the train
file for that. http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

~~~bash
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P data/
$ cd data/
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
~~~

## References
1. [imm_dataset](http://www.imm.dtu.dk/~aam/datasets/datasets.html, "Imm dataset")
2. [coots](https://www.cs.cmu.edu/~efros/courses/AP06/Papers/cootes-eccv-98.pdf, "Coots")
