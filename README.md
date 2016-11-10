## Prerequisites
Run the following command
~~~~
$ make
$ source bin/activate
$ make show_reconstruction
~~~~

Will get the data needed for this tool.

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
