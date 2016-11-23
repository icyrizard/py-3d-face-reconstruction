FROM smvanveen/computer-vision:20161109143812

# install python requirements
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# extra packages:
# graphviz: for cProfiling using pycallgraph.
# libeigen3-dev: for eos: 3D morphable face model fitting library.
RUN apt-get install -y \
    graphviz \
    libeigen3-dev

WORKDIR /libs

# install dlib
RUN git clone https://github.com/davisking/dlib.git
RUN (cd dlib; python setup.py install --yes USE_AVX_INSTRUCTIONS)

# install eos (face-recosntruction, (3D Morphable Face Model fitting library)
RUN git clone --recursive https://github.com/patrikhuber/eos.git

# remove dependency on opencv 2.4.3, opencv 3.0 works fine
WORKDIR /libs/eos
RUN sed -i 's/2.4.3//g' CMakeLists.txt
RUN mkdir build
WORKDIR /libs/eos/build
RUN cmake ../ \
    -DCMAKE_INSTALL_PREFIX=/usr/local/eos \
    -DGENERATE_PYTHON_BINDINGS=on \
    -DBUILD_UTILS=on \
    -DPYTHON_EXECUTABLE=/usr/bin/python

RUN make && make install
ENV PYTHONPATH=/usr/local/eos/bin/:$PYTHONPATH

WORKDIR /src
