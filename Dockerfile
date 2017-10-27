FROM smvanveen/computer-vision:20161109143812

# install python requirements
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687 -y
RUN apt-get update -y
RUN apt-get install libsuitesparse-dev -y

# extra packages:
# graphviz: for cProfiling using pycallgraph.
# libeigen3-dev: for eos: 3D morphable face model fitting library.
RUN apt-get install -y \
    graphviz \
    libeigen3-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev


WORKDIR /libs

# install dlib
RUN git clone https://github.com/davisking/dlib.git
RUN (cd dlib; python setup.py install --yes USE_AVX_INSTRUCTIONS)

RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN (cd ceres-solver; make -j3)
RUN (cd ceres-solver; make install)

# install eos (face-recosntruction, (3D Morphable Face Model fitting library)
RUN git clone --recursive \
    https://github.com/patrikhuber/eos.git

# remove dependency on opencv 2.4.3, opencv 3.0 works fine
WORKDIR /libs/eos
RUN git checkout devel

# needed for master branch
#RUN sed -i 's/2.4.3//g' CMakeLists.txt

RUN mkdir build
WORKDIR /libs/eos/build
RUN cmake ../ \
    -DCMAKE_INSTALL_PREFIX=/usr/local/eos \
    -DEOS_GENERATE_PYTHON_BINDINGS=on \
    -DEOS_BUILD_CERES_EXAMPLE=on \
    -DEOS_BUILD_UTILS=on \
    -DEOS_BUILD_EXAMPLES=on

RUN make && make install

ENV PYTHONPATH=/usr/local/eos/bin/:$PYTHONPATH

WORKDIR /libs
RUN git clone https://github.com/pybind/pybind11.git
RUN (cd pybind11; mkdir build; cd build; cmake -DPYBIND11_PYTHON_VERSION=2.7 ..);
RUN (cd pybind11/build; make -j4 && make install);

#TODO, remove the tmp libs folder in production?

WORKDIR /src
