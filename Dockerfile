FROM smvanveen/computer-vision:20161109143812
RUN git clone https://github.com/davisking/dlib.git
RUN (cd dlib; python setup.py install --yes USE_AVX_INSTRUCTIONS)
RUN apt-get install graphviz -y
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
WORKDIR /src
