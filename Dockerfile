FROM smvanveen/computer-vision:20160120173546
RUN pip install dlib
WORKDIR /src

#COPY requirements.txt /tmp
#RUN python --version
