FROM ubuntu
ADD requirements.txt /
RUN apt-get update && apt-get install -y python3 python3-dev python3-pip python3-virtualenv
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt
ADD foobar /foobar
ENV PYTHONPATH ${PYTHONPATH}:/
