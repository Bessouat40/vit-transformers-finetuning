FROM ubuntu:bionic

RUN apt-get update
RUN apt-get install curl -y
RUN apt-get install python3 python3-pip -y
RUN pip3 install --upgrade pip

WORKDIR /

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY ./src /

CMD python3 onnx_inferences.py