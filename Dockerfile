FROM pytorch/pytorch:latest

ARG cuda_devices
ENV CUDA_VISIBLE_DEVICES=$cuda_devices 

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install wget git vim sudo -y
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip -qq
RUN pip3 install -U s3fs fsspec
RUN pip3 install ipdb shyaml

RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1000 user
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN chmod -R a+rwx /home/user

USER root

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt -qq

USER user

