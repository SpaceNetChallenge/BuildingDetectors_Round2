FROM ubuntu:16.04

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        software-properties-common

# Install Java 
RUN add-apt-repository ppa:webupd8team/java -y && \
        apt-get update && \
        echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
        apt-get install -y oracle-java8-installer && \
        apt-get clean

RUN mkdir wleite

COPY . /opt/wleite/

WORKDIR /opt/wleite


